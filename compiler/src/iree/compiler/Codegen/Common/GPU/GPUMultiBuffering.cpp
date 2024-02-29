// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/PassDetail.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/IR/Dominance.h"

namespace mlir::iree_compiler {

namespace {
/// Filter to decide which contraction ops need allocations.
static bool writeToWorkgroupMemory(linalg::GenericOp op) {
  // If write to workgroup memory.
  if (!hasMarker(op, getCopyToWorkgroupMemoryMarker()))
    return false;
  for (auto dpsInit : op.getDpsInits()) {
    if (auto dpsInitMemrefType =
            llvm::dyn_cast<MemRefType>(dpsInit.getType())) {
      if (hasSharedMemoryAddressSpace(dpsInitMemrefType))
        return true;
    }
  }
  return false;
}

struct GPUMultiBufferingPass
    : public GPUMultiBufferingBase<GPUMultiBufferingPass> {
  GPUMultiBufferingPass(unsigned numBuffers) : numBuffers(numBuffers) {}

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();

    // First hoist all shared memory allocations to the entry block of the
    // function. We can see memref.alloc in loops after bufferizing scf.forall
    // with promoted shared memory usage inside.

    SmallVector<memref::AllocOp> allocs;
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      if (hasSharedMemoryAddressSpace(allocOp.getType()))
        allocs.push_back(allocOp);
    });

    assert(funcOp.getBlocks().size() == 1);
    for (memref::AllocOp allocOp : allocs) {
      if (allocOp->getParentOp() != funcOp)
        allocOp->moveBefore(&*funcOp.begin()->begin());
    }

    // Then perform multibuffering transformations.

    allocs.clear();
    // Collect all the alloc operations.
    funcOp.walk([&](memref::AllocOp allocOp) {
      // Skip allocations not used in a loop.
      for (Operation *user : allocOp->getUsers()) {
        auto loop = user->getParentOfType<scf::ForOp>();
        if (!loop)
          return WalkResult::advance();
      }
      allocs.push_back(allocOp);
      return WalkResult::advance();
    });
    // Apply multi-buffering to all of them.
    for (memref::AllocOp alloc : allocs) {
      if (failed(memref::multiBuffer(alloc, numBuffers))) {
        // Error out and stop if any buffer cannot be multi buffered, as future
        // software pipelining transformations will assume this happened.
        alloc.emitOpError("cannot be multi-buffered");
        return signalPassFailure();
      }
    }

    // Add sync for genericOps with writes to shared memory.
    SmallVector<linalg::GenericOp> opsToSync;
    funcOp.walk([&](linalg::GenericOp op) {
      if (writeToWorkgroupMemory(op))
        opsToSync.push_back(op);
    });
    for (linalg::GenericOp genericOp : opsToSync) {
      OpBuilder builder(genericOp);
      // Insert Barrier before write to wait for memory allocs.
      builder.setInsertionPoint(genericOp);
      builder.create<gpu::BarrierOp>(genericOp->getLoc());
      // Insert Barrier after write to ensure it's ready.
      builder.setInsertionPointAfter(genericOp);
      builder.create<gpu::BarrierOp>(genericOp->getLoc());
    }
  }

private:
  unsigned numBuffers;
};
} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createGPUMultiBuffering(unsigned numBuffers) {
  return std::make_unique<GPUMultiBufferingPass>(numBuffers);
}

} // namespace mlir::iree_compiler
