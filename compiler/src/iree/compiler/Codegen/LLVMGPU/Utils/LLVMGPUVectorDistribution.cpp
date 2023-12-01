// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

namespace {

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, RewriterBase &rewriter,
                     LayoutProvider *provider, VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), provider(provider), rewriter(rewriter) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;
  }

  void distribute() {
    root->walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);

      if (provider->specializedDistribution(op)) {
        return;
      }

      TypeSwitch<Operation *, void>(op)
          .Case<vector::ContractionOp>([&](auto contractOp) {
            distributeContractions(contractOp);
            return;
          })
          .Case<vector::TransferReadOp>([&](auto transferReadOp) {
            distributeTransferReads(transferReadOp);
            return;
          })
          .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {
            distributeTransferWrites(transferWriteOp);
            return;
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            distributeConstants(constantOp);
            return;
          })
          .Default([&](auto op) {});
    });
  }

private:
  void distributeTransferWrites(vector::TransferWriteOp transferWriteOp);

  void distributeTransferReads(vector::TransferReadOp transferReadOp);

  void distributeContractions(vector::ContractionOp contractionOp) {}

  void distributeConstants(arith::ConstantOp constantOp) {}

  SmallVector<Value> getThreadIds() {
    return {rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::x),
            rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::y),
            rewriter.create<gpu::ThreadIdOp>(root.getLoc(), gpu::Dimension::z)};
  }

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider *provider;
  RewriterBase &rewriter;
  IRMapping simdToSimt;
}; // namespace

} // namespace

/// Iterate over all indices of the given shape, and call the callback with the
/// indices. The callback returns a stride to move by.
static void
iterateOverShape(ArrayRef<int64_t> shape,
                 std::function<int64_t(ArrayRef<int64_t>)> callback) {
  SmallVector<int64_t> indices(shape.size(), 0);
  while (true) {
    int64_t stride = callback(indices);
    // Move by the given stride.
    int64_t remaining = stride;
    for (int64_t i = indices.size() - 1; i >= 0; --i) {
      indices[i] += remaining;
      remaining = indices[i] / shape[i];
      indices[i] %= shape[i];

      if (remaining == 0)
        break;
    }

    if (remaining != 0)
      break;
  }
}

void VectorDistribution::distributeTransferReads(
    vector::TransferReadOp transferReadOp) {
  TypedValue<VectorType> result = transferReadOp.getResult();
  SmallVector<Value> gpuThreadIds = getThreadIds();
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(result);
  for (auto it : distributedShape) {
    llvm::errs() << it << "x";
  }
  llvm::errs() << "\n";
  analysis.getLayout<Attribute>(result).dump();

  // Iterate over the given shape with a stride of 1.
  iterateOverShape(distributedShape, [&](ArrayRef<int64_t> iterate) -> int64_t {
    SmallVector<AffineMap> indexMaps =
        provider->getDistributedIndex(result, iterate);
    // Get the indices for the load.
    SmallVector<Value> loadIndices;
    // Iterate on the indices in the permutation order.
    for (auto index : transferReadOp.getPermutationMap().getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(index);
      if (!dimExpr) {
        emitError(transferReadOp.getLoc())
            << "Non-dim expr in permutation map\n";
        continue;
      }

      int64_t dim = dimExpr.getPosition();
      llvm::errs() << indexMaps[dim] << "\n";
    }

    // Store from loadIndices with a width provided by the provider.
    int64_t storeWidth = provider->getStoreWidth(result, iterate);

    return storeWidth;
  });
}

void VectorDistribution::distributeTransferWrites(
    vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();
  SmallVector<Value> gpuThreadIds = getThreadIds();
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(vector);

  for (auto it : distributedShape) {
    llvm::errs() << it << "x";
  }
  llvm::errs() << "\n";
  analysis.getLayout<Attribute>(vector).dump();

  // Iterate over the given shape with a stride of 1.
  iterateOverShape(distributedShape, [&](ArrayRef<int64_t> iterate) -> int64_t {
    SmallVector<AffineMap> indexMaps =
        provider->getDistributedIndex(vector, iterate);
    // Get the indices for the load.
    SmallVector<Value> loadIndices;
    // Iterate on the indices in the permutation order.
    for (auto index : transferWriteOp.getPermutationMap().getResults()) {
      auto dimExpr = dyn_cast<AffineDimExpr>(index);
      if (!dimExpr) {
        emitError(transferWriteOp.getLoc())
            << "Non-dim expr in permutation map\n";
        continue;
      }

      int64_t dim = dimExpr.getPosition();
      llvm::errs() << indexMaps[dim] << "\n";
    }

    // Store from loadIndices with a width provided by the provider.
    int64_t storeWidth = provider->getStoreWidth(vector, iterate);

    return storeWidth;
  });
}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
