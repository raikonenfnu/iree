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

  // Get GPU Thread IDs.
  SmallVector<Value> gpuThreadIds = {
      rewriter.create<gpu::ThreadIdOp>(transferReadOp.getLoc(),
                                       gpu::Dimension::x),
      rewriter.create<gpu::ThreadIdOp>(transferReadOp.getLoc(),
                                       gpu::Dimension::y),
      rewriter.create<gpu::ThreadIdOp>(transferReadOp.getLoc(),
                                       gpu::Dimension::z)};

  // Ask the provider for the distributed shape.
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(result);
  // Ask the provider for distribution maps.
  SmallVector<AffineMap> simdIndices =
      provider->getSIMDIndexFromDistributedIndex(result);
  // Iterate over the given shape with a stride of 1.
  iterateOverShape(distributedShape, [&](ArrayRef<int64_t> iterate) -> int64_t {
    // Get the indices for the load.
    SmallVector<Value> loadIndices;
    for (auto [index, indice] : enumerate(iterate)) {
      Value indiceVal = rewriter.create<arith::ConstantIndexOp>(
          transferReadOp.getLoc(), indice);

      // (indice)[threadx, thready, threadz]
      SmallVector<Value> applyOperands;
      applyOperands.push_back(indiceVal);
      for (auto gpuThreadId : gpuThreadIds) {
        applyOperands.push_back(gpuThreadId);
      }

      Value loadIndex = rewriter.create<affine::AffineApplyOp>(
          transferReadOp.getLoc(), applyOperands);

      loadIndices.push_back(loadIndex);
    }

    // Load from loadIndices with a width provided by the provider.
    int64_t loadWidth = provider->getLoadWidth(result, iterate);

    return loadWidth;
  });
}

void VectorDistribution::distributeTransferWrites(
    vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();

  // Get GPU Thread IDs.
  SmallVector<Value> gpuThreadIds = {
      rewriter.create<gpu::ThreadIdOp>(transferWriteOp.getLoc(),
                                       gpu::Dimension::x),
      rewriter.create<gpu::ThreadIdOp>(transferWriteOp.getLoc(),
                                       gpu::Dimension::y),
      rewriter.create<gpu::ThreadIdOp>(transferWriteOp.getLoc(),
                                       gpu::Dimension::z)};

  // Ask the provider for the distributed shape.
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(vector);
  // Ask the provider for distribution maps.
  SmallVector<AffineMap> simdIndices =
      provider->getSIMDIndexFromDistributedIndex(vector);
  // Iterate over the given shape with a stride of 1.
  iterateOverShape(distributedShape, [&](ArrayRef<int64_t> iterate) -> int64_t {
    // Get the indices for the load.
    SmallVector<Value> loadIndices;
    for (auto [index, indice] : enumerate(iterate)) {
      Value indiceVal = rewriter.create<arith::ConstantIndexOp>(
          transferWriteOp.getLoc(), indice);

      // (indice)[threadx, thready, threadz]
      SmallVector<Value> applyOperands;
      applyOperands.push_back(indiceVal);
      for (auto gpuThreadId : gpuThreadIds) {
        applyOperands.push_back(gpuThreadId);
      }

      Value loadIndex = rewriter.create<affine::AffineApplyOp>(
          transferWriteOp.getLoc(), applyOperands);

      loadIndices.push_back(loadIndex);
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
