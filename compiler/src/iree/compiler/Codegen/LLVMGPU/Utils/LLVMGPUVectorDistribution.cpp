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

  void distributeContractions(vector::ContractionOp contractionOp);

  void distributeConstants(arith::ConstantOp constantOp) {}

  SmallVector<Value> getIndices(LayoutAttr &layout, LayoutAttr::Iterator &iterator,
                                SmallVector<Value> indices, AffineMap permutationMap,
                                Location loc, OpBuilder &rewriter);

  // We delinearize threadIdx to 2 thread indices.
  Value getThreadIds(PerDimLayoutAttr &layout, Value lastShape,
                     DenseMap<int64_t, Value> &laneIds, int64_t key, LayoutDimension dim) {
    Location loc = root.getLoc();
    Value threadX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    auto possibleShape = layout.getShape(dim);
    if (possibleShape) {
      switch (dim) {
        case LayoutDimension::LANEX: {
          Value shapeValue = rewriter.create<arith::ConstantIndexOp>(loc, *possibleShape);
          laneIds[key] = rewriter.create<arith::RemUIOp>(loc, threadX, shapeValue);
          return shapeValue;
        }
        case LayoutDimension::LANEY:
          laneIds[key] = rewriter.create<arith::DivUIOp>(loc, threadX, lastShape);
          break;
        default:
          break;
      }
    }
    return lastShape;
  }

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider *provider;
  RewriterBase &rewriter;
  IRMapping simdToSimt;
}; // namespace

} // namespace

static SmallVector<Value> handlePermutations(SmallVector<Value> &indices,
                                             AffineMap permutationMap) {
  SmallVector<Value> newIndices{indices.begin(), indices.end()};
  int laneDim = 0;
  for (AffineExpr expr : permutationMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr)
      continue;
    unsigned pos = dimExpr.getPosition();
    newIndices[pos] = indices[laneDim++];
  }
  return newIndices;
}

SmallVector<Value> VectorDistribution::getIndices(LayoutAttr &layout, LayoutAttr::Iterator &iterator,
                                                  SmallVector<Value> indices, AffineMap permutationMap,
                                                  Location loc, OpBuilder &rewriter) {
  SmallVector<Value> simdIndices;
  Value shape;
  DenseMap<int64_t, Value> laneIds;
  for (LayoutDimension dim : {LayoutDimension::LANEX, LayoutDimension::LANEY}) {
    for (auto pair : llvm::enumerate(layout.getLayouts())) {
      PerDimLayoutAttr perDimLayout = pair.value();
      shape = getThreadIds(perDimLayout, shape, laneIds, pair.index(), dim);
    }
  }

  int i{0};
  for (auto pair : llvm::zip(layout.getLayouts(), iterator.states)) {
    PerDimLayoutAttr perDimLayout = std::get<0>(pair);
    PerDimLayoutAttr::Iterator iterator = std::get<1>(pair);
    Value index = rewriter.create<affine::AffineApplyOp>(loc, perDimLayout.computeSIMDIndex(iterator), laneIds[i]);
    index = rewriter.create<arith::AddIOp>(loc, index, indices[i++]);
    simdIndices.push_back(index);
  }
  simdIndices = handlePermutations(simdIndices, permutationMap);
  return simdIndices;
}

void VectorDistribution::distributeTransferReads(vector::TransferReadOp transferReadOp) {
  TypedValue<VectorType> result = transferReadOp.getResult();
  Value source = transferReadOp.getSource();
  Type elementType = llvm::cast<ShapedType>(source.getType()).getElementType();
  auto vectorType = VectorType::get(provider->getDistributedShape(result), elementType);
  Location loc = transferReadOp.getLoc();
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  int64_t loadElements = layout.getTransferElements();
  auto loadFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<Value> indices =
            getIndices(layout, iterator, transferReadOp.getIndices(),
                       transferReadOp.getPermutationMap(), loc, rewriter);
    auto vectorType = VectorType::get({loadElements}, elementType);
    if (loadElements == 1) {
      Value element = rewriter.create<memref::LoadOp>(loc, source, indices);
        Value broadcasted =
            rewriter.create<vector::BroadcastOp>(loc, vectorType, element);
        vector = rewriter.create<vector::InsertStridedSliceOp>(
            loc, broadcasted, vector, layout.computeSIMTIndex(iterator, provider->getSIMTLabels()),
            SmallVector<int64_t>{1});
    } else {
      Value element = rewriter.create<vector::LoadOp>(loc, vectorType, source,
            indices);
      vector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, element, vector, layout.computeSIMTIndex(iterator, provider->getSIMTLabels()),
          SmallVector<int64_t>{1});
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  steps[LayoutDimension::VECTORX] = loadElements;
  LayoutAttr::Iterator iterator = layout.getIterator(steps);
  layout.map(loadFn, iterator);
  simdToSimt.map(result, vector);
}

void VectorDistribution::distributeTransferWrites(vector::TransferWriteOp transferWriteOp) {
}

void VectorDistribution::distributeContractions(vector::ContractionOp contractOp) {
}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
