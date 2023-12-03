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

  void distributeConstants(arith::ConstantOp constantOp);

  SmallVector<Value> getIndices(LayoutAttr &layout,
                                LayoutAttr::Iterator &iterator,
                                SmallVector<Value> indices,
                                AffineMap permutationMap, Location loc,
                                OpBuilder &rewriter);

  // We delinearize threadIdx to 2 thread indices.
  Value getThreadIds(PerDimLayoutAttr &layout, Value lastShape,
                     DenseMap<int64_t, Value> &laneIds, int64_t key,
                     LayoutDimension dim) {
    Location loc = root.getLoc();
    Value threadX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
    auto possibleShape = layout.getShape(dim);
    if (possibleShape) {
      switch (dim) {
      case LayoutDimension::LANEX: {
        Value shapeValue =
            rewriter.create<arith::ConstantIndexOp>(loc, *possibleShape);
        laneIds[key] =
            rewriter.create<arith::RemUIOp>(loc, threadX, shapeValue);
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

SmallVector<Value> VectorDistribution::getIndices(
    LayoutAttr &layout, LayoutAttr::Iterator &iterator,
    SmallVector<Value> indices, AffineMap permutationMap, Location loc,
    OpBuilder &rewriter) {
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
    Value index = rewriter.create<affine::AffineApplyOp>(
        loc, perDimLayout.computeSIMDIndex(iterator), laneIds[i]);
    index = rewriter.create<arith::AddIOp>(loc, index, indices[i++]);
    simdIndices.push_back(index);
  }
  simdIndices = handlePermutations(simdIndices, permutationMap);
  return simdIndices;
}

void VectorDistribution::distributeTransferReads(
    vector::TransferReadOp transferReadOp) {
  TypedValue<VectorType> result = transferReadOp.getResult();
  Value source = transferReadOp.getSource();
  Type elementType = llvm::cast<ShapedType>(source.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Location loc = transferReadOp.getLoc();
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  int64_t loadElements = layout.getTransferElements();
  auto loadFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<Value> simdIndices =
        getIndices(layout, iterator, transferReadOp.getIndices(),
                   transferReadOp.getPermutationMap(), loc, rewriter);
    SmallVector<int64_t> simtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels());
    auto vectorType = VectorType::get({loadElements}, elementType);
    if (loadElements == 1) {
      Value element = rewriter.create<memref::LoadOp>(loc, source, simdIndices);
      Value broadcasted =
          rewriter.create<vector::BroadcastOp>(loc, vectorType, element);
      vector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, broadcasted, vector, simtIndices, SmallVector<int64_t>{1});
    } else {
      Value element =
          rewriter.create<vector::LoadOp>(loc, vectorType, source, simdIndices);
      vector = rewriter.create<vector::InsertStridedSliceOp>(
          loc, element, vector, simtIndices, SmallVector<int64_t>{1});
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  steps[LayoutDimension::VECTORX] = loadElements;
  LayoutAttr::Iterator iterator = layout.getIterator(steps);
  layout.map(loadFn, iterator);
  simdToSimt.map(result, vector);
}

void VectorDistribution::distributeTransferWrites(
    vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();
  if (!simdToSimt.lookupOrNull(vector))
    return;
  Value source = transferWriteOp.getSource();
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(vector);
  Location loc = transferWriteOp.getLoc();
  int64_t storeElements = layout.getTransferElements();
  auto storeFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<Value> simdIndices =
        getIndices(layout, iterator, transferWriteOp.getIndices(),
                   transferWriteOp.getPermutationMap(), loc, rewriter);
    SmallVector<int64_t> simtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels());
    if (storeElements == 1) {
      Value result = rewriter.create<vector::ExtractOp>(
          loc, simdToSimt.lookup(vector), simtIndices);
      rewriter.create<memref::StoreOp>(
          loc, result, source,
          getIndices(layout, iterator, transferWriteOp.getIndices(),
                     transferWriteOp.getPermutationMap(), loc, rewriter));
    } else {
      SmallVector<int64_t> strides(simtIndices.size(), 1);
      SmallVector<int64_t> shapes(simtIndices.size(), 1);
      shapes[shapes.size() - 1] = storeElements;
      Value result = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, simdToSimt.lookup(vector), simtIndices, shapes, strides);
      result = rewriter.create<vector::ExtractOp>(
          loc, result, SmallVector<int64_t>(simtIndices.size() - 1, 0));
      rewriter.create<vector::StoreOp>(loc, result, source, simdIndices);
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  steps[LayoutDimension::VECTORX] = storeElements;
  LayoutAttr::Iterator iterator = layout.getIterator(steps);
  layout.map(storeFn, iterator);
}

void VectorDistribution::distributeContractions(
    vector::ContractionOp contractOp) {
  TypedValue<VectorType> lhs = contractOp.getLhs();
  TypedValue<VectorType> rhs = contractOp.getRhs();
  Value acc = contractOp.getAcc();
  if (!simdToSimt.lookupOrNull(lhs) || !simdToSimt.lookupOrNull(rhs) ||
      !simdToSimt.lookupOrNull(acc))
    return;
  Location loc = contractOp.getLoc();
  TypedValue<VectorType> result =
      cast<TypedValue<VectorType>>(contractOp.getResult());
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  LayoutAttr lhsLayout = analysis.getLayout<LayoutAttr>(lhs);
  int K = provider->getKDimension(lhsLayout.getBatchDim(0),
                                  lhsLayout.getBatchDim(1));
  Type elementType = llvm::cast<ShapedType>(acc.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  auto contractFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<int64_t> simtIndices = layout.computeIteratorProjectedSIMTIndex(
        iterator, provider->getSIMTLabels());
    Value dMatrix = rewriter.create<vector::ExtractOp>(
        loc, simdToSimt.lookup(acc), simtIndices);
    for (int k = 0; k < K; k++) {
      Value aMatrix = rewriter.create<vector::ExtractOp>(
          loc, simdToSimt.lookup(lhs),
          provider->getContractIndices(ContractMatrixType::A, simtIndices[0],
                                       k));
      Value bMatrix = rewriter.create<vector::ExtractOp>(
          loc, simdToSimt.lookup(rhs),
          provider->getContractIndices(ContractMatrixType::B, k,
                                       simtIndices[1]));
      dMatrix = provider->computeMMA(aMatrix, bMatrix, dMatrix, loc, rewriter);
    }
    vector =
        rewriter.create<vector::InsertOp>(loc, dMatrix, vector, simtIndices);
  };
  LayoutAttr::Iterator iterator = layout.getBatchIterator();
  layout.map(contractFn, iterator);
  simdToSimt.map(result, vector);
}

void VectorDistribution::distributeConstants(arith::ConstantOp constantOp) {
  Value constantResult = constantOp.getResult();
  if (!isa<VectorType>(constantResult.getType()))
    return;
  auto constant = cast<TypedValue<VectorType>>(constantResult);
  auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());
  // Only handle splat values for now
  if (!attr.isSplat())
    return;
  Type elementType =
      llvm::cast<VectorType>(constant.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(constant), elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      constantOp.getLoc(), vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
  simdToSimt.map(constant, result);
}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
