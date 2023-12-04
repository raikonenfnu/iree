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

TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider *provider) {
  // If this is a result of a "to_simd" op, use the source value of it.
  if (auto toSIMD = value.getDefiningOp<IREE::VectorExt::ToSIMDOp>()) {
    value = cast<TypedValue<VectorType>>(toSIMD.getInput());
    return value;
  }
  // Create a "to_simt" op to convert the value to the distributed layout.
  SmallVector<int64_t> distributedShape = provider->getDistributedShape(value);
  VectorType distributedType =
      VectorType::get(distributedShape, value.getType().getElementType());
  auto toSIMT = rewriter.create<IREE::VectorExt::ToSIMTOp>(
      value.getLoc(), distributedType, value);
  return toSIMT.getResult();
}

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider *provider,
                                    ValueRange values) {
  // Replace all OpResults with the given values.
  SmallVector<Value> replacements;
  for (OpResult opResult : op->getOpResults()) {
    Value replacement = values[opResult.getResultNumber()];
    // If this value is a vector type, it must be converted back to simd.
    if (isa<VectorType>(replacement.getType())) {
      auto oldResult = cast<TypedValue<VectorType>>(opResult);
      // Create a toSIMD op to convert the value back to the simd.
      rewriter.setInsertionPointAfterValue(oldResult);
      auto toSIMD = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldResult.getLoc(), oldResult.getType(), replacement);
      // Clone the layout to the new value.
      provider->getAnalysis().cloneLayoutInformationToNewValue(
          oldResult, toSIMD.getResult());
      // Add to replacements.
      replacement = toSIMD.getResult();
    }
    replacements.push_back(replacement);
  }

  rewriter.replaceOp(op, replacements);
}

namespace {

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, RewriterBase &rewriter,
                     LayoutProvider *provider, VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), provider(provider), rewriter(rewriter) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;
    analysis.dump();
  }

  void distribute() {
    // 1: Collect all operations that need to be distributed.
    SmallVector<Operation *> worklist;
    root->walk([&](Operation *op) {
      bool needsDistribution = false;
      // Check if this operation has any operands with a vector type. If so,
      // then they need to have a layout.
      for (Value operand : op->getOperands()) {
        if (isa<VectorType>(operand.getType())) {
          if (!analysis.getLayout<Attribute>(operand)) {
            llvm::report_fatal_error("operand of operation " +
                                     op->getName().getStringRef() +
                                     " does not have a layout");
          }
          needsDistribution = true;
        }
      }

      // Check if this operation has any results with a vector type. If so,
      // then they need to have a layout.
      for (OpResult result : op->getResults()) {
        if (isa<VectorType>(result.getType())) {
          if (!analysis.getLayout<Attribute>(result)) {
            llvm::report_fatal_error("result of operation " +
                                     op->getName().getStringRef() +
                                     " does not have a layout");
          }
          needsDistribution = true;
        }
      }

      if (needsDistribution) {
        worklist.push_back(op);
      }
    });

    // 2. Distribute all operations in the worklist. Each pattern currently
    // only replaces a single operation, which means we can iterate only once.
    // TODO: Add a rewriter which can handle multiple replacements.
    for (unsigned i = 0; i < worklist.size(); ++i) {
      Operation *op = worklist[i];
      rewriter.setInsertionPoint(op);

      if (provider->specializedDistribution(rewriter, op)) {
        continue;
      }

      bool distributed =
          TypeSwitch<Operation *, bool>(op)
              .Case<vector::TransferReadOp>([&](auto transferReadOp) {
                distributeTransferReads(transferReadOp);
                return true;
              })
              .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {
                distributeTransferWrites(transferWriteOp);
                return true;
              })
              .Case<arith::ConstantOp>([&](auto constantOp) {
                distributeConstants(constantOp);
                return true;
              })
              .Case<IREE::VectorExt::LayoutConflictResolutionOp>(
                  [&](auto resolutionOp) {
                    distributeResolutions(resolutionOp);
                    return true;
                  })
              .Default([&](Operation *op) { return false; });

      // If the operation was distributed, continue with the next one.
      if (distributed) {
        continue;
      }

      if (OpTrait::hasElementwiseMappableTraits(op)) {
        distributeElementwise(op);
        continue;
      }
    }

    // 3. Ideally, we should error out here if everything was not distributed.
    // Currently, I'm not adding it for debugging purposes.
    // TODO: Add a check here if something was not distributed.
  }

private:
  void distributeTransferWrites(vector::TransferWriteOp transferWriteOp);

  void distributeTransferReads(vector::TransferReadOp transferReadOp);

  void distributeConstants(arith::ConstantOp constantOp);

  void distributeElementwise(Operation *op);

  void distributeResolutions(
      IREE::VectorExt::LayoutConflictResolutionOp resolutionOp);

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
  replaceOpWithDistributedValues(rewriter, transferReadOp, provider, vector);
}

void VectorDistribution::distributeTransferWrites(
    vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();
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
          loc, getDistributed(rewriter, vector, provider), simtIndices);
      rewriter.create<memref::StoreOp>(
          loc, result, source,
          getIndices(layout, iterator, transferWriteOp.getIndices(),
                     transferWriteOp.getPermutationMap(), loc, rewriter));
    } else {
      SmallVector<int64_t> strides(simtIndices.size(), 1);
      SmallVector<int64_t> shapes(simtIndices.size(), 1);
      shapes[shapes.size() - 1] = storeElements;
      Value result = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, getDistributed(rewriter, vector, provider), simtIndices, shapes,
          strides);
      result = rewriter.create<vector::ExtractOp>(
          loc, result, SmallVector<int64_t>(simtIndices.size() - 1, 0));
      rewriter.create<vector::StoreOp>(loc, result, source, simdIndices);
    }
  };
  DenseMap<LayoutDimension, int64_t> steps;
  steps[LayoutDimension::VECTORX] = storeElements;
  LayoutAttr::Iterator iterator = layout.getIterator(steps);
  layout.map(storeFn, iterator);
  replaceOpWithDistributedValues(rewriter, transferWriteOp, provider,
                                 ValueRange());
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
  Type elementType = constant.getType().getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(constant), elementType);
  replaceOpWithNewDistributedOp<arith::ConstantOp>(
      provider, rewriter, constantOp, vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
}

void VectorDistribution::distributeElementwise(Operation *op) {
  assert(OpTrait::hasElementwiseMappableTraits(op) &&
         "expected elementwise op");
  // Replace vector operands with their distributed counter-parts.
  SmallVector<Value> operands;
  for (Value operand : op->getOperands()) {
    if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
      operand = getDistributed(rewriter, vectorOperand, provider);
    }
    operands.push_back(operand);
  }

  // Replace vector types with their distributed counter-parts.
  SmallVector<Type> resultTypes;
  for (Value result : op->getResults()) {
    if (auto vectorResult = dyn_cast<TypedValue<VectorType>>(result)) {
      // Distribute vector result types.
      auto newType =
          VectorType::get(provider->getDistributedShape(vectorResult),
                          vectorResult.getType().getElementType());
      resultTypes.push_back(newType);
    } else {
      resultTypes.push_back(result.getType());
    }
  }

  // Replace the original op with the distributed op.
  Operation *distributedOp =
      rewriter.create(op->getLoc(), op->getName().getIdentifier(), operands,
                      resultTypes, op->getAttrs());
  replaceOpWithDistributedValues(rewriter, op, provider,
                                 distributedOp->getResults());
}

void VectorDistribution::distributeResolutions(
    IREE::VectorExt::LayoutConflictResolutionOp resolutionOp) {}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
