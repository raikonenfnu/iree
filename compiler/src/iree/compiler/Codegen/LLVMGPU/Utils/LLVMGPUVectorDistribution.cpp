// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>
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
#include "mlir/IR/Verifier.h"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

static bool canDistribute(Operation *op, VectorLayoutAnalysis &analysis) {
  bool needsDistribution = false;
  // Check if this operation has any operands with a vector type. If so,
  // then they need to have a layout.
  for (Value operand : op->getOperands()) {
    if (isa<VectorType>(operand.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(operand)) {
        return false;
      }
    }
  }

  // Check if this operation has any results with a vector type. If so,
  // then they need to have a layout.
  for (OpResult result : op->getResults()) {
    if (isa<VectorType>(result.getType())) {
      needsDistribution = true;
      if (!analysis.getLayout<Attribute>(result)) {
        return false;
      }
    }
  }

  return needsDistribution;
}

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

class DistributionRewriter : public IRRewriter, public RewriterBase::Listener {
public:
  DistributionRewriter(MLIRContext *ctx, DenseSet<Operation *> &erasedOps,
                       SmallVector<Operation *> &worklist,
                       VectorLayoutAnalysis &analysis)
      : IRRewriter(ctx), erasedOps(erasedOps), worklist(worklist),
        analysis(analysis) {
    setListener(this);
  }

protected:
  void notifyOperationRemoved(Operation *op) override { erasedOps.insert(op); }

  void notifyOperationInserted(Operation *op) override {
    // For now, do nothing.
    // TODO: Check if the operation can still be distributed and try to
    // distribute it.
  }

private:
  // A reference to the set of operations that have been erased.
  DenseSet<Operation *> &erasedOps;
  // A reference to the worklist of operations that need to be distributed.
  SmallVector<Operation *> &worklist;
  // A reference to the analysis that provides the layout information.
  VectorLayoutAnalysis &analysis;
};

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, LayoutProvider *provider,
                     VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), provider(provider) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;
  }

  void distribute() {
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> erasedOps;
    DistributionRewriter rewriter(root.getContext(), erasedOps, worklist,
                                  analysis);

    // 1: Collect all operations that need to be distributed.
    root->walk([&](Operation *op) {
      if (canDistribute(op, analysis)) {
        worklist.push_back(op);
      }
    });

    // 2. Set the insertion point to the beginning of the root, and build the
    // thread grid.
    rewriter.setInsertionPointToStart(&root.getBody().getBlocks().front());
    threadGrid = provider->getThreadGrid(rewriter);

    // 3. Distribute all operations in the worklist until we reach a fixed
    // point.
    bool changed = true;
    while (changed) {
      changed = false;
      for (unsigned i = 0; i < worklist.size(); ++i) {
        Operation *op = worklist[i];
        if (erasedOps.count(op))
          continue;

        rewriter.setInsertionPoint(op);

        if (provider->specializedDistribution(rewriter, op).succeeded()) {
          changed = true;
          continue;
        }

        LogicalResult distributed =
            TypeSwitch<Operation *, LogicalResult>(op)
                .Case<vector::TransferReadOp>([&](auto transferReadOp) {
                  return distributeTransferReads(rewriter, transferReadOp);
                })
                .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {
                  return distributeTransferWrites(rewriter, transferWriteOp);
                })
                .Case<arith::ConstantOp>([&](auto constantOp) {
                  return distributeConstants(rewriter, constantOp);
                })
                .Case<IREE::VectorExt::LayoutConflictResolutionOp>(
                    [&](auto resolutionOp) {
                      return distributeResolutions(rewriter, resolutionOp);
                    })
                .Case<scf::ForOp>([&](auto forOp) {
                  return distributeScfFor(rewriter, forOp);
                })
                .Case<scf::YieldOp>([&](auto yieldOp) {
                  return distributeScfYield(rewriter, yieldOp);
                })
                .Case<vector::MultiDimReductionOp>([&](auto reductionOp) {
                  return distributeReductions(rewriter, reductionOp);
                })
                .Case<vector::BroadcastOp>([&](auto broadcastOp) {
                  return distributeBroadcasts(rewriter, broadcastOp);
                })
                .Default([&](Operation *op) { return failure(); });

        // If the operation was distributed, continue with the next one.
        if (distributed.succeeded()) {
          changed = true;
          continue;
        }

        if (OpTrait::hasElementwiseMappableTraits(op)) {
          if (distributeElementwise(rewriter, op).succeeded()) {
            changed = true;
          }
          continue;
        }
      }
    }

    // 4. Ideally, we should error out here if everything was not distributed.
    // Currently, I'm not adding it for debugging purposes.
    // TODO: Add a check here if something was not distributed.
  }

private:
  LogicalResult
  distributeTransferWrites(RewriterBase &rewriter,
                           vector::TransferWriteOp transferWriteOp);

  LogicalResult distributeTransferReads(RewriterBase &rewriter,
                                        vector::TransferReadOp transferReadOp);

  LogicalResult distributeConstants(RewriterBase &rewriter,
                                    arith::ConstantOp constantOp);

  LogicalResult distributeElementwise(RewriterBase &rewriter, Operation *op);

  LogicalResult distributeResolutions(
      RewriterBase &rewriter,
      IREE::VectorExt::LayoutConflictResolutionOp resolutionOp);

  LogicalResult distributeTripToSharedMem(
      RewriterBase &rewriter,
      IREE::VectorExt::LayoutConflictResolutionOp resolutionOp);

  LogicalResult distributeScfFor(RewriterBase &rewriter, scf::ForOp forOp);

  LogicalResult distributeScfYield(RewriterBase &rewriter,
                                   scf::YieldOp yieldOp);

  LogicalResult distributeReductions(RewriterBase &rewriter,
                                     vector::MultiDimReductionOp reductionOp);

  LogicalResult distributeBroadcasts(RewriterBase &rewriter,
                                     vector::BroadcastOp broadcastOp);

  TypedValue<VectorType> reshapeVector(RewriterBase &rewriter,
                                       TypedValue<VectorType> src,
                                       LayoutAttr &currentLayout,
                                       LayoutAttr &targetLayout,
                                       Type elementType);

  SmallVector<Value> getIndices(RewriterBase &rewriter, LayoutAttr &layout,
                                LayoutAttr::Iterator &iterator,
                                SmallVector<Value> indices,
                                AffineMap permutationMap, Location loc);

  ArrayRef<Value> getThreadIds() { return threadGrid; }

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider *provider;
  SmallVector<Value> threadGrid;
}; // namespace

} // namespace

static SmallVector<Value>
handlePermutationsAndLeadingUnitDims(RewriterBase &rewriter,
                                     SmallVector<Value> &indices,
                                     AffineMap permutationMap) {
  SmallVector<Value> newIndices(permutationMap.getNumDims());
  // Not all dims are used in the permutation map.
  int unitLeadingDims =
      permutationMap.getNumDims() - permutationMap.getNumResults();
  int laneDim = 0;
  for (AffineExpr expr : permutationMap.getResults()) {
    auto dimExpr = dyn_cast<AffineDimExpr>(expr);
    if (!dimExpr) {
      llvm::report_fatal_error("invalid permutation map");
    }
    unsigned pos = dimExpr.getPosition();
    assert(pos >= unitLeadingDims && "invalid permutation map");
    newIndices[pos] = indices[laneDim++];
  }
  // TODO: Fix unknown loc here.
  for (unsigned i = 0; i < unitLeadingDims; ++i) {
    newIndices[i] =
        rewriter.create<arith::ConstantIndexOp>(rewriter.getUnknownLoc(), 0);
  }

  return newIndices;
}

SmallVector<Value> VectorDistribution::getIndices(
    RewriterBase &rewriter, LayoutAttr &layout, LayoutAttr::Iterator &iterator,
    SmallVector<Value> indices, AffineMap permutationMap, Location loc) {
  SmallVector<Value> simdIndices;

  ArrayRef<Value> threadIds = getThreadIds();
  SmallVector<Value> laneIds;
  for (PerDimLayoutAttr dimLayout : layout.getLayouts()) {
    for (LayoutDimensionAttr label : dimLayout.getLabels()) {
      switch (label.getValue()) {
      case LayoutDimension::LANEX:
        laneIds.push_back(threadIds[0]);
        break;
      case LayoutDimension::LANEY:
        laneIds.push_back(threadIds[1]);
        break;
      case LayoutDimension::LANEZ:
        laneIds.push_back(threadIds[2]);
        break;
      default:
        break;
      }
    }
  }

  int i{0};
  for (auto pair : llvm::zip_equal(layout.getLayouts(), iterator.states)) {
    PerDimLayoutAttr perDimLayout = std::get<0>(pair);
    PerDimLayoutAttr::Iterator iterator = std::get<1>(pair);
    Value index = rewriter.create<affine::AffineApplyOp>(
        loc, perDimLayout.computeSIMDIndex(iterator), laneIds[i]);
    index = rewriter.create<arith::AddIOp>(loc, index, indices[i++]);
    simdIndices.push_back(index);
  }
  simdIndices = handlePermutationsAndLeadingUnitDims(rewriter, simdIndices,
                                                     permutationMap);
  return simdIndices;
}

LogicalResult VectorDistribution::distributeTransferReads(
    RewriterBase &rewriter, vector::TransferReadOp transferReadOp) {
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
        getIndices(rewriter, layout, iterator, transferReadOp.getIndices(),
                   transferReadOp.getPermutationMap(), loc);
    SmallVector<int64_t> simtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
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
  return success();
}

LogicalResult VectorDistribution::distributeTransferWrites(
    RewriterBase &rewriter, vector::TransferWriteOp transferWriteOp) {
  TypedValue<VectorType> vector = transferWriteOp.getVector();
  Value source = transferWriteOp.getSource();
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(vector);
  Location loc = transferWriteOp.getLoc();
  int64_t storeElements = layout.getTransferElements();
  auto storeFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<Value> simdIndices =
        getIndices(rewriter, layout, iterator, transferWriteOp.getIndices(),
                   transferWriteOp.getPermutationMap(), loc);
    SmallVector<int64_t> simtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
    if (storeElements == 1) {
      Value result = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, vector, provider), simtIndices);
      rewriter.create<memref::StoreOp>(
          loc, result, source,
          getIndices(rewriter, layout, iterator, transferWriteOp.getIndices(),
                     transferWriteOp.getPermutationMap(), loc));
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
  // TODO: This is wrong. We should be using the innermost dimension for
  // calculating this step instead.
  steps[LayoutDimension::VECTORX] = storeElements;
  LayoutAttr::Iterator iterator = layout.getIterator(steps);
  layout.map(storeFn, iterator);
  replaceOpWithDistributedValues(rewriter, transferWriteOp, provider,
                                 ValueRange());
  return success();
}

LogicalResult
VectorDistribution::distributeConstants(RewriterBase &rewriter,
                                        arith::ConstantOp constantOp) {
  Value constantResult = constantOp.getResult();
  if (!isa<VectorType>(constantResult.getType()))
    return failure();
  auto constant = cast<TypedValue<VectorType>>(constantResult);
  auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());
  // Only handle splat values for now
  if (!attr.isSplat())
    return failure();
  Type elementType = constant.getType().getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(constant), elementType);
  replaceOpWithNewDistributedOp<arith::ConstantOp>(
      provider, rewriter, constantOp, vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
  return success();
}

LogicalResult VectorDistribution::distributeElementwise(RewriterBase &rewriter,
                                                        Operation *op) {
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
  return success();
}

TypedValue<VectorType> VectorDistribution::reshapeVector(
    RewriterBase &rewriter, TypedValue<VectorType> src,
    LayoutAttr &currentLayout, LayoutAttr &targetLayout, Type elementType) {
  SmallVector<int64_t> targetShape =
      targetLayout.getSIMTVectorShape(provider->getSIMTLabels(targetLayout));
  auto newVectorType = VectorType::get(targetShape, elementType);
  Location loc = root->getLoc();
  arith::ConstantOp constantOp = rewriter.create<arith::ConstantOp>(
      loc, newVectorType, rewriter.getZeroAttr(newVectorType));
  auto newVector = cast<TypedValue<VectorType>>(constantOp.getResult());

  SmallVector<int64_t> currentShape =
      currentLayout.getSIMTVectorShape(provider->getSIMTLabels(currentLayout));
  int64_t innermostDim = targetShape.size() - 1;
  int64_t step =
      std::min(targetShape[innermostDim], currentShape[innermostDim]);
  DenseMap<LayoutDimension, int64_t> steps;
  LayoutDimension vecDim = provider->getInnerMostVecDim();
  steps[vecDim] = step;
  auto srcIterator = currentLayout.getIterator(steps);
  auto targetIterator = targetLayout.getIterator(steps);
  do {
    auto srcOffset = currentLayout.computeSIMTIndex(
        srcIterator, provider->getSIMTLabels(currentLayout));
    SmallVector<int64_t> sliceSize(srcOffset.size(), 1);
    SmallVector<int64_t> sliceStride(srcOffset.size(), 1);
    sliceSize[sliceSize.size() - 1] = step;
    Value slice = rewriter.create<vector::ExtractStridedSliceOp>(
        loc, src, srcOffset, sliceSize, sliceStride);
    auto targetOffset = targetLayout.computeSIMTIndex(
        targetIterator, provider->getSIMTLabels(targetLayout));
    newVector = rewriter.create<vector::InsertStridedSliceOp>(
        loc, slice, newVector, targetOffset, sliceStride);
  } while (!srcIterator.next() && !targetIterator.next());
  return newVector;
}

LogicalResult VectorDistribution::distributeResolutions(
    RewriterBase &rewriter,
    IREE::VectorExt::LayoutConflictResolutionOp resolutionOp) {
  TypedValue<VectorType> vector = resolutionOp.getInput();
  TypedValue<VectorType> result = resolutionOp.getOutput();
  LayoutAttr currentLayout = cast<LayoutAttr>(resolutionOp.getSourceLayout());
  LayoutAttr targetLayout = cast<LayoutAttr>(resolutionOp.getDesiredLayout());
  if (currentLayout.hasLaneConflictWith(targetLayout))
    return distributeTripToSharedMem(rewriter, resolutionOp);
  SmallVector<int64_t> currentVecShape =
      currentLayout.getSIMTVectorShape(provider->getSIMTLabels(currentLayout));
  SmallVector<int64_t> targetVecShape =
      targetLayout.getSIMTVectorShape(provider->getSIMTLabels(targetLayout));
  if (currentVecShape.size() != targetVecShape.size())
    return failure();
  auto numElements = [](ArrayRef<int64_t> vector) {
    return std::accumulate(vector.begin(), vector.end(), 1,
                           std::multiplies<int64_t>());
  };
  if (numElements(currentVecShape) != numElements(targetVecShape))
    return failure();
  Type elementType = llvm::cast<VectorType>(result.getType()).getElementType();
  Value newVector =
      reshapeVector(rewriter, getDistributed(rewriter, vector, provider),
                    currentLayout, targetLayout, elementType);
  replaceOpWithDistributedValues(rewriter, resolutionOp, provider, newVector);
  return success();
}

/// Helper function for loop distribution. Given a list of bbArgs of the new
/// (distributed) loop op, wrap the distributed vector args (now distributed)
/// into ToSIMDOps, so that the block body can be moved over to the new op.
static SmallVector<Value> getBbArgsReplacements(RewriterBase &rewriter,
                                                Block::BlockArgListType bbArgs,
                                                ValueRange oldInits) {
  SmallVector<Value> replacements;
  for (auto [bbArg, oldInit] : llvm::zip_equal(bbArgs, oldInits)) {
    Value val = bbArg;
    if (auto oldVectorInit = dyn_cast<TypedValue<VectorType>>(oldInit)) {
      val = rewriter.create<IREE::VectorExt::ToSIMDOp>(
          oldVectorInit.getLoc(), oldVectorInit.getType(), val);
    }
    replacements.push_back(val);
  }
  return replacements;
}

LogicalResult VectorDistribution::distributeScfFor(RewriterBase &rewriter,
                                                   scf::ForOp forOp) {
  Block *oldLoopBody = forOp.getBody();

  // The new vector init_args of the loop.
  SmallVector<Value> newInitArgs;
  for (Value initArg : forOp.getInitArgs()) {
    if (auto vectorInitArg = dyn_cast<TypedValue<VectorType>>(initArg)) {
      initArg = getDistributed(rewriter, vectorInitArg, provider);
    }
    newInitArgs.push_back(initArg);
  }

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newForOp->setAttrs(forOp->getAttrs());
  Block *loopBody = newForOp.getBody();

  // Set up new iter_args. The loop body uses SIMD, so wrap the SIMD iter_args
  // of the new loop op into ToSIMDOps.
  rewriter.setInsertionPointToStart(loopBody);
  SmallVector<Value> iterArgs = getBbArgsReplacements(
      rewriter, newForOp.getRegionIterArgs(), forOp.getInitArgs());
  iterArgs.insert(iterArgs.begin(), newForOp.getInductionVar());

  // Move loop body to new loop.
  rewriter.mergeBlocks(oldLoopBody, loopBody, iterArgs);

  // Rpleace loop results.
  replaceOpWithDistributedValues(rewriter, forOp, provider,
                                 newForOp.getResults());
  return success();
}

LogicalResult VectorDistribution::distributeScfYield(RewriterBase &rewriter,
                                                     scf::YieldOp yieldOp) {
  SmallVector<Value> newOperands;
  for (Value operand : yieldOp.getOperands()) {
    if (auto vectorOperand = dyn_cast<TypedValue<VectorType>>(operand)) {
      operand = getDistributed(rewriter, vectorOperand, provider);
    }
    newOperands.push_back(operand);
  }
  rewriter.replaceOpWithNewOp<scf::YieldOp>(yieldOp, newOperands);
  return success();
}

LogicalResult VectorDistribution::distributeReductions(
    RewriterBase &rewriter, vector::MultiDimReductionOp reductionOp) {
  Location loc = reductionOp.getLoc();
  auto reductionDims = llvm::to_vector<4>(
      reductionOp.getReductionDims().getAsRange<IntegerAttr>());
  // Only support reduction on one dimension
  if (reductionDims.size() > 1)
    return failure();
  int reductionDim = reductionDims[0].getInt();
  // TODO: The following assumes only a 2D tensor
  int parallelDim = !reductionDim;
  vector::CombiningKind combiningKind = reductionOp.getKind();
  auto acc = cast<TypedValue<VectorType>>(reductionOp.getAcc());
  TypedValue<VectorType> source = reductionOp.getSource();

  Type elementType = llvm::cast<ShapedType>(acc.getType()).getElementType();
  int bitWidth = elementType.getIntOrFloatBitWidth();
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(source);

  // Get information for reduction from layout
  uint64_t offset{0};
  auto maybeReductionLaneDim = layout.getLaneId(reductionDim);
  if (!maybeReductionLaneDim)
    return failure();
  int64_t laneSize = *layout.getShape(*maybeReductionLaneDim);
  switch (*maybeReductionLaneDim) {
  case LayoutDimension::LANEX:
    offset = 1;
    break;
  case LayoutDimension::LANEY:
    offset = *layout.getShape(LayoutDimension::LANEX);
    break;
  case LayoutDimension::LANEZ:
    offset = (*layout.getShape(LayoutDimension::LANEX)) *
             (*layout.getShape(LayoutDimension::LANEY));
    break;
  default:
    break;
  }

  auto resultVec = dyn_cast<TypedValue<VectorType>>(reductionOp.getResult());
  // For now, fail if result is not a vector.
  // TODO: Support this.
  if (!resultVec) {
    return failure();
  }
  auto storeVectorType =
      VectorType::get(provider->getDistributedShape(resultVec), elementType);
  Value storeVec = rewriter.create<arith::ConstantOp>(
      loc, storeVectorType, rewriter.getZeroAttr(storeVectorType));

  auto reduceFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<int64_t> parallelSimtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
    auto projectedIndices = layout.projectSIMTVector(
        provider->getSIMTLabels(layout), parallelSimtIndices, reductionDim);
    Value mEmpty = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, acc, provider), projectedIndices);
    int count{0};
    Value tmp, result, mask;
    if (bitWidth == 32) {
      tmp = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(VectorType::get({1}, elementType)));
    } else {
      tmp = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getZeroAttr(VectorType::get({2}, elementType)));
    }

    auto reduceLocalFn = [&](LayoutAttr::Iterator &iterator) {
      SmallVector<int64_t> reductionSimtIndices =
          layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
      SmallVector<int64_t> indices;
      for (auto [a, b] : llvm::zip(parallelSimtIndices, reductionSimtIndices))
        indices.push_back(a + b);
      Value x = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, source, provider), indices);
      if (bitWidth == 32) {
        tmp = rewriter.create<vector::InsertOp>(loc, x, tmp,
                                                SmallVector<int64_t>{0});
      } else {
        int index = (count++) % 2;
        tmp = rewriter.create<vector::InsertOp>(loc, x, tmp,
                                                SmallVector<int64_t>{index});
        if (index == 0)
          return;
      }
      result = !result ? tmp
                       : makeArithReduction(rewriter, loc, combiningKind,
                                            result, tmp, mask);
    };

    LayoutAttr::Iterator reductionIterator =
        layout.getDimIterator(reductionDim);
    layout.map(reduceLocalFn, reductionIterator);

    auto reduceGlobalFn = [&]() {
      uint32_t size{32};
      for (uint64_t i = offset; i < offset * laneSize; i <<= 1) {
        Value packed = packVectorToSupportedWidth(loc, rewriter, result);
        auto shuffleOp = rewriter.create<gpu::ShuffleOp>(loc, packed, i, size,
                                                         gpu::ShuffleMode::XOR);
        Value unpacked =
            unpackToVector(loc, rewriter, shuffleOp.getShuffleResult(),
                           result.getType().cast<VectorType>());
        result = makeArithReduction(rewriter, loc, combiningKind, unpacked,
                                    result, mask);
      }

      // Convert to f16 or f32
      if (bitWidth == 32) {
        Value v0 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{0});
        result =
            makeArithReduction(rewriter, loc, combiningKind, v0, mEmpty, mask);
      } else {
        Value v0 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{0});
        Value v1 = rewriter.create<vector::ExtractOp>(loc, result,
                                                      SmallVector<int64_t>{1});
        result = makeArithReduction(rewriter, loc, combiningKind, v0, v1, mask);
        result = makeArithReduction(rewriter, loc, combiningKind, result,
                                    mEmpty, mask);
      }
    };
    reduceGlobalFn();
    reductionIterator = layout.getDimIterator(reductionDim);
    storeVec = rewriter.create<vector::InsertOp>(loc, result, storeVec,
                                                 projectedIndices);
  };

  LayoutAttr::Iterator parallelIterator = layout.getDimIterator(parallelDim);
  layout.map(reduceFn, parallelIterator);
  replaceOpWithDistributedValues(rewriter, reductionOp, provider, storeVec);
  return success();
}

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDCDNAGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, &layoutProvider, analysis);
  distribution.distribute();
}

// Distributed a broadcast + transpose which returns a 1D vector
// to its original 2D shape (prior to be folded along the reduction dimension).
LogicalResult
VectorDistribution::distributeBroadcasts(RewriterBase &rewriter,
                                         vector::BroadcastOp broadcastOp) {
  // Check if this is a broadcast with one use that is a transpose.
  if (!broadcastOp.getResult().hasOneUse())
    return failure();
  vector::TransposeOp transposeOp;
  for (Operation *user : broadcastOp.getResult().getUsers()) {
    transposeOp = dyn_cast<vector::TransposeOp>(user);
    if (transposeOp) {
      break;
    }
  }
  if (!transposeOp)
    return failure();

  TypedValue<VectorType> result = transposeOp.getResult();
  TypedValue<VectorType> source =
      cast<TypedValue<VectorType>>(broadcastOp.getSource());
  auto layout = analysis.getLayout<LayoutAttr>(result);
  Type elementType = llvm::cast<ShapedType>(result.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Location loc = broadcastOp.getLoc();
  Value broadcastedVector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  // Since this is a broadcast + transpose, reduction dimension = 1.
  // For just a broadcast, reduction dimension = 0.
  int64_t reductionDim = 1;
  int64_t parallelDim = 0;
  auto broadcastFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<int64_t> parallelSimtIndices =
        layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
    auto projectedIndices = layout.projectSIMTVector(
        provider->getSIMTLabels(layout), parallelSimtIndices, reductionDim);
    Value value = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, source, provider), projectedIndices);

    auto broadcastValue = [&](LayoutAttr::Iterator &iterator) {
      SmallVector<int64_t> reductionSimtIndices =
          layout.computeSIMTIndex(iterator, provider->getSIMTLabels(layout));
      SmallVector<int64_t> indices;
      for (auto [a, b] : llvm::zip(parallelSimtIndices, reductionSimtIndices))
        indices.push_back(a + b);
      broadcastedVector = rewriter.create<vector::InsertOp>(
          loc, value, broadcastedVector, indices);
    };
    LayoutAttr::Iterator reductionIterator =
        layout.getDimIterator(reductionDim);
    layout.map(broadcastValue, reductionIterator);
  };
  LayoutAttr::Iterator parallelIterator = layout.getDimIterator(parallelDim);
  layout.map(broadcastFn, parallelIterator);
  replaceOpWithDistributedValues(rewriter, transposeOp, provider,
                                 broadcastedVector);
  return success();
}

LogicalResult VectorDistribution::distributeTripToSharedMem(
    RewriterBase &rewriter,
    IREE::VectorExt::LayoutConflictResolutionOp resolutionOp) {
  TypedValue<VectorType> vector = resolutionOp.getInput();
  TypedValue<VectorType> result = resolutionOp.getOutput();
  VectorLayoutInterface targetLayout =
      analysis.getLayout<VectorLayoutInterface>(result);

  // We do a SIMD -> SIMD rewrite, and then distribute the result to get
  // SIMD -> SIMT.
  // TODO: Ideally, the distribution should know what is SIMD and what is SIMT.
  // So we should just be able to do a SIMD -> SIMD rewrite here and the
  // distribution should pickup on that.

  // Create a alloca which can store the vector in shared memory.
  // The alloca shape can be taken from the type of the output.
  ArrayRef<int64_t> shape = result.getType().getShape();
  Type elementType = result.getType().getElementType();
  auto memSpace =
      rewriter.getAttr<gpu::AddressSpaceAttr>(gpu::AddressSpace::Workgroup);
  auto memType = MemRefType::get(shape, elementType,
                                 MemRefLayoutAttrInterface{}, memSpace);
  auto allocaOp =
      rewriter.create<memref::AllocOp>(resolutionOp.getLoc(), memType);

  // Create a transfer_write to this new alloca.
  auto zeroIndex =
      rewriter.create<arith::ConstantIndexOp>(resolutionOp.getLoc(), 0);
  SmallVector<Value> indices(shape.size(), zeroIndex);
  auto transferWriteOp = rewriter.create<vector::TransferWriteOp>(
      resolutionOp.getLoc(), vector, allocaOp.getResult(), indices);

  // Create a barrier over this transfer_read.
  rewriter.create<gpu::BarrierOp>(resolutionOp.getLoc());

  // Create a transfer_write from this new alloca.
  SmallVector<VectorLayoutInterface> readLayouts = {targetLayout};
  auto transferReadOp = createSIMDOp<vector::TransferReadOp>(
      provider, rewriter, ArrayRef(readLayouts), resolutionOp.getLoc(),
      result.getType(), allocaOp.getResult(), indices);

  // Replace with the conflict result with the transfer_read result.
  rewriter.replaceOp(resolutionOp, transferReadOp.getResult());

  // Distribute the transfer_write.
  rewriter.setInsertionPoint(transferWriteOp);
  (void)distributeTransferWrites(rewriter, transferWriteOp);

  // Distribute the transfer_read.
  rewriter.setInsertionPoint(transferReadOp);
  (void)distributeTransferReads(rewriter, transferReadOp);

  return success();
}

} // namespace mlir::iree_compiler
