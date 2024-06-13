// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/Utils/IndexingUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::IREE::LinalgExt {

namespace {

OpFoldResult getPadding(RewriterBase &rewriter, Location loc,
                        OpFoldResult bound, int64_t padMultiple) {
  AffineExpr s0;
  bindSymbols(rewriter.getContext(), s0);
  AffineExpr padByExpr = (s0).ceilDiv(padMultiple) * padMultiple - s0;
  return affine::makeComposedFoldedAffineApply(rewriter, loc, padByExpr,
                                               {bound});
}

int64_t getPaddedDim(RewriterBase &rewriter, Location loc, OpFoldResult bound,
                     int64_t padMultiple) {
  std::optional<int64_t> maybeStaticDim = getConstantIntValue(bound);
  if (!maybeStaticDim.has_value()) {
    return ShapedType::kDynamic;
  }
  int64_t paddedDims = maybeStaticDim.value();
  if (padMultiple != 0) {
    paddedDims =
        llvm::divideCeil(maybeStaticDim.value(), padMultiple) * padMultiple;
  }
  return paddedDims;
}

static Value
getPaddedValue(RewriterBase &rewriter, Location loc, Value padSource,
               ArrayRef<OpFoldResult> padding,
               std::optional<TypedAttr> padValueAttr = std::nullopt) {
  auto sourceType = cast<RankedTensorType>(padSource.getType());
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        std::optional<int64_t> padInt = getConstantIntValue(std::get<1>(it));
        if (ShapedType::isDynamic(std::get<0>(it)) || !padInt) {
          return ShapedType::kDynamic;
        }
        return std::get<0>(it) + padInt.value();
      });
  if (!padValueAttr.has_value()) {
    padValueAttr = rewriter.getZeroAttr(sourceType.getElementType());
  }
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  Value paddingValue =
      rewriter.create<arith::ConstantOp>(loc, padValueAttr.value());
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, padding, paddingValue);
  return paddedResult;
}

struct PadAttentionPass : public PadAttentionBase<PadAttentionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<
        affine::AffineDialect, IREE::LinalgExt::IREELinalgExtDialect,
        linalg::LinalgDialect, scf::SCFDialect, tensor::TensorDialect>();
  }
  void runOnOperation() override;
};

} // namespace

/// Pads iree_linalg_ext.attention.
IREE::LinalgExt::AttentionOp padAttention(IREE::LinalgExt::AttentionOp attnOp,
                                          SmallVectorImpl<Operation *> &ops,
                                          RewriterBase &rewriter,
                                          ArrayRef<int64_t> padToMultipleOf) {
  SmallVector<AffineMap> maps = attnOp.getIndexingMapsArray();
  FailureOr<IREE::LinalgExt::AttentionOpDetail> maybeOpInfo =
      IREE::LinalgExt::AttentionOpDetail::get(maps);
  assert(succeeded(maybeOpInfo) && "failed to infer attention dims");
  auto opInfo = maybeOpInfo.value();
  Location loc = attnOp.getLoc();
  rewriter.setInsertionPoint(attnOp);

  assert(padToMultipleOf.size() == opInfo.getDomainRank() &&
         "Expected pad_to_multiple_of to have same rank as dimensions of "
         "attention.");
  assert(opInfo.getDomainRank() == 5 &&
         "Currently only support base-case of attention dims.");
  SmallVector<Range> bounds = attnOp.getIterationDomain(rewriter);

  int64_t batchIdx = opInfo.getBatchDims().back();
  int64_t mIdx = opInfo.getMDims().back();
  int64_t k1Idx = opInfo.getK1Dims().back();
  int64_t k2Idx = opInfo.getK2Dims().back();
  int64_t nIdx = opInfo.getNDims().back();

  SmallVector<OpFoldResult> padValues(opInfo.getDomainRank(),
                                      rewriter.getIndexAttr(0));
  SmallVector<int64_t> paddedDims(opInfo.getDomainRank(), 0);
  for (auto [idx, bound] : enumerate(bounds)) {
    if (padToMultipleOf[idx] != 0) {
      padValues[idx] =
          getPadding(rewriter, loc, bound.size, padToMultipleOf[idx]);
    }
    paddedDims[idx] =
        getPaddedDim(rewriter, loc, bound.size, padToMultipleOf[idx]);
  }

  Value paddedQuery = attnOp.getQuery();
  Value paddedKey = attnOp.getKey();
  Value paddedValue = attnOp.getValue();
  Value paddedAcc = attnOp.getOutput();
  Value scale = attnOp.getScale();

  OpFoldResult zero = rewriter.getIndexAttr(0);

  // Padding Q, K, V, Acc if required.
  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[mIdx], 0) ||
      !isConstantIntValue(padValues[k1Idx], 0)) {
    paddedQuery = getPaddedValue(
        rewriter, loc, paddedQuery,
        {padValues[batchIdx], padValues[mIdx], padValues[k1Idx]});
  }

  if (!isConstantIntValue(padValues[k2Idx], 0)) {
    Type keyElType = attnOp.getKeyType().getElementType();
    auto negInfApFloat =
        APFloat::getInf(llvm::cast<FloatType>(keyElType).getFloatSemantics(),
                        /*Negative=*/true);
    auto negInfAttr = rewriter.getFloatAttr(keyElType, negInfApFloat);
    paddedKey = getPaddedValue(rewriter, loc, paddedKey,
                               {zero, padValues[k2Idx], zero}, negInfAttr);
  }

  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[k1Idx], 0)) {
    paddedKey = getPaddedValue(rewriter, loc, paddedKey,
                               {padValues[batchIdx], zero, padValues[k1Idx]});
  }

  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[k2Idx], 0) ||
      !isConstantIntValue(padValues[nIdx], 0)) {
    paddedValue = getPaddedValue(
        rewriter, loc, paddedValue,
        {padValues[batchIdx], padValues[k2Idx], padValues[nIdx]});
  }

  if (!isConstantIntValue(padValues[batchIdx], 0) ||
      !isConstantIntValue(padValues[mIdx], 0) ||
      !isConstantIntValue(padValues[nIdx], 0)) {
    if (llvm::dyn_cast_or_null<tensor::EmptyOp>(paddedAcc.getDefiningOp())) {
      SmallVector<OpFoldResult> paddedQueryShape =
          tensor::getMixedSizes(rewriter, loc, paddedQuery);
      SmallVector<OpFoldResult> paddedValueShape =
          tensor::getMixedSizes(rewriter, loc, paddedValue);
      SmallVector<OpFoldResult> paddedOutputShape = {
          paddedQueryShape[0], paddedQueryShape[1], paddedValueShape[2]};
      paddedAcc = rewriter.create<tensor::EmptyOp>(
          loc, paddedOutputShape, attnOp.getOutputType().getElementType());
    } else {
      paddedAcc = getPaddedValue(
          rewriter, loc, paddedAcc,
          {padValues[batchIdx], padValues[mIdx], padValues[nIdx]});
    }
  }

  auto paddedAttnOp = rewriter.create<IREE::LinalgExt::AttentionOp>(
      loc, paddedAcc.getType(),
      SmallVector<Value>{paddedQuery, paddedKey, paddedValue, scale},
      paddedAcc);

  ops.push_back(paddedAttnOp);

  // Extract subtensor result.
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(3, zero);
  SmallVector<OpFoldResult> strides(3, one);
  SmallVector<OpFoldResult> sizes = llvm::map_to_vector(
      attnOp.getOutputType().getShape(),
      [&](int64_t dim) -> OpFoldResult { return rewriter.getIndexAttr(dim); });
  Operation *extracted = rewriter.create<tensor::ExtractSliceOp>(
      loc, paddedAttnOp->getResults()[0], offsets, sizes, strides);
  ops.push_back(extracted);

  rewriter.replaceOp(attnOp, extracted);

  return paddedAttnOp;
}

void PadAttentionPass::runOnOperation() {
  MLIRContext *context = &getContext();
  IRRewriter rewriter(context);
  getOperation().walk([&](AttentionOp attnOp) {
    SmallVector<Operation *> ops;
    padAttention(attnOp, ops, rewriter, padToMultipleOf);
  });
}

std::unique_ptr<Pass> createPadAttentionPass() {
  return std::make_unique<PadAttentionPass>();
}

} // namespace mlir::iree_compiler::IREE::LinalgExt
