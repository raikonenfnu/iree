// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/Transforms/RegionOpUtils.h"
#include "iree/compiler/GlobalOptimization/PassDetail.h"
#include "iree/compiler/GlobalOptimization/Passes.h"
#include "iree/compiler/GlobalOptimization/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "iree-global-opt-convert-memory-efficient-qk"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

class ConvertMemoryEfficientKVUpdatePattern final
    : public OpRewritePattern<flow::TensorUpdateOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(flow::TensorUpdateOp updateOp,
                                PatternRewriter &rewriter) const override {
    //   %expanded_1351 = tensor.expand_shape %extracted_slice_1350 [[0, 1, 2],
    //   [3]] : tensor<32x128xf16> into tensor<1x1x32x128xf16>
    auto expandOp = dyn_cast_or_null<tensor::ExpandShapeOp>(
        updateOp.getSource().getDefiningOp());
    if (!expandOp)
      return failure();

    auto extractOp = dyn_cast_or_null<tensor::ExtractSliceOp>(
        expandOp.getSrc().getDefiningOp());
    if (!extractOp)
      return failure();

    auto insertKSliceOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        extractOp.getSource().getDefiningOp());
    if (!extractOp)
      return failure();

    auto insertKCacheOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        insertKSliceOp.getDest().getDefiningOp());
    if (!insertKCacheOp)
      return failure();
    llvm::outs() << updateOp << "\n";
    return failure();
  }
};

// This pattern does a basic fusion of two matmuls and three linalg.generics.
// The pattern matches only on a DAG representing:
// output = Silu(matmul(A, B) * matmul(A, C).
class ConvertMemoryEfficientQKPattern final
    : public OpRewritePattern<linalg::BatchMatmulOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulOp bmmOp,
                                PatternRewriter &rewriter) const override {
    // Check that correct pattern of Ops matches.
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(
        bmmOp.getDpsInits()[0].getDefiningOp());
    if (!fillOp)
      return failure();

    auto initEmptyOp = dyn_cast_or_null<tensor::EmptyOp>(
        fillOp.getDpsInits()[0].getDefiningOp());
    if (!initEmptyOp)
      return failure();

    auto collapseOp = dyn_cast_or_null<tensor::CollapseShapeOp>(
        bmmOp.getInputs()[1].getDefiningOp());
    if (!collapseOp)
      return failure();

    auto insertNewKSliceOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        collapseOp.getSrc().getDefiningOp());
    if (!insertNewKSliceOp)
      return failure();

    auto insertPastKCacheOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        insertNewKSliceOp.getDest().getDefiningOp());
    if (!insertPastKCacheOp)
      return failure();

    auto transposePastKCacheOp = dyn_cast_or_null<linalg::GenericOp>(
        insertPastKCacheOp.getSource().getDefiningOp());
    if (!transposePastKCacheOp)
      return failure();
    auto transposeIndexingMaps = transposePastKCacheOp.getIndexingMapsArray();
    if (!transposeIndexingMaps[0].isIdentity())
      return failure();
    if (!transposeIndexingMaps[1].isPermutation())
      return failure();
    if (transposePastKCacheOp.getNumDpsInits() != 1 ||
        transposePastKCacheOp.getNumDpsInputs() != 1) {
      return failure();
    }
    auto yieldOp =
        cast<linalg::YieldOp>(transposePastKCacheOp.getBody()->getTerminator());
    if (yieldOp->getOperand(0) !=
        transposePastKCacheOp.getBody()->getArgument(0)) {
      return failure();
    }

    // Check rank and shapes of bmm.
    auto bmmRhsShape =
        dyn_cast<TensorType>(bmmOp.getInputs()[1].getType()).getShape();
    int64_t numHeads = bmmRhsShape[0];
    int64_t hiddenDim = bmmRhsShape[2];

    // Check rank and shapes of transpose.
    auto transposeInType =
        dyn_cast<TensorType>(transposePastKCacheOp.getInputs()[0].getType());
    if (!transposeInType)
      return failure();
    if (transposeInType.getRank() != 3)
      return failure();
    auto transposeInShape = transposeInType.getShape();
    if (transposeInShape[1] != numHeads || transposeInShape[2] != hiddenDim)
      return failure();

    // Check shapes and offsets of insertPastKCacheOp.
    if (llvm::any_of(insertPastKCacheOp.getStaticOffsets(),
                     [&](int64_t offset) { return offset != 0; })) {
      return failure();
    }
    ArrayRef<int64_t> pastKCacheStaticSize =
        insertPastKCacheOp.getStaticSizes();
    if (pastKCacheStaticSize.size() != 4)
      return failure();
    if (pastKCacheStaticSize[0] != 1 || pastKCacheStaticSize[1] != numHeads ||
        pastKCacheStaticSize[3] != hiddenDim)
      return failure();
    if (insertPastKCacheOp.getSizes().size() != 1)
      return failure();
    auto seqLen = insertPastKCacheOp.getSizes()[0];

    // Check shapes and offsets of insertNewKSliceOp.
    ArrayRef<int64_t> newKSliceStaticSize = insertNewKSliceOp.getStaticSizes();
    if (newKSliceStaticSize.size() != 4)
      return failure();
    if (newKSliceStaticSize[1] != numHeads || newKSliceStaticSize[2] != 1 ||
        newKSliceStaticSize[3] != hiddenDim)
      return failure();
    ArrayRef<int64_t> newKSliceStaticOffsets =
        insertNewKSliceOp.getStaticOffsets();
    if (newKSliceStaticOffsets[1] != 0 || newKSliceStaticOffsets[3] != 0)
      return failure();

    auto newKSliceSourceType =
        dyn_cast<TensorType>(insertNewKSliceOp.getSource().getType());
    if (!newKSliceSourceType)
      return failure();
    if (newKSliceSourceType.getRank() != 2)
      return failure();
    if (newKSliceSourceType.getShape()[0] != numHeads ||
        newKSliceSourceType.getShape()[1] != hiddenDim)
      return failure();

    // Check that newKSlice is indeed concat to past K cache.
    if (insertNewKSliceOp.getOffsets().size() != 1)
      return failure();
    if (insertNewKSliceOp.getOffsets()[0] != seqLen) {
      return failure();
    }

    // Setting up values for new computations.
    auto loc = bmmOp.getLoc();
    Value query = bmmOp.getInputs()[0];
    Value PastKCache = transposePastKCacheOp.getInputs()[0];
    auto bmmResultType = dyn_cast<TensorType>(bmmOp.getResults()[0].getType());
    if (!bmmResultType)
      return failure();
    int64_t bmmRank = 3;
    // Slice query in K-dim.
    SmallVector<OpFoldResult> queryForKCacheOffsets(bmmRank,
                                                    rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> queryForKCacheStrides(bmmRank,
                                                    rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> queryForKCacheSizes =
        tensor::getMixedSizes(rewriter, loc, query);
    queryForKCacheSizes[bmmRank - 1] = seqLen;
    auto querySliceForKCache = rewriter.create<tensor::ExtractSliceOp>(
        loc, query, queryForKCacheOffsets, queryForKCacheSizes,
        queryForKCacheStrides);

    SmallVector<OpFoldResult> queryForNewKSliceOffsets(
        bmmRank, rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> queryForNewKSliceStrides(
        bmmRank, rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> queryForNewKSliceSizes =
        tensor::getMixedSizes(rewriter, loc, query);
    queryForNewKSliceOffsets[bmmRank - 1] = seqLen;
    queryForNewKSliceSizes[bmmRank - 1] = rewriter.getIndexAttr(1);
    auto querySliceForNewKSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, query, queryForNewKSliceOffsets, queryForNewKSliceSizes,
        queryForNewKSliceStrides);

    // TODO: Compose transpose affineMap to generate new matmul.
    // TODO: Add check if seqLen is dynamic or no.
    // Create new generic to represent bmm with past K cache with collapse_shape
    // baked in. Creates empty and fill for init.
    // %169 = linalg.batch_matmul ins(%expanded_142, %collapsed_148 :
    // tensor<32x1x?xf16>, tensor<32x?x128xf16>) outs(%168 :
    // tensor<32x1x128xf16>) -> tensor<32x1x128xf16> util.return %169 :
    // tensor<32x1x?xf16> tensor<32x1x?xf16>, tensor<?x32x128xf16> ->
    // tensor<32x1x128xf16> (d0, d1, d2, d3) -> (d0, d1, d3), (d3, d0, d2), (d0,
    // d1, d2)
    llvm::SmallVector<Value> QKcacheDynSizes = {};
    Value QKCacheAcc =
        rewriter.create<tensor::EmptyOp>(loc, bmmResultType, QKcacheDynSizes);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(bmmResultType.getElementType()));
    Value QKcacheZeroAcc =
        rewriter.create<linalg::FillOp>(loc, zero, QKCacheAcc).getResults()[0];

    // Setting up iterator types and indexing maps.
    int KCacheRank = pastKCacheStaticSize.size();
    SmallVector<utils::IteratorType> QKCacheIterators(
        KCacheRank, utils::IteratorType::parallel);
    QKCacheIterators[pastKCacheStaticSize.size() - 1] =
        utils::IteratorType::reduction;

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);
    AffineExpr d2 = rewriter.getAffineDimExpr(2);
    AffineExpr d3 = rewriter.getAffineDimExpr(3);

    SmallVector<AffineExpr> QKCacheLhsExprs = {d0, d1, d3};
    SmallVector<AffineExpr> KCacheExprs = {d3, d0, d2};
    SmallVector<AffineExpr> QKCacheOutexprs = {d0, d1, d2};
    SmallVector<AffineMap> QKCacheAffineMaps{
        AffineMap::get(KCacheRank, 0, QKCacheLhsExprs, rewriter.getContext()),
        AffineMap::get(KCacheRank, 0, KCacheExprs, rewriter.getContext()),
        AffineMap::get(KCacheRank, 0, QKCacheOutexprs, rewriter.getContext())};

    // Create the new compute for QKCache.
    auto QKCache =
        rewriter
            .create<linalg::GenericOp>(
                loc, QKcacheZeroAcc.getType(),
                ValueRange{querySliceForKCache, PastKCache},
                ValueRange{QKcacheZeroAcc}, QKCacheAffineMaps, QKCacheIterators,
                [=](OpBuilder &b, Location loc, ValueRange args) {
                  Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
                  Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
                  b.create<linalg::YieldOp>(loc, add);
                })
            .getResult(0);

    // Create new generic to represent bmm with new K slice.
    auto newKSlice = insertNewKSliceOp.getSource();
    SmallVector<ReassociationIndices> reassoc = {ReassociationIndices{0, 1},
                                                 ReassociationIndices{2}};
    auto expandedType = RankedTensorType::get({numHeads, 1, hiddenDim},
                                              bmmResultType.getElementType());
    Value expandedKSlice = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, newKSlice, reassoc);
    auto QNewKSlice =
        rewriter
            .create<linalg::BatchMatmulOp>(
                loc, ValueRange{querySliceForNewKSlice, expandedKSlice},
                ValueRange{QKcacheZeroAcc})
            .getResult(0);
    SmallVector<AffineMap> combinedMaps{
        rewriter.getMultiDimIdentityMap(bmmRank),
        rewriter.getMultiDimIdentityMap(bmmRank),
        rewriter.getMultiDimIdentityMap(bmmRank)};
    SmallVector<utils::IteratorType> combineIterators(
        bmmRank, utils::IteratorType::parallel);
    auto combined =
        rewriter
            .create<linalg::GenericOp>(
                loc, QKcacheZeroAcc.getType(), ValueRange{QKCache, QNewKSlice},
                ValueRange{QKcacheZeroAcc}, combinedMaps, combineIterators,
                [=](OpBuilder &b, Location loc, ValueRange args) {
                  Value add = b.create<arith::AddFOp>(loc, args[0], args[1]);
                  b.create<linalg::YieldOp>(loc, add);
                })
            .getResult(0);
    rewriter.replaceOp(bmmOp, combined);
    return success();
  }
};

// This pattern does a basic fusion of two matmuls and three linalg.generics.
// The pattern matches only on a DAG representing:
// output = Silu(matmul(A, B) * matmul(A, C).
class ConvertMemoryEfficientQKTransposePattern final
    : public OpRewritePattern<linalg::BatchMatmulTransposeBOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::BatchMatmulTransposeBOp bmmtOp,
                                PatternRewriter &rewriter) const override {
    // Check that correct pattern of Ops matches.
    auto fillOp = dyn_cast_or_null<linalg::FillOp>(
        bmmtOp.getDpsInits()[0].getDefiningOp());
    if (!fillOp)
      return failure();

    auto initEmptyOp = dyn_cast_or_null<tensor::EmptyOp>(
        fillOp.getDpsInits()[0].getDefiningOp());
    if (!initEmptyOp)
      return failure();

    auto collapseOp = dyn_cast_or_null<tensor::CollapseShapeOp>(
        bmmtOp.getInputs()[1].getDefiningOp());
    if (!collapseOp)
      return failure();

    auto insertNewKSliceOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        collapseOp.getSrc().getDefiningOp());
    if (!insertNewKSliceOp)
      return failure();

    auto insertPastKCacheOp = dyn_cast_or_null<tensor::InsertSliceOp>(
        insertNewKSliceOp.getDest().getDefiningOp());
    if (!insertPastKCacheOp)
      return failure();

    auto transposePastKCacheOp = dyn_cast_or_null<linalg::GenericOp>(
        insertPastKCacheOp.getSource().getDefiningOp());
    if (!transposePastKCacheOp)
      return failure();
    auto transposeIndexingMaps = transposePastKCacheOp.getIndexingMapsArray();
    if (!transposeIndexingMaps[0].isIdentity())
      return failure();
    if (!transposeIndexingMaps[1].isPermutation())
      return failure();
    if (transposePastKCacheOp.getNumDpsInits() != 1 ||
        transposePastKCacheOp.getNumDpsInputs() != 1) {
      return failure();
    }
    auto yieldOp =
        cast<linalg::YieldOp>(transposePastKCacheOp.getBody()->getTerminator());
    if (yieldOp->getOperand(0) !=
        transposePastKCacheOp.getBody()->getArgument(0)) {
      return failure();
    }

    // Check rank and shapes of bmm.
    auto bmmtLhsShape =
        dyn_cast<TensorType>(bmmtOp.getInputs()[0].getType()).getShape();
    int64_t numHeads = bmmtLhsShape[0];
    int64_t hiddenDim = bmmtLhsShape[2];

    // Check rank and shapes of transpose.
    auto transposeInType =
        dyn_cast<TensorType>(transposePastKCacheOp.getInputs()[0].getType());
    if (!transposeInType)
      return failure();
    if (transposeInType.getRank() != 3)
      return failure();
    auto transposeInShape = transposeInType.getShape();
    if (transposeInShape[1] != numHeads || transposeInShape[2] != hiddenDim)
      return failure();

    // Check shapes and offsets of insertPastKCacheOp.
    if (llvm::any_of(insertPastKCacheOp.getStaticOffsets(),
                     [&](int64_t offset) { return offset != 0; })) {
      return failure();
    }
    ArrayRef<int64_t> pastKCacheStaticSize =
        insertPastKCacheOp.getStaticSizes();
    if (pastKCacheStaticSize.size() != 4)
      return failure();
    if (pastKCacheStaticSize[0] != 1 || pastKCacheStaticSize[1] != numHeads ||
        pastKCacheStaticSize[3] != hiddenDim)
      return failure();
    if (insertPastKCacheOp.getSizes().size() != 1)
      return failure();
    auto seqLen = insertPastKCacheOp.getSizes()[0];

    // Check shapes and offsets of insertNewKSliceOp.
    ArrayRef<int64_t> newKSliceStaticSize = insertNewKSliceOp.getStaticSizes();
    if (newKSliceStaticSize.size() != 4)
      return failure();
    if (newKSliceStaticSize[1] != numHeads || newKSliceStaticSize[2] != 1 ||
        newKSliceStaticSize[3] != hiddenDim)
      return failure();
    ArrayRef<int64_t> newKSliceStaticOffsets =
        insertNewKSliceOp.getStaticOffsets();
    if (newKSliceStaticOffsets[1] != 0 || newKSliceStaticOffsets[3] != 0)
      return failure();

    auto newKSliceSourceType =
        dyn_cast<TensorType>(insertNewKSliceOp.getSource().getType());
    if (!newKSliceSourceType)
      return failure();
    if (newKSliceSourceType.getRank() != 2)
      return failure();
    if (newKSliceSourceType.getShape()[0] != numHeads ||
        newKSliceSourceType.getShape()[1] != hiddenDim)
      return failure();

    // Check that newKSlice is indeed concat to past K cache.
    if (insertNewKSliceOp.getOffsets().size() != 1)
      return failure();
    if (insertNewKSliceOp.getOffsets()[0] != seqLen) {
      return failure();
    }

    // Setting up values for new computations.
    auto loc = bmmtOp.getLoc();
    Value query = bmmtOp.getInputs()[0];
    Value PastKCache = transposePastKCacheOp.getInputs()[0];
    auto bmmtResultType =
        dyn_cast<TensorType>(bmmtOp.getResults()[0].getType());
    if (!bmmtResultType)
      return failure();
    // TODO: Compose transpose affineMap to generate new matmul.
    // TODO: Add check if seqLen is dynamic or no.
    // Create new generic to represent bmm with past K cache with collapse_shape
    // baked in. Creates empty and fill for init.
    llvm::SmallVector<Value> QKcacheDynSizes = {seqLen};
    llvm::SmallVector<int64_t> QKcacheStaticSizes = {numHeads, 1,
                                                     ShapedType::kDynamic};
    auto resultQKcacheType = RankedTensorType::get(
        QKcacheStaticSizes, bmmtResultType.getElementType());
    Value QKCacheAcc = rewriter.create<tensor::EmptyOp>(loc, resultQKcacheType,
                                                        QKcacheDynSizes);
    Value zero = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(bmmtResultType.getElementType()));
    Value QKcacheZeroAcc =
        rewriter.create<linalg::FillOp>(loc, zero, QKCacheAcc).getResults()[0];

    // Setting up iterator types and indexing maps.
    int KCacheRank = pastKCacheStaticSize.size();
    SmallVector<utils::IteratorType> QKCacheIterators(
        KCacheRank, utils::IteratorType::parallel);
    QKCacheIterators[pastKCacheStaticSize.size() - 1] =
        utils::IteratorType::reduction;

    AffineExpr d0 = rewriter.getAffineDimExpr(0);
    AffineExpr d1 = rewriter.getAffineDimExpr(1);
    AffineExpr d2 = rewriter.getAffineDimExpr(2);
    AffineExpr d3 = rewriter.getAffineDimExpr(3);

    SmallVector<AffineExpr> QKCacheLhsExprs = {d0, d1, d3};
    SmallVector<AffineExpr> KCacheExprs = {d2, d0, d3};
    SmallVector<AffineExpr> QKCacheOutexprs = {d0, d1, d2};
    SmallVector<AffineMap> QKCacheAffineMaps{
        AffineMap::get(KCacheRank, 0, QKCacheLhsExprs, rewriter.getContext()),
        AffineMap::get(KCacheRank, 0, KCacheExprs, rewriter.getContext()),
        AffineMap::get(KCacheRank, 0, QKCacheOutexprs, rewriter.getContext())};

    // Create the new compute for QKCache.
    auto QKCache =
        rewriter
            .create<linalg::GenericOp>(
                loc, QKcacheZeroAcc.getType(), ValueRange{query, PastKCache},
                ValueRange{QKcacheZeroAcc}, QKCacheAffineMaps, QKCacheIterators,
                [=](OpBuilder &b, Location loc, ValueRange args) {
                  Value mul = b.create<arith::MulFOp>(loc, args[0], args[1]);
                  Value add = b.create<arith::AddFOp>(loc, mul, args[2]);
                  b.create<linalg::YieldOp>(loc, add);
                })
            .getResult(0);

    // Create new generic to represent bmm with new K slice.
    llvm::SmallVector<Value> dyn = {};
    auto resultQNewKSliceType = RankedTensorType::get(
        {numHeads, bmmtLhsShape[1], newKSliceStaticSize[2]},
        bmmtResultType.getElementType());
    Value QNewKSliceAcc =
        rewriter.create<tensor::EmptyOp>(loc, resultQNewKSliceType, dyn);
    Value QNewKSliceZeroAcc =
        rewriter.create<linalg::FillOp>(loc, zero, QNewKSliceAcc)
            .getResults()[0];
    auto newKSlice = insertNewKSliceOp.getSource();
    SmallVector<ReassociationIndices> reassoc = {ReassociationIndices{0, 1},
                                                 ReassociationIndices{2}};
    auto expandedType = RankedTensorType::get({numHeads, 1, hiddenDim},
                                              bmmtResultType.getElementType());
    Value expandedKSlice = rewriter.create<tensor::ExpandShapeOp>(
        loc, expandedType, newKSlice, reassoc);
    auto QNewKSlice = rewriter
                          .create<linalg::BatchMatmulTransposeBOp>(
                              loc, ValueRange{query, expandedKSlice},
                              ValueRange{QNewKSliceZeroAcc})
                          .getResult(0);

    // Combine them into a single slice.
    auto indexOne = rewriter.getIndexAttr(1);
    auto indexZero = rewriter.getIndexAttr(0);
    int64_t bmmRank = 3;
    SmallVector<OpFoldResult> QKCacheInsertstrides(bmmRank, indexOne);
    SmallVector<OpFoldResult> QKCacheInsertoffsets(bmmRank, indexZero);
    SmallVector<OpFoldResult> QKCacheInsertsizes = {
        rewriter.getIndexAttr(numHeads), indexOne, seqLen};
    Value insertQKCache = rewriter.create<tensor::InsertSliceOp>(
        loc, QKCache, initEmptyOp, QKCacheInsertoffsets, QKCacheInsertsizes,
        QKCacheInsertstrides);

    SmallVector<OpFoldResult> QNewKSliceInsertstrides(bmmRank, indexOne);
    SmallVector<OpFoldResult> QNewKSliceInsertoffsets = {indexZero, indexZero,
                                                         seqLen};
    SmallVector<OpFoldResult> QNewKSliceInsertsizes = {
        rewriter.getIndexAttr(numHeads), indexOne, indexOne};
    rewriter.replaceOpWithNewOp<tensor::InsertSliceOp>(
        bmmtOp, QNewKSlice, insertQKCache, QNewKSliceInsertoffsets,
        QNewKSliceInsertsizes, QNewKSliceInsertstrides);
    return success();
  }
};

struct ConvertMemoryEfficientQKPass
    : public ConvertMemoryEfficientQKBase<ConvertMemoryEfficientQKPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  ConvertMemoryEfficientQKPass() {}
  ConvertMemoryEfficientQKPass(const ConvertMemoryEfficientQKPass &pass)
      : ConvertMemoryEfficientQKPass() {}

  void runOnOperation() override;
};

} // namespace

void ConvertMemoryEfficientQKPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.insert<ConvertMemoryEfficientQKPattern,
                  ConvertMemoryEfficientQKTransposePattern,
                  ConvertMemoryEfficientKVUpdatePattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createConvertMemoryEfficientQKPass() {
  return std::make_unique<ConvertMemoryEfficientQKPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
