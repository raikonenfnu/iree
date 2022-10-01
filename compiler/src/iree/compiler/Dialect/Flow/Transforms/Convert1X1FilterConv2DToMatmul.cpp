// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

static Value createAdd(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt) return builder.create<arith::AddIOp>(loc, x, y);
  return builder.create<arith::AddFOp>(loc, x, y);
}

static Value createMul(Location loc, Value x, Value y, bool isInt,
                       OpBuilder &builder) {
  if (isInt) return builder.create<arith::MulIOp>(loc, x, y);
  return builder.create<arith::MulFOp>(loc, x, y);
}

// Converts linalg.conv_2d_input_nhwc_filter_nhwc op to linalg.matmul
template <typename Conv2DOpType>
class Convert1x1FilterConvToMatmul : public OpRewritePattern<Conv2DOpType> {
 public:
  using OpRewritePattern<Conv2DOpType>::OpRewritePattern;

  LogicalResult matchAndRewrite(Conv2DOpType convOp,
                                PatternRewriter &rewriter) const override {
    auto inputShapeType = convOp.getInputOperand(0)
                              ->get()
                              .getType()
                              .template dyn_cast<RankedTensorType>();
    auto filterShapeType = convOp.getInputOperand(1)
                               ->get()
                               .getType()
                               .template dyn_cast<RankedTensorType>();
    auto outputShapeType = convOp.getOutputOperand(0)
                               ->get()
                               .getType()
                               .template dyn_cast<RankedTensorType>();

    const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(convOp);
    const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(convOp);
    if (!isNCHW & !isNHWC) return failure();

    if (!inputShapeType || !filterShapeType || !outputShapeType)
      return failure();

    auto inputShape = inputShapeType.getShape();
    auto filterShape = filterShapeType.getShape();
    auto outputShape = outputShapeType.getShape();

    const bool isBatched = inputShape[0] >= 1;

    // Adjusting dimension indices based on Conv2DOpType.
    const int nIndex = 0;
    const int kcIndex = isNHWC ? 2 : 1;
    const int kfIndex = isNHWC ? 3 : 0;
    const int khIndex = isNHWC ? 0 : 2;
    const int kwIndex = isNHWC ? 1 : 3;
    const int ohIndex = isNHWC ? 1 : 2;
    const int owIndex = isNHWC ? 2 : 3;
    const int ocIndex = isNHWC ? 3 : 1;

    bool isInputHWDynamic = inputShape[ohIndex] == ShapedType::kDynamicSize &&
                            inputShape[owIndex] == ShapedType::kDynamicSize;

    // We cannot merge the width and height if they are both dynamic as we
    // cannot expand them back to their dynamic values.
    if (isInputHWDynamic) return failure();

    if (filterShape[khIndex] != 1 || filterShape[kwIndex] != 1)
      return failure();

    // TODO(ataei): Support conversion to linalg.batch_matmul.
    if (isBatched & isNHWC) return failure();

    if (!llvm::all_of(convOp.getStrides(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();
    if (!llvm::all_of(convOp.getDilations(), [](APInt element) {
          return element.getSExtValue() == 1;
        }))
      return failure();

    auto combineDims = [](int64_t a, int64_t b) {
      if (a == ShapedType::kDynamicSize || b == ShapedType::kDynamicSize)
        return ShapedType::kDynamicSize;
      return a * b;
    };

    SmallVector<ReassociationIndices, 4> reassociationInputOutputIndices;
    SmallVector<ReassociationIndices, 4> reassociationFilterIndices;
    SmallVector<int64_t> reshapedInputShape(2, 0);
    SmallVector<int64_t> reshapedFilterShape(2, 0);
    SmallVector<int64_t> reshapedOutputShape(2, 0);
    if (isNHWC) {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ohIndex, owIndex}, {ocIndex}};
      reassociationFilterIndices = {{khIndex, kwIndex, kcIndex}, {kfIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          combineDims(inputShape[ohIndex], inputShape[owIndex]),
          inputShape[ocIndex]};
      reshapedFilterShape = {filterShape[kcIndex], filterShape[kfIndex]};
      reshapedOutputShape = {
          combineDims(outputShape[ohIndex], outputShape[owIndex]),
          outputShape[ocIndex]};
    } else if (isNCHW) {
      if(isBatched) {
        // Generate reassociation indices.
        reassociationInputOutputIndices = {{nIndex}, {ocIndex}, {ohIndex, owIndex}};
        reassociationFilterIndices = {{kfIndex}, {kcIndex, khIndex, kwIndex}};

        // Generate matmul shapes from 1x1 conv.
        reshapedInputShape = {
            inputShape[0],
            inputShape[ocIndex],
            combineDims(inputShape[ohIndex], inputShape[owIndex])};
        reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
        reshapedOutputShape = {
            outputShape[0],
            outputShape[ocIndex],
            combineDims(outputShape[ohIndex], outputShape[owIndex])};
      } else {
      // Generate reassociation indices.
      reassociationInputOutputIndices = {{nIndex, ocIndex}, {ohIndex, owIndex}};
      reassociationFilterIndices = {{kfIndex}, {kcIndex, khIndex, kwIndex}};

      // Generate matmul shapes from 1x1 conv.
      reshapedInputShape = {
          inputShape[ocIndex],
          combineDims(inputShape[ohIndex], inputShape[owIndex])};
      reshapedFilterShape = {filterShape[kfIndex], filterShape[kcIndex]};
      reshapedOutputShape = {
          outputShape[ocIndex],
          combineDims(outputShape[ohIndex], outputShape[owIndex])};
      }
    }

    auto reshapedInputType = RankedTensorType::get(
        reshapedInputShape, inputShapeType.getElementType());

    auto reshapedFilterType = RankedTensorType::get(
        reshapedFilterShape, filterShapeType.getElementType());

    auto reshapedOutputType = RankedTensorType::get(
        reshapedOutputShape, outputShapeType.getElementType());

    Value input = convOp.getInputOperand(0)->get();
    Value filter = convOp.getInputOperand(1)->get();
    Value output = convOp.getOutputOperand(0)->get();
    auto loc = convOp.getLoc();

    Value reshapedInput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedInputType, input, reassociationInputOutputIndices);
    Value reshapedFilter = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedFilterType, filter, reassociationFilterIndices);
    Value reshapedOutput = rewriter.create<tensor::CollapseShapeOp>(
        loc, reshapedOutputType, output, reassociationInputOutputIndices);

    SmallVector<Value, 2> matmulInput;
    if (isNHWC) {
      matmulInput = {reshapedInput, reshapedFilter};
    } else if (isNCHW) {
      matmulInput = {reshapedFilter, reshapedInput};
    }
    Value matmulResult;
    if(isBatched) {
      MLIRContext* ctx = rewriter.getContext();
      AffineExpr bDim, mDim, nDim, kDim;
      bindDims(ctx, bDim, mDim, nDim, kDim);
      auto lhsMap = AffineMap::get(4, 0, {mDim, kDim}, ctx);
      auto rhsMap = AffineMap::get(4, 0, {bDim, kDim, nDim}, ctx);
      auto resultMap = AffineMap::get(4, 0, {bDim, mDim, nDim}, ctx);
      StringRef parallel = getParallelIteratorTypeName();
      StringRef reduction = getReductionIteratorTypeName();
      SmallVector<StringRef> genericIterators = {parallel, parallel, parallel,
                                                 reduction};
      bool isInt = outputShapeType.getElementType().template isa<IntegerType>();
      matmulResult = rewriter.create<linalg::GenericOp>(
          loc, reshapedOutputType,
          /*inputs=*/matmulInput,
          /*outputs=*/ValueRange{reshapedOutput},
          ArrayRef<AffineMap>{lhsMap, rhsMap, resultMap}, genericIterators,
          [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
            Value mul = createMul(loc, args[0], args[1], isInt, nestedBuilder);
            Value add = createAdd(loc, mul, args[2], isInt, nestedBuilder);
            nestedBuilder.create<linalg::YieldOp>(nestedLoc, add);
          }).getResults().front();
    } else {
    matmulResult = rewriter.create<linalg::MatmulOp>(
        loc, reshapedOutputType, matmulInput, ArrayRef<Value>{reshapedOutput}).getResults().front();
    }

    auto reshapedResult = rewriter.create<tensor::ExpandShapeOp>(
        loc, outputShapeType, matmulResult,
        reassociationInputOutputIndices);

    rewriter.replaceOp(convOp, ArrayRef<Value>{reshapedResult});

    return success();
  }
};

struct Convert1X1FilterConv2DToMatmulPass
    : public Convert1X1FilterConv2DToMatmulBase<
          Convert1X1FilterConv2DToMatmulPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(&getContext());
    patterns.insert<Convert1x1FilterConvToMatmul<linalg::Conv2DNhwcHwcfOp>,
                    Convert1x1FilterConvToMatmul<linalg::Conv2DNchwFchwOp>>(
        context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace

std::unique_ptr<Pass> createConvert1X1FilterConv2DToMatmulPass() {
  return std::make_unique<Convert1X1FilterConv2DToMatmulPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
