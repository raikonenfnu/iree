// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Utils/Utils.h"
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

#define DEBUG_TYPE "iree-global-opt-simplify-ext-argmax"
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

namespace mlir {
namespace iree_compiler {
namespace GlobalOptimization {

namespace {

// This pattern does a basic fusion of two matmuls and three linalg.generics.
// The pattern matches only on a DAG representing:
// output = Silu(matmul(A, B) * matmul(A, C).
class SimplifyExtArgmaxPattern final
    : public OpRewritePattern<linalg::GenericOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter &rewriter) const override {
    // Check that correct pattern of Ops matches.
    if (failed(isArgmaxOp(op))) {
      return failure();
    }
    auto extractSliceOp =
        dyn_cast<tensor::ExtractSliceOp>(op.getDpsInputs()[0].getDefiningOp());
    if (!extractSliceOp)
      return failure();
    // Does not support dynamic slice as input to argmax.
    if (extractSliceOp.getSizes().size() != 0)
      return failure();

    auto expandOp = dyn_cast<tensor::ExpandShapeOp>(
        extractSliceOp.getSource().getDefiningOp());
    if (!expandOp)
      return failure();
    auto expandSrcType = dyn_cast<TensorType>(expandOp.getSrc().getType());
    if (!expandSrcType)
      return failure();
    auto expandSrcShape = expandSrcType.getShape();
    auto expandDstType = dyn_cast<TensorType>(expandOp.getResult().getType());
    if (!expandDstType)
      return failure();
    auto expandDstShape = expandDstType.getShape();
    if (expandDstShape.size() - expandSrcShape.size() != 1) {
      return failure();
    }
    // Check we are only expanding on the first dimension with unit dim.
    for (int i = 0; i < expandSrcShape.size(); i++) {
      if (expandSrcShape[i] != expandDstShape[i + 1]) {
        return failure();
      }
    }

    auto extGenericOp =
        dyn_cast<linalg::GenericOp>(expandOp.getSrc().getDefiningOp());
    if (!extGenericOp)
      return failure();
    if (extGenericOp.getNumDpsInits() != 1 ||
        extGenericOp.getNumDpsInputs() != 1) {
      return failure();
    }
    // Matching ExtFOp
    auto extGenericIndexingMaps = extGenericOp.getIndexingMapsArray();
    if (!extGenericIndexingMaps[0].isIdentity() ||
        !extGenericIndexingMaps[1].isIdentity())
      return failure();
    auto yieldOp =
        cast<linalg::YieldOp>(extGenericOp.getBody()->getTerminator());
    Value producerOutput = yieldOp->getOperand(0);
    Operation *producer = producerOutput.getDefiningOp();
    if (!producer || producer->getNumOperands() == 0) {
      return failure();
    }
    if (producer->getOperand(0) != extGenericOp.getBody()->getArguments()[0]) {
      return failure();
    }
    if (!matchPattern(producer, m_Op<arith::ExtFOp>())) {
      return failure();
    }
    // Create extract slice on source of extfop with dropped dim.
    auto loc = op.getLoc();
    auto srcElType = dyn_cast<TensorType>(extGenericOp.getInputs()[0].getType())
                         .getElementType();
    auto dstElType =
        dyn_cast<TensorType>(extGenericOp.getOutputs()[0].getType())
            .getElementType();
    SmallVector<OpFoldResult> originalOffsets =
        extractSliceOp.getMixedOffsets();
    SmallVector<OpFoldResult> originalSizes = extractSliceOp.getMixedSizes();
    SmallVector<OpFoldResult> originalStrides =
        extractSliceOp.getMixedStrides();
    SmallVector<OpFoldResult> offsets;
    SmallVector<OpFoldResult> strides;
    SmallVector<OpFoldResult> sizes;
    for (int i = 0; i < expandSrcShape.size(); i++) {
      offsets.push_back(originalOffsets[i + 1]);
      strides.push_back(originalStrides[i + 1]);
      sizes.push_back(originalSizes[i + 1]);
    }
    auto originalExtractSliceType =
        dyn_cast<TensorType>(extractSliceOp.getResult().getType());
    if (!originalExtractSliceType)
      return failure();
    auto newSliceType =
        RankedTensorType::get(originalExtractSliceType.getShape(), srcElType);
    auto newSliceOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, newSliceType, extGenericOp.getInputs()[0], offsets, sizes,
        strides);
    // Create new extfop.
    llvm::SmallVector<Value> emptyDyn = {};
    SmallVector<AffineMap> extAffineMaps{
        rewriter.getMultiDimIdentityMap(
            originalExtractSliceType.getShape().size()),
        rewriter.getMultiDimIdentityMap(
            originalExtractSliceType.getShape().size())};
    SmallVector<utils::IteratorType> extIterators(
        originalExtractSliceType.getShape().size(),
        utils::IteratorType::parallel);
    auto newExtType = RankedTensorType::get(
        dyn_cast<TensorType>(newSliceOp.getResult().getType()).getShape(),
        dstElType);
    Value newExtInit =
        rewriter.create<tensor::EmptyOp>(loc, newExtType, emptyDyn);
    auto newExtFOp =
        rewriter
            .create<linalg::GenericOp>(
                loc, newExtType, ValueRange{newSliceOp}, ValueRange{newExtInit},
                extAffineMaps, extIterators,
                [=](OpBuilder &b, Location loc, ValueRange args) {
                  Value extf = b.create<arith::ExtFOp>(loc, dstElType, args[0]);
                  b.create<linalg::YieldOp>(loc, extf);
                })
            .getResult(0);
    // replace dpsInit of .
    rewriter.startOpModification(op);
    op.setOperand(0, newExtFOp);
    rewriter.finalizeOpModification(op);
    return success();
  }
};

struct SimplifyExtArgmaxPass
    : public SimplifyExtArgmaxBase<SimplifyExtArgmaxPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, IREE::Flow::FlowDialect,
                    math::MathDialect>();
  }
  SimplifyExtArgmaxPass() {}
  SimplifyExtArgmaxPass(const SimplifyExtArgmaxPass &pass)
      : SimplifyExtArgmaxPass() {}

  void runOnOperation() override;
};

} // namespace

void SimplifyExtArgmaxPass::runOnOperation() {
  MLIRContext *context = &getContext();

  RewritePatternSet patterns(context);
  patterns.insert<SimplifyExtArgmaxPattern>(context);
  if (failed(
          applyPatternsAndFoldGreedily(getOperation(), std::move(patterns)))) {
    return signalPassFailure();
  }
}

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createSimplifyExtArgmaxPass() {
  return std::make_unique<SimplifyExtArgmaxPass>();
}

} // namespace GlobalOptimization
} // namespace iree_compiler
} // namespace mlir
