// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <numeric>

#include "iree/compiler/Codegen/Common/PassDetail.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "mlir/Dialect/Affine/Transforms/Transforms.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/MathExtras.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

/// Pattern to combine instructions across ForOp boundary. It is common when
/// doing incremental lowering to generate transient ops that cancel each others
/// out. Canonicalization usually clean up those operations. When the value is
/// loop carried, MLIR canonicalization currently doesn't remove the redundant
/// operations.
///
/// This pass allow to workaround MLIR limitation and does ad hoc clean up of
/// instructions found in IREE. Once we have a more general mechanism in MLIR
/// this pass can be completely removed.
/// This pass does this kind of transformation:
/// ```
/// %21 = vector.shape_cast %20 : vector<4xf32> to vector<1x4xf32>
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %21)
///    -> vector<1x4xf32> {
///    [...]
///    %100 = vector.shape_cast %arg4 : vector<1x4xf32> to vector<4xf32>
///    [...]
///    %109 = vector.shape_cast %108 : vector<4xf32> to vector<1x4xf32>
///    scf.yield %109 : vector<1x4xf32>
///  }
///  %24 = vector.shape_cast %22 : vector<1x4xf32> to vector<4xf32>
/// ```
/// ->
/// ```
/// %22 = scf.for %arg3 = %c0 to %c4096 step %c4 iter_args(%arg4 = %20)
///    -> vector<4xf32> {
///    [...]
///    scf.yield %108 : vector<4xf32>
///  }
/// ```
struct SinkAndCollapseForOpToChildForOp final
    : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  // Transfer the body of `source` into `dest` and update the terminator of
  // `dest` to use the specified results. The result list also replaces
  // any block arguments from `source` with the corresponding block argument
  // in `dest` and returns the updated result list.
  SmallVector<Value> transferBody(Block *source, Block *dest,
                                  ArrayRef<Value> results,
                                  PatternRewriter &rewriter) const {
    // Collect the old block arguments before merging.
    SmallVector<std::optional<int64_t>> maybeBlockArgNum;
    for (auto res : results) {
      if (auto blockArg = dyn_cast<BlockArgument>(res)) {
        maybeBlockArgNum.push_back(blockArg.getArgNumber());
      } else {
        maybeBlockArgNum.push_back(std::nullopt);
      }
    }
    // Move all operations to the destination block.
    rewriter.mergeBlocks(source, dest, dest->getArguments());
    // Replace the yield op by one that returns only the used values.
    auto yieldOp = cast<scf::YieldOp>(dest->getTerminator());
    // Create a new result set with the updated block arguments.
    SmallVector<Value> newResults;
    for (auto [index, argNum] : llvm::enumerate(maybeBlockArgNum)) {
      if (argNum) {
        newResults.push_back(dest->getArgument(*argNum));
      } else {
        newResults.push_back(results[index]);
      }
    }
    rewriter.modifyOpInPlace(
        yieldOp, [&]() { yieldOp.getOperation()->setOperands(newResults); });
    return newResults;
  }

  LogicalResult matchAndRewrite(scf::ForOp parentForOp,
                                PatternRewriter &rewriter) const override {
    // Preconditons:
    // 1. Only single forOp per level, i.e no sibling forOps.
    // 2. For extra security, only handle if iterArgs between parent and child
    // forOps matches perfectly.
    // 3. Both lower bound needs to be 0.
    // 4. (TODO) Output of childForOp goes back to yield of parentForOp.
    // 4. UB and Steps are static and constant.
    SmallVector<scf::ForOp> childForOps;
    for (auto childOp : parentForOp.getOps<scf::ForOp>()) {
      childForOps.push_back(childOp);
    }
    if (childForOps.size() != 1) {
      return failure();
    }
    auto &childForOp = childForOps.back();
    if (childForOp->getParentOp() != parentForOp) {
      return failure();
    }

    // Matching iter args between parentOp and childOp.
    auto parentIterArgs = ValueRange(parentForOp.getRegionIterArgs());
    auto childIterArgs = ValueRange(childForOp.getRegionIterArgs());
    if (parentIterArgs.size() != childIterArgs.size()) {
      return failure();
    }
    for (auto parentIterArg : llvm::enumerate(parentIterArgs)) {
      if (!parentIterArg.value().hasOneUse()) {
        return failure();
      }
      if (childForOp.getInitArgs()[parentIterArg.index()] !=
          parentIterArg.value()) {
        return failure();
      }
    }
    // Ensure lower and upper bounds are static and valid for both forOps.
    auto parentLbInt = getConstantIntValue(parentForOp.getLowerBound());
    if (!parentLbInt)
      return failure();
    if (*parentLbInt != 0)
      return failure();

    auto childLbInt = getConstantIntValue(childForOp.getLowerBound());
    if (!childLbInt)
      return failure();
    if (*childLbInt != 0)
      return failure();

    auto parentUbInt = getConstantIntValue(parentForOp.getUpperBound());
    if (!parentUbInt)
      return failure();

    auto childUbInt = getConstantIntValue(childForOp.getUpperBound());
    if (!childUbInt)
      return failure();

    // Sink non-forOps of parentForOp into the childForOp.
    // TODO: Maybe make two pattern, 1.Sink, 2.Collapse, to prevent one-shotty.
    int numSunkenOps = 0;
    Operation *recentChildOp = &childForOp.getBody()->front();
    for (Operation &childOp :
         llvm::make_early_inc_range(parentForOp.getOps())) {
      if (&childOp == childForOp.getOperation() ||
          dyn_cast<scf::YieldOp>(childOp)) {
        continue;
      }
      rewriter.moveOpBefore(&childOp, recentChildOp);
      numSunkenOps++;
    }
    if (numSunkenOps < 1)
      return failure();

    // Flatten the loop bounds and steps.
    int64_t collapsedUbInt = childUbInt.value() * parentUbInt.value();
    auto collapsedUbValue = rewriter.create<arith::ConstantOp>(
        parentForOp.getLoc(),
        rewriter.getIntegerAttr(parentForOp.getUpperBound().getType(),
                                collapsedUbInt));

    auto parentStep = getConstantIntValue(parentForOp.getStep());
    if (!parentStep)
      return failure();

    auto childStep = getConstantIntValue(childForOp.getStep());
    if (!childStep)
      return failure();

    int64_t collapsedStepInt = childStep.value() * parentStep.value();
    auto collapsedStepValue = rewriter.create<arith::ConstantOp>(
        parentForOp.getLoc(),
        rewriter.getIntegerAttr(parentForOp.getStep().getType(),
                                collapsedStepInt));

    // Create delinearization of indices using the induction variable.
    auto parentIv = parentForOp.getInductionVar();
    auto childIv = childForOp.getInductionVar();
    rewriter.setInsertionPointToStart(childForOp.getBody());
    SmallVector<OpFoldResult> basisIndexAttr = {
        rewriter.getIndexAttr(parentUbInt.value()),
        rewriter.getIndexAttr(childUbInt.value())};
    SmallVector<Value> delinearizedIdx =
        rewriter
            .create<affine::AffineDelinearizeIndexOp>(childForOp.getLoc(),
                                                      childIv, basisIndexAttr)
            .getMultiIndex();
    rewriter.replaceAllUsesWith(parentIv, delinearizedIdx[0]);
    rewriter.replaceAllUsesExcept(childIv, delinearizedIdx[1],
                                  delinearizedIdx[1].getDefiningOp());

    // Create new "collapsed" loop with calculated params and replace original
    // forOp.
    rewriter.setInsertionPoint(parentForOp);
    auto newLoop = rewriter.create<scf::ForOp>(
        parentForOp.getLoc(), parentForOp.getLowerBound(), collapsedUbValue,
        collapsedStepValue, parentForOp.getInitArgs());
    auto terminator = cast<scf::YieldOp>(childForOp.getBody()->getTerminator());
    auto returnValues = llvm::to_vector<8>(terminator.getOperands());
    SmallVector<Value> newReturnVals = transferBody(
        childForOp.getBody(), newLoop.getBody(), returnValues, rewriter);
    rewriter.replaceOp(parentForOp, newLoop.getResults());
    return success();
  }
};

struct CollapseNestedTiledForOpsPass
    : public CollapseNestedTiledForOpsBase<CollapseNestedTiledForOpsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<scf::SCFDialect, vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto fn = getOperation();
    // These patterns collide so we apply them one after another. The
    // canonicalization pattern will be blocked by the packing pattern
    // so we apply that first.
    RewritePatternSet canonPatterns(&getContext());
    canonPatterns.insert<SinkAndCollapseForOpToChildForOp>(fn.getContext());
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(canonPatterns)))) {
      return signalPassFailure();
    }
    RewritePatternSet expandPatterns(&getContext());
    affine::populateAffineExpandIndexOpsPatterns(expandPatterns);
    if (failed(applyPatternsAndFoldGreedily(fn, std::move(expandPatterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<InterfacePass<mlir::FunctionOpInterface>>
createCollapseNestedTiledForOpsPass() {
  return std::make_unique<CollapseNestedTiledForOpsPass>();
}

} // namespace mlir::iree_compiler
