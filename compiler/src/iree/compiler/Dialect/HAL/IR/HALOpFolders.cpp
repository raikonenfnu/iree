// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// hal.tensor.import/export
//===----------------------------------------------------------------------===//

OpFoldResult TensorImportOp::fold(ArrayRef<Attribute> operands) {
  if (auto exportOp = getSource().getDefiningOp<TensorExportOp>()) {
    if (exportOp.getSource().getType() == getTarget().getType() &&
        exportOp.getSourceEncoding() == getTargetEncoding()) {
      return exportOp.getSource();
    }
  }
  return {};
}

OpFoldResult TensorExportOp::fold(ArrayRef<Attribute> operands) {
  if (auto importOp = getSource().getDefiningOp<TensorImportOp>()) {
    if (importOp.getSource().getType() == getTarget().getType() &&
        importOp.getTargetEncoding() == getSourceEncoding()) {
      return importOp.getSource();
    }
  }
  return {};
}

//===----------------------------------------------------------------------===//
// hal.buffer_view.*
//===----------------------------------------------------------------------===//

namespace {

/// Folds hal.buffer.subspans into buffer view creation subspans.
struct FoldBufferViewCreateSubspan
    : public OpRewritePattern<BufferViewCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(BufferViewCreateOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = op.getSourceOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getSourceBuffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getSourceOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
    });
    return success();
  }
};

}  // namespace

void BufferViewCreateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<FoldBufferViewCreateSubspan>(context);
}

namespace {

/// Skips a hal.buffer_view.buffer accessor when the buffer view was created in
/// the same scope and we know the origin buffer.
struct SkipBufferViewBufferOp : public OpRewritePattern<BufferViewBufferOp> {
  using OpRewritePattern<BufferViewBufferOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(BufferViewBufferOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<BufferViewCreateOp>(
            op.getBufferView().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.getSourceBuffer());
      return success();
    }
    return failure();
  }
};

}  // namespace

void BufferViewBufferOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                     MLIRContext *context) {
  results.insert<SkipBufferViewBufferOp>(context);
}

//===----------------------------------------------------------------------===//
// hal.command_buffer.*
//===----------------------------------------------------------------------===//

namespace {

/// Skips a hal.command_buffer.device accessor when the device was created in
/// the same scope.
struct SkipCommandBufferDeviceOp
    : public OpRewritePattern<CommandBufferDeviceOp> {
  using OpRewritePattern<CommandBufferDeviceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferDeviceOp op,
                                PatternRewriter &rewriter) const override {
    if (auto createOp = dyn_cast_or_null<CommandBufferCreateOp>(
            op.getCommandBuffer().getDefiningOp())) {
      rewriter.replaceOp(op, createOp.getDevice());
      return success();
    }
    return failure();
  }
};

}  // namespace

void CommandBufferDeviceOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<SkipCommandBufferDeviceOp>(context);
}

namespace {

/// Folds hal.buffer.subspans into buffer fill offsets.
struct FoldCommandBufferFillBufferSubspans
    : public OpRewritePattern<CommandBufferFillBufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferFillBufferOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newTargetBuffer = op.getTargetBuffer();
    auto newTargetOffset = op.getTargetOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getTargetBuffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.getSourceBuffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getTargetOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getTargetBufferMutable().assign(newTargetBuffer);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

}  // namespace

void CommandBufferFillBufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferFillBufferSubspans>(context);
}

namespace {

/// Folds hal.buffer.subspans into buffer copy offsets.
struct FoldCommandBufferCopyBufferSubspans
    : public OpRewritePattern<CommandBufferCopyBufferOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferCopyBufferOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto newSourceBuffer = op.getSourceBuffer();
    auto newSourceOffset = op.getSourceOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getSourceBuffer().getDefiningOp())) {
      newSourceBuffer = subspanOp.getSourceBuffer();
      newSourceOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getSourceOffset());
      needsUpdate = true;
    }
    auto newTargetBuffer = op.getTargetBuffer();
    auto newTargetOffset = op.getTargetOffset();
    if (auto subspanOp = dyn_cast_or_null<BufferSubspanOp>(
            op.getTargetBuffer().getDefiningOp())) {
      newTargetBuffer = subspanOp.getSourceBuffer();
      newTargetOffset = rewriter.createOrFold<mlir::arith::AddIOp>(
          subspanOp.getLoc(), subspanOp.getSourceOffset(),
          op.getTargetOffset());
      needsUpdate = true;
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      op.getSourceBufferMutable().assign(newSourceBuffer);
      op.getSourceOffsetMutable().assign(newSourceOffset);
      op.getTargetBufferMutable().assign(newTargetBuffer);
      op.getTargetOffsetMutable().assign(newTargetOffset);
    });
    return success();
  }
};

}  // namespace

void CommandBufferCopyBufferOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferCopyBufferSubspans>(context);
}

namespace {

/// Folds hal.buffer.subspans into push descriptor bindings.
/// The binding range is always equal to or a subset of the subspan.
struct FoldCommandBufferPushDescriptorSetBufferSubspan
    : public OpRewritePattern<CommandBufferPushDescriptorSetOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(CommandBufferPushDescriptorSetOp op,
                                PatternRewriter &rewriter) const override {
    auto ip = rewriter.saveInsertionPoint();
    rewriter.setInsertionPoint(op);
    bool needsUpdate = false;
    auto bindingBuffers = llvm::to_vector<4>(op.getBindingBuffers());
    auto bindingOffsets = llvm::to_vector<4>(op.getBindingOffsets());
    for (size_t i = 0; i < bindingBuffers.size(); ++i) {
      auto *definingOp = bindingBuffers[i].getDefiningOp();
      if (!definingOp) continue;
      if (auto subspanOp = dyn_cast<BufferSubspanOp>(definingOp)) {
        needsUpdate = true;
        bindingBuffers[i] = subspanOp.getSourceBuffer();
        bindingOffsets[i] = rewriter.createOrFold<mlir::arith::AddIOp>(
            subspanOp.getLoc(), subspanOp.getSourceOffset(), bindingOffsets[i]);
      }
    }
    rewriter.restoreInsertionPoint(ip);
    if (!needsUpdate) return failure();
    rewriter.updateRootInPlace(op, [&]() {
      auto mutableBindingBuffers = op.getBindingBuffersMutable();
      mutableBindingBuffers.clear();
      mutableBindingBuffers.append(bindingBuffers);
      auto mutableBindingOffsets = op.getBindingOffsetsMutable();
      mutableBindingOffsets.clear();
      mutableBindingOffsets.append(bindingOffsets);
    });
    return success();
  }
};

}  // namespace

void CommandBufferPushDescriptorSetOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<FoldCommandBufferPushDescriptorSetBufferSubspan>(context);
}

//===----------------------------------------------------------------------===//
// hal.device.switch
//===----------------------------------------------------------------------===//

// TODO(benvanik): fold conditions with the same IR tree.
// TODO(benvanik): remove duplicate conditions.
// TODO(benvanik): fold condition expressions (any(always, ...) -> always, etc).
// TODO(benvanik): completely replace switches with just one always block.
// TODO(benvanik): remove conditions with no side-effects.

//===----------------------------------------------------------------------===//
// hal.device.match.id
//===----------------------------------------------------------------------===//

// TODO(benvanik): fold matches that are known true based on device config.

//===----------------------------------------------------------------------===//
// hal.executable.*
//===----------------------------------------------------------------------===//

namespace {

// Returns a set of fused locations for each result from all return sites.
static SmallVector<Location> gatherResultLocations(int numResults,
                                                   Region &region) {
  SmallVector<SmallVector<Location>> allLocs;
  allLocs.resize(numResults);
  for (auto returnOp : region.getOps<IREE::HAL::ReturnOp>()) {
    for (auto [i, result] : llvm::enumerate(returnOp.getOperands())) {
      allLocs[i].push_back(result.getLoc());
    }
  }
  return llvm::to_vector(llvm::map_range(allLocs, [&](auto resultLocs) {
    return FusedLoc::get(region.getContext(), resultLocs);
  }));
}

// Rewrites |region| to have a single hal.return with all prior return sites
// branching to it. Upon return the exit block may not be the last!
static void rewriteToOneReturn(int numResults, Region &region,
                               PatternRewriter &rewriter) {
  // Get all of the return ops - if there's only one then the requirement is
  // already satisfied and we can exit early.
  auto returnOps = llvm::to_vector(region.getOps<IREE::HAL::ReturnOp>());
  if (returnOps.size() <= 1) return;  // no-op
  SmallVector<Location> returnLocs;
  for (auto returnOp : returnOps) returnLocs.push_back(returnOp.getLoc());

  // Create the new exit block with arguments matching 1:1 with results.
  auto anyReturnOp = returnOps.front();
  auto resultLocs = gatherResultLocations(anyReturnOp.getNumOperands(), region);
  auto &exitBlock = region.emplaceBlock();
  exitBlock.addArguments(anyReturnOp.getOperandTypes(), resultLocs);
  OpBuilder::atBlockBegin(&exitBlock)
      .create<IREE::HAL::ReturnOp>(
          FusedLoc::get(region.getContext(), returnLocs),
          exitBlock.getArguments());

  // Rewrite all return ops to branch to the exit block.
  for (auto returnOp : returnOps) {
    OpBuilder(returnOp).create<cf::BranchOp>(returnOp.getLoc(), &exitBlock,
                                             returnOp.getOperands());
    rewriter.eraseOp(returnOp);
  }
}

/// Merges hal.executable.constant.block ops together into one.
/// Duplicate keys are ignored and will be cleaned up by
/// DeduplicateExecutableConstantBlockKeys.
struct MergeExecutableConstantBlocks
    : public OpRewritePattern<ExecutableVariantOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExecutableVariantOp variantOp,
                                PatternRewriter &rewriter) const override {
    auto blockOps =
        llvm::to_vector(variantOp.getOps<ExecutableConstantBlockOp>());
    if (blockOps.size() <= 1) {
      return rewriter.notifyMatchFailure(variantOp,
                                         "not enough blocks to merge");
    }

    rewriter.startRootUpdate(variantOp);

    // Gather all constants initialized by the blocks.
    SmallVector<Location> blockLocs;
    bool anyRequireDevice = false;
    SmallVector<Type> resultTypes;
    SmallVector<Attribute> resultKeys;
    SmallVector<Location> resultLocs;
    for (auto blockOp : blockOps) {
      blockLocs.push_back(blockOp.getLoc());
      if (blockOp.getNumArguments() > 0) anyRequireDevice = true;
      llvm::append_range(resultTypes, blockOp.getResultTypes());
      llvm::append_range(resultKeys, blockOp.getKeys().getValue());
      llvm::append_range(
          resultLocs,
          gatherResultLocations(blockOp.getNumResults(), blockOp.getRegion()));
    }
    SmallVector<Type> inputTypes;
    if (anyRequireDevice) {
      inputTypes.push_back(IREE::HAL::DeviceType::get(rewriter.getContext()));
    }

    // Create the new combined block op at the location of the first block to
    // keep things in a deterministic order; this makes it look like we are
    // merging all subsequent blocks into the first but without having to worry
    // about making that work.
    rewriter.setInsertionPoint(blockOps.front());
    auto fusedLoc = rewriter.getFusedLoc(blockLocs);
    auto newBlockOp = rewriter.create<ExecutableConstantBlockOp>(
        fusedLoc, rewriter.getFunctionType(inputTypes, resultTypes),
        rewriter.getArrayAttr(resultKeys));

    // Create the entry block that captures the optional device argument and
    // the exit block that returns the final flattened set of keys.
    auto &targetRegion = newBlockOp.getRegion();
    auto *preBlock = newBlockOp.addEntryBlock();
    SmallVector<Block *> targetBlocks;
    for (size_t i = 0; i < blockOps.size(); ++i) {
      targetBlocks.push_back(&targetRegion.emplaceBlock());
    }
    auto *postBlock = &targetRegion.emplaceBlock();
    OpBuilder::atBlockBegin(preBlock).create<cf::BranchOp>(
        blockOps.front().getLoc(), targetBlocks.front());

    // Inline all source constant block regions (which may have multiple
    // Blocks).
    SmallVector<Value> resultValues;
    for (unsigned i = 0; i < targetBlocks.size(); ++i) {
      auto *headerBlock = targetBlocks[i];
      auto *nextBlock =
          i < targetBlocks.size() - 1 ? targetBlocks[i + 1] : postBlock;
      auto blockOp = blockOps[i];
      auto &sourceRegion = blockOp.getRegion();

      // Ensure there's only one hal.return in the region.
      // This makes it easier to splice in as we can capture the returned values
      // for use in our combined return.
      rewriteToOneReturn(resultTypes.size(), sourceRegion, rewriter);

      // Inline the entire CFG of the constant block into the target.
      rewriter.cloneRegionBefore(sourceRegion, nextBlock);

      // Branch from the header block into the first block of the region. Note
      // that it may have a %device argument.
      Block *firstBlock = headerBlock->getNextNode();
      SmallVector<Value> firstBranchOperands;
      if (firstBlock->getNumArguments() > 0) {
        firstBranchOperands.push_back(newBlockOp.getArgument(0));
      }
      OpBuilder::atBlockEnd(headerBlock)
          .create<cf::BranchOp>(newBlockOp.getLoc(), firstBlock,
                                firstBranchOperands);

      // Find the single expected return, capture its operands, and rewrite it
      // to branch to the next block.
      for (auto returnOp : llvm::make_early_inc_range(
               targetRegion.getOps<IREE::HAL::ReturnOp>())) {
        llvm::append_range(resultValues, returnOp.getOperands());
        OpBuilder(returnOp).create<cf::BranchOp>(returnOp.getLoc(), nextBlock);
        returnOp.erase();
      }
    }

    // Return from the constant block with all operands.
    OpBuilder::atBlockBegin(postBlock).create<IREE::HAL::ReturnOp>(
        fusedLoc, resultValues);

    rewriter.finalizeRootUpdate(variantOp);

    // Erase all the old blocks.
    for (auto blockOp : blockOps) {
      rewriter.eraseOp(blockOp);
    }

    return success();
  }
};

}  // namespace

void ExecutableVariantOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<MergeExecutableConstantBlocks>(context);
}

namespace {

static void filterReturnOperands(ExecutableConstantBlockOp blockOp,
                                 const BitVector &preservedIndices) {
  for (auto returnOp :
       llvm::make_early_inc_range(blockOp.getOps<IREE::HAL::ReturnOp>())) {
    SmallVector<Value> operands;
    for (auto [i, operand] : llvm::enumerate(returnOp.getOperands())) {
      if (preservedIndices.test(i)) operands.push_back(operand);
    }
    returnOp.operandsMutable().assign(operands);
  }
}

/// Drops the %device argument of a constant block region if unused.
struct DropUnusedExecutableConstantBlockDeviceArg
    : public OpRewritePattern<ExecutableConstantBlockOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExecutableConstantBlockOp blockOp,
                                PatternRewriter &rewriter) const override {
    if (blockOp.getNumArguments() == 0) return failure();
    auto deviceArg = blockOp.getArgument(0);
    if (!deviceArg.use_empty()) return failure();
    rewriter.updateRootInPlace(blockOp, [&]() {
      blockOp.eraseArgument(0);
      blockOp.setFunctionTypeAttr(TypeAttr::get(
          rewriter.getFunctionType(/*inputs=*/{}, blockOp.getResultTypes())));
    });
    return success();
  }
};

/// Deduplicates constant values that have matching keys, choosing the first
/// one found. There's no verification that the values produced are the same
/// as users are expected to uniquely name their keys.
struct DeduplicateExecutableConstantBlockKeys
    : public OpRewritePattern<ExecutableConstantBlockOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(ExecutableConstantBlockOp blockOp,
                                PatternRewriter &rewriter) const override {
    // Build a set of preserved result indices (those with unique keys).
    BitVector resultIndices(blockOp.getNumResults(), /*t=*/false);
    SmallVector<Type> resultTypes;
    SetVector<Attribute> resultKeys;
    int i = 0;
    for (auto [resultKey, resultType] :
         llvm::zip(blockOp.getKeys().getValue(), blockOp.getResultTypes())) {
      if (resultKeys.insert(resultKey)) {
        resultIndices.set(i);
        resultTypes.push_back(resultType);
      }
      ++i;
    }

    // If all results are preserved this is a no-op.
    if (resultIndices.all()) {
      return rewriter.notifyMatchFailure(blockOp, "no duplicate keys");
    }

    // Update function in-place.
    rewriter.updateRootInPlace(blockOp, [&]() {
      // Update metadata.
      blockOp.setFunctionTypeAttr(TypeAttr::get(
          rewriter.getFunctionType(blockOp.getArgumentTypes(), resultTypes)));
      blockOp.setKeysAttr(rewriter.getArrayAttr(resultKeys.takeVector()));
      // Drop all unneeded results from each return.
      filterReturnOperands(blockOp, resultIndices);
    });
    return success();
  }
};

}  // namespace

void ExecutableConstantBlockOp::getCanonicalizationPatterns(
    RewritePatternSet &results, MLIRContext *context) {
  results.insert<DropUnusedExecutableConstantBlockDeviceArg>(context);
  results.insert<DeduplicateExecutableConstantBlockKeys>(context);
}

//===----------------------------------------------------------------------===//
// hal.fence.create
//===----------------------------------------------------------------------===//

namespace {

/// Replaces a fence with no timepoints with a null value.
struct ElideEmptyFenceCreate : public OpRewritePattern<FenceCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceCreateOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0) return failure();
    rewriter.replaceOpWithNewOp<IREE::Util::NullOp>(op,
                                                    op.getResult().getType());
    return success();
  }
};

/// Deduplicates timepoints by taking the maximum payload value of any that
/// share the same semaphore.
struct DeduplicateFenceCreateTimepoints
    : public OpRewritePattern<FenceCreateOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceCreateOp op,
                                PatternRewriter &rewriter) const override {
    // Check to see if the fence is over a single (semaphore, value) timepoint.
    if (op.getSemaphores().size() <= 1) {
      return failure();  // just 0 or 1 timepoint
    }

    // Build a map of all timepoints keyed on semaphore.
    // This will implicitly deduplicate the semaphores and the values for each.
    llvm::MapVector<Value, SetVector<Value>> timepoints;
    for (auto it : llvm::zip(op.getSemaphores(), op.getMinValues())) {
      auto semaphore = std::get<0>(it);
      auto minValue = std::get<1>(it);
      timepoints[semaphore].insert(minValue);
    }

    // Check for no-op when we don't deduplicate anything.
    if (timepoints.size() == op.getSemaphores().size()) return failure();

    // Build the timepoints.
    // A single semaphore may have multiple values and we need to take the max.
    SmallVector<Value> semaphores;
    SmallVector<Value> minValues;
    semaphores.reserve(timepoints.size());
    minValues.reserve(timepoints.size());
    for (auto it : timepoints) {
      semaphores.push_back(it.first);
      if (it.second.size() == 1) {
        // Single timepoint.
        minValues.push_back(it.second.front());
      } else {
        // Join timepoints. This will fold if constant.
        minValues.push_back(rewriter.createOrFold<IREE::Util::RangeMaxOp>(
            op.getLoc(), it.second.takeVector()));
      }
    }

    // Build new op. The map/set vectors we used will ensure the relative order
    // of the timepoints matches the original.
    rewriter.replaceOpWithNewOp<FenceCreateOp>(op, op.getResult().getType(),
                                               semaphores, minValues);
    return success();
  }
};

}  // namespace

void FenceCreateOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                                MLIRContext *context) {
  results.insert<ElideEmptyFenceCreate>(context);
  results.insert<DeduplicateFenceCreateTimepoints>(context);
}

//===----------------------------------------------------------------------===//
// hal.fence.join
//===----------------------------------------------------------------------===//

namespace {

/// Replaces a fence join with no operands with a null value.
struct ElideEmptyFenceJoin : public OpRewritePattern<FenceJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceJoinOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getNumOperands() != 0) return failure();
    rewriter.replaceOpWithNewOp<IREE::Util::NullOp>(op,
                                                    op.getResult().getType());
    return success();
  }
};

// Produces a deduplicated and null-elided operand list.
// Returns None if nothing changed.
static Optional<std::vector<Value>> deduplicateFenceOperands(
    ValueRange operands) {
  SetVector<Value> newOperands;
  for (auto operand : operands) {
    if (isa_and_nonnull<IREE::Util::NullOp>(operand.getDefiningOp())) {
      // Drop null values as they don't mean anything. Ideally we'd reach back
      // a little further here but that's best done in an IPO pass.
      continue;
    }
    newOperands.insert(operand);
  }

  if (newOperands.size() == operands.size()) return None;
  return newOperands.takeVector();
}

/// Deduplicates fence join operands and drops nulls.
struct DeduplicateFenceJoinFences : public OpRewritePattern<FenceJoinOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceJoinOp op,
                                PatternRewriter &rewriter) const override {
    auto newOperands = deduplicateFenceOperands(op.getFences());
    if (!newOperands) return failure();
    rewriter.replaceOpWithNewOp<FenceJoinOp>(op, op.getResult().getType(),
                                             newOperands.value());
    return success();
  }
};

}  // namespace

void FenceJoinOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                              MLIRContext *context) {
  results.insert<ElideEmptyFenceJoin>(context);
  results.insert<DeduplicateFenceJoinFences>(context);
}

//===----------------------------------------------------------------------===//
// hal.fence.await
//===----------------------------------------------------------------------===//

namespace {

/// Elides a fence await with no fences.
struct ElideEmptyFenceAwait : public OpRewritePattern<FenceAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceAwaitOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getFences().empty()) return failure();
    rewriter.replaceOpWithNewOp<arith::ConstantIntOp>(op, /*ok=*/0, 32);
    return success();
  }
};

/// Deduplicates fence await operands and drops nulls.
struct DeduplicateFenceAwaitFences : public OpRewritePattern<FenceAwaitOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(FenceAwaitOp op,
                                PatternRewriter &rewriter) const override {
    auto newOperands = deduplicateFenceOperands(op.getFences());
    if (newOperands == None) return failure();
    rewriter.replaceOpWithNewOp<FenceAwaitOp>(op, op.getStatus().getType(),
                                              op.getTimeoutMillis(),
                                              newOperands.value());
    return success();
  }
};

}  // namespace

void FenceAwaitOp::getCanonicalizationPatterns(RewritePatternSet &results,
                                               MLIRContext *context) {
  results.insert<ElideEmptyFenceAwait>(context);
  results.insert<DeduplicateFenceAwaitFences>(context);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
