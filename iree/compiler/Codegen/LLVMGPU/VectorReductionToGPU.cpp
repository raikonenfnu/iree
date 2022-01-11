// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstddef>
#include <cstdint>

#include "iree/compiler/Codegen/LLVMGPU/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Dialect/GPU/Passes.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Vector/VectorOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {

static Value createElementwiseOp(OpBuilder& b, Location loc, Value lhs,
                                 Value rhs, vector::CombiningKind kind) {
  switch (kind) {
    case vector::CombiningKind::ADD:
      return b.create<arith::AddFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MAXF:
      return b.create<arith::MaxFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MINF:
      return b.create<arith::MinFOp>(loc, lhs, rhs);
    case vector::CombiningKind::MUL:
      return b.create<arith::MulFOp>(loc, lhs, rhs);
    default:
      break;
  }
  return nullptr;
}

static Value warpReduce(OpBuilder& b, Location loc, Value val,
                        vector::CombiningKind kind) {
  std::array<Type, 2> shuffleType = {val.getType(), b.getI1Type()};
  auto downAttr = b.getStringAttr("down");
  Value activeWidth =
      b.create<arith::ConstantIntOp>(loc, 0xFFFFFFFF, b.getI32Type());
  Value value = val;
  for (int i = kWarpSize / 2; i > 0; i >>= 1) {
    Value offset = b.create<arith::ConstantIntOp>(loc, i, b.getI32Type());
    auto shuffleOp = b.create<gpu::ShuffleOp>(loc, shuffleType, value, offset,
                                              activeWidth, downAttr);
    value = createElementwiseOp(b, loc, value, shuffleOp->getResult(0), kind);
  }
  return value;
}

static Value broadcastToAllLanes(OpBuilder& b, Location loc, Value val) {
  std::array<Type, 2> shuffleType = {val.getType(), b.getI1Type()};
  Value activeWidth =
      b.create<arith::ConstantIntOp>(loc, 0xFFFFFFFF, b.getI32Type());
  auto idAttr = b.getStringAttr("idx");
  Value zero = b.create<arith::ConstantIntOp>(loc, 0, b.getI32Type());
  return b
      .create<gpu::ShuffleOp>(loc, shuffleType, val, zero, activeWidth, idAttr)
      .getResult(0);
}

struct ConvertReduceToGPU final
    : public OpRewritePattern<vector::MultiDimReductionOp> {
  using OpRewritePattern<vector::MultiDimReductionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::MultiDimReductionOp op,
                                PatternRewriter& rewriter) const override {
    if (op.getSourceVectorType().getNumElements() != kWarpSize ||
        (op.getDestType().isa<VectorType>() &&
         op.getDestType().cast<VectorType>().getNumElements() != 1))
      return failure();
    auto funcOp = op->getParentOfType<FuncOp>();
    if (!funcOp) return failure();
    auto vecType = VectorType::get(
        SmallVector<int64_t>(op.getSourceVectorType().getRank(), 1),
        op.getSourceVectorType().getElementType());
    rewriter.setInsertionPoint(&funcOp.front(), funcOp.front().begin());
    mlir::Value laneId = rewriter.create<mlir::gpu::ThreadIdOp>(
        op.getLoc(), rewriter.getIndexType(), "x");
    rewriter.setInsertionPoint(op);
    mlir::AffineExpr d0 = rewriter.getAffineDimExpr(0);
    laneId = mlir::makeComposedAffineApply(
        rewriter, op.getLoc(), d0 % rewriter.getAffineConstantExpr(kWarpSize),
        {laneId});
    // Distribute the value on the warp lanes.
    Value distributedVal = rewriter.create<vector::ExtractMapOp>(
        op.getLoc(), vecType, op.source(), laneId);
    distributedVal = rewriter.create<vector::ExtractOp>(
        op.getLoc(), distributedVal,
        SmallVector<int64_t>(vecType.getRank(), 0));
    Value v = warpReduce(rewriter, op.getLoc(), distributedVal, op.kind());
    v = broadcastToAllLanes(rewriter, op.getLoc(), v);
    if (op.getType().isa<VectorType>())
      v = rewriter.create<vector::BroadcastOp>(op.getLoc(), op.getType(), v);
    rewriter.replaceOp(op, v);
    return success();
  }
};

struct LLVMGPUReduceToGPUPass
    : public LLVMGPUReduceToGPUBase<LLVMGPUReduceToGPUPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    RewritePatternSet patterns(funcOp.getContext());
    patterns.insert<ConvertReduceToGPU>(funcOp.getContext());
    vector::populatePropagateVectorDistributionPatterns(patterns);
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // anonymous namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertVectorReductionToGPUPass() {
  return std::make_unique<LLVMGPUReduceToGPUPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
