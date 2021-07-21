// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Analysis/DependenceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

enum MatmulOpType { INVALID, MATMUL, BATCH_MATMUL };
MatmulOpType GetMatmulOpType(ArrayRef<int64_t> lhs_shape,
                             ArrayRef<int64_t> rhs_shape,
                             ArrayRef<int64_t> out_shape) {
  auto lhs_rank = lhs_shape.size();
  auto rhs_rank = rhs_shape.size();
  auto out_rank = out_shape.size();

  // Matmul and batch matmul need equal rank.
  if (out_rank != 2 && out_rank != 3) return INVALID;
  if (lhs_rank != rhs_rank || lhs_rank != out_rank) return INVALID;

  bool is_bmm = false;
  // Initialzie matrix indices.
  int i = 0;
  int j = 1;
  // If 3-D check that batch dim is equal.
  if (out_rank == 3) {
    if (lhs_shape[0] != rhs_shape[0] || lhs_shape[0] != out_shape[0]) {
      return INVALID;
    }
    is_bmm = true;
    // Offset indices if matmul operation is bmm.
    i++;
    j++;
  }

  // Check validity of matmul shapes
  if (lhs_shape[i] != out_shape[i] || rhs_shape[j] != out_shape[j] ||
      lhs_shape[j] != rhs_shape[i]) {
    return INVALID;
  }

  if (is_bmm) {
    return BATCH_MATMUL;
  } else {
    return MATMUL;
  }
}

// Pattern to canonialize LinalgOps with contraction to matmul.
class ContractionToMatmul : public OpRewritePattern<linalg::GenericOp> {
 public:
  using OpRewritePattern<linalg::GenericOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::GenericOp op,
                                PatternRewriter& rewriter) const override {
    // Check if it is contraction op and has only 2 inputs
    if (!isaContractionOpInterface(op) || op.inputs().size() != 2 ||
        op.outputs().size() != 1)
      return failure();

    Location loc = op.getLoc();
    auto lhs = op.inputs()[0];
    auto rhs = op.inputs()[1];
    auto init_tensor = op.outputs()[0];

    auto result_ty = op.getType(0).dyn_cast<RankedTensorType>();
    auto lhs_ty = lhs.getType().dyn_cast<RankedTensorType>();
    auto rhs_ty = rhs.getType().dyn_cast<RankedTensorType>();
    if (!lhs_ty || !rhs_ty || !result_ty) return failure();
    auto lhs_shape = lhs_ty.getShape();
    auto rhs_shape = rhs_ty.getShape();
    auto out_shape = result_ty.getShape();

    // TODO(raikonenfnu): Add the cases for matmul type
    MatmulOpType matmul_type = GetMatmulOpType(lhs_shape, rhs_shape, out_shape);
    if (matmul_type == BATCH_MATMUL) {
      rewriter.replaceOpWithNewOp<linalg::BatchMatmulOp>(
          op, /*resultTensorTypes=*/TypeRange{result_ty},
          /*inputs=*/ValueRange{lhs, rhs},
          /*outputBuffers=*/ValueRange{init_tensor});
    } else if (matmul_type == MATMUL) {
      rewriter.replaceOpWithNewOp<linalg::MatmulOp>(
          op, /*resultTensorTypes=*/TypeRange{result_ty},
          /*inputs=*/ValueRange{lhs, rhs},
          /*outputBuffers=*/ValueRange{init_tensor});
    } else {
      return failure();
    }
    return success();
  }
};

struct ConvertContractionToMatmulPass
    : public ConvertContractionToMatmulBase<ConvertContractionToMatmulPass> {
  void runOnOperation() override {
    FuncOp funcOp = getOperation();
    OwningRewritePatternList patterns(&getContext());
    patterns.add<ContractionToMatmul>(&getContext());
    (void)applyPatternsAndFoldGreedily(funcOp, std::move(patterns));
  }
};

}  // namespace

std::unique_ptr<OperationPass<FuncOp>> createConvertContractionToMatmulPass() {
  return std::make_unique<ConvertContractionToMatmulPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
