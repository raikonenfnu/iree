// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Nod/IR/nod_dialect.h"
#include "iree/compiler/Dialect/Nod/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Nod/Transforms/Passes.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace Nod {

struct LinalgMatmulToNodConversion final
    : public OpRewritePattern<linalg::MatmulOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::MatmulOp matmulOp,
                                PatternRewriter &rewriter) const override {
    auto loc = matmulOp.getLoc();
    auto lhs = matmulOp.inputs()[0];
    auto rhs = matmulOp.inputs()[1];
    auto result = matmulOp.outputs()[0];
    rewriter.replaceOpWithNewOp<IREE::Nod::MatmulTensorOp>(matmulOp, result.getType(), lhs, rhs, result);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

namespace {
struct LinalgToNodPass
    : public LinalgToNodBase<LinalgToNodPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<IREE::Nod::NodDialect, linalg::LinalgDialect, StandardOpsDialect, math::MathDialect,
                memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();

    OwningRewritePatternList patterns(context);
    patterns.insert<LinalgMatmulToNodConversion>(context);
    if (failed(applyPatternsAndFoldGreedily(getOperation(),
                                            std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
}  // namespace


std::unique_ptr<OperationPass<FuncOp>> createLinalgToNodPass() {
  return std::make_unique<LinalgToNodPass>();
}

}  // namespace linalg_ext
}  // namespace iree_compiler
}  // namespace mlir
