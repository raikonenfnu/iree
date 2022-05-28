// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Flow/IR/FlowDialect.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Flow/IR/FlowTypes.h"
#include "iree/compiler/Dialect/Flow/Transforms/PassDetail.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"


#define DEBUG_TYPE "iree-flow-convert-out-of-range-demotion"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Flow {

namespace {

/// Converts linalg operations that can map to flow.tensor.* operations.
struct ConvertOutOfRangeDemotionPass
    : public ConvertOutOfRangeDemotionBase<ConvertOutOfRangeDemotionPass> {
  ConvertOutOfRangeDemotionPass() = default;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, tensor::TensorDialect,
                    linalg::LinalgDialect, mlir::func::FuncDialect,
                    mlir::arith::ArithmeticDialect, mlir::math::MathDialect>();
  }

  void runOnOperation() override {
    auto funcOp = getOperation();
    MLIRContext *context = funcOp->getContext();
    funcOp.walk([&](Operation* nestedOp) {
        llvm::outs()<<"op:"<<nestedOp->getName()<<"\n";
    });
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::func::FuncOp>>
createConvertOutOfRangeDemotionPass() {
  return std::make_unique<ConvertOutOfRangeDemotionPass>();
}

}  // namespace Flow
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
