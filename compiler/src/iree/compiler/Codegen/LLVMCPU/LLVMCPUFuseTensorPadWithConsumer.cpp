// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LLVMCPUFuseTensorPadWithConsumerPass final
    : public LLVMCPUFuseTensorPadWithConsumerBase<
          LLVMCPUFuseTensorPadWithConsumerPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    func::FuncOp funcOp = getOperation();

    RewritePatternSet patterns(context);
    patterns.insert<linalg::ExtractSliceOfPadTensorSwapPattern>(
        context, [](tensor::ExtractSliceOp) { return false; });
    if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

}  // namespace

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUFuseTensorPadWithConsumerPass() {
  return std::make_unique<LLVMCPUFuseTensorPadWithConsumerPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
