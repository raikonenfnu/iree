// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace iree_compiler {

namespace {

struct LLVMCPUInjectCodePass
    : public LLVMCPUInjectCodeBase<
          LLVMCPUInjectCodePass> {
  LLVMCPUInjectCodePass() = default;

  void runOnOperation() override {
    //   TODO: From old torch->hipkl conversion, add insertion of FuncOp at front of entryPoint.
    //  TODO: Use Libm example to insert and declare func https://github.com/llvm/llvm-project/blob/a48adc565864e0ce10becf301de5455308bd7d6c/mlir/lib/Conversion/MathToLibm/MathToLibm.cpp#L113-L139
    // TODO: Use linalg standard example for name/flaysymbolRef attribute https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/LinalgToStandard/LinalgToStandard.cpp#L53-L54
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    // OpBuilder builder(context);
    // builder.setInsertionPointToStart(&moduleOp->getRegion(0).front());
    OpBuilder builder(moduleOp.getBodyRegion());    // Adding Func Declaration
    std::string fnName = "OpName";
    FlatSymbolRefAttr fnNameAttr =
          SymbolRefAttr::get(context, fnName);
    auto OperandTypes = TypeRange();
    auto ResultTypes = TypeRange();
    auto opFunctionTy = FunctionType::get(context, OperandTypes, ResultTypes);
    SymbolOpInterface opFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(), fnNameAttr.getValue(),
                                           opFunctionTy);
    opFunc->setAttr("llvm.emit_c_interface", UnitAttr::get(context));
    opFunc.setPrivate();

    // Injecting call into Kernel.
    func::FuncOp funcOp;
    for (const auto &op : moduleOp.getOps<func::FuncOp>()) {
        funcOp = op;
    }
    Block *entryBlock;
    Operation *op = funcOp.getOperation();
    for (Region &region : op->getRegions()) {
      for(Block &block : region.getBlocks()) {
        if(block.isEntryBlock()) {
          entryBlock = &block;
          break;
        }
      }
    }
    SmallVector<Value> operandList;
    builder.setInsertionPointToStart(entryBlock);
    builder.create<func::CallOp>(funcOp->getLoc(), fnNameAttr.getValue(), ResultTypes, operandList);
    // funcOp.walk([&](Operation* nestedOp) {
    //     llvm::outs()<<"op:"<<nestedOp->getName()<<"\n";
    // });
    // if (moduleOp.lookupSymbol(fnNameAttr.getAttr())) {
      // llvm::outs()<<"Found it man!\n";
    // }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUInjectCodePass() {
  return std::make_unique<LLVMCPUInjectCodePass>();
}

}  // namespace iree_compiler
}  // namespace mlir

