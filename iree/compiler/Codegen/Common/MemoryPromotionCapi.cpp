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

struct MemoryPromotionCapiPass
    : public MemoryPromotionCapiBase<
          MemoryPromotionCapiPass> {
  MemoryPromotionCapiPass() = default;

  void runOnOperation() override {
    //   TODO: From old torch->hipkl conversion, add insertion of FuncOp at front of entryPoint.
    //  TODO: Use Libm example to insert and declare func https://github.com/llvm/llvm-project/blob/a48adc565864e0ce10becf301de5455308bd7d6c/mlir/lib/Conversion/MathToLibm/MathToLibm.cpp#L113-L139
    // TODO: Use linalg standard example for name/flaysymbolRef attribute https://github.com/llvm/llvm-project/blob/main/mlir/lib/Conversion/LinalgToStandard/LinalgToStandard.cpp#L53-L54
    auto moduleOp = getOperation();
    MLIRContext *context = &getContext();
    OpBuilder builder(moduleOp.getBodyRegion());    // Adding Func Declaration
    std::string fnName = "mem_promote";
    FlatSymbolRefAttr fnNameAttr =
          SymbolRefAttr::get(context, fnName);
    auto unrankedMemrefType = UnrankedMemRefType::get(builder.getF16Type(), 0);
    // auto unrankedMemrefType = UnrankedMemRefType::get(builder.getF32Type(), 0);
    auto OperandTypes = TypeRange({unrankedMemrefType});
    auto ResultTypes = TypeRange();
    auto opFunctionTy = FunctionType::get(context, OperandTypes, ResultTypes);
    SymbolOpInterface opFunc = builder.create<func::FuncOp>(builder.getUnknownLoc(), fnNameAttr.getValue(),
                                           opFunctionTy);
    opFunc->setAttr("llvm.emit_c_interface", UnitAttr::get(context));
    opFunc.setPrivate();

    // Searching for main func region, to inject the kernel onto.
    func::FuncOp funcOp;
    for (const auto &op : moduleOp.getOps<func::FuncOp>()) {
        funcOp = op;
    }
    // TODO: Within cpp try print the data
    // TODO: within cpp try check that pointer stays the same.
    // Searching for
    funcOp.walk([&](Operation* nestedOp) {
        if(dyn_cast<vector::TransferReadOp>(nestedOp)) {
          // Cast Op into unranked and then insert memory promtoe
          builder.setInsertionPoint(nestedOp);
          Value subTensor = nestedOp->getOperand(0);
          auto inputOp = subTensor.getDefiningOp();
          auto inputMemrefType = subTensor.getType().cast<BaseMemRefType>();
          unrankedMemrefType = UnrankedMemRefType::get(inputMemrefType.getElementType(),
                                           inputMemrefType.getMemorySpace());
          Value urankInput = builder.create<memref::CastOp>(nestedOp->getLoc(), unrankedMemrefType, inputOp->getResult(0));
          SmallVector<Value, 1> operandList = {urankInput};
          builder.create<func::CallOp>(nestedOp->getLoc(), fnNameAttr.getValue(), ResultTypes, operandList);
        }
    });
    // if (moduleOp.lookupSymbol(fnNameAttr.getAttr())) {
      // llvm::outs()<<"Found it man!\n";
    // }
  }
};

}  // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createMemoryPromotionCapiPass() {
  return std::make_unique<MemoryPromotionCapiPass>();
}

}  // namespace iree_compiler
}  // namespace mlir

