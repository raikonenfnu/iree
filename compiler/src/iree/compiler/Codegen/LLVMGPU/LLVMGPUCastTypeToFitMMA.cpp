// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/VectorOpUtils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

namespace {

struct UpcastContractOutput final : OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp contractOp,
                                PatternRewriter &rewriter) const override {
    VectorContractOpInfo opInfo(contractOp);

    auto srcCType = dyn_cast<VectorType>(contractOp.getAccType());
    if (!srcCType) {
      return rewriter.notifyMatchFailure(contractOp, "unhandled scalar case");
    }
    auto srcAType = contractOp.getLhsType();
    auto srcBType = contractOp.getRhsType();

    auto intrinsic = contractOp->getAttrOfType<IREE::GPU::MmaInterfaceAttr>(
        "iree.amdgpu.mma");
    if (!intrinsic) {
      return rewriter.notifyMatchFailure(
          contractOp, "could not find iree.amdgpu.mma attribute on contract");
    }
    auto [dstAElemType, dstBElemType, dstCElemType] =
        intrinsic.getABCElementTypes();

    auto srcCElemFType = dyn_cast<FloatType>(srcCType.getElementType());
    auto dstCElemFType = dyn_cast<FloatType>(dstCElemType);
    if (!srcCElemFType || !dstCElemFType ||
        srcCElemFType.getWidth() > dstCElemFType.getWidth()) {
      return rewriter.notifyMatchFailure(
          contractOp, "unhandled non-floating point or non-upcasting case");
    }

    Value lhs = contractOp.getLhs();
    Value rhs = contractOp.getRhs();

    Location loc = contractOp.getLoc();
    auto dstAType = srcAType.clone(dstAElemType);
    auto dstBType = srcBType.clone(dstBElemType);
    auto dstCType = srcCType.clone(dstCElemFType);

    if (srcAType.getElementType() != dstAElemType) {
	lhs = rewriter.create<arith::TruncFOp>(loc, dstAType, lhs);
    }

    if (srcBType.getElementType() != dstBElemType) {
	rhs = rewriter.create<arith::TruncFOp>(loc, dstBType, rhs);
    }

    auto extOp =
        rewriter.create<arith::ExtFOp>(loc, dstCType, contractOp.getAcc());
    auto newContractOp = rewriter.create<vector::ContractionOp>(
        loc, lhs, rhs, extOp,
        contractOp.getIndexingMaps(), contractOp.getIteratorTypes());
    rewriter.replaceOpWithNewOp<arith::TruncFOp>(contractOp, srcCType,
                                                 newContractOp);
    return success();
  }
};

struct LLVMGPUCastTypeToFitMMAPass
    : public LLVMGPUCastTypeToFitMMABase<LLVMGPUCastTypeToFitMMAPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<vector::VectorDialect>();
    registry.insert<arith::ArithDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
      if (configDict) {
        scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
            configDict.get(scheduleAttrName));
      }
    }

    // Import mma type from dispatch schedule attribute if present.
    if (scheduleAttr) {
      func.walk([&](vector::ContractionOp contract) {
        if (!contract->hasAttr("iree.amdgpu.mma")) {
          contract->setAttr("iree.amdgpu.mma", scheduleAttr.getIntrinsic());
        }
      });
    }

    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    patterns.add<UpcastContractOutput>(context);

    if (failed(applyPatternsAndFoldGreedily(func, std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace
std::unique_ptr<InterfacePass<FunctionOpInterface>>
createLLVMGPUCastTypeToFitMMAPass() {
  return std::make_unique<LLVMGPUCastTypeToFitMMAPass>();
}

} // namespace mlir::iree_compiler
