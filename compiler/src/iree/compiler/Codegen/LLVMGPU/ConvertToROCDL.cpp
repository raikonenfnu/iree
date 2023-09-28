// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUPatterns.h"
#include "iree/compiler/Codegen/LLVMGPU/ConvertToLLVM.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ComplexToLLVM/ComplexToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/GPUToROCDL/GPUToROCDLPass.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-convert-to-rocdl"

namespace mlir {
namespace iree_compiler {

namespace {

struct FastIntToHalfFloat final
    : public OpRewritePattern<LLVM::UIToFPOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(LLVM::UIToFPOp op,
                                PatternRewriter &rewriter) const override {
    auto zextOp = op.getArg().getDefiningOp<LLVM::ZExtOp>();
    if (!zextOp) return failure();
    Value source = zextOp.getArg();

    // Check Vector Types.
    auto sourceVectorType = source.getType().dyn_cast<VectorType>();
    auto dstVectorType = op.getResult().getType().dyn_cast<VectorType>();
    if (!sourceVectorType) return failure();
    if (!dstVectorType) return failure();

    // Check elemType prerequisites.
    if (!sourceVectorType.getElementType().isIntOrIndex()) return failure();
    if (!dstVectorType.getElementType().isF16()) return failure();
    int64_t srcElemBitwidth = sourceVectorType.getElementTypeBitWidth();
    if (srcElemBitwidth != 4 && srcElemBitwidth != 8)
      return failure();
    if (sourceVectorType.getNumElements() != dstVectorType.getNumElements()) return failure();

    // Algorithm based on https://arxiv.org/pdf/2211.10017.pdf:
    //  a = 1024 | src : vector<8xi4>
    //  b = zext a : vector<8xi16>
    //  c = bitcast b : vector<8xi16>
    //  d = c - 1024 : vector<8xi16>

    // Step 1. zext
    auto intHalfType = VectorType::get({sourceVectorType.getNumElements()},
                                    rewriter.getIntegerType(16));
    Value zextToHalfInt = rewriter.create<LLVM::ZExtOp>(op.getLoc(), intHalfType, source);

    // Step 2. apply or mask.
    auto maskVals = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), intHalfType,
        DenseIntElementsAttr::get(intHalfType, IntegerAttr::get(rewriter.getIntegerType(16), 1024).getValue()));
    Value orVal = rewriter.create<LLVM::OrOp>(op.getLoc(), intHalfType, maskVals, zextToHalfInt);

    // Step 3. Bitcast
    auto zextToHalfOp = rewriter.create<LLVM::BitcastOp>(op.getLoc(), dstVectorType, orVal);

    // Step 4. Renormalize
    // auto offsetBufferType = VectorType::get({dstVectorType.getNumElements()},
    //                             rewriter.getF16Type());
    // SmallVector<float16_t> offsetArray(dstVectorType.getNumElements());
    // for (int32_t i = 0; i < dstVectorType.getNumElements(); i++) {
    //   offsetArray[i] = 1024.0;
    // }
    auto offsetVals = rewriter.create<LLVM::ConstantOp>(
        op.getLoc(), dstVectorType,
        DenseFPElementsAttr::get(dstVectorType, FloatAttr::get(dstVectorType.getElementType(), 1024.0).getValue()));
    rewriter.replaceOpWithNewOp<LLVM::FSubOp>(op, dstVectorType, zextToHalfOp, offsetVals);
    return success();
  }
};

/// A pass that replaces all occurrences of GPU device operations with their
/// corresponding ROCDL equivalent.
///
/// This pass only handles device code and is not meant to be run on GPU host
/// code.
struct ConvertToROCDLPass : public ConvertToROCDLBase<ConvertToROCDLPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect, ROCDL::ROCDLDialect>();
  }
  void runOnOperation() override {
    ModuleOp m = getOperation();

    /// Customize the bitwidth used for the device side index computations.
    LowerToLLVMOptions options(m.getContext(), DataLayout(m));
    options.overrideIndexBitwidth(32);
    LLVMTypeConverter converter(m.getContext(), options);
    populateGpuMemorySpaceAttributeConversions(
        converter, [](gpu::AddressSpace space) {
          switch (space) {
          case gpu::AddressSpace::Global:
            return 1;
          case gpu::AddressSpace::Workgroup:
            return 3;
          case gpu::AddressSpace::Private:
            return 5;
          }
          llvm_unreachable("unknown address space enum value");
          return 0;
        });

      LLVM_DEBUG({
        llvm::dbgs() << "--- Original ---\n";
        m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
        llvm::dbgs() << "\n\n";
      });

    // Apply in-dialect lowering first. In-dialect lowering will replace ops
    // which need to be lowered further, which is not supported by a single
    // conversion pass.
    // Run Vector -> Vector transformations ahead of conversion to LLVM.
    {
      RewritePatternSet patterns(&getContext());
      populateDropSharedMemoryDeallocOpPatterns(patterns);
      populateScalarizeMathOps(patterns);
      populateConvertSharedMemoryAllocOps(patterns);
      vector::populateVectorToVectorCanonicalizationPatterns(patterns);
      vector::populateVectorBroadcastLoweringPatterns(patterns);
      vector::populateVectorContractLoweringPatterns(
          patterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(patterns);
      vector::populateVectorShapeCastLoweringPatterns(patterns);
      // TODO: doubtful that the "default" does what one want here, it is likely
      // better to use something else.
      vector::populateVectorTransposeLoweringPatterns(
          patterns, vector::VectorTransformsOptions());
      vector::populateVectorTransferLoweringPatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Vector-to-Vector transformation ---\n";
      m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });


    {
      RewritePatternSet patterns(&getContext());
      populateGpuRewritePatterns(patterns);
      if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After GPU Rewrite Patterns ---\n";
      m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    {
      RewritePatternSet llvmPatterns(&getContext());
      populateLowerHALInterfaceOp(llvmPatterns);
      populateLLVMConversionPatterns(&getContext(), llvmPatterns, converter);
      populateComplexToLLVMConversionPatterns(converter, llvmPatterns);
      populateMathToLLVMConversionPatterns(converter, llvmPatterns);
      memref::populateExpandStridedMetadataPatterns(llvmPatterns);
      populateFinalizeMemRefToLLVMConversionPatterns(converter, llvmPatterns);
      populateFuncToLLVMConversionPatterns(converter, llvmPatterns);
      cf::populateControlFlowToLLVMConversionPatterns(converter, llvmPatterns);
      arith::populateArithToLLVMConversionPatterns(converter, llvmPatterns);
      populateVectorToLLVMConversionPatterns(converter, llvmPatterns);
      populateGpuToROCDLConversionPatterns(converter, llvmPatterns,
                                           gpu::amd::Runtime::Unknown);
      LLVMConversionTarget target(getContext());
      populateFuncToLLVMFuncOpConversionPattern(converter, llvmPatterns);
      configureGpuToROCDLConversionLegality(target);
      if (failed(applyPartialConversion(m, target, std::move(llvmPatterns))))
        signalPassFailure();
    }
    LLVM_DEBUG({
      llvm::dbgs() << "--- After LLVM Conversion Rewrite Patterns ---\n";
      m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // {
    //   RewritePatternSet patterns(&getContext());
    //   patterns.add<FastIntToHalfFloat>(&getContext());
    //   if (failed(applyPatternsAndFoldGreedily(m, std::move(patterns)))) {
    //     return signalPassFailure();
    //   }
    // }
    // LLVM_DEBUG({
    //   llvm::dbgs() << "--- After Fast low-P int to half float ---\n";
    //   m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    //   llvm::dbgs() << "\n\n";
    // });

    ConvertToDynamicSharedMemory(m);

    LLVM_DEBUG({
      llvm::dbgs() << "--- After Converting do dynamic Shared Memory ---\n";
      m.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });
  }
};

} // anonymous namespace

std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass() {
  return std::make_unique<ConvertToROCDLPass>();
}

} // namespace iree_compiler
} // namespace mlir
