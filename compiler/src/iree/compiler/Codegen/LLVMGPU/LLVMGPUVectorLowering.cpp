// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/PassDetail.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorTransforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-llvmgpu-vector-lowering"

namespace mlir {
namespace iree_compiler {

//====---------------------------------------------------------------------===//
// Patterns for late vector op lowering.
//====---------------------------------------------------------------------===//

// Helper functions for vector op lowering.
//====---------------------------------------------------------------------===//

int getComputeVectorSize(int64_t size) {
  for (int i : {4, 3, 2}) {
    if (size % i == 0)
      return i;
  }
  return 1;
}

int getMemoryVectorSize(Value source, Type scalarType, int64_t size) {
  int bitwidth = scalarType.getIntOrFloatBitWidth();
  while (auto sliceOp = source.getDefiningOp<tensor::ExtractSliceOp>())
    source = sliceOp.getSource();
  if (!matchPattern(source, m_Constant())) {
    // If we are not reading from a constant array that is embedded in the
    // kernel, try to use a large vector size matching the bitwidth to read in
    // 128-bit chunks. This helps with memory access performance. Such vector
    // sizes are not native in SPIR-V though; this relies on following passes to
    // bitcast them to 32-bit 4-element vectors to be valid.
    if (bitwidth <= 8 && size % 16 == 0)
      return 16;
    if (bitwidth <= 16 && size % 8 == 0)
      return 8;
  }
  if (bitwidth <= 32 && size % 4 == 0)
    return 4;
  return size % 2 == 0 ? 2 : 1;
}

SmallVector<int64_t> getNativeVectorShapeImpl(VectorTransferOpInterface op) {
  auto vecType = op.getVectorType();
  SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
  for (const auto &[index, dim] :
       llvm::enumerate(op.getPermutationMap().getResults())) {
    if (auto dimExpr = dim.dyn_cast<AffineDimExpr>()) {
      if (dimExpr.getPosition() == op.getPermutationMap().getNumDims() - 1) {
        nativeSize[index] = getMemoryVectorSize(
            op.source(), vecType.getElementType(), vecType.getShape()[index]);
      }
    }
  }
  return nativeSize;
}

SmallVector<int64_t> getNativeVectorShapeImpl(vector::ReductionOp op) {
  VectorType srcVectorType = op.getSourceVectorType();
  assert(srcVectorType.getRank() == 1); // Guaranteed by semantics
  int64_t vectorSize = getComputeVectorSize(srcVectorType.getDimSize(0));
  return {vectorSize};
}

std::optional<SmallVector<int64_t>>
getNativeVectorShape(Operation *op) {
  if (OpTrait::hasElementwiseMappableTraits(op) && op->getNumResults() == 1) {
    if (auto vecType = llvm::dyn_cast<VectorType>(op->getResultTypes()[0])) {
      SmallVector<int64_t> nativeSize(vecType.getRank(), 1);
      nativeSize.back() = getComputeVectorSize(vecType.getShape().back());
      return nativeSize;
    }
  }

  return TypeSwitch<Operation *, std::optional<SmallVector<int64_t>>>(op)
      .Case<VectorTransferOpInterface, vector::ReductionOp>(
          [](auto typedOp) { return getNativeVectorShapeImpl(typedOp); })
      .Default([](Operation *) { return std::nullopt; });
}


/// Adds patterns to unroll vector ops to SPIR-V native vector size.
void populateVectorUnrollPatterns(RewritePatternSet &patterns) {
  auto options = vector::UnrollVectorOptions().setNativeShapeFn(
      [=](auto op) { return getNativeVectorShape(op); });
  vector::populateVectorUnrollPatterns(patterns, options);
}

namespace {
struct LLVMGPUVectorLoweringPass
    : public LLVMGPUVectorLoweringBase<LLVMGPUVectorLoweringPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<affine::AffineDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<vector::VectorDialect>();
    registry.insert<scf::SCFDialect>();
  }
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Special peephole optimizations to clean up IR before further processing.
    {
      RewritePatternSet patterns(funcOp.getContext());
      // Pull in patterns to shuffle broadcast/transpose ops around in order to
      // cancel them or embed into contract ops. Embedding in the flexible
      // contract ops will help to sustain the structure through various
      // transformations.
      vector::populateVectorReductionToContractPatterns(patterns);
      // Pull in patterns to canonicalize transfer ops.
      vector::populateVectorTransferPermutationMapLoweringPatterns(patterns);
      // Fold consumer add ops into the contraction op itself.
      vector::ContractionOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
      // Fold transpose ops if possible as we cannot unroll it later.
      vector::TransposeOp::getCanonicalizationPatterns(patterns, funcOp.getContext());

      if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
        return signalPassFailure();
      }
    }

    LLVM_DEBUG({
      llvm::dbgs() << "--- After peephole optimization ---\n";
      funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
      llvm::dbgs() << "\n\n";
    });

    // // Fold tensor.extract_slice/insert_slice ops into transfer ops. This helps
    // // to remove those tensor slice ops so that we can enable further vector op
    // // transformations.
    // {
    //   RewritePatternSet patterns(funcOp.getContext());
    //   vector::TransferReadOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
    //   vector::TransferWriteOp::getCanonicalizationPatterns(patterns, funcOp.getContext());
    //   populateVectorTransferTensorSliceTransforms(patterns);

    //   if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    //     return signalPassFailure();
    //   }
    // }

    // LLVM_DEBUG({
    //   llvm::dbgs() << "--- After folding tensor extract/insert slice ops ---\n";
    //   funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    //   llvm::dbgs() << "\n\n";
    // });

    // // Then unroll vectors to native vector size. We try to use 128-bit
    // // vectors for memory access and 4/2/1 vector sizes for computation.
    // {
    //   RewritePatternSet patterns(funcOp.getContext());
    //   populateVectorUnrollPatterns(patterns);
    //   if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    //     return signalPassFailure();
    //   }
    // }

    // LLVM_DEBUG({
    //   llvm::dbgs() << "--- After unrolling vector ---\n";
    //   funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    //   llvm::dbgs() << "\n\n";
    // });

    // Next run canonicalization to cast away leading size-1 dimensions. They
    // can be generated from vector unrolling and generally cause issues to
    // cancel corresponding read/write or insert/extract op pairs. This also
    // need to happen before hoisting, where we would make certain vectors loop
    // carried. Once that's done, it's hard to handle the leading size-1
    // dimensions across regions.
    // {
    //   auto context = funcOp.getContext();
    //   RewritePatternSet patterns(context);

    //   // We need to pull in casting way leading one dims to allow cancelling
    //   // some read/write ops.
    //   vector::populateCastAwayVectorLeadingOneDimPatterns(patterns);

    //   // We may have vector.insert_strided_slice inserting 1-D native vectors
    //   // into n-D larger vectors with the above. Break that down too. This is a
    //   // companion transformation of unrolling.
    //   vector::populateVectorInsertExtractStridedSliceDecompositionPatterns(
    //       patterns);
    //   vector::ExtractOp::getCanonicalizationPatterns(patterns, context);

    //   // Trimming leading unit dims may generate broadcast/shape_cast ops. Clean
    //   // them up.
    //   vector::BroadcastOp::getCanonicalizationPatterns(patterns, context);
    //   vector::ShapeCastOp::getCanonicalizationPatterns(patterns, context);

    //   vector::TransferReadOp::getCanonicalizationPatterns(patterns, context);
    //   vector::TransferWriteOp::getCanonicalizationPatterns(patterns, context);
    //   populateVectorTransferTensorSliceTransforms(patterns);

    //   if (failed(applyPatternsAndFoldGreedily(funcOp, std::move(patterns)))) {
    //     return signalPassFailure();
    //   }
    // }

    // LLVM_DEBUG({
    //   llvm::dbgs() << "--- After trimming leading unit dims ---\n";
    //   funcOp.print(llvm::dbgs(), OpPrintingFlags().useLocalScope());
    //   llvm::dbgs() << "\n\n";
    // });

    {
      // Lower high level vector operations like contract or multidim reduce ops
      // to lower level vector ops.
      RewritePatternSet contractLoweringPatterns(funcOp.getContext());
      vector::populateVectorTransferPermutationMapLoweringPatterns(
          contractLoweringPatterns);
      vector::TransposeOp::getCanonicalizationPatterns(contractLoweringPatterns,
                                                       funcOp.getContext());
      vector::populateVectorBroadcastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorContractLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorTransformsOptions().setVectorTransformsOptions(
              vector::VectorContractLowering::OuterProduct));
      vector::populateVectorMaskOpLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorShapeCastLoweringPatterns(contractLoweringPatterns);
      vector::populateVectorMultiReductionLoweringPatterns(
          contractLoweringPatterns,
          vector::VectorMultiReductionLowering::InnerParallel);
      if (failed(applyPatternsAndFoldGreedily(
              funcOp, std::move(contractLoweringPatterns)))) {
        return signalPassFailure();
      }
    }

    RewritePatternSet vectorToLoopsPatterns(&getContext());
    VectorTransferToSCFOptions vectorToSCFOptions;
    vectorToSCFOptions.enableFullUnroll();
    populateVectorToSCFConversionPatterns(vectorToLoopsPatterns,
                                          vectorToSCFOptions);
    memref::populateFoldMemRefAliasOpPatterns(vectorToLoopsPatterns);
    vector::populateVectorTransferLoweringPatterns(vectorToLoopsPatterns);
    if (failed(applyPatternsAndFoldGreedily(
            funcOp, std::move(vectorToLoopsPatterns)))) {
      return signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorLoweringPass() {
  return std::make_unique<LLVMGPUVectorLoweringPass>();
}

} // namespace iree_compiler
} // namespace mlir
