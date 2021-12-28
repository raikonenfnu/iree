// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Sandbox/PassDetail.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Linalg/Transforms/CodegenStrategy.h"
#include "mlir/Dialect/Linalg/Transforms/Hoisting.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Dialect/Vector/VectorTransforms.h"
#include "mlir/Dialect/X86Vector/Transforms.h"
#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace mlir::linalg;

#define DEBUG_TYPE "iree-linalg-tensor-codegen-driver"

//===----------------------------------------------------------------------===//
// IREE specific functions
//===----------------------------------------------------------------------===//

/// Default method to initialize the tiling options in IREE. These could be
/// overriden by the command line options if specified. For now the sentinel
/// -1 is used for avoiding querying the lowering config.
static bool getTilingOptionsFromConfig(int64_t tilingLevel,
                                       LinalgTilingOptions &tilingOptions) {
  if (tilingLevel != -1) {
    tilingOptions.setTileSizeComputationFunction(
        [tilingLevel](OpBuilder &builder,
                      Operation *operation) -> SmallVector<Value, 4> {
          return ::mlir::iree_compiler::getTileSizes(builder, operation,
                                                     tilingLevel);
        });
    return true;
  }
  return false;
}

/// Default method to initialize the tiling options for fusion in IREE. These
/// could be ovveridden by the command line options if specified.
static FailureOr<LinalgTilingAndFusionOptions> getTileAndFuseOptionsFromConfig(
    FuncOp funcOp, int64_t tilingLevel) {
  SmallVector<Operation *> computeOps;
  SmallVector<mlir::iree_compiler::LoopTilingAndDistributionInfo> tiledLoops;
  mlir::iree_compiler::IREE::Codegen::LoweringConfigAttr loweringConfig;
  if (tilingLevel != -1 &&
      succeeded(getComputeOps(funcOp, computeOps, tiledLoops))) {
    for (auto op : computeOps) {
      if (auto currLoweringConfig = iree_compiler::getLoweringConfig(op)) {
        if (loweringConfig) {
          return LogicalResult(funcOp.emitOpError(
              "unhandled multiple lowering configurations in compute ops"));
        }
        loweringConfig = currLoweringConfig;
      }
    }
  }
  if (!loweringConfig) {
    return LinalgTilingAndFusionOptions();
  }
  LinalgTilingAndFusionOptions options;
  options.tileSizes.assign(loweringConfig.getTileSizeVals(tilingLevel));
  return options;
}

//===----------------------------------------------------------------------===//
// From Sandbox
//===----------------------------------------------------------------------===//

namespace {
struct LinalgFusePass : public LinalgFuseBase<LinalgFusePass> {
  LinalgFusePass(int64_t tilingLevel = -1, bool vectorize = false) {
    this->tilingLevel.setValue(tilingLevel);
    this->vectorize.setValue(vectorize);
  }
  LinalgFusePass(const LinalgFusePass &pass) {}
  void runOnOperation() override;
};

struct LinalgSingleTilingExpertPass
    : public LinalgSingleTilingExpertBase<LinalgSingleTilingExpertPass> {
  LinalgSingleTilingExpertPass(int64_t tilingLevel = -1,
                               bool vectorize = false) {
    this->tilingLevel.setValue(tilingLevel);
    this->vectorize.setValue(vectorize);
  }
  LinalgSingleTilingExpertPass(const LinalgSingleTilingExpertPass &pass) {}

  /// Function pass entry point.
  void runOnOperation() override;
};

struct LinalgVectorLoweringPass
    : public LinalgVectorLoweringBase<LinalgVectorLoweringPass> {
  LinalgVectorLoweringPass(int64_t vectorLoweringStage = 0) {
    this->vectorLoweringStage.setValue(vectorLoweringStage);
  }
  LinalgVectorLoweringPass(const LinalgVectorLoweringPass &pass) {}

  void runOnOperation() override;
};
}  // namespace

/// Return the neutral element as a new Value.
/// For now, just assume it is the zero of type.
/// In the future, it should be the zero of type + op.
static Value getNeutralOfLinalgOp(OpBuilder &b, OpOperand &op) {
  auto t = getElementTypeOrSelf(op.get().getType());
  return b.create<arith::ConstantOp>(op.getOwner()->getLoc(), t,
                                     b.getZeroAttr(t));
}

/// Collect all Linalg ops, they must all have tensor semantics.
/// For now this just fuses everything.
// TODO: finer control.
void LinalgFusePass::runOnOperation() {
  FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  FailureOr<LinalgTilingAndFusionOptions> defaultTilingOptions =
      getTileAndFuseOptionsFromConfig(funcOp, tilingLevel);
  if (failed(defaultTilingOptions)) {
    return signalPassFailure();
  }
  LinalgTilingAndFusionOptions tilingOptions = defaultTilingOptions.getValue();
  bool doTiling = !tilingOptions.tileSizes.empty();
  if (!tileSizes.empty()) {
    doTiling = true;
    tilingOptions.tileSizes = {tileSizes.begin(), tileSizes.end()};
  }
  if (!tileInterchange.empty()) {
    tilingOptions.tileInterchange = {tileInterchange.begin(),
                                     tileInterchange.end()};
  }

  // Set up padding options.
  // TODO: Replace the lambdas by either functions defined in MLIR core or even
  // adapt the LinalgPaddingOptions to take the `hoistPaddings` and
  // `packPaddings` arrays directly.
  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < packPaddings.size()
               ? packPaddings[opOperand.getOperandNumber()]
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < hoistPaddings.size()
               ? hoistPaddings[opOperand.getOperandNumber()]
               : 0;
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);

  CodegenStrategy strategy;
  strategy.tileAndFuseIf(doTiling, anchorOpName, tilingOptions)
      .padIf(pad, "", paddingOptions)
      .vectorizeIf(vectorize, "", nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());

  if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
}

void LinalgSingleTilingExpertPass::runOnOperation() {
  FuncOp funcOp = getOperation();

  // Set up tiling and vectorization options.
  LinalgTilingOptions tilingOptions;
  bool doTiling = getTilingOptionsFromConfig(tilingLevel, tilingOptions);
  if (!tileSizes.empty()) {
    doTiling = true;
    tilingOptions = tilingOptions.setTileSizes(tileSizes);
  }
  if (!tileInterchange.empty())
    tilingOptions = tilingOptions.setInterchange(
        SmallVector<unsigned>(tileInterchange.begin(), tileInterchange.end()));
  if (scalarizeDynamicDims) {
    doTiling = true;
    tilingOptions = tilingOptions.scalarizeDynamicDims();
  }
  tilingOptions = tilingOptions.setPeeledLoops(peeledLoops);

  // Set up padding options.
  // TODO: Replace the lambdas by either functions defined in MLIR core or even
  // adapt the LinalgPaddingOptions to take the `hoistPaddings` and
  // `packPaddings` arrays directly.
  auto packFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < packPaddings.size()
               ? packPaddings[opOperand.getOperandNumber()]
               : false;
  };
  auto hoistingFunc = [&](OpOperand &opOperand) {
    return opOperand.getOperandNumber() < hoistPaddings.size()
               ? hoistPaddings[opOperand.getOperandNumber()]
               : 0;
  };
  LinalgPaddingOptions paddingOptions;
  paddingOptions.setPaddingValueComputationFunction(getNeutralOfLinalgOp);
  paddingOptions.setPaddingNoFoldComputationFunction(packFunc);
  paddingOptions.setPaddingHoistComputationFunction(hoistingFunc);

  CodegenStrategy strategy;
  StringRef genericOpName = GenericOp::getOperationName();
  strategy.tileIf(doTiling, anchorOpName, tilingOptions)
      .padIf(pad, anchorOpName, paddingOptions)
      .generalizeIf(generalize, anchorOpName)
      // TODO: decomposeToLowerDimIf when the need arises.
      .interchangeIf(!iteratorInterchange.empty(), iteratorInterchange)
      .vectorizeIf(vectorize, generalize ? genericOpName : anchorOpName,
                   nullptr, vectorizePadding);

  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());

  if (decomposeToLowerDimOp) {
    dynamicPM.addPass(createLinalgStrategyDecomposePass());
  }

  if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
}

void LinalgVectorLoweringPass::runOnOperation() {
  LLVM_DEBUG(llvm::dbgs() << "\n ---- Stage : " << vectorLoweringStage;);
  vector::VectorTransposeLowering vectorTransposeLowering =
      llvm::StringSwitch<vector::VectorTransposeLowering>(
          lowerVectorTransposeTo.getValue())
          .Case("eltwise", vector::VectorTransposeLowering::EltWise)
          .Case("flat_transpose", vector::VectorTransposeLowering::Flat)
          .Case("shuffle", vector::VectorTransposeLowering::Shuffle)
          .Default(vector::VectorTransposeLowering::EltWise);
  vector::VectorMultiReductionLowering vectorMultiReductionLowering =
      llvm::StringSwitch<vector::VectorMultiReductionLowering>(
          lowerVectorMultiReductionTo.getValue())
          .Case("innerreduction",
                vector::VectorMultiReductionLowering::InnerReduction)
          .Default(vector::VectorMultiReductionLowering::InnerParallel);
  vector::VectorContractLowering vectorContractLowering =
      llvm::StringSwitch<vector::VectorContractLowering>(
          lowerVectorContractionTo.getValue())
          .Case("matrixintrinsics", vector::VectorContractLowering::Matmul)
          .Case("dot", vector::VectorContractLowering::Dot)
          .Case("outerproduct", vector::VectorContractLowering::OuterProduct)
          .Default(vector::VectorContractLowering::OuterProduct);
  vector::VectorTransferSplit vectorTransferSplit =
      llvm::StringSwitch<vector::VectorTransferSplit>(
          splitVectorTransfersTo.getValue())
          .Case("none", vector::VectorTransferSplit::None)
          .Case("linalg-copy", vector::VectorTransferSplit::LinalgCopy)
          .Case("vector-transfers", vector::VectorTransferSplit::VectorTransfer)
          .Default(vector::VectorTransferSplit::None);

  // Per-function lowering pipeline.
  vector::VectorTransformsOptions vectorTransformOptions =
      vector::VectorTransformsOptions()
          .setVectorTransposeLowering(vectorTransposeLowering)
          .setVectorTransformsOptions(vectorContractLowering)
          .setVectorMultiReductionLowering(vectorMultiReductionLowering)
          .setVectorTransferSplit(vectorTransferSplit);
  VectorTransferToSCFOptions vectorTransferToSCFOptions =
      VectorTransferToSCFOptions()
          .enableFullUnroll(unrollVectorTransfers)
          .enableLowerPermutationMaps();

  LinalgVectorLoweringOptions vectorLoweringOptions =
      LinalgVectorLoweringOptions()
          // Lowering of vector contractions.
          .enableContractionLowering(vectorLoweringStage >= 0)
          // Lowering of vector multi_reduction.
          .enableMultiReductionLowering(vectorLoweringStage >= 1)
          // Whether to split full/partial vector.transfer ops.
          .enableTransferPartialRewrite(vectorLoweringStage >= 2 &&
                                        vectorTransferSplit !=
                                            vector::VectorTransferSplit::None)
          // Set the maximum vector load / store rank.
          .setMaxTransferRank(maxTransferRank)
          // Lower vector.transfer to vector.transfer of max rank.
          .enableTransferLowering(vectorLoweringStage >= 3)
          // Conversion to scf.
          .enableTransferToSCFConversion(vectorLoweringStage >= 4)
          .setVectorTransferToSCFOptions(vectorTransferToSCFOptions)
          // Lowering of vector.shape_cast.
          .enableShapeCastLowering(vectorLoweringStage >= 5)
          // Lowering of vector.transpose.
          .enableVectorTransposeLowering(vectorLoweringStage >= 6)
          .setVectorTransformsOptions(vectorTransformOptions)
          .enableAVX2Lowering(lowerVectorTransposeToAVX2)
          .setAVX2LoweringOptions(
              x86vector::avx2::LoweringOptions().setTransposeOptions(
                  x86vector::avx2::TransposeLoweringOptions()
                      .lower4x8xf32(lowerVectorTransposeToAVX2)
                      .lower8x8xf32(lowerVectorTransposeToAVX2)));

  CodegenStrategy strategy;
  strategy.vectorLowering(vectorLoweringOptions);
  // Created a nested OpPassManager and run.
  OpPassManager dynamicPM(FuncOp::getOperationName());
  FuncOp funcOp = getOperation();
  strategy.configurePassPipeline(dynamicPM, funcOp.getContext());
  if (failed(runPipeline(dynamicPM, funcOp))) return signalPassFailure();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgFusePass() {
  return std::make_unique<LinalgFusePass>();
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::createLinalgSingleTilingExpertPass() {
  return std::make_unique<LinalgSingleTilingExpertPass>();
}

std::unique_ptr<OperationPass<FuncOp>> mlir::createLinalgVectorLoweringPass(
    int64_t vectorLoweringStage) {
  return std::make_unique<LinalgVectorLoweringPass>(vectorLoweringStage);
}

//===----------------------------------------------------------------------===//
// Transforms
//===----------------------------------------------------------------------===//

void mlir::addLowerToVectorTransforms(OpPassManager &passManager) {
  passManager.addPass(createLinalgVectorLoweringPass(0));
  passManager.addPass(createLinalgVectorLoweringPass(1));
  passManager.addPass(createLinalgVectorLoweringPass(2));
  passManager.addPass(createLinalgVectorLoweringPass(3));
  passManager.addPass(createLinalgVectorLoweringPass(4));
  passManager.addPass(createLinalgVectorLoweringPass(5));
  passManager.addPass(createLinalgVectorLoweringPass(6));
}

//===----------------------------------------------------------------------===//
// IREE specific pass creation methods to allow invocation from within IREEs
// backend pipelines
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<FuncOp>>
mlir::iree_compiler::createLinalgFusePass(int64_t tilingLevel, bool vectorize) {
  return std::make_unique<LinalgSingleTilingExpertPass>(tilingLevel, vectorize);
}

std::unique_ptr<OperationPass<FuncOp>>
mlir::iree_compiler::createLinalgSingleTilingExpertPass(int64_t tilingLevel,
                                                        bool vectorize) {
  return std::make_unique<LinalgSingleTilingExpertPass>(tilingLevel, vectorize);
}

namespace mlir {
/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Sandbox/Passes.h.inc"
}  // namespace mlir

void mlir::iree_compiler::registerSandboxPasses() { registerPasses(); }
