// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/Sandbox/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/Passes.h.inc"
}  // namespace

void registerCodegenPasses() {
  // Generated.
  registerPasses();
  registerSandboxPasses();

  static PassPipelineRegistration<> LinalgLLVMVPipeline(
      "iree-codegen-linalg-to-llvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to LLVM",
      [](OpPassManager &passManager) {
        buildLLVMCPUCodegenPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager) {
        buildLLVMGPUTransformPassPipeline(passManager, false);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline",
      "Runs the progressive lowering pipeline from Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildLLVMGPUTransformPassPipeline(passManager, true);
      });

  static PassPipelineRegistration<> LinalgSPIRVPipeline(
      "iree-codegen-linalg-to-spirv-pipeline",
      "Runs the progressive lowering pipeline from linalg to SPIR-V",
      [](OpPassManager &passManager) {
        buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false,
                                      /*use64bitIndex=*/false);
      });
}

/// Hook to verify the lowering configuration and translation info for an
/// operation.
LogicalResult verifyLoweringConfiguration(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize) {
  switch (translationInfo.getDispatchLoweringPassPipeline()) {
    case IREE::Codegen::DispatchLoweringPassPipeline::
        CPUAArchDoubleTilingExpert:
      return verifyDoubleTilingExpertPassPipelineConfig(op, loweringConfig,
                                                        translationInfo);
    case IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt:
      return verifyGPUMatmulSimtPassPipeline(op, loweringConfig,
                                             translationInfo, workgroupSize);
    default:
      break;
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
