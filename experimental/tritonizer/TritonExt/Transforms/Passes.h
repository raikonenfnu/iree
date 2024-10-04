// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//===----------------------------------------------------------------------===//
//
// This file includes the LLVMGPU Passes.
//
//===----------------------------------------------------------------------===//

#ifndef IREE_COMPILER_TRITON_PASSES_H_
#define IREE_COMPILER_TRITON_PASSES_H_

#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

using IREE::GPU::GPUPipelineOptions;

//----------------------------------------------------------------------------//
// Triton backend Pass Pipelines.
//----------------------------------------------------------------------------//

/// Lowering based on vector distribution patterns.
void addGPUTritonPassPipeline(OpPassManager &funcPassManager,
                              const GPUPipelineOptions &options);

/// Populates passes needed to preprocess and select the translation strategy.
void buildTritonCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManagery);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via
/// the structured ops path. The pass manager `pm` in here should operate on
/// the module within the IREE::HAL::ExecutableOp.
void buildTritonCodegenPassPipeline(OpPassManager &variantPassManagery);

//----------------------------------------------------------------------------//
// Register LLVMGPU Passes
//----------------------------------------------------------------------------//

#define GEN_PASS_DECL
#include "experimental/tritonizer/TritonExt/Transforms/Passes.h.inc"  // IWYU pragma: keep

void registerCodegenTritonPasses();

}  // namespace mlir::iree_compiler

#endif  // IREE_COMPILER_TRITON_PASSES_H_
