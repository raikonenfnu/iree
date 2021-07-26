// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Passes.h"

#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"
#include "iree/compiler/Dialect/Shape/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/SCFToStandard/SCFToStandard.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/StandardOps/Transforms/Passes.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

static Value gpuAllocationFunction(OpBuilder &builder, Location loc,
                                   ArrayRef<int64_t> staticShape,
                                   Type elementType,
                                   ArrayRef<Value> dynamicSizes) {
  MemRefType allocType = MemRefType::get(staticShape, elementType, {}, 3);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes);
}

void addGPUVectorizationPassPipeline(OpPassManager &pm) {
  // Convert tensor to buffers.
  addLinalgBufferizePasses(pm.nest<ModuleOp>(), gpuAllocationFunction);
  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addPass(createLLVMGPUTileAndDistributeToThreads());
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  pm.nest<ModuleOp>().addNestedPass<FuncOp>(
      createLLVMGPURemoveSingleIterationLoopPass());

  // Linalg -> vector
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createLLVMGPUVectorizationPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &pm) {
  // Convert tensor to buffers.
  addLinalgBufferizePasses(pm.nest<ModuleOp>(), gpuAllocationFunction);

  //===--------------------------------------------------------------------===//
  // Initial clean up.
  //===--------------------------------------------------------------------===//
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  pm.addPass(createLLVMGPUTileAndDistributeToThreads());
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  pm.nest<ModuleOp>().addNestedPass<FuncOp>(
      createLLVMGPURemoveSingleIterationLoopPass());
}

static void addLowerToLLVMGPUPasses(OpPassManager &pm, bool useROCM) {
  pm.addNestedPass<ModuleOp>(createLowerAffinePass());
  pm.addNestedPass<ModuleOp>(createCanonicalizerPass());
  pm.addNestedPass<ModuleOp>(createCSEPass());

  // LinalgExt -> SCF
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(
      linalg_ext::createLinalgExtToLoopsPass());

  // Linalg -> SCF
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createConvertLinalgToLoopsPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  // Handled tensor-type constants.
  pm.addNestedPass<ModuleOp>(createTensorConstantBufferizePass());
  pm.addNestedPass<ModuleOp>(createFoldTensorExtractOpPass());

  // SCF -> STD
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createLowerToCFGPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCanonicalizerPass());
  pm.nest<ModuleOp>().addNestedPass<FuncOp>(createCSEPass());

  pm.addNestedPass<ModuleOp>(createLowerAffinePass());

  // Strip out the debug info for the kernel as CUDA driver doesn't diggest PTX
  // debug info well.
  pm.addNestedPass<ModuleOp>(createStripDebugInfoPass());
  if (useROCM) {
    // convert to ROCDL.
    pm.addNestedPass<ModuleOp>(createConvertToROCDLPass());
  } else {
    // convert to NVVM.
    pm.addNestedPass<ModuleOp>(createConvertToNVVMPass());
  }
}

void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM) {
  // pm.nest<ModuleOp>().addNestedPass<FuncOp>(createConvertContractionToMatmulPass());
  pm.addPass(createLLVMGPULowerExecutableTargetPass());
  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(pm, useROCM);
}

}  // namespace iree_compiler
}  // namespace mlir
