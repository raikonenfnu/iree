// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
#define IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_

#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/HAL/Target/TargetBackend.h"
#include "llvm/ADT/StringMap.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

// Adds a set of passes to the given pass manager that run the required HAL
// transforms in the canonical order.
//
// Most translation code should prefer to use this instead of manually adding
// the passes themselves to ensure that expected pass ordering is observed.
//
// The expected usage is:
//   <run conversion to flow/sequencer/etc>
//   buildHALTransformPassPipeline & run
//   <run conversion from HAL to vm/etc>
void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions);

// Adds a set of passes to the given pass manager that run the head of the HAL
// pipeline to assign devices, materialize interfaces, and translate
// executables. The host portion of the program is annotated but not modified.
void buildHALConfigurationPassPipeline(OpPassManager &passManager,
                                       const TargetOptions &targetOptions);

void registerHALTransformPassPipeline();
void registerHALConfigurationPassPipeline();

//===----------------------------------------------------------------------===//
// Conversion
//===----------------------------------------------------------------------===//

// Converts input flow/std/etc dialects to the IREE HAL dialect.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createConvertToHALPass();

// Materializes timelines for device queues.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMaterializeTimelinesPass();

//===----------------------------------------------------------------------===//
// Device management
//===----------------------------------------------------------------------===//

// Verifies that the target execution environment is valid.
// #hal.device.target and #hal.executable.target attribute placement and
// definition will be checked as well along with other structural requirements.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createVerifyTargetEnvironmentPass();

// Assigns the HAL devices the module will target to the given list of targets.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createAssignTargetDevicesPass(
    ArrayRef<std::string> targets);

// Applies fixups to the program for when using legacy HAL devices that only
// support synchronous execution. Once all devices support async this will be
// removed.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createFixupLegacySyncPass();

// Outlines hal.device.switch conditions into functions and inlines conditions.
std::unique_ptr<OperationPass<void>> createInlineDeviceSwitchesPass();

// Finds hal.device.query ops and creates variables initialized on startup.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createMemoizeDeviceQueriesPass();

//===----------------------------------------------------------------------===//
// Executable translation
//===----------------------------------------------------------------------===//

// Defines hal.executables and hal.interfaces for flow.executable ops based on
// usage within the module. Target backends are queried to check for support and
// device placements are made.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeInterfacesPass();

// Dumps individual hal.executable source listings to |path|.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createDumpExecutableSourcesPass(
    StringRef path);

// Dumps standalone hal.executable benchmarks to |path|.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createDumpExecutableBenchmarksPass(StringRef path);

// Translates hal.executable.variant ops via a nested translation pipeline.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createTranslateExecutablesPass();

// Translates hal.executable.variant ops for the specified |target| backend.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTranslateTargetExecutableVariantsPass(StringRef target);

// Calls into each target backend to have it link multiple hal.executables
// together (if that makes sense). For example, the LLVM AOT backend may combine
// all executable targets for the same architecture into a single executable and
// link it as a shared library.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkExecutablesPass();

// Links executables for the specified |target| backend.
std::unique_ptr<OperationPass<mlir::ModuleOp>> createLinkTargetExecutablesPass(
    StringRef target);

// Resolves hal.executable.export references to ordinals.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createResolveExportOrdinalsPass();

// Converts hal.executable.variants to one or more hal.executable.binary ops.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeExecutablesPass(int debugLevel = 2,
                               std::string dumpIntermediatesPath = "",
                               std::string dumpBinariesPath = "");

// Serializes executables for the specified |target| backend.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableOp>>
createSerializeTargetExecutablesPass(StringRef target, int debugLevel = 2,
                                     std::string dumpIntermediatesPath = "",
                                     std::string dumpBinariesPath = "");

//===----------------------------------------------------------------------===//
// Resource initialization, caching, and optimization
//===----------------------------------------------------------------------===//

// Finds all resource lookups (such as hal.executable.lookup), materializes
// their cache storage and initialization, and rewrites the lookups to
// references.
std::unique_ptr<OperationPass<mlir::ModuleOp>>
createMaterializeResourceCachesPass(TargetOptions targetOptions);

// Elides stateful command buffer ops that set redundant state.
std::unique_ptr<OperationPass<void>> createElideRedundantCommandsPass();

// Repeats dispatches `iree-hal-repeat-dispatch-num` times, which is 1 by
// default.
std::unique_ptr<OperationPass<func::FuncOp>> createBenchmarkBatchDispatchesPass(
    unsigned repeatCount);

//===----------------------------------------------------------------------===//
// Register all Passes
//===----------------------------------------------------------------------===//

inline void registerHALPasses() {
  registerHALTransformPassPipeline();
  registerHALConfigurationPassPipeline();
  auto targetOptions = TargetOptions::FromFlags::get();
  createAssignTargetDevicesPass({});
  createBenchmarkBatchDispatchesPass(/*repeatCount=*/1);
  createConvertToHALPass();
  createDumpExecutableSourcesPass("");
  createElideRedundantCommandsPass();
  createInlineDeviceSwitchesPass();
  createFixupLegacySyncPass();
  createLinkExecutablesPass();
  createLinkTargetExecutablesPass("");
  createMaterializeInterfacesPass();
  createMaterializeResourceCachesPass(targetOptions);
  createMaterializeTimelinesPass();
  createMemoizeDeviceQueriesPass();
  createResolveExportOrdinalsPass();
  createSerializeExecutablesPass();
  createSerializeTargetExecutablesPass("");
  createTranslateExecutablesPass();
  createTranslateTargetExecutableVariantsPass("");
  createVerifyTargetEnvironmentPass();
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TRANSFORMS_PASSES_H_
