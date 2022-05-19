// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include <iostream>

#include <memory>

#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

namespace {

struct TransformOptions : public PassPipelineOptions<TransformOptions> {
  // TODO(benvanik): replace the global iree-hal-target-backends flag with this.
  // ListOption<std::string> targets{
  //     *this, "targets", llvm::cl::desc("One or more HAL devices to target."),
  //     llvm::cl::ZeroOrMore};
  Option<bool> serializeExecutables{
      *this, "serialize-executables",
      llvm::cl::desc("Whether to serialize hal.executable.variant ops to "
                     "hal.executable.binary ops."),
      llvm::cl::init(true)};
  Option<bool> linkExecutables{
      *this, "link-executables",
      llvm::cl::desc("Whether to link hal.executable ops together."),
      llvm::cl::init(true)};
};

// TODO(#7277): move this to stream dialect (and add options for concurrency).
static llvm::cl::opt<unsigned> benchmarkDispatchRepeatCount{
    "iree-hal-benchmark-dispatch-repeat-count",
    llvm::cl::desc(
        "The number of times to repeat each hal.command_buffer.dispatch op. "
        "This simply duplicates the dispatch op and inserts barriers. It's "
        "meant for command buffers having linear dispatch structures."),
    llvm::cl::init(1)};

}  // namespace

static void addCleanupPatterns(OpPassManager &passManager) {
  // Standard MLIR cleanup.
  passManager.addPass(mlir::createCanonicalizerPass());
  passManager.addPass(mlir::createCSEPass());

  // Simplify util.global accesses; this can help with data flow tracking as
  // redundant store-loads are removed.
  passManager.addNestedPass<IREE::Util::InitializerOp>(
      IREE::Util::createSimplifyGlobalAccessesPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      IREE::Util::createSimplifyGlobalAccessesPass());

  // Cleanup and canonicalization of util.global (and other util ops).
  passManager.addPass(IREE::Util::createApplyPatternsPass());
  passManager.addPass(IREE::Util::createFoldGlobalsPass());
  passManager.addPass(IREE::Util::createFuseGlobalsPass());
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions,
                                   const TransformOptions &transformOptions) {
  //----------------------------------------------------------------------------
  // Input cleanup and simplification
  //----------------------------------------------------------------------------

  // Perform cleanup upon entry so that our IR is in a good state for assignment
  // and initial interface analysis (we rely on CSE and such having been run).
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Device assignment and interface materialization
  //----------------------------------------------------------------------------

  // The HAL must know its targets early on in the process. This pass discovers/
  // derives/specifies the target devices and annotates the module with that
  // information. This allows subsequent passes to lookup which devices they are
  // targeting.
  if (!targetOptions.targets.empty()) {
    // Today we just assign devices from parameters but we should instead be
    // performing analysis at the flow level and then doing magic device
    // database lookups here.
    passManager.addPass(createAssignTargetDevicesPass(targetOptions.targets));
  }
  passManager.addPass(createVerifyTargetEnvironmentPass());

  // Pack dispatch operands on stream.executable into i32 values.
  // We do this prior to materializing interfaces so we can easily add/remove
  // operands. By not doing this afterward on hal ops we can have stronger
  // type verification. Though we're manipulating stream ops we need to use our
  // target information we only have after device assignment to know what data
  // types are supported and how many push constants we can use.
  //
  // TODO(benvanik): re-evaluate moving this up in to streams and making the
  // requirements universal. It's a leak of HAL behavior (i32 push constants)
  // but would fit better up in there. We need to re-evaluate once there are
  // multiple devices with different data type support or host/device index
  // width mismatches.
  passManager.addPass(createPackDispatchOperandsPass());

  // TODO(benvanik): when we spill push constants spill to staging buffers. But
  // maybe up in stream first? Need to know push constant limit but that could
  // be specified as a stream option (max operand count).

  // Each executable needs a hal.interface to specify how the host and
  // device communicate across the ABI boundary.
  passManager.addPass(createMaterializeInterfacesPass());

  // Dump a source listing of each hal.executable and update the source
  // locations in the IR. This will allow us to easily inspect each executable
  // and give downstream tools that can display source information something
  // more useful and slim than the entire original source model.
  if (!targetOptions.sourceListingPath.empty()) {
    passManager.addPass(
        createDumpExecutableSourcesPass(targetOptions.sourceListingPath));
  }

  // Dump standalone hal.executable benchmark modules.
  // Today this only works for executables that have static dispatch parameters
  // and is only useful for basic microbenchmarking.
  if (!targetOptions.executableBenchmarksPath.empty()) {
    passManager.addPass(createDumpExecutableBenchmarksPass(
        targetOptions.executableBenchmarksPath));
  }

  // TODO(benvanik): move translation after conversion; today translation
  // inserts the workgroup count logic we need to convert but we could instead
  // insert placeholder ops that are expanded after translation.
  //
  // Translate each executable variant to its target IR form.
  // It's extremely important this runs parallelized as it's where a large
  // majority of our compilation time lives (we invoke LLVM and lld and such).
  //
  // After this point the executables are opaque blobs and we cannot change
  // their interfaces.
  passManager.addNestedPass<IREE::HAL::ExecutableOp>(
      createTranslateExecutablesPass());

  //----------------------------------------------------------------------------
  // Host program conversion
  //----------------------------------------------------------------------------

  // Convert supported input dialects (std, stream, etc) into the HAL dialect.
  passManager.addPass(createConvertToHALPass());
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Executable packing and runtime loading
  //----------------------------------------------------------------------------

  // TODO(benvanik): move translation down to here.

  // After all executables are translated and before resolving entry point
  // ordinals, we allow the backends to link executables together. For example,
  // the LLVM AOT backend may combine all executable targets for the same
  // architecture into a single executable and link it as a shared library.
  if (transformOptions.linkExecutables) {
    passManager.addPass(createLinkExecutablesPass());
  }

  // Resolve entry point ordinals from nested symbol references prior to
  // serialization. As this pass creates lookup ops it should run before
  // MaterializeResourceCachesPass.
  passManager.addPass(createResolveEntryPointOrdinalsPass());

  // Gather cachable resources such as executables and descriptor sets and
  // cache them at initialization-time.
  passManager.addPass(createMaterializeResourceCachesPass(targetOptions));

  //----------------------------------------------------------------------------
  // Device management and specialization
  //----------------------------------------------------------------------------

  // Inline hal.device.switch ops and memoize their queries such that we can
  // better CSE/fold dispatch logic.
  passManager.addNestedPass<IREE::Util::InitializerOp>(
      createInlineDeviceSwitchesPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createInlineDeviceSwitchesPass());

  // Memoize device queries such that we don't need to repeatedly ask the same
  // information at runtime.
  passManager.addPass(createMemoizeDeviceQueriesPass());

  // Big cleanup after all our conversion and materialization.
  addCleanupPatterns(passManager);

  // HACK: repeat dispatch ops for benchmarks.
  if (benchmarkDispatchRepeatCount != 1) {
    passManager.addNestedPass<mlir::func::FuncOp>(
        createBenchmarkBatchDispatchesPass(benchmarkDispatchRepeatCount));
  }

  // Elide redundant command buffer state ops created during conversion.
  passManager.addNestedPass<IREE::Util::InitializerOp>(
      createElideRedundantCommandsPass());
  passManager.addNestedPass<mlir::func::FuncOp>(
      createElideRedundantCommandsPass());

  // Fixup workgroup count calculations that may have used the affine dialect.
  // Kind of random here but can happen if the benchmarking code does things.
  passManager.addPass(createLowerAffinePass());

  // Combine the initializers we emitted during resource cache materialization.
  passManager.addPass(IREE::Util::createCombineInitializersPass());
  addCleanupPatterns(passManager);

  //----------------------------------------------------------------------------
  // Executable serialization
  //----------------------------------------------------------------------------

  // Happens at the very end as IR is much more debuggable with the executable
  // contents not turned into a big base64 string.
  if (transformOptions.serializeExecutables) {
    passManager.addNestedPass<IREE::HAL::ExecutableOp>(
        createSerializeExecutablesPass());

    // NOTE: symbol DCE will destroy executable target contents, so only run it
    // if we serialized things.
    passManager.addPass(createSymbolDCEPass());
  }
}

void buildHALTransformPassPipeline(OpPassManager &passManager,
                                   const TargetOptions &targetOptions) {
  TransformOptions transformOptions;
  buildHALTransformPassPipeline(passManager, targetOptions, transformOptions);
}

void registerHALTransformPassPipeline() {
  PassPipelineRegistration<TransformOptions>(
      "iree-hal-transformation-pipeline",
      "Runs the full IREE HAL dialect transformation pipeline",
      [](OpPassManager &passManager, const TransformOptions &transformOptions) {
        buildHALTransformPassPipeline(
            passManager, TargetOptions::FromFlags::get(), transformOptions);
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
