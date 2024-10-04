// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include <cstdint>

#include "experimental/tritonizer/TritonExt/Transforms/Passes.h"
#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Utils/PassUtils.h"
#include "llvm/ADT/STLForwardCompat.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/CommandLine.h"
#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ComplexToStandard/ComplexToStandard.h"
#include "mlir/Conversion/MathToLLVM/MathToLLVM.h"
#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "iree-triton-lowering-pass-pipeline"

namespace mlir::iree_compiler {

static llvm::cl::opt<ReorderWorkgroupsStrategy> clReorderWorkgroupsStrategy(
    "iree-triton-reorder-workgroups-strategy",
    llvm::cl::desc("Reorder workgroup IDs using the selected strategy"),
    llvm::cl::values(clEnumValN(ReorderWorkgroupsStrategy::None, "none",
                                "No workgroup reordering"),
                     clEnumValN(ReorderWorkgroupsStrategy::Swizzle, "swizzle",
                                "Swizzle"),
                     clEnumValN(ReorderWorkgroupsStrategy::Transpose,
                                "transpose", "Transpose")),
    llvm::cl::init(ReorderWorkgroupsStrategy::None));

static llvm::cl::opt<unsigned> clReorderWorkgroupsLogSwizzleTile(
    "iree-triton-reorder-workgroups-log-swizzle-tile",
    llvm::cl::desc("Reorder workgroups: log tile size to use"),
    llvm::cl::init(3));

static llvm::cl::opt<bool> clLLVMGPUUseIgemm(
    "iree-triton-llvmgpu-use-igemm",
    llvm::cl::desc("Enable implicit gemm for convolutions."),
    llvm::cl::init(false));

//===----------------------------------------------------------------------===//
// Bufferization Configuration
//===----------------------------------------------------------------------===//

static bool hasThreadMapping(scf::ForallOp forall) {
  if (!forall.getMapping().has_value()) {
    return false;
  }
  return llvm::any_of(*forall.getMapping(),
                      llvm::IsaPred<gpu::GPUThreadMappingAttr>);
  ;
}

// All pipelines that use this allocation function distribute scf.forall ops
// after bufferizing. This means that to differentiate between an allocation in
// function memory and workgroup memory, we need to look for a parent
// scf.forall op with a thread mapping. If not present, we allocate workgroup
// memory. Pipelines that choose to distribute in a different order will have
// to use a different allocation function.
static FailureOr<Value> gpuAllocationFn(OpBuilder &builder, Location loc,
                                        MemRefType memRefType,
                                        ValueRange dynamicSizes,
                                        unsigned alignment) {
  Block *insertionBlock = builder.getInsertionBlock();
  Operation *parent = insertionBlock->getParentOp();
  scf::ForallOp enclosingForall = dyn_cast<scf::ForallOp>(parent);
  if (!enclosingForall) {
    enclosingForall = parent->getParentOfType<scf::ForallOp>();
  }
  if (enclosingForall && hasThreadMapping(enclosingForall)) {
    auto addressSpace = gpu::AddressSpaceAttr::get(
        builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
    auto allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), addressSpace);
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }

  auto addressSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getWorkgroupAddressSpace());
  auto allocType =
      MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                      AffineMap(), addressSpace);
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

// Barriers are only needed when copying to/from workgroup memory. The only
// other kind of memory that can be allocated is function memory, which is local
// to a thread.
static LogicalResult gpuCopyFn(OpBuilder &builder, Location loc, Value from,
                               Value to) {
  bool needsBarrier = false;
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(from.getType()))) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier) builder.create<gpu::BarrierOp>(loc);
  Operation *copy = builder.create<memref::CopyOp>(loc, from, to);
  if (needsBarrier) {
    setMarker(copy, getCopyToWorkgroupMemoryMarker());
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

// Returns success when workgroup reordering is supported / enabled for
// `funcOp`. On ROCm, we require workgroup counts to be static.
static LogicalResult canReorderWorkgroups(FunctionOpInterface funcOp) {
  auto target = IREE::HAL::ExecutableTargetAttr::lookup(funcOp);
  if (!target) {
    return failure();
  }
  if (target.getBackend() != "rocm") return success();

  // Workgroup reordering on ROCm currently requires all workgrup counts to be
  // static.
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  if (llvm::any_of(workgroupCounts, ShapedType::isDynamic)) return failure();

  // This is further restricted to 2D+ grids as we reorder along the X and Y
  // workgroup IDs.
  return success(workgroupCounts.size() >= 2);
}

// Reconciles workgroup reordering strategy based on the pipeline `option` and
// the CLI flag.
static ReorderWorkgroupsStrategy getReorderWorkgroupsStrategy(
    const std::optional<ReorderWorkgroupsStrategy> &option) {
  return option.value_or(clReorderWorkgroupsStrategy);
}

//===----------------------------------------------------------------------===//
// Common Pass Recipes
//===----------------------------------------------------------------------===//

static void tileAndDistributeToWorkgroup(
    OpPassManager &funcPassManager,
    std::optional<ConvertToDestinationPassingStylePassOptions>
        convertToDpsOptions = ConvertToDestinationPassingStylePassOptions{}) {
  funcPassManager.addPass(createTileAndDistributeToWorkgroupsPass(
      kNumMaxParallelDims,
      linalg::DistributionMethod::CyclicNumProcsEqNumIters));
  funcPassManager.addPass(createCSEPass());

  if (convertToDpsOptions) {
    funcPassManager.addPass(
        createConvertToDestinationPassingStylePass(*convertToDpsOptions));
  }
  // TODO(#16421): Disable decomposition due to failure in bufferization.
  // funcPassManager.addPass(
  //     IREE::LinalgExt::createTileAndDecomposeAttentionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

static void addGPUVectorizationPasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeIm2colPass());
  funcPassManager.addPass(
      IREE::VectorExt::createVectorizeIREEVectorExtOpsPass());
  // Vectorize.
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeGatherAccesses = true;
  options.enableCleanup = false;
  options.foldCastIntoContract = true;
  funcPassManager.addPass(createGenericVectorizationPass(options));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  // Run subset hoisting to convert iter_args to vectors.
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Triton Pipeline
//===---------------------------------------------------------------------===//

static void addVectorBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUTritonPassPipeline(OpPassManager &funcPassManager,
                              const GPUPipelineOptions &options) {
  tileAndDistributeToWorkgroup(funcPassManager);

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(createReorderWorkgroups(
      reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));
  funcPassManager.addPass(
      IREE::LinalgExt::createConvertAttentionToOnlineAttentionPass());

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Tile to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Set anchors at tensor level for vector distribution later and hoist out
  // loop invariant anchors.
  funcPassManager.addPass(createLLVMGPUConfigureTensorLayoutsPass());
  funcPassManager.addPass(createLoopInvariantCodeMotionPass());

  // Generalize all named ops so that we can fold away unit extent dims. By this
  // point, all tiling is finished so the tiling configurations on those ops can
  // be safely dropped. This additionally allows vectorization of convolution to
  // `vector.contract` as filter dimensions are expected to be tiled to 1 by
  // this point.
  funcPassManager.addPass(createLinalgGeneralizeNamedOpsPass());

  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Linalg -> Vector
  addGPUVectorizationPasses(funcPassManager);

  // Allocate tensors for copies to shared memory.
  funcPassManager.addPass(createGPUVectorAllocPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createGPUCombineValueBarriersPass());

  // Tensor -> Memref
  addVectorBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());

  // Preprocessing for vector distribution.
  funcPassManager.addPass(createLLVMGPUCastTypeToFitMMAPass());

  // Vector SIMD -> Vector SIMT
  funcPassManager.addPass(createLLVMGPUConfigureVectorLayoutsPass());
  funcPassManager.addPass(createLLVMGPUVectorDistributePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  if (options.prefetchSharedMemory) {
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

// Add passes to make the address computation more explicit and optimize them.
//
// The idea here is to be less dependent on what the LLVM backend is able to do,
// by heavy lifting most of the work while we still have the information about
// loops.
//
// Note that this needs to run before SCF -> CF.
static void addLowerAndOptimizeAddressComputationPasses(
    FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createExtractAddressComputationGPUPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      .addPass(createLoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      .addPass(createLoopInvariantCodeMotionPass)
      .addPass(createLowerAffinePass);
}

static void addLowerToTritonPasses(OpPassManager &modulePassManager) {
  modulePassManager.addPass(
      createConvertHALDescriptorTypeToGPUAddressSpacePass());
  modulePassManager.addPass(createCanonicalizerPass());
  modulePassManager.addPass(createCSEPass());

  modulePassManager.addPass(createLowerUKernelOpsToCallsPass());

  FunctionLikeNest(modulePassManager)
      // LinalgExt -> SCF
      .addPass(IREE::LinalgExt::createLinalgExtToLoopsPass)

      // Linalg -> SCF
      .addPass(createMemrefCopyToLinalgPass)
      .addPass(createConvertLinalgToLoopsPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)

      // Pad allocations with dynamic dimension after linalg lowering but before
      // lowering SCF and affine ops.
      .addPass(createPadDynamicAllocPass)

      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Handled tensor constants.
  addConstantBufferizePasses(modulePassManager);

  FunctionLikeNest funcPassManager(modulePassManager);
  funcPassManager.addPass(createFoldTensorExtractOpPass)
      .addPass(createLLVMGPUVectorLoweringPass)
      .addPass(createExpandGPUOpsPass);

  // This pass needs to run before SCF -> CF.
  addLowerAndOptimizeAddressComputationPasses(funcPassManager);

  // Run checks on shared memory usage.
  funcPassManager
      .addPass([&]() {
        auto getIndexBitwidth = [](mlir::FunctionOpInterface) { return 64; };
        return createGPUCheckResourceUsagePass(getIndexBitwidth);
      })
      // SCF -> CF
      .addPass(createConvertSCFToCFPass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass)
      // Handle complex operation conversion.
      .addPass(createConvertComplexToStandardPass)
      // Convert BF16 operations to occur as F32.
      .addPass(createConvertBf16ArithToF32Pass)
      .addPass(createConvertBf16ToUInt16BuffersPass)
      // Convert math dialect elementry functions to polynomial form.
      .addPass(createPolynomialApproximationPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      .addPass(createEmulateNarrowTypePass)
      .addPass(affine::createAffineExpandIndexOpsPass)
      .addPass(createLowerAffinePass)
      .addPass(createCanonicalizerPass)
      .addPass(createCSEPass);

  // Strip out the debug info for the kernel.
  modulePassManager.addPass(createStripDebugInfoPass());
  // Cast address spaces of all function arguments to generic.
  modulePassManager.addPass(createLLVMGPUCastAddressSpaceFunctionPass());
}

//===----------------------------------------------------------------------===//
// Common Pass Pipelines
//===----------------------------------------------------------------------===//

static LogicalResult igemmConfigFn(linalg::GenericOp genericOp,
                                   IREE::LinalgExt::Im2colOp im2colOp) {
  auto funcOp = genericOp->getParentOfType<FunctionOpInterface>();
  if (!funcOp) {
    return genericOp.emitError("cannot find parent funcOp");
  }
  IREE::GPU::TargetAttr target = getGPUTargetAttr(funcOp);
  if (!target) {
    return funcOp.emitError("missing GPU target in parent funcOp");
  }
  if (failed(IREE::GPU::setMatmulLoweringConfig(target, funcOp, genericOp))) {
    return IREE::GPU::setTileAndFuseLoweringConfig(target, funcOp, genericOp);
  }
  return success();
}

static void buildTritonCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPredicatedPass(clLLVMGPUUseIgemm, []() {
      return createConvolutionToIGEMMPass(igemmConfigFn);
    });
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
    addEncodingToNopPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());
  modulePassManager.addPass(createLLVMGPUSelectLoweringStrategyPass());
}

void buildTritonCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  buildTritonCodegenConfigurationPassPipelineImpl(
      variantPassManager.nest<ModuleOp>());
}

void buildTritonCodegenPassPipeline(OpPassManager &variantPassManager) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    FunctionLikeNest(modulePassManager)
        .addPass(createTritonLowerExecutableTargetPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToTritonPasses(variantPassManager.nest<ModuleOp>());
}

//===---------------------------------------------------------------------===//
// Triton Pass Registration
//===---------------------------------------------------------------------===//

namespace triton {
#define GEN_PASS_REGISTRATION
#include "experimental/tritonizer/TritonExt/Transforms/Passes.h.inc"
}  // namespace triton

void registerCodegenTritonPasses() {
  triton::registerPasses();

  static PassPipelineRegistration<> TritonConfigPipeline(
      "iree-codegen-triton-configuration-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for GPU "
      "on all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildTritonCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgTritonPipeline(
      "iree-codegen-linalg-to-triton-pipeline",
      "Runs the progressive lowering pipeline from Linalg to TritonIR",
      [](OpPassManager &passManager) {
        buildTritonCodegenPassPipeline(passManager);
      });
}

}  // namespace mlir::iree_compiler
