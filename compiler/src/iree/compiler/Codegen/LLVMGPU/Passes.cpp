// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/LinalgExt/Transforms/Passes.h"

#include <cstdint>

#include "iree-dialects/Dialect/LinalgTransform/Passes.h"
#include "iree/compiler/Codegen/Common/GPU/Passes.h"
#include "iree/compiler/Codegen/Common/Passes.h"
#include "iree/compiler/Codegen/Dialect/GPU/TargetUtils/ConfigUtils.h"
#include "iree/compiler/Codegen/Dialect/GPU/Transforms/Passes.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/Transforms/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h"
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

#define DEBUG_TYPE "iree-llvm-gpu-lowering-pass-pipeline"

namespace mlir::iree_compiler {

constexpr int64_t kDefaultSubgroupSize = 32;

static llvm::cl::opt<ReorderWorkgroupsStrategy> clReorderWorkgroupsStrategy(
    "iree-codegen-reorder-workgroups-strategy",
    llvm::cl::desc("Reorder workgroup IDs using the selected strategy"),
    llvm::cl::values(clEnumValN(ReorderWorkgroupsStrategy::None, "none",
                                "No workgroup reordering"),
                     clEnumValN(ReorderWorkgroupsStrategy::Swizzle, "swizzle",
                                "Swizzle"),
                     clEnumValN(ReorderWorkgroupsStrategy::Transpose,
                                "transpose", "Transpose")),
    llvm::cl::init(ReorderWorkgroupsStrategy::None));

static llvm::cl::opt<unsigned> clReorderWorkgroupsLogSwizzleTile(
    "iree-codegen-reorder-workgroups-log-swizzle-tile",
    llvm::cl::desc("Reorder workgroups: log tile size to use"),
    llvm::cl::init(3));

static llvm::cl::opt<int64_t> clLLVMGPUSharedMemoryLimit(
    "iree-llvmgpu-shared-memory-limit",
    llvm::cl::desc("specify the maximum amount of shared memory allowed to be "
                   "allocated for the given target"),
    llvm::cl::init(163 * 1024));

static llvm::cl::opt<bool>
    clLLVMGPUUseIgemm("iree-codegen-llvmgpu-use-igemm",
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
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
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
  if (target.getBackend() != "rocm")
    return success();

  // Workgroup reordering on ROCm currently requires all workgrup counts to be
  // static.
  SmallVector<int64_t> workgroupCounts = getStaticNumWorkgroups(funcOp);
  if (llvm::any_of(workgroupCounts, ShapedType::isDynamic))
    return failure();

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

static void addBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

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

static void tileAndBufferize(OpPassManager &funcPassManager) {
  ConvertToDestinationPassingStylePassOptions options;
  options.useWARForCooperativeMatrixCodegen = true;
  tileAndDistributeToWorkgroup(funcPassManager, options);
  addBufferizePasses(funcPassManager);
}

static void addGPUVectorizationPasses(OpPassManager &funcPassManager,
                                      bool vectorizeCopies = true) {
  funcPassManager.addPass(createDecomposeConvolutionToLowerDimOpsPass());
  funcPassManager.addPass(IREE::LinalgExt::createDecomposeIm2colPass());
  funcPassManager.addPass(createLLVMGPUVectorizeSpecialOpsPass());
  funcPassManager.addPass(
      IREE::VectorExt::createVectorizeIREEVectorExtOpsPass());
  // Vectorize.
  GenericVectorizationPassOptions options;
  options.vectorizePadding = true;
  options.vectorizeCopies = vectorizeCopies;
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
// Default Vectorization
//===---------------------------------------------------------------------===//

void addGPUVectorizationPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createGPUTensorTilePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createGPUDistributePass());

  // Post bufferization optimizations.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// Tile and Fuse
//===---------------------------------------------------------------------===//

static FailureOr<Value> gpuRequireMemSpaceAllocationFn(OpBuilder &builder,
                                                       Location loc,
                                                       MemRefType memRefType,
                                                       ValueRange dynamicSizes,
                                                       unsigned alignment) {
  Attribute memorySpace = memRefType.getMemorySpace();
  // Bail out if the memref type specifies a nonnull memory space that is not
  // #gpu.address_space.
  if (memorySpace && !llvm::isa<gpu::AddressSpaceAttr>(memorySpace)) {
    return failure();
  }

  MemRefType allocType = memRefType;
  auto privateSpace = gpu::AddressSpaceAttr::get(
      builder.getContext(), gpu::GPUDialect::getPrivateAddressSpace());
  if (!memorySpace) {
    allocType =
        MemRefType::get(memRefType.getShape(), memRefType.getElementType(),
                        AffineMap(), privateSpace);
    memorySpace = privateSpace;
  }

  if (memorySpace == privateSpace) {
    return builder.create<memref::AllocaOp>(loc, allocType, dynamicSizes)
        .getResult();
  }
  return builder.create<memref::AllocOp>(loc, allocType, dynamicSizes)
      .getResult();
}

static void addGPUBufferizePasses(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createEliminateEmptyTensorsPass());
  funcPassManager.addPass(bufferization::createEmptyTensorToAllocTensorPass());
  funcPassManager.addPass(createGPUInferMemorySpacePass());
  BufferizationOptions::AllocationFn allocationFn =
      gpuRequireMemSpaceAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = [](OpBuilder &builder, Location loc,
                                               Value from, Value to) {
    builder.create<memref::CopyOp>(loc, from, to);
    return success();
  };
  funcPassManager.addPass(
      createIREEComprehensiveBufferizePass(allocationFn, memcpyFn));
  addIREEPostBufferizationPasses(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUTileAndFusePassPipeline(OpPassManager &funcPassManager,
                                   const GPUPipelineOptions &pipelineOptions) {
  tileAndDistributeToWorkgroup(funcPassManager,
                               /*convertToDpsOptions=*/std::nullopt);

  // Step 1. Promote matmul operands and pack to intrinsic shapes.
  funcPassManager.addPass(createGPUPromoteMatmulOperandsPass());
  funcPassManager.addPass(IREE::GPU::createPackToIntrinsicsPass());

  // Step 1.5. Expand result shapes of MultiMmaOps before reduction tiling.
  {
    IREE::GPU::ConcretizeMmaShapesPassOptions options;
    options.concretizeInputs = false;
    options.concretizeResult = true;
    funcPassManager.addPass(IREE::GPU::createConcretizeMmaShapesPass());
  }

  // Step 2. Tile and fuse tileable ops to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  // Step 3. Decompose pack and unpack ops and propagate the resulting reshapes.
  funcPassManager.addPass(
      createDecomposePackUnPackOpsPass(/*tileOuterToOne=*/false,
                                       /*useOnlyReshapes=*/true,
                                       /*controlFn=*/std::nullopt));

  // Step 3.5. Expand the inner dimensions of MultiMma ops in preparation for
  // distribution to lanes.
  {
    IREE::GPU::ConcretizeMmaShapesPassOptions options;
    options.concretizeInputs = true;
    options.concretizeResult = false;
    funcPassManager.addPass(IREE::GPU::createConcretizeMmaShapesPass());
  }

  funcPassManager.addPass(createPropagateReshapesByExpansionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      /*useWARForCooperativeMatrixCodegen=*/false));

  // Step 4. Tile and fuse tileable ops to subgroups/threads.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Thread;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Subgroup;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
  }
  funcPassManager.addPass(IREE::GPU::createDistributeMmaToLanesPass());

  // Normalize loop bounds for later lowerings.
  funcPassManager.addPass(iree_compiler::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // TODO: This LICM instance is load bearing due to brittleness of the
  // hoisting and fusion pass, as well as a lack of a fallback distribution
  // pass.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  {
    OptimizeTensorInsertExtractSlicesPassOptions options;
    options.foldIdentitySlices = true;
    funcPassManager.addPass(
        createOptimizeTensorInsertExtractSlicesPass(options));
  }

  // Step 5. Greedily fuse parallel loops and hoist from serial loops.
  funcPassManager.addPass(IREE::GPU::createFuseAndHoistParallelLoopsPass());
  funcPassManager.addPass(createGPUGreedilyDistributeToThreadsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(IREE::GPU::createCombineBarrierRegionsPass());

  // Step 6. Lower special ops and vectorize.
  funcPassManager.addPass(IREE::GPU::createVectorizeIREEGPUOpsPass());
  addGPUVectorizationPasses(funcPassManager, /*vectorizeCopies=*/false);
  funcPassManager.addPass(createCleanupBufferAllocViewPass());
  funcPassManager.addPass(createGPUCombineValueBarriersPass());

  // Step 7. Bufferize.
  addGPUBufferizePasses(funcPassManager);

  // Step 8. Resolve remaining parallel loops.
  funcPassManager.addPass(iree_compiler::createNormalizeLoopBoundsPass(
      NormalizeLoopBoundsPassOptions{/*normalizeFor=*/false,
                                     /*normalizeForall=*/true}));
  funcPassManager.addPass(createGPUVerifyDistributionPass());
  funcPassManager.addPass(createGPUDistributeForallPass());

  // Vectorize copies that came out of bufferization.
  funcPassManager.addPass(createVectorizeMemrefCopyPass());

  // Step 8. Unroll operations to native intrinsic widths.
  funcPassManager.addPass(IREE::GPU::createUnrollToIntrinsicsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Step 9. Remaining post-bufferization optimizations/lowerings.
  funcPassManager.addPass(IREE::GPU::createLowerIREEGPUOpsPass());
  funcPassManager.addPass(createUnrollAnnotatedLoopsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  if (pipelineOptions.enableReduceSharedMemoryBankConflicts) {
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 64;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }
  if (pipelineOptions.prefetchSharedMemory) {
    funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
    funcPassManager.addPass(createLLVMGPUPrefetchSharedMemoryPass());
  }

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  {
    OptimizeVectorTransferPassOptions options;
    // Disable redundant vector transfer hoisting because it does not
    // properly consider distributed code on memrefs.
    options.redundantHoisting = false;
    funcPassManager.addPass(createOptimizeVectorTransferPass());
  }
  funcPassManager.addPass(createHoistStaticallyBoundAllocationsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Winograd Vectorize
//===---------------------------------------------------------------------===//

void addGPUWinogradVectorizePassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(createGPUTilePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      IREE::LinalgExt::createDecomposeWinogradTransformPass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);
  GPUDistributeScfForPassOptions options;
  options.useBlockDims = false;
  funcPassManager.addPass(createGPUDistributeScfForPass(options));

  // Post bufferization optimizations.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
}

//===---------------------------------------------------------------------===//
// MatmulSIMT
//===---------------------------------------------------------------------===//

void addGPUMatmulSimtPassPipeline(OpPassManager &funcPassManager,
                                  const GPUPipelineOptions &options) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTensorTileToSerialLoopsPass());
  funcPassManager.addPass(createGPUTensorAlloc());
  funcPassManager.addPass(createGPUTensorTilePass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);

  // tensor to memref
  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistributePass());

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    funcPassManager.addPass(createGPUReduceBankConflictsPass());
  }

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(createReorderWorkgroups(
      reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Even though we vectorize before bufferization we are not able to hoist
  // accumulator load/store out of the K loop until distribution. This is
  // because we materialize the fill and the matmul in two different scf.forall
  // regions, when they should be in the same scf.forall. Newer pipelines
  // like TileAndFuse don't have this problem, because they coalesce these
  // scf.forall regions into a single scf.forall.
  //
  // Therefore we still rely on buffer level transformations for transfer ops
  // hoisting and store to load forwarding. This relies on shacky alias
  // analysis and we need to move this to tensor level once we have better
  // abstractions.
  funcPassManager.addPass(createOptimizeVectorTransferPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  funcPassManager.addPass(createGPUPipeliningPass());
}

//===---------------------------------------------------------------------===//
// Matmul Tensor Core
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCorePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth}));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(createReorderWorkgroups(
      reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::WMMA));
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  if (options.enableReduceSharedMemoryBankConflicts) {
    funcPassManager.addPass(createGPUReduceBankConflictsPass());
  }

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createLLVMGPUVectorToGPUPass(GPUTensorCoreType::WMMA));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = false;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex =
      llvm::to_underlying(PipeliningSchedulingStrategy::loadGlobalStage0);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
}

//===---------------------------------------------------------------------===//
// Matmul MMA.Sync
//===---------------------------------------------------------------------===//

void addGPUMatmulTensorCoreMmaSyncPassPipeline(
    OpPassManager &funcPassManager, const GPUPipelineOptions &options,
    unsigned pipelineDepth) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto warps within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/true));
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
  if (pipelineDepth > 1) {
    funcPassManager.addPass(createGPUMultiBufferingPass(
        GPUMultiBufferingPassOptions{pipelineDepth}));
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());

  ReorderWorkgroupsStrategy reorderStrategy =
      getReorderWorkgroupsStrategy(options.reorderStrategy);
  funcPassManager.addPass(createReorderWorkgroups(
      reorderStrategy, clReorderWorkgroupsLogSwizzleTile,
      canReorderWorkgroups));

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  funcPassManager.addPass(
      createLLVMGPUTensorCoreVectorizationPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // Distribute shared memory copies.
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Vector -> MMA ops
  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(
      createLLVMGPUVectorToGPUPass(GPUTensorCoreType::MMA_SYNC));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Hoist loop invariant code to avoid pipelining it.
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  // Pipeline memory operations.
  GPUPipeliningPassOptions pipelieningOptions = {};
  pipelieningOptions.epiloguePeeling = false;
  pipelieningOptions.depth = pipelineDepth;
  pipelieningOptions.scheduleIndex =
      llvm::to_underlying(PipeliningSchedulingStrategy::nvidiaTensorCore);
  funcPassManager.addPass(createGPUPipeliningPass(pipelieningOptions));
  // Optimize shared memory usage.
  funcPassManager.addPass(createLLVMGPUPackSharedMemoryAllocPass());
}

//===---------------------------------------------------------------------===//
// Transpose
//===---------------------------------------------------------------------===//

void addGPUTransposePassPipeline(OpPassManager &funcPassManager,
                                 const GPUPipelineOptions &options) {
  tileAndDistributeToWorkgroup(funcPassManager);

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(
      createGPUTensorAlloc(GPUPromoteSharedMemPattern::TransposeOpPattern));
  funcPassManager.addPass(createGPUTensorTilePass());

  // Linalg -> vector
  addGPUVectorizationPasses(funcPassManager);
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());

  // tensor to memref
  addBufferizePasses(funcPassManager);

  // distribute foreach threads
  funcPassManager.addPass(createGPUDistributePass());

  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createGPUDistributeSharedMemoryCopyPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  if (options.enableReduceSharedMemoryBankConflicts) {
    // May or may not need to reduce shared mememory conflicts.
    GPUReduceBankConflictsPassOptions options = {};
    options.paddingBits = 32;
    funcPassManager.addPass(createGPUReduceBankConflictsPass(options));
  }

  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

//===---------------------------------------------------------------------===//
// Vector Distribution
//===---------------------------------------------------------------------===//

// Matmul pipeline using vector distribution patterns to map to various tensor
// core operations. The current implementation below is unstable and is missing
// a few crucial pieces for performance (primarily software pipelining). The
// current flow is as follows.
//
// 1. Tile + fuse and distribute to workgroups.
// 2. Problem specific tiling, namely tiling the K dimension of the GEMM.
// 3. Vectorize
// 4. Materialize shared memory allocations as vectorized copies.
// 5. Bufferize
//
// * Distribution to warps should happen here, but right now this pipeline
//   is single subgroup. Pending improvements to vector distribution to allow
//   distribution to warps.
//
// 6. Distribute to virtual lanes (i.e. threads in this case).
//
// Note that a few pieces here are subject to change in the immediate future.
// First, the shared memory promotion done here is in a sense a stopgap, as it
// won't compose well with what's available for bufferization/pipelining today.
// Second, distribution to more than one warp depends on either layout changes,
// or explicit distribution using `scf.forall`. For now this keeps it simple
// and gives us a starting point for generating code for matmuls in the first
// place.

// We use vector ops to do the copy for this pipeline because distribution is
// vector based.
static LogicalResult gpuVectorCopyFn(OpBuilder &builder, Location loc,
                                     Value from, Value to) {
  bool needsBarrier = false;
  MemRefType fromType = llvm::cast<MemRefType>(from.getType());
  if (hasSharedMemoryAddressSpace(fromType)) {
    needsBarrier = true;
  }
  if (hasSharedMemoryAddressSpace(llvm::cast<MemRefType>(to.getType()))) {
    needsBarrier = true;
  }
  if (needsBarrier)
    builder.create<gpu::BarrierOp>(loc);
  VectorType vectorType =
      VectorType::get(fromType.getShape(), fromType.getElementType());
  Value c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
  SmallVector<Value> indices(vectorType.getRank(), c0);
  SmallVector<bool> inBounds(vectorType.getRank(), true);
  Value read = builder.create<vector::TransferReadOp>(loc, vectorType, from,
                                                      indices, inBounds);
  builder.create<vector::TransferWriteOp>(loc, read, to, indices, inBounds);
  if (needsBarrier) {
    builder.create<gpu::BarrierOp>(loc);
  }
  return success();
}

static void addVectorBufferizePasses(OpPassManager &funcPassManager) {
  BufferizationOptions::AllocationFn allocationFn = gpuAllocationFn;
  BufferizationOptions::MemCpyFn memcpyFn = gpuCopyFn;
  addIREEComprehensiveBufferizePasses(funcPassManager, allocationFn, memcpyFn);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUVectorDistributePassPipeline(OpPassManager &funcPassManager,
                                        const GPUPipelineOptions &options,
                                        bool usePadToModelSharedMemcpy) {
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

  if (usePadToModelSharedMemcpy) {
    LLVMGPUMatmulPadOption option = LLVMGPUMatmulPadOption::ParallelDims;
    funcPassManager.addPass(createLLVMGPUPromoteMatmulToFitMMAPass(option));
  }

  // Tile to reduction loops.
  {
    GPUApplyTilingLevelPassOptions options;
    options.tilingLevel = IREE::GPU::TilingLevel::Reduction;
    funcPassManager.addPass(createGPUApplyTilingLevelPass(options));
    funcPassManager.addPass(affine::createLoopCoalescingPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

  if (usePadToModelSharedMemcpy) {
    LLVMGPUMatmulPadOption option = LLVMGPUMatmulPadOption::ReductionDims;
    funcPassManager.addPass(createLLVMGPUPromoteMatmulToFitMMAPass(option));
  }

  funcPassManager.addPass(IREE::LinalgExt::createDecomposeAttentionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Set anchors at tensor level for vector distribution later and hoist out
  // loop invariant anchors.
  funcPassManager.addPass(createLLVMGPUConfigureTensorLayoutsPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());

  // Generalize all named ops so that we can fold away unit extent dims. By this
  // point, all tiling is finished so the tiling configurations on those ops can
  // be safely dropped. This additionally allows vectorization of convolution to
  // `vector.contract` as filter dimensions are expected to be tiled to 1 by
  // this point.
  funcPassManager.addPass(createLinalgGeneralizeNamedOpsPass());
  if (!usePadToModelSharedMemcpy) {
    LinalgFoldUnitExtentDimsPassOptions options;
    options.useRankReducingSlices = true;
    funcPassManager.addPass(
        IREE::VectorExt::createVectorExtFoldUnitExtentDimsPass());
    funcPassManager.addPass(mlir::createLinalgFoldUnitExtentDimsPass(options));
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }

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

void addGPUWarpReductionPassPipeline(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);
  funcPassManager.addPass(createRematerializeParallelOpsPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createGPUTileReductionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  // Linalg -> vector
  {
    GenericVectorizationPassOptions options;
    options.enableVectorMasking = true;
    options.useConfiguredVectorSizes = false;
    options.vectorizePadding = true;
    options.vectorizeGatherAccesses = true;
    options.enableCleanup = false;
    options.generateContract = false;
    funcPassManager.addPass(createGenericVectorizationPass(options));
    funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
    funcPassManager.addPass(createCanonicalizerPass());
    funcPassManager.addPass(createCSEPass());
  }
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);

  funcPassManager.addPass(memref::createFoldMemRefAliasOpsPass());
  funcPassManager.addPass(createOptimizeVectorTransferPass());
  funcPassManager.addPass(createOptimizeTensorInsertExtractSlicesPass());
  funcPassManager.addPass(createIREELoopInvariantCodeMotionPass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  funcPassManager.addPass(createForOpCanonicalizationPass());
  funcPassManager.addPass(createCanonicalizerPass());

  // vector -> simt gpu + vector
  funcPassManager.addPass(createConvertVectorReductionToGPUPass(
      /*expandSubgroupReduction=*/true));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
}

void addGPUPackUnPackPasses(OpPassManager &funcPassManager) {
  tileAndDistributeToWorkgroup(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createGPUTensorTilePass());
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(
      createDecomposePackUnPackOpsPass(/*tileOuterToOne=*/true,
                                       /*useOnlyReshapes=*/false,
                                       /*controlFn=*/std::nullopt));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());
  addGPUVectorizationPasses(funcPassManager);

  addBufferizePasses(funcPassManager);

  funcPassManager.addPass(createGPUDistributePass());
}

void addGPUSimpleDistributePassPipeline(OpPassManager &funcPassManager) {
  tileAndBufferize(funcPassManager);

  // Distribute linalg onto threads within the workgroup.
  funcPassManager.addPass(
      createLLVMGPUTileAndDistributePass(/*distributeToWarp=*/false));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
}

void addGPUDefaultPassPipeline(OpPassManager &funcPassManager,
                               const GPUPipelineOptions &options) {
  ConvertToDestinationPassingStylePassOptions dpsOptions;
  dpsOptions.useWARForCooperativeMatrixCodegen = true;
  tileAndDistributeToWorkgroup(funcPassManager, dpsOptions);
  if (options.enableUkernels) {
    funcPassManager.addPass(createGPULowerToUKernelsPass());
  }
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
}

void addGPUBaseLoweringPassPipeline(OpPassManager &funcPassManager) {
  funcPassManager.addPass(createConvertToDestinationPassingStylePass(
      /*useWARForCooperativeMatrixCodegen=*/false));
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  addBufferizePasses(funcPassManager);
  funcPassManager.addPass(createCanonicalizerPass());
  funcPassManager.addPass(createCSEPass());

  funcPassManager.addPass(IREE::LinalgExt::createLinalgExtToLoopsPass());
  funcPassManager.addPass(createMemrefCopyToLinalgPass());
  funcPassManager.addPass(createConvertLinalgToLoopsPass());
  funcPassManager.addPass(createRemoveSingleIterationLoopPass());
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
static void
addLowerAndOptimizeAddressComputationPasses(FunctionLikeNest &funcPassManager) {
  funcPassManager.addPass(createExtractAddressComputationGPUPass)
      .addPass(memref::createExpandOpsPass)
      .addPass(memref::createFoldMemRefAliasOpsPass)
      .addPass(memref::createExpandStridedMetadataPass)
      // Hoist loop invariant variables to give affine decomposition pass the
      // right loop dependencies.
      .addPass(createIREELoopInvariantCodeMotionPass)
      // Decompose affine ops.
      .addPass(createDecomposeAffineOpsPass)
      // Get rid of the redundant computations.
      .addPass(createCSEPass)
      // Hoist the resulting decompositions.
      .addPass(createIREELoopInvariantCodeMotionPass)
      .addPass(createLowerAffinePass);
}

static void addLowerToLLVMGPUPasses(OpPassManager &modulePassManager,
                                    bool forROCDL) {
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

  if (forROCDL) {
    // convert to ROCDL.
    modulePassManager.addPass(createConvertToROCDLPass());
    modulePassManager.addNestedPass<LLVM::LLVMFuncOp>(
        createROCDLAnnotateKernelForTranslationPass());
  } else {
    // convert to NVVM.
    modulePassManager.addPass(createConvertToNVVMPass());
  }
}

void addGPUTransformDialectPasses(OpPassManager &funcPassManager,
                                  StringRef entryPoint) {
  funcPassManager.addPass(
      mlir::iree_compiler::createTransformDialectInterpreterPass(entryPoint));

  // Dropping the schedule is needed:
  //   1. if we want to embed the transform in the module: we should drop the
  //      schedule once applied.
  //   2. if transform.do_not_dce_operands ops are introduced.
  funcPassManager.addPass(createDropSchedulePass());
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

static void buildLLVMGPUCodegenConfigurationPassPipelineImpl(
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

void buildLLVMGPUCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  buildLLVMGPUCodegenConfigurationPassPipelineImpl(
      variantPassManager.nest<ModuleOp>());
}

void buildLLVMGPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool useROCM) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createLLVMGPULowerExecutableTargetPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass());

  //===--------------------------------------------------------------------===//
  // Convert Linalg ops to LLVM+NVVM/ROCDL ops.
  //
  // Post-conditions:
  //   - All Linalg/Loops/GPU/Affine/Standard ops are converted away.
  //   - The module contains the final llvm.module ready to be serialized.
  //===--------------------------------------------------------------------===//
  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(), useROCM);

  LLVM_DEBUG({
    llvm::dbgs() << "Using LLVMGPU pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===----------------------------------------------------------------------===//
// ROCDL Pass Pipelines
//===----------------------------------------------------------------------===//

static void buildROCDLCodegenConfigurationPassPipelineImpl(
    OpPassManager &modulePassManager) {
  {
    FunctionLikeNest funcPassManager(modulePassManager);
    funcPassManager.addPredicatedPass(clLLVMGPUUseIgemm, []() {
      return createConvolutionToIGEMMPass(igemmConfigFn);
    });
    funcPassManager.addPass(createGPUGeneralizeNamedOpsPass);
    addCommonTargetExecutablePreprocessingPasses(funcPassManager);
  }
  modulePassManager.addPass(createMaterializeUserConfigsPass());

  modulePassManager.addPass(createROCDLSelectLoweringStrategyPass());
}

void buildROCDLCodegenConfigurationPassPipeline(
    OpPassManager &variantPassManager) {
  OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
  buildROCDLCodegenConfigurationPassPipelineImpl(modulePassManager);
}

void buildROCDLCodegenPassPipeline(OpPassManager &variantPassManager) {
  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createROCDLLowerExecutableTargetPass);
  }
  variantPassManager.addPass(createReconcileTranslationInfoPass());
  variantPassManager.addPass(IREE::Util::createDropCompilerHintsPass());

  addLowerToLLVMGPUPasses(variantPassManager.nest<ModuleOp>(),
                          /*forROCDL=*/true);

  LLVM_DEBUG({
    llvm::dbgs() << "Using ROCDL pass pipeline:\n";
    variantPassManager.printAsTextualPipeline(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

//===---------------------------------------------------------------------===//
// Common Pass Registration
//===---------------------------------------------------------------------===//

namespace common {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"
} // namespace common

void registerCodegenLLVMGPUPasses() {
  // Generated.
  common::registerPasses();

  static PassPipelineRegistration<> LLVMGPUConfigPipeline(
      "iree-codegen-llvmgpu-configuration-pipeline",
      "Runs the translation strategy configuration pipeline on Linalg for GPU "
      "on all functions in a module",
      [](OpPassManager &modulePassManager) {
        buildLLVMGPUCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgNVVMPipeline(
      "iree-codegen-linalg-to-nvvm-pipeline",
      "Runs the progressive lowering pipeline from Linalg to NVVM",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenPassPipeline(passManager, false);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline",
      "Runs the progressive lowering pipeline from Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildLLVMGPUCodegenPassPipeline(passManager, true);
      });
}

//===---------------------------------------------------------------------===//
// ROCDL Pass Registration
//===---------------------------------------------------------------------===//

namespace rocdl {
#define GEN_PASS_REGISTRATION
#include "iree/compiler/Codegen/LLVMGPU/ROCDLPasses.h.inc"
} // namespace rocdl

void registerCodegenROCDLPasses() {
  // Generated.
  rocdl::registerPasses();

  static PassPipelineRegistration<> ROCDLConfigPipeline(
      "iree-codegen-rocdl-configuration-pipeline",
      "Runs pass pipeline to select a suitable lowering strategy for ROCDL",
      [](OpPassManager &modulePassManager) {
        buildROCDLCodegenConfigurationPassPipelineImpl(modulePassManager);
      });

  static PassPipelineRegistration<> LinalgROCDLPipeline(
      "iree-codegen-linalg-to-rocdl-pipeline2",
      "Runs pass pipeline to progressively lower Linalg to ROCDL",
      [](OpPassManager &passManager) {
        buildROCDLCodegenPassPipeline(passManager);
      });

  static PassPipelineRegistration<> LLVMGPUBufferizePipeline(
      "iree-codegen-llvmgpu-bufferization-pipeline",
      "Runs pass pipeline to bufferize for llvmgpu backends",
      [](OpPassManager &passManager) { addBufferizePasses(passManager); });
}

} // namespace mlir::iree_compiler
