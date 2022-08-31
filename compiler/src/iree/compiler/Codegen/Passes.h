// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_PASSES_H_
#define IREE_COMPILER_CODEGEN_PASSES_H_

#include <memory>

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Utils/CustomKernelsTargetInfo.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassOptions.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {

/// Registers all conversion passes in this directory.
void registerCodegenPasses();

/// Verify that the configuration used for compilation is valid.
LogicalResult verifyLoweringConfiguration(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});

//------------------------------------------------------------------------------
// Misc/common conversions
//------------------------------------------------------------------------------

using bufferization::BufferizationOptions;
void addIREEComprehensiveBufferizePasses(
    OpPassManager &passManager,
    Optional<BufferizationOptions::AllocationFn> allocationFn = None,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn = None,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn = None);

/// Pass to perform canonicalizations/cleanups related to HAL interface/buffer
/// allocations and view operations.
std::unique_ptr<OperationPass<func::FuncOp>> createCleanupBufferAllocViewPass();

/// Pass to bufferize dispatches that are copying from one interface to another.
/// This will create a `linalg.generic` op which is a copy that can then be
/// used by backends to handle appropriately.
std::unique_ptr<OperationPass<ModuleOp>>
createBufferizeCopyOnlyDispatchesPass();

// Decomposes linalg generics on tensors into generics containing no more than
// one op in the body.
std::unique_ptr<Pass> createDecomposeLinalgGenericPass();

/// Flattens n-D MemRef subspan ops to 1-D MemRef and folds the byte offsets on
/// subspan ops to the consumer load/store ops, in preparation for lowering to
/// backends that require linearized access.
std::unique_ptr<OperationPass<ModuleOp>> createFlattenMemRefSubspanPass();

/// Creates a pass to fold `affine.min` ops in tiled and distributed loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createFoldAffineMinInDistributedLoopsPass();

/// After running the upstream TensorConstantBufferize pass, remove tensor_loads
/// introduced for use only in tensor_extract. These can be folded to use a load
/// of the created memref object that holds the constant values.
std::unique_ptr<OperationPass<>> createFoldTensorExtractOpPass();

/// An ad-hoc pass to canonicalize selected loop carried dependencies on
/// scf.for.
std::unique_ptr<OperationPass<func::FuncOp>> createForOpCanonicalizationPass();

/// Pass to perform linalg on tensor bufferization. The function passed into the
/// pass through the `allocationFn` argument is invoked whenever a new buffer is
/// to be created. The callback will be passed the Values for the dynamic
/// dimensions in the memref type that is to be allocated.  The callback is
/// expected to return a MemRefType Value.  When no `allocationFn` is specified,
/// the default allocator generates an `std.alloc` instruction with the
/// allocated MemRefType having no stride map (i.e. default row-major striding)
/// and default memory space.
std::unique_ptr<OperationPass<ModuleOp>> createIREEComprehensiveBufferizePass(
    Optional<BufferizationOptions::AllocationFn> allocationFn = None,
    Optional<BufferizationOptions::DeallocationFn> deallocationFn = None,
    Optional<BufferizationOptions::MemCpyFn> memCpyFn = None);

/// Creates a pass to remove single iteration distributed loops.
std::unique_ptr<OperationPass<func::FuncOp>>
createRemoveSingleIterationLoopPass();

/// Converts entry point function within dispatch regions to use
/// destination-passing style, which is better suited for the upstream
/// comprehensive bufferization pass.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertToDestinationPassingStylePass();

/// Creates a pass to vectorize a very specific form of linalg.conv ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createLinalgToVectorVectorizeConvPass();

/// Creates a pass to vectorize a very specific form of tensor.pad ops with
/// control flows.
std::unique_ptr<OperationPass<func::FuncOp>> createVectorizePadPass();

/// Pass to optimize vector transfer_read and transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>> createOptimizeVectorTransferPass(
    bool flatten = false);

/// Pass to test Partitionable loop interface
std::unique_ptr<OperationPass<void>>
createTestPartitionableLoopsInterfacePass();

/// Pass to tile and distribute to workgroups.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createTileAndDistributeToWorkgroupsPass();

/// Pass to propagate type to avoid generating load/stores of illegal types.
std::unique_ptr<OperationPass<func::FuncOp>> createTypePropagationPass();

/// Pass to convert math operations to their polynomial approximation.
std::unique_ptr<OperationPass<>> createPolynomialApproximationPass();

/// Creates a pass to convert memref.copy to linalg op.
std::unique_ptr<OperationPass<func::FuncOp>> createMemrefCopyToLinalgPass();

/// Convert GPU shared memory copies to distributed
/// transfer_read/transfer_write.
std::unique_ptr<OperationPass<func::FuncOp>>
createGPUDistributeSharedMemoryCopy();

/// Apply software pipelining.
std::unique_ptr<OperationPass<func::FuncOp>> createGPUPipeliningPass(
    unsigned depth = 1);

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createWorkGroupSwizzle(
    unsigned swizzleLogTile = 0);

/// Pad dynamic alloc op to convert them into static one.
std::unique_ptr<OperationPass<func::FuncOp>> createPadDynamicAlloc();

/// Create an IREE-specific Transform dialect interpreter pass with all
/// registrations necessary for IREE.
std::unique_ptr<Pass> createTransformDialectInterpreterPass(
    llvm::StringRef transformFileName = llvm::StringRef());

//----------------------------------------------------------------------------//
// Common codegen patterns.
//----------------------------------------------------------------------------//

/// Populates `patterns` with patterns to fold `affine.min` ops in tiled and
/// distributed loops.
void populateFoldAffineMinInDistributedLoopsPatterns(
    RewritePatternSet &patterns);

/// Populates `patterns` with a very specific pattern that vectorizes a
/// linalg.conv op for a single thread. The linalg.conv should compute on
/// static-sized subviews. To match, output shape must be 1x1xWoxCo, where Co
/// Co is a multiple of 4, and filter shape must be 1x1x4xCo.
void populateLinalgToVectorVectorizeConvPatterns(MLIRContext *context,
                                                 RewritePatternSet &patterns);

/// Populates `patterns` with patterns that vectorize tensor.pad with static
/// result shape by generating control flows to guard against vector transfer
/// read ops to make sure they are in bounds.
///
/// Such conversions are needed for correctness when the tensor.pad op has
/// dynamic low padding values and also beneficial for eventually lowering to
/// hardware targets without native support for vector transfer read ops with
/// out of bound semantics.
void populateVectorizePadPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

//------------------------------------------------------------------------------
// LLVMCPU
//------------------------------------------------------------------------------

// Verifies that only supported IR constructs are passed to the compiler (like
// no Linalg transform markers are set).
std::unique_ptr<OperationPass<ModuleOp>>
createVerifyLinalgTransformLegalityPass();

/// Performs the final conversion to LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToLLVMPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUEmitVectorizationRemarksPass();

/// Checks CPU backend specific IR constraints (like no stack allocations)
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUCheckIRBeforeLLVMConversionPass();

/// Pass to lower the module an hal.executable.variant operation to external
/// dialect. Currently this pass lowers to LLVM dialect, but could be
/// generalized to lower to any "final" dialect like SPIR-V/NVVM, etc.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMCPULowerExecutableTargetPass();

/// Synchronizes LLVM linkage with MLIR symbol visibility.
std::unique_ptr<OperationPass<ModuleOp>>
createLLVMCPUSynchronizeSymbolVisibilityPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUAArch64VectorLoweringPass();

/// Replaces llvm.intr.fma with its unfused mul and add ops.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUUnfuseFMAOpsPass();

/// A pass that converts certain vector.contract ops to custom kernels.
std::unique_ptr<OperationPass<func::FuncOp>>
createVectorContractCustomKernelsPass();

/// Fuses tensor.pad ops into their consumer ops' tiled loop nests.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUFuseTensorPadWithConsumerPass();

std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMCPUPadTilePass();

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMCPUVectorizePadPass();

//------------------------------------------------------------------------------
// LLVMCPU Codegen specific patterns.
//------------------------------------------------------------------------------

/// Populates `patterns` to convert certain vector.contract ops to special
/// "kernels" written either in SIMD intrinsics or inline assembly.
void populateVectorContractCustomKernelsPatterns(
    const CustomKernelsTargetInfo &targetInfo, RewritePatternSet &patterns);

void populateUnfusedFMAOpsPassPatterns(MLIRContext *context,
                                       RewritePatternSet &patterns);

void populateLLVMCPUVectorizePadPatterns(RewritePatternSet &patterns,
                                  PatternBenefit baseBenefit = 1);

//----------------------------------------------------------------------------//
// LLVMCPU backend Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates the passes to lower to scalars operations for linalg based
/// code-generation. This pipeline does not vectorize, but instead just converts
/// to memrefs
void addCPUDefaultPassPipeline(OpPassManager &passManager);

/// Populates the passes to lower to tiled/distributed/bufferized ops, suitable
/// for library call dispatch and lowering to loops.
void addVMVXDefaultPassPipeline(OpPassManager &passManager);

/// Populates the passes to lower linalg ops on buffers. Currenly this pipeline
/// is only used for dispatches that just copy data from input interfaces to
/// output interface.
void addCPUBufferOpsTileAndVectorizePipeline(OpPassManager &passManager);

/// Populates the passes needed to multi level tile and lowering of linalg ops
/// on tensors to vectors operations.
LogicalResult verifyTensorToVectorsPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addTensorToVectorsPassPipeline(OpPassManager &passManager,
                                    bool lowerToVectors = true);

/// Populates the passes needed to do two-level tile + vectorize of linalg ops
/// using the Codegen drivers from sandbox.
LogicalResult verifyDoubleTilingExpertPassPipelineConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addMultiTilingExpertPassPipeline(OpPassManager &passManager,
                                      int64_t numLevels, bool enablePeeling,
                                      bool lowerToAVX2 = false);
void addDoubleTilingPadExpertPassPipeline(OpPassManager &passManager);

// Populates the passes needed to do tiling, decomposing, and vectorizing the
// convolution ops using the Codegen drivers from sandbox.
LogicalResult verifyConvTileAndDecomposeExpertConfig(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize = {});
void addConvTileAndDecomposeExpertPassPipeline(OpPassManager &passManager);

/// Populates the passes from Sandbox for testing transformations from sandbox.
/// Unlike other pipelines this pass mangaer is nested at the
/// `hal.executable.variant` op.
void addTransformDialectInterpreterPasses(OpPassManager &passManager);

/// Populates the passes needed to multi level tile, fuse and vectorize lowering
/// of linalg ops on tensors to vectors operations.
void addCPUAArchDoubleTilingExpertPassPipeline(OpPassManager &passManager);

//----------------------------------------------------------------------------//
// LLVMCPU Pass Pipelines for lowering to LLVM dialect.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower a XLA HLO op to LLVM dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMCPUCodegenPassPipeline(OpPassManager &passManager);

//------------------------------------------------------------------------------
// LLVMGPU
//------------------------------------------------------------------------------

/// Lowering calling vectorization patterns. Expects pass manager to be a
/// module-level pass manager.
void addGPUVectorizationPassPipeline(OpPassManager &pm);

/// Lowering calling vectorization patterns.
LogicalResult verifyGPUMatmulSimtPassPipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addGPUMatmulSimtPassPipeline(OpPassManager &pm);

/// Lowering using tensorcore operations.
LogicalResult verifyGPUMatmulTensorCorePipeline(
    Operation *op, IREE::Codegen::LoweringConfigAttr loweringConfig,
    IREE::Codegen::TranslationInfoAttr translationInfo,
    ArrayRef<int64_t> workgroupSize);
void addGPUMatmulTensorCorePassPipeline(OpPassManager &pm,
                                        unsigned pipelineDepth);

/// Lowering reductions to warp reductions.
void addGPUWarpReductionPassPipeline(OpPassManager &pm);

/// Experimental path for transform dialect.
void addGPUTransformDialectInterpreterPasses(OpPassManager &pm);

/// Simple lowering only distributute linalg ops on blocks and threads. This
/// will result in scalar operations. Expects pass manager to be a module-level
/// pass manager.
void addGPUSimpleDistributePassPipeline(OpPassManager &pm);

/// Populates passes needed to lower a XLA HLO op to NVVM/ROCDL dialect via the
/// structured ops path. The pass manager `pm` in here should operate on the
/// module within the IREE::HAL::ExecutableOp.
void buildLLVMGPUTransformPassPipeline(OpPassManager &pm, bool useROCM);

/// Performs the final conversion to NNVM+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToNVVMPass();

/// Performs the final conversion to ROCDL+LLVM dialect.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToROCDLPass();

/// Perform tiling and distribution to threads.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileAndDistribute(
    bool distributeToWarp = false);

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUTileTensor(
    bool distributeToWarp = false);

std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUDistribute();

/// Create pass calling the dynamic pipeline for LLVMGPU.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createLLVMGPULowerExecutableTargetPass();

/// Convert Linalg ops to Vector.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorizationPass(
    int64_t nativeVector = 4, bool generateContract = true);

/// Convert Linalg ops to Vector and prepare converstion to GPU MMA ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUTensorCoreVectorizationPass();

/// Lower vector ops before convertion to LLVM.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorLoweringPass();

/// Apply multi-buffering transformation.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUMultiBuffering(
    unsigned numBuffers = 5);

/// Apply transformation to reduce the number of bank conflicts when accessing
/// shared memory.
std::unique_ptr<OperationPass<func::FuncOp>>
createLLVMGPUReduceSharedMemoryBankConflicts();

/// Converts vector ops to gpu dialect.
std::unique_ptr<OperationPass<func::FuncOp>> createLLVMGPUVectorToGPU();

// Distribute vector ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createConvertVectorReductionToGPUPass();

//------------------------------------------------------------------------------
// SPIR-V Passes
//------------------------------------------------------------------------------

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V scalar code. Additionally performs
/// distribution to threads without vectorization.
void addSPIRVTileAndDistributePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V scalar and vector code. Additionally
/// performs distribution to threads with vectorization.
void addSPIRVTileAndVectorizePassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V cooperative matrix code. Additionally
/// performs distribution to threads with vectorization.
void addSPIRVTileAndVectorizeToCooperativeOpsPassPipeline(OpPassManager &pm);

/// Pass pipeline to lower IREE HAL executables with workgroup tiled and
/// distributed Linalg ops to SPIR-V scalar and vector code. Additionally
/// performs distribution to threads with vectorization and promotion to use
/// workgroup memory.
void addSPIRVTileAndVectorizeWithWorkgroupMemoryPassPipeline(OpPassManager &pm);

/// Pass to perform the final conversion to SPIR-V dialect.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass();

/// Creates a pass to fold processor ID uses where possible.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFoldProcessorIDUsesPass();

/// Main pass to lower executables to scalar + vector code on SPIR-V path.
/// Invokes one of the pass pipelines that translate the executable to
/// scalar + vector code.
std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createSPIRVLowerExecutableTargetPass();

/// Pass to tile and distribute Linalg ops with buffer semantics to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndDistributePass();

/// Pass to promote Linalg ops with buffer semantics to use workgroup memory and
/// then tile to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTileAndPromotePass();

/// Pass to tile Linalg ops with buffer semantics to subgroups and vectorize to
/// vector ops suitable for lowering to SPIR-V cooperative ops.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVTileAndVectorizeToCooperativeOpsPass();

/// Pass to convert vector read/write/arithmetic operations to the corresponding
/// cooperative matrix ops when possible.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVVectorToCooperativeOpsPass();

/// Pass to tile Linalg ops with tensor semantics to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVTilePass();

/// Pass to distribute tiled loop nests to invocations.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVDistributePass();

/// Pass to vectorize Linalg ops with buffer semantics.
std::unique_ptr<OperationPass<func::FuncOp>> createSPIRVVectorizePass();

/// Converts memref of scalar to memref of vector of efficent size. This will
/// allow to convert memory accesses to vector load/store in SPIR-V without
/// having pointer bitcast.
std::unique_ptr<OperationPass<ModuleOp>> createSPIRVVectorizeLoadStore();

/// Fuses tensor.pad ops into their consumer ops' tiled loop nests.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVFuseTensorPadWithConsumerPass();

// Uses `tensor.pad` ops as anchors to create separate fast and slow paths
// inside the kernel. The fast path is for inner tiles where we don't need
// padding, while the slow path is for boundary tiles where we do need padding.
std::unique_ptr<OperationPass<func::FuncOp>>
createSPIRVCreateFastSlowPathPass();

//----------------------------------------------------------------------------//
// SPIRV Codegen Pass Pipelines.
//----------------------------------------------------------------------------//

/// Populates passes needed to lower a XLA HLO op to SPIR-V dialect via the
/// structured ops path. The pass manager `pm` in here operate on the module
/// within the IREE::HAL::ExecutableOp. The `workGroupSize` can be used to
/// control the work group size used in the code generation and is intended for
/// testing purposes only. The pass pipeline will set an appropriate workgroup
/// size.
/// TODO: Are both of these needed and does this one still work on HLO?
void buildSPIRVCodegenPassPipeline(OpPassManager &pm,
                                   bool useKernelCapability = false);

//------------------------------------------------------------------------------
// VMVX passes
//------------------------------------------------------------------------------

// Lowers high level library calls from named ops and generics. This operates
// at the bufferized linalg level.
std::unique_ptr<Pass> createVMVXLowerLinalgMicrokernelsPass();

//------------------------------------------------------------------------------
// Test passes
//------------------------------------------------------------------------------

std::unique_ptr<OperationPass<ModuleOp>> createTestLLVMGPULegalizePass();

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_CODEGEN_PASSES_H_
