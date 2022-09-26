// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/SPIRV/KernelConfig.h"

#include <functional>
#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Transforms/Transforms.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVAttributes.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVEnums.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#define DEBUG_TYPE "iree-spirv-kernel-config"

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utility Functions
//===----------------------------------------------------------------------===//

bool isMatmulOrBatchMatmul(linalg::LinalgOp linalgOp) {
  return linalg::isaContractionOpInterface(linalgOp) &&
         llvm::is_contained({2u, 3u}, linalgOp.getNumParallelLoops());
}

//===----------------------------------------------------------------------===//
// Convolution Default Configuration
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult setConvOpConfig(linalg::LinalgOp linalgOp,
                              const int64_t subgroupSize,
                              const int64_t bestTilingFactor) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as convolution...\n");
  Type inputType = linalgOp.getInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = inputType.cast<ShapedType>().getShape();
  Type outputType = linalgOp.getOutputOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = outputType.cast<ShapedType>().getShape();

  if (isa<linalg::Conv2DNhwcHwcfOp>(*linalgOp) &&
      ShapedType::isDynamic(inputShape[3])) {
    return success();
  }
  if (llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return success();
  }

  int64_t ic = inputShape[3];
  int64_t oh = outputShape[1], ow = outputShape[2], oc = outputShape[3];

  // The conversion pipeline requires the input channel dimension to be some
  // multipler of four, or less than four.
  if (!(ic % 4 == 0 || ic < 4)) return success();

  // The core idea is to distribute the convolution OH/OW/OC dimension to the
  // workgroup Z/Y/X dimension, with each thread in a workgroup handling
  // multiple vector elements. We try to 1) utilize all threads in a subgroup,
  // and 2) handle an optimal tile size along each dimension.

  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1);     // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 0);   // (N, OH, OW, OC)
  SmallVector<int64_t> invocationTileSizes(4, 0);  // (N, OH, OW, OC)

  // Deduce the configuration for the OC dimension.
  for (int64_t x = residualThreads; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = 4;
    if (oc % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[3] = x * chosenTileSize;
      invocationTileSizes[3] = chosenTileSize;
      residualThreads /= x;
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[3] == 0) return success();

  // Deduce the configruation for the OW and OH dimension. Try to make them even
  // if possible given we typically have images with the same height and width.
  bool tileToSquare = false;
  unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (ow == oh && residualThreads != 1 && log2Threads % 2 == 0) {
    int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      workgroupSize[1] = workgroupSize[2] = yz;
      workgroupTileSizes[2] = workgroupTileSizes[1] = yz * chosenTileSize;
      invocationTileSizes[2] = invocationTileSizes[1] = chosenTileSize;
      tileToSquare = true;
    }
  }

  // Otherwise treat OW and OH separately to allow them to have different number
  // of threads and tiling size.
  if (!tileToSquare) {
    // Decide the tiling and distribution parameters for one dimension.
    auto decideOneDim = [&](int64_t inputDim, int64_t &wgDimSize,
                            int64_t &wgTileSize, int64_t &invoTileSize) {
      for (int64_t dim = residualThreads; dim >= 1; dim >>= 1) {
        int64_t chosenTileSize = 0;
        for (int64_t t = residualTilingFactor; t >= 1; t >>= 1) {
          if (inputDim % (dim * t) == 0) {
            chosenTileSize = t;
            break;
          }
        }
        if (chosenTileSize) {
          wgDimSize = dim;
          wgTileSize = dim * chosenTileSize;
          invoTileSize = chosenTileSize;
          residualThreads /= dim;
          residualTilingFactor /= chosenTileSize;
          return true;
        }
      }
      return false;
    };

    if (!decideOneDim(ow, workgroupSize[1], workgroupTileSizes[2],
                      invocationTileSizes[2]) ||
        !decideOneDim(oh, workgroupSize[2], workgroupTileSizes[1],
                      invocationTileSizes[1])) {
      return success();
    }
  }

  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize;
  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(invocationTileSizes);
  // Tiling along reduction dimensions
  if (isa<linalg::Conv2DNhwcHwcfOp>(linalgOp)) {
    tileSizes.push_back({0, 0, 0, 0, 1, 1, 4});
  } else if (isa<linalg::DepthwiseConv2DNhwcHwcOp>(linalgOp)) {
    tileSizes.push_back({0, 0, 0, 0, 1, 1});
  } else {
    return success();
  }

  auto funcOp = linalgOp->getParentOfType<func::FuncOp>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes,
                                               pipeline, workgroupSize);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// Matmul Default Configuration
//===----------------------------------------------------------------------===//

namespace detail {

LogicalResult setMatmulOpConfig(linalg::LinalgOp op, int64_t subgroupSize,
                                std::array<int64_t, 2> bestWorkgroupSizeXY,
                                std::array<int64_t, 3> bestThreadTileSizeMNK,
                                bool useWorkgroupMemory) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as matmul...\n");
  OpOperand *lhs = op.getInputOperand(0);
  OpOperand *rhs = op.getInputOperand(1);

  auto lhsType = lhs->get().getType().cast<ShapedType>();
  auto rhsType = rhs->get().getType().cast<ShapedType>();
  auto elementBits = lhsType.getElementType().getIntOrFloatBitWidth();
  if (elementBits != 16 && elementBits != 32) return success();

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  if (llvm::any_of(lhsShape, ShapedType::isDynamic)) return success();
  if (llvm::any_of(rhsShape, ShapedType::isDynamic)) return success();

  assert(llvm::is_contained({2u, 3u}, op.getNumParallelLoops()));
  const bool isBM = op.getNumParallelLoops() == 3;

  auto lhsLoopIndices = llvm::to_vector(llvm::map_range(
      llvm::seq<int>(0, lhsShape.size()),
      [&](int i) { return op.getTiedIndexingMap(lhs).getDimPosition(i); }));
  auto rhsLoopIndices = llvm::to_vector(llvm::map_range(
      llvm::seq<int>(0, rhsShape.size()),
      [&](int i) { return op.getTiedIndexingMap(rhs).getDimPosition(i); }));

  // Figure out what dimension each loop corresponds to.
  int bIndex = -1, mIndex = -1, nIndex = -1, kIndex = -1;
  int lastParallelDim = -1;
  for (unsigned i = 0; i < op.getNumLoops(); ++i) {
    if (linalg::isReductionIterator(op.getIteratorTypes()[i])) {
      kIndex = i;
      continue;
    }

    const bool inLHS = llvm::is_contained(lhsLoopIndices, i);
    const bool inRHS = llvm::is_contained(rhsLoopIndices, i);
    if (inLHS && inRHS) {
      bIndex = i;
    } else if (inLHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the LHS, treat the outer one of them as the batch dimension.
      if (mIndex >= 0 && bIndex < 0) bIndex = mIndex;
      mIndex = i;
    } else if (inRHS) {
      // For cases where we have two parallel dimensions only accessed by
      // the RHS, treat the outer one of them as the batch dimension.
      if (nIndex >= 0 && bIndex < 0) bIndex = nIndex;
      nIndex = i;
    }
    lastParallelDim = i;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "bIndex = " << bIndex << "\n";
    llvm::dbgs() << "mIndex = " << mIndex << "\n";
    llvm::dbgs() << "kIndex = " << kIndex << "\n";
    llvm::dbgs() << "nIndex = " << nIndex << "\n";
  });
  if (mIndex < 0 || nIndex < 0 || kIndex < 0) return success();

  SmallVector<int64_t, 4> loopRanges = op.getStaticLoopRanges();
  const unsigned numLoops = loopRanges.size();

  const int64_t dimM = loopRanges[mIndex];
  const int64_t dimK = loopRanges[kIndex];
  const int64_t dimN = loopRanges[nIndex];

  // The core idea is to distribute the matmul M/N dimension to the workgroup
  // Y/X dimension, with each thread in a workgroup handling multiple vector
  // elements. We start from the best (X, Y) and the tiling sizes for (M, N, K)
  // and try different configurations by scaling them down until we find a
  // configuration that can perfectly tile the input matmul.

  const int64_t bestThreadM = bestThreadTileSizeMNK[0],
                bestThreadN = bestThreadTileSizeMNK[1],
                bestThreadK = bestThreadTileSizeMNK[2];

  int64_t bestX = bestWorkgroupSizeXY[0], bestY = bestWorkgroupSizeXY[1];
  // We will deduce a configuration first for x and then y. But look at y here
  // to see if the problem size is too small; for such cases, "shift" the
  // parallelism to x.
  if (dimM < bestThreadM) {
    int64_t factor = llvm::PowerOf2Ceil(llvm::divideCeil(bestThreadM, dimM));
    bestX *= factor;
    bestY = llvm::divideCeil(bestY, factor);
  }

  int64_t residualThreads = bestX * bestY;
  int64_t residualTilingFactor = (bestThreadM + bestThreadK) * bestThreadN;

  SmallVector<int64_t, 3> workgroupSize(3, 1);  // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(numLoops, 0);
  SmallVector<int64_t> invocationTileSizes(numLoops, 0);
  SmallVector<int64_t> reductionTileSizes(numLoops, 0);

  if (isBM) workgroupTileSizes[bIndex] = invocationTileSizes[bIndex] = 1;

  // Deduce the configuration for the N dimension. Start with the best workgroup
  // X size, and reduce by a factor of two each time.
  for (int64_t x = bestX; x >= 2; x >>= 1) {
    // Handle 4 elements per thread for the innermost dimension. We need this
    // for vectorized load.
    int64_t chosenTileSize = bestThreadN;
    if (dimN % (x * chosenTileSize) == 0) {
      workgroupSize[0] = x;
      workgroupTileSizes[nIndex] = x * chosenTileSize;
      invocationTileSizes[nIndex] = chosenTileSize;
      residualThreads /= x;
      assert(residualTilingFactor % chosenTileSize == 0);
      residualTilingFactor /= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[nIndex] == 0) return success();

  // Don't overshoot when using workgroup memory to avoid blowing up workgroup
  // memory size.
  if (useWorkgroupMemory) residualThreads = std::min(residualThreads, bestY);

  // Deduce the configuration for the M dimension. Start with the best workgroup
  // Y size, and reduce by a factor of two each time.
  for (int64_t y = residualThreads; y >= 1; y >>= 1) {
    int64_t chosenTileSize = 0;
    // Reduce the thread tiling size by one each time. We read one row each
    // time; so it's fine to not be some power of two here.
    for (int64_t t = bestThreadM; t >= 1; --t) {
      if (dimM % (y * t) == 0) {
        chosenTileSize = t;
        break;
      }
    }
    if (chosenTileSize) {
      workgroupSize[1] = y;
      workgroupTileSizes[mIndex] = y * chosenTileSize;
      invocationTileSizes[mIndex] = chosenTileSize;
      assert(residualTilingFactor > chosenTileSize);
      residualTilingFactor -= chosenTileSize;
      break;
    }
  }
  if (workgroupTileSizes[mIndex] == 0) return success();

  // Deduce the configuration for the K dimension. We need some power of two
  // here so that we can do vector load.
  for (int64_t t = llvm::PowerOf2Floor(residualTilingFactor); t >= 2; t >>= 1) {
    if (dimK % t == 0) {
      reductionTileSizes[kIndex] = t;
      break;
    }
  }
  if (reductionTileSizes[kIndex] == 0) return success();

  auto totalThreads =
      std::accumulate(workgroupSize.begin(), workgroupSize.end(), 1,
                      std::multiplies<int64_t>());
  auto pipeline =
      (useWorkgroupMemory && totalThreads > subgroupSize)
          ? IREE::Codegen::DispatchLoweringPassPipeline::
                SPIRVVectorizeWithWorkgroupMemory
          : IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize;

  TileSizesListType tileSizes;
  workgroupTileSizes.resize(lastParallelDim + 1);
  invocationTileSizes.resize(lastParallelDim + 1);
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(invocationTileSizes);
  tileSizes.push_back(reductionTileSizes);

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes, pipeline,
      workgroupSize);
}

}  // namespace detail

//===----------------------------------------------------------------------===//
// FFT Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setFftOpConfig(spirv::ResourceLimitsAttr limits,
                                    IREE::LinalgExt::FftOp op) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as fft...\n");
  const int subgroupSize = limits.getSubgroupSize();
  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

  std::array<int64_t, 3> workgroupSize = {subgroupSize, 1, 1};

  SmallVector<StringRef> loopIteratorTypes = op.getLoopIteratorTypes();
  unsigned loopDepth = loopIteratorTypes.size();
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);

  // Tiling along partitioned loops with size 1.
  for (auto iteratorType : llvm::enumerate(loopIteratorTypes)) {
    if (iteratorType.value() == getParallelIteratorTypeName()) {
      workgroupTileSize[iteratorType.index()] = 1;
    }
  }
  auto rank = op.getOperandRank();
  if (workgroupTileSize.size() >= rank && workgroupTileSize[rank - 1] != 0) {
    APInt value;
    if (matchPattern(op.getStage(), m_ConstantInt(&value))) {
      workgroupTileSize[rank - 1] = 1ll << value.getSExtValue();
    } else {
      op.emitError("non-constant stage might not work for fft op");
      return failure();
    }
  }
  TileSizesListType tileSizes = {workgroupTileSize};
  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes, pipeline,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Reduction Default Configuration
//===----------------------------------------------------------------------===//

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult setReductionConfig(const spirv::TargetEnv &targetEnv,
                                        linalg::GenericOp op) {

  // TODO: Fix/support Warp Reduction for LevelZero.
  if (targetEnv.allows(spirv::Capability::Kernel))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "trying to deduce config as reduction...\n");
  if (op.hasDynamicShape()) return failure();
  // This pipeline eventually generates non-uniform group shuffle ops, which
  // requires special capability.
  if (!targetEnv.allows(spirv::Capability::GroupNonUniformShuffle))
    return failure();

  SmallVector<unsigned> reductionDims;
  op.getReductionDims(reductionDims);
  if (reductionDims.size() != 1 || reductionDims[0] != op.getNumLoops() - 1)
    return failure();
  if (op.getRegionOutputArgs().size() != 1) return failure();

  // Only support projected permutation for now. This could be extended to
  // projected permutated with broadcast.
  if (llvm::any_of(op.getInputOperands(), [&](OpOperand *input) {
        return !op.getTiedIndexingMap(input).isProjectedPermutation();
      })) {
    return failure();
  }

  // Only support single combiner operations for now.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1) {
    return failure();
  }

  const int subgroupSize = targetEnv.getResourceLimits().getSubgroupSize();
  Optional<int64_t> dimSize = op.getStaticLoopRanges()[reductionDims[0]];
  if (!dimSize || *dimSize % subgroupSize != 0) return failure();

  // Let each thread handle `vectorSize` elements.
  unsigned vectorSize = 4;
  while ((*dimSize / vectorSize) % subgroupSize != 0) vectorSize /= 2;

  std::array<int64_t, 3> workgroupSize = {*dimSize / vectorSize, 1, 1};

  // Tile all the parallel dimension to 1.
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
  partitionedLoopsSet.insert(partitionedLoops.begin(), partitionedLoops.end());
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);

  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level

  return setOpConfigAndEntryPointFnTranslation(
      op->getParentOfType<func::FuncOp>(), op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::SPIRVSubgroupReduce,
      workgroupSize);
}

//===----------------------------------------------------------------------===//
// Everything Default Configuration
//===----------------------------------------------------------------------===//

static LogicalResult setDefaultOpConfig(spirv::ResourceLimitsAttr limits,
                                        Operation *op,
                                        bool allowVectorization = true) {
  LLVM_DEBUG(llvm::dbgs() << "trying to deduce as default op...\n");
  func::FuncOp funcOp = op->getParentOfType<func::FuncOp>();
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);

  // Special case for not tiled ops.
  if (partitionedLoops.empty()) {
    // No tiled loops means we cannot tile (and distribute) at all. Use just one
    // single thread to run everything.
    auto pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;
    std::array<int64_t, 3> workgroupSize = {1, 1, 1};
    return setOpConfigAndEntryPointFnTranslation(funcOp, op, {}, pipeline,
                                                 workgroupSize);
  }

  const int subgroupSize = limits.getSubgroupSize();
  const unsigned loopDepth = partitionedLoops.back() + 1;

  // Configurations we need to decide.
  std::array<int64_t, 3> workgroupSize;
  SmallVector<int64_t> workgroupTileSizes;
  SmallVector<int64_t> threadTileSizes;

  // Initialize the configuration.
  auto initConfiguration = [&]() {
    workgroupSize = {subgroupSize, 1, 1};
    workgroupTileSizes.resize(loopDepth, 0);
    threadTileSizes.resize(loopDepth, 0);

    // Initialize tiling along all partitioned loops with size 1.
    for (int64_t loopIndex : partitionedLoops) {
      workgroupTileSizes[loopIndex] = threadTileSizes[loopIndex] = 1;
    }
    // Override the innermost dimension to distribute to threads in a subgroup.
    workgroupTileSizes.back() = subgroupSize;
    threadTileSizes.back() = 1;
  };

  // Special case for non-linalg ops.
  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  if (!linalgOp || linalgOp.getNumOutputs() != 1) {
    auto pipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

    initConfiguration();
    TileSizesListType tileSizes;
    tileSizes.push_back(workgroupTileSizes);
    tileSizes.push_back(threadTileSizes);

    return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes,
                                                 pipeline, workgroupSize);
  }

  // Common case for all linalg ops.

  // The core idea is to distribute the partitioned loops to the workgroup
  // dimensions. The goal is to fill up the GPU as much as possible, which means
  // 1) distributing to as many threads as possible, and 2) avoid assigning too
  // many threads to handle out-of-bound elements (thus idle).

  // Returns true if the given `operand` has 32-bit element type.
  auto has32BitElementType = [](Value operand) {
    auto shapedType = operand.getType().dyn_cast<ShapedType>();
    Type elementType =
        (shapedType ? shapedType.getElementType() : operand.getType());
    return elementType.isa<FloatType>() || elementType.isInteger(32);
  };

  // Whether we can try to use the vectorization pipeline.
  SmallVector<int64_t, 4> loopBounds = linalgOp.getStaticLoopRanges();
  bool vectorizable =
      allowVectorization &&
      // The vectorization pipeline assumes tensor semantics for tiling.
      linalgOp.hasTensorSemantics() && !linalgOp.hasIndexSemantics() &&
      // Require all affine maps to be projected permutation so that we can
      // generate vector transfer ops.
      llvm::all_of(
          linalgOp.getIndexingMapsArray(),
          [](AffineMap map) { return map.isProjectedPermutation(); }) &&
      // TODO: Fix non-32-bit element type vectorization and remove this.
      llvm::all_of(linalgOp->getOperands(), has32BitElementType) &&
      llvm::none_of(loopBounds, ShapedType::isDynamic);

  // Distribute workload to the given `numThreads` by allowing a potental loss.
  auto distributeToThreads = [&](int64_t numThreads,
                                 Optional<int64_t> lossFactor = llvm::None) {
    LLVM_DEBUG(llvm::dbgs() << "\nLoss factor: " << lossFactor << "\n");
    initConfiguration();

    // Scan from the innermost shape dimension and try to deduce the
    // configuration for the corresponding GPU workgroup dimension.
    int64_t wgDim = 0;
    for (auto shapeDim : llvm::reverse(partitionedLoops)) {
      int64_t loopBound = loopBounds[shapeDim];
      // Skip dynamic dimensions.
      if (ShapedType::isDynamic(loopBound)) continue;

      // Try to find some power of two that can devide the current shape dim
      // size. This vector keeps the candidate tile sizes.
      SmallVector<int64_t, 8> candidates;

      // For the inner most workgroup dim, try to see if we can have 4
      // elements per thread. This enables vectorization.
      if (vectorizable && wgDim == 0 && !lossFactor) {
        candidates.push_back(4 * numThreads);
      }
      // Try all power of two numbers upto the subgroup size.
      for (unsigned i = numThreads; i >= 1; i >>= 1) {
        candidates.push_back(i);
      }
      LLVM_DEBUG({
        llvm::dbgs() << "Candidate tile sizes: [";
        llvm::interleaveComma(candidates, llvm::dbgs());
        llvm::dbgs() << "]\n";
      });

      for (int64_t candidate : candidates) {
        if (loopBound % candidate != 0) {
          if (!lossFactor) continue;
          // Skip this candidate if it causes many threads to be idle.
          int64_t idleThreads = candidate - (loopBound % candidate);
          if (idleThreads > candidate / *lossFactor) continue;
        }
        // If the workload is too small and we cannot distribute to more than 2
        // workgroups, try a smaller tile size to increase parallelism.
        if (partitionedLoops.size() == 1 && candidate > subgroupSize &&
            llvm::divideCeil(loopBound, candidate) <= 2) {
          continue;
        }

        // Found a suitable candidate. Try to let each thread handle 4
        // elements if this is the workgroup x dimension.
        workgroupTileSizes[shapeDim] = candidate;
        LLVM_DEBUG(llvm::dbgs() << "Chosen tile size: " << candidate << "\n");
        if (vectorizable && wgDim == 0 && !lossFactor && candidate % 4 == 0) {
          // Use size-1 vectors to increase parallelism if larger ones causes
          // idle threads in the subgroup.
          bool hasIdleThreads =
              partitionedLoops.size() == 1 && candidate <= subgroupSize;
          int vectorSize = hasIdleThreads ? 1 : 4;
          LLVM_DEBUG(llvm::dbgs() << "Use vector size: " << vectorSize << "\n");
          threadTileSizes[shapeDim] = vectorSize;
          workgroupSize[wgDim] = candidate / vectorSize;
          assert(numThreads % (candidate / vectorSize) == 0);
          numThreads /= candidate / vectorSize;
        } else {
          if (wgDim == 0) vectorizable = false;
          threadTileSizes[shapeDim] = 1;
          workgroupSize[wgDim] = candidate;
          assert(numThreads % candidate == 0);
          numThreads /= candidate;
        }
        assert(numThreads >= 1);
        break;
      }

      // Stop if we have distributed all threads.
      if (numThreads == 1) break;
      wgDim++;
    }
    return numThreads;
  };

  // First try to see if we can use up all threads without any loss.
  if (distributeToThreads(subgroupSize) != 1) {
    // Otherwise, allow larger and larger loss factor.

    // Threads for distribution. Use 32 at least.
    int64_t numThreads = std::max(subgroupSize, 32);
    // We can tolerate (1 / lossFactor) of threads in the workgroup to be idle.
    int64_t lossFactor = 32;

    for (; lossFactor >= 1; lossFactor >>= 1) {
      if (distributeToThreads(numThreads, lossFactor) == 1) break;
    }
  }

  auto pipeline =
      vectorizable
          ? IREE::Codegen::DispatchLoweringPassPipeline::SPIRVVectorize
          : IREE::Codegen::DispatchLoweringPassPipeline::SPIRVDistribute;

  TileSizesListType tileSizes;
  tileSizes.push_back(workgroupTileSizes);
  tileSizes.push_back(threadTileSizes);

  if (vectorizable) {
    // Try to tile all reductions by size 4 if possible. This gives us a chance
    // to perform vector4 load if an input has its innnermost dimension being
    // reduction. It also avoids generating too many instructions when unrolling
    // vector later. Similarly, also try to tile other untiled parallel
    // dimensions by 4 to avoid instruction bloat.
    SmallVector<int64_t> loopTileSizes(linalgOp.getNumLoops(), 0);
    for (const auto &it : llvm::enumerate(linalgOp.getIteratorTypes())) {
      auto i = it.index();
      if (loopBounds[i] % 4 != 0) continue;
      if (linalg::isReductionIterator(it.value()) ||
          workgroupTileSizes[i] == 0) {
        loopTileSizes[it.index()] = 4;
      }
    }
    if (llvm::any_of(loopTileSizes, [](int64_t s) { return s != 0; })) {
      tileSizes.push_back(loopTileSizes);
    }
  }

  return setOpConfigAndEntryPointFnTranslation(funcOp, op, tileSizes, pipeline,
                                               workgroupSize);
}

//===----------------------------------------------------------------------===//
// Configuration Dispatcher
//===----------------------------------------------------------------------===//

/// Sets the CodeGen configuration as attributes to the given `rootOp` if it's a
/// known Linalg matmul/convolution op with good configurations.
static LogicalResult setSPIRVOpConfig(const spirv::TargetEnv &targetEnv,
                                      func::FuncOp entryPointFn,
                                      Operation *rootOp) {
  if (IREE::Codegen::CompilationInfoAttr compilationInfo =
          getCompilationInfo(rootOp)) {
    // If the op already has a lowering configuration specified from the
    // original source by the user, then use it directly.
    return setUserConfig(entryPointFn, rootOp, compilationInfo);
  }

  LogicalResult result = success();
  // First try to find a proper CodeGen configuration to tile and vectorize for
  // the current target architecture.
  switch (targetEnv.getVendorID()) {
    case spirv::Vendor::AMD:
      result = detail::setAMDCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::Apple:
      result = detail::setAppleCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::ARM:
      result = detail::setMaliCodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::NVIDIA:
      result = detail::setNVIDIACodeGenConfig(targetEnv, rootOp);
      break;
    case spirv::Vendor::Qualcomm:
      result = detail::setAdrenoCodeGenConfig(targetEnv, rootOp);
      break;
    default:
      break;
  }

  if (failed(result)) return result;
  // Check whether there is actually a configuration found. If so, it's done.
  if (getLoweringConfig(rootOp)) return result;

  // Otherwise fallback to use a default configuration that tiles and
  // distributes/vectorizes.
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();
  return TypeSwitch<Operation *, LogicalResult>(rootOp)
      .Case<linalg::BatchMatmulOp, linalg::MatmulOp>([limits](auto op) {
        // Try to tile and vectorize first. It's common to see 32 threads
        // per subgroup for GPUs.
        std::array<int64_t, 2> workgroupXY = {32, 2};
        std::array<int64_t, 3> threadMNK = {8, 8, 4};
        auto result = detail::setMatmulOpConfig(op, /*subgroupSize=*/32,
                                                workgroupXY, threadMNK);
        if (failed(result)) return result;
        if (getLoweringConfig(op)) return result;

        // If unsuccessful, try to tile and distribute.
        return setDefaultOpConfig(limits, op);
      })
      .Case<linalg::Conv2DNhwcHwcfOp, linalg::DepthwiseConv2DNhwcHwcOp>(
          [limits](auto op) {
            // Try to tile and vectorize first. It's common to see 32 threads
            // per subgroup for GPUs.
            auto result = detail::setConvOpConfig(op, /*subgroupSize=*/32,
                                                  /*bestTilingFactor=*/32);
            if (failed(result)) return result;
            if (getLoweringConfig(op)) return result;

            // If unsuccessful, try to tile and distribute.
            return setDefaultOpConfig(limits, op);
          })
      .Case<linalg::ConvolutionOpInterface>([limits](auto op) {
        // Other convolution/pooling op vectorization is not wired up.
        return setDefaultOpConfig(limits, op, /*allowVectorization=*/false);
      })
      .Case<linalg::GenericOp>([&](linalg::GenericOp op) {
        LLVM_DEBUG(llvm::dbgs() << "figuring configuration for generic op\n");
        if (succeeded(setReductionConfig(targetEnv, op))) return success();

        // If a generic op has reduction iterator types, it can be treated as a
        // root op for configuration as well. Use the default configuration,
        // which will mark it as a root.
        if (op.getNumLoops() != op.getNumParallelLoops()) {
          return setDefaultOpConfig(limits, op);
        }
        return success();
      })
      .Case<IREE::LinalgExt::FftOp>([limits](IREE::LinalgExt::FftOp op) {
        return setFftOpConfig(limits, op);
      })
      .Default([](Operation *) { return success(); });
};

//===----------------------------------------------------------------------===//
// Entry Point
//===----------------------------------------------------------------------===//

LogicalResult initSPIRVLaunchConfig(ModuleOp module) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(module);
  spirv::TargetEnvAttr targetEnvAttr = getSPIRVTargetEnvAttr(module);
  if (!targetEnvAttr) {
    return module.emitOpError(
        "expected parent hal.executable.variant to have spv.target_env "
        "attribute");
  }
  spirv::TargetEnv targetEnv(targetEnvAttr);
  spirv::ResourceLimitsAttr limits = targetEnv.getResourceLimits();

  for (auto funcOp : module.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;

    SmallVector<Operation *> computeOps;
    SmallVector<LoopTilingAndDistributionInfo> tiledLoops;
    if (failed(getComputeOps(funcOp, computeOps, tiledLoops))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    if (computeOps.empty()) {
      return funcOp.emitOpError(
          "unhandled translation of function without compute ops");
    }

    Operation *rootOperation = nullptr;
    // Try to find a configuration according to a matmul/convolution op and use
    // it as the root op.
    for (Operation *computeOp : computeOps) {
      if (failed(setSPIRVOpConfig(targetEnv, funcOp, computeOp)))
        return failure();

      // Check if the op configuration was set.
      if (!getLoweringConfig(computeOp)) continue;

      if (rootOperation) {
        return computeOp->emitOpError(
            "unhandled multiple roots in dispatch region");
      }
      rootOperation = computeOp;
    }

    if (!rootOperation) {
      // If there are still no root op, check for any linalg.generic op.
      Operation *computeOp = computeOps.back();
      if (failed(setDefaultOpConfig(limits, computeOp))) return failure();

      // Check if the op configuration was set.
      if (!getLoweringConfig(computeOp)) {
        return computeOp->emitOpError(
            "without known roots, the last compute operation in the tiled "
            "loop body is expected to be set as root");
      }
      rootOperation = computeOp;
    }

    // Propogate the `lowering_config` attribute to the other ops.
    // TODO(ravishankarm, antiagainst): This is a very specific use (and
    // fragile). In general, this should not be needed. Things are already tiled
    // and distributed. The rest of the compilation must be structured to either
    // use `TileAndFuse` or they are independent configurations that are
    // determined based on the op.
    IREE::Codegen::LoweringConfigAttr config = getLoweringConfig(rootOperation);
    for (auto op : computeOps) {
      if (op == rootOperation) continue;
      setLoweringConfig(op, config);
    }
  }
  return success();
}

}  // namespace iree_compiler
}  // namespace mlir
