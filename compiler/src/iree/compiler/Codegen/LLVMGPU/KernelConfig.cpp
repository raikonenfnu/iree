// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/KernelConfig.h"

#include <numeric>

#include "iree-dialects/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "iree/compiler/Codegen/Common/LinalgOpInfo.h"
#include "iree/compiler/Codegen/Common/UserConfig.h"
#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/LLVMGPU/TransposeUtils.h"
#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Types.h"
#include "mlir/IR/Value.h"

using namespace mlir;
using namespace mlir::iree_compiler;

static constexpr unsigned cudaWarpSize = 32;
static constexpr StringLiteral kCudaTarget = "cuda";
namespace mlir {
namespace iree_compiler {
llvm::cl::opt<std::string> clGPUCodegenTransformDialectFileName(
    "iree-codegen-llvmgpu-use-transform-dialect",
    llvm::cl::desc(
        "MLIR file containing a transform dialect specification to apply"),
    llvm::cl::init(""));

llvm::cl::list<int64_t> clGPUCodegenTransformDialectTileSizes(
    "iree-codegen-llvmgpu-workgroup-tile-sizes",
    llvm::cl::desc("Fixed tile sizes when using the transform dialect starting "
                   "from IR already workgroup distributed"),
    llvm::cl::CommaSeparated);
}  // namespace iree_compiler
}  // namespace mlir

namespace {
struct TileWorkgroupSizePair {
  // How many scalar elements each workgroup should handle along each dimension.
  std::array<int64_t, 3> tileSize;
  std::array<int64_t, 3> workgroupSize;
};

// Software pipeline depths
constexpr unsigned softwarePipelineDepthTensorCore = 4;
// Simt codegen does not do software pipelining.
constexpr unsigned softwarePipelineDepthSimt = 0;
}  // namespace

/// Return the best combination of tile size and wg size. It will then used to
/// pick the best size aligned with the shape dimension.
static void getMatmulConfig(SmallVectorImpl<TileWorkgroupSizePair> &tileSizes) {
  // Pick tile size so that M*K and K*N dividible by wgSize * \*vecSize=*\4.
  // This way workgroup memory copy don't need to be masked. Once we support
  // masked load we can get performance out of more configuration.
  tileSizes.push_back(TileWorkgroupSizePair({{32, 128, 32}, {32, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{128, 64, 8}, {16, 8, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 256, 32}, {64, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{8, 32, 32}, {8, 8, 1}}));

  tileSizes.push_back(TileWorkgroupSizePair({{8, 128, 4}, {32, 1, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{16, 64, 4}, {16, 2, 1}}));
  tileSizes.push_back(TileWorkgroupSizePair({{1, 128, 8}, {32, 1, 1}}));
}

/// Return the best combination of tile size and wg size when using tensorcore
/// operations.
static void getTensorCoreConfig(
    SmallVectorImpl<TileWorkgroupSizePair> &tileSizes, bool isFp16) {
  // Tile sizes are skewed towards small matmul for now. Long term the plan is
  // to not rely on hardcoded configurations.
  if (isFp16) {
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 32}, {64, 2, 1}}));
  } else {
    tileSizes.push_back(TileWorkgroupSizePair({{32, 32, 16}, {64, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 32, 16}, {64, 1, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{32, 16, 16}, {32, 2, 1}}));
    tileSizes.push_back(TileWorkgroupSizePair({{16, 16, 16}, {32, 1, 1}}));
  }
}

static std::string getTargetArch(func::FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    if (auto config = targetAttr.getConfiguration()) {
      if (auto attr = config.getAs<StringAttr>("target_arch")) {
        return attr.getValue().str();
      }
    }
  }
  return "";
}

bool isCudaTarget(func::FuncOp entryPoint) {
  if (auto variantOp =
          entryPoint->getParentOfType<IREE::HAL::ExecutableVariantOp>()) {
    IREE::HAL::ExecutableTargetAttr targetAttr = variantOp.getTarget();
    if (auto backend = targetAttr.getBackend()) {
      return backend.getValue().str() == kCudaTarget;
    }
  }
  return false;
}

static bool supportsTensorCore(func::FuncOp entryPoint, linalg::LinalgOp op) {
  // Limit tensor core pipeline to matmul as not all combinations of transpose
  // are supported upstream.
  // TODO(thomasraoux): Enable batchMatmul and generic contraction.
  if (getTargetArch(entryPoint) != "sm_80") return false;
  if (!(isa<linalg::MatmulOp>(op) || isa<linalg::BatchMatmulOp>(op))) {
    assert(linalg::isaContractionOpInterface(op));
    // If this is not a named op matmul check some properties to make sure that
    // we can map it to tensorcore ops. We should have only mulAdd in the region
    // and the output map should have no permutation and the last dimension
    // should be a reduce.
    Region &body = op->getRegion(0);
    Region::OpIterator it = body.op_begin();
    if (it == body.op_end() || !isa<arith::MulFOp>(*(it++))) return false;
    if (it == body.op_end() || !isa<arith::AddFOp>(*(it++))) return false;
    if (it == body.op_end() || !isa<linalg::YieldOp>(*(it++))) return false;
    AffineMap outputMap = op.getMatchingIndexingMap(op.getOutputOperand(0));
    if (outputMap.getNumResults() != outputMap.getNumDims() - 1) return false;
    OpBuilder b(op);
    for (unsigned i = 0, e = outputMap.getNumResults(); i < e - 1; i++) {
      if (outputMap.getResult(i) != b.getAffineDimExpr(i)) return false;
    }
  }
  return true;
}

static LogicalResult setContractConfig(func::FuncOp entryPoint,
                                       linalg::LinalgOp op) {
  auto setMatmulConfig =
      [&entryPoint, &op](int64_t tileX, int64_t tileY, int64_t tileK,
                         llvm::ArrayRef<int64_t> workgroupSize,
                         unsigned softwarePipelineDepth,
                         IREE::Codegen::DispatchLoweringPassPipeline pipeline) {
        TileSizesListType tileSizes;
        unsigned numParallelLoops = op.getNumParallelLoops();
        SmallVector<int64_t> workgroupTileSizes(numParallelLoops - 2, 1);
        workgroupTileSizes.append({tileX, tileY});
        workgroupTileSizes.append(op.getNumReductionLoops(), tileK);

        SmallVector<unsigned> partitionedLoops =
            cast<PartitionableLoopsInterface>(op.getOperation())
                .getPartitionableLoops(kNumMaxParallelDims);
        llvm::SmallDenseSet<unsigned, 4> partitionedLoopsSet;
        partitionedLoopsSet.insert(partitionedLoops.begin(),
                                   partitionedLoops.end());
        for (auto loopID : llvm::seq<unsigned>(0, numParallelLoops)) {
          if (!partitionedLoopsSet.count(loopID)) {
            workgroupTileSizes[loopID] = 0;
          }
        }

        tileSizes.emplace_back(
            std::move(workgroupTileSizes));  // Workgroup level.
        return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                     pipeline, workgroupSize,
                                                     softwarePipelineDepth);
      };
  // Infer the MxN size of the matmul based on operands and indexing maps.
  auto lhsShape =
      op.getInputOperand(0)->get().getType().cast<ShapedType>().getShape();
  auto rhsShape =
      op.getInputOperand(1)->get().getType().cast<ShapedType>().getShape();
  int64_t sizeM = ShapedType::kDynamicSize;
  int64_t sizeN = ShapedType::kDynamicSize;
  int64_t sizeK = ShapedType::kDynamicSize;
  auto outputMap = op.getMatchingIndexingMap(op.getOutputOperand(0));
  for (unsigned i = 0; i < lhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getInputOperand(0)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 2)) {
      sizeM = lhsShape[i];
      break;
    }
  }
  for (unsigned i = 0; i < rhsShape.size(); i++) {
    if (op.getMatchingIndexingMap(op.getInputOperand(1)).getDimPosition(i) ==
        outputMap.getDimPosition(outputMap.getNumResults() - 1)) {
      sizeN = rhsShape[i];
      break;
    }
  }
  SmallVector<unsigned> exprs;
  op.getReductionDims(exprs);
  if (exprs.size() == 1) {
    for (unsigned i = 0; i < lhsShape.size(); i++) {
      if (op.getMatchingIndexingMap(op.getInputOperand(0)).getDimPosition(i) ==
          exprs[0]) {
        sizeK = lhsShape[i];
        break;
      }
    }
  }
  bool isStaticSize = sizeM != ShapedType::kDynamicSize &&
                      sizeN != ShapedType::kDynamicSize &&
                      sizeK != ShapedType::kDynamicSize;
  if (isStaticSize) {
    /// Try tensorcore config first.
    if (supportsTensorCore(entryPoint, op)) {
      SmallVector<TileWorkgroupSizePair> TCtileSizeConfig;

      getTensorCoreConfig(TCtileSizeConfig, op.getInputOperand(0)
                                                ->get()
                                                .getType()
                                                .cast<RankedTensorType>()
                                                .getElementType()
                                                .isF16());
      // Pick the best configuration where the original shape is aligned on the
      // tile size.
      for (TileWorkgroupSizePair &config : TCtileSizeConfig) {
        if (sizeK % config.tileSize[2] == 0 &&
            sizeN % config.tileSize[1] == 0 &&
            sizeM % config.tileSize[0] == 0) {
          return setMatmulConfig(
              config.tileSize[0], config.tileSize[1], config.tileSize[2],
              config.workgroupSize,
              sizeK == config.tileSize[2] ? 1 : softwarePipelineDepthTensorCore,
              IREE::Codegen::DispatchLoweringPassPipeline::
                  LLVMGPUMatmulTensorCore);
        }
      }
    }
    // Special case for very small matrices.
    if (sizeM * sizeN <= cudaWarpSize) {
      return setMatmulConfig(
          sizeN, sizeM, 4, {sizeM, sizeN, 1}, softwarePipelineDepthSimt,
          IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
    }
    // simt matmul case
    SmallVector<TileWorkgroupSizePair> tileSizeConfig;
    // Query the best configuration.
    getMatmulConfig(tileSizeConfig);
    // Pick the best configuration where the original shape is aligned on the
    // tile size.
    for (TileWorkgroupSizePair &config : tileSizeConfig) {
      if (sizeN % config.tileSize[1] == 0 && sizeM % config.tileSize[0] == 0) {
        return setMatmulConfig(
            config.tileSize[0], config.tileSize[1], config.tileSize[2],
            config.workgroupSize, softwarePipelineDepthSimt,
            IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
      }
    }
  }
  // If we haven't found any config, use the best tile size hoping that
  // the workgroup specialization handles the main tile path efficiently.
  SmallVector<TileWorkgroupSizePair> tileSizeConfig;
  // Query the best configuration.
  getMatmulConfig(tileSizeConfig);
  constexpr size_t configIndex = 0;
  const TileWorkgroupSizePair &config = tileSizeConfig[configIndex];
  const int64_t tileX = config.tileSize[0];
  const int64_t tileY = config.tileSize[1];
  const int64_t tileK = config.tileSize[2];
  const std::array<int64_t, 3> workgroupSize{config.workgroupSize[0],
                                             config.workgroupSize[1],
                                             config.workgroupSize[2]};
  return setMatmulConfig(
      tileX, tileY, tileK, workgroupSize, softwarePipelineDepthSimt,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUMatmulSimt);
}

static LogicalResult setFftConfig(func::FuncOp entryPoint,
                                  IREE::LinalgExt::FftOp op) {
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  unsigned loopDepth = partitionedLoops.back() + 1;
  SmallVector<int64_t> workgroupTileSize(loopDepth, 0);
  SmallVector<int64_t, 3> workgroupSize = {cudaWarpSize, 1, 1};

  // Tiling along partitioned loops with size 1.
  for (int64_t loopIndex : partitionedLoops) {
    workgroupTileSize[loopIndex] = 1;
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
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

static LogicalResult setSortConfig(func::FuncOp entryPoint, Operation *op) {
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(
        entryPoint, op, tileSizes,
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
        {1, 1, 1});
  }
  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  // Tile to have one element per thread.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0];
      break;
    }
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      workgroupSize);
}

// Basic default properties for linalg ops that haven't been tuned.
static LogicalResult setRootDefaultConfig(func::FuncOp entryPoint,
                                          Operation *op) {
  IREE::Codegen::DispatchLoweringPassPipeline passPipeline =
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute;
  TileSizesListType tileSizes;
  auto interfaceOp = cast<PartitionableLoopsInterface>(*op);
  auto partitionedLoops =
      interfaceOp.getPartitionableLoops(kNumMaxParallelDims);
  if (partitionedLoops.empty()) {
    tileSizes.push_back({});
    return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                                 passPipeline, {1, 1, 1});
  }

  size_t numLoops = partitionedLoops.back() + 1;
  // To get peak occupancy we need a workgroup size of at least two warps
  std::array<int64_t, 3> workgroupSize = {2 * cudaWarpSize, 1, 1};
  unsigned vectorSize = 4;
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  // Set all non-parallel loops to zero tile size.
  llvm::DenseSet<unsigned> partitionedLoopsSet(partitionedLoops.begin(),
                                               partitionedLoops.end());
  for (auto depth : llvm::seq<int64_t>(0, numLoops)) {
    if (!partitionedLoopsSet.count(depth)) {
      workgroupTileSizes[depth] = 0;
    }
  }

  if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
    for (auto outputOperand : enumerate(genericOp.getOutputOperands())) {
      if (!genericOp.getMatchingIndexingMap(outputOperand.value())
               .isProjectedPermutation()) {
        vectorSize = 1;
        break;
      }
      ArrayRef<int64_t> shape = cast<linalg::LinalgOp>(op)
                                    .getOutputOperand(outputOperand.index())
                                    ->get()
                                    .getType()
                                    .cast<ShapedType>()
                                    .getShape();
      if (llvm::any_of(shape, ShapedType::isDynamic)) {
        vectorSize = 1;
        break;
      }
      // Since we vectorize along the most inner dimension, make sure if can be
      // dividied by number of threads * vectorSize.
      while (vectorSize > 1 &&
             shape.back() % (workgroupSize[0] * vectorSize) != 0) {
        vectorSize /= 2;
      }
      int64_t problemSize = std::accumulate(
          shape.begin(), shape.end(), 1,
          [](const int64_t &a, const int64_t &b) { return a * b; });
      if ((problemSize / (cudaWarpSize * vectorSize)) < 64) {
        vectorSize = 1;
        break;
      }
    }
  }

  auto linalgOp = dyn_cast<linalg::LinalgOp>(op);
  // Pick a vectorSize of 1 for op that we know won't get vectorizedd.
  // Also skip vectorization for linalg on memref (no result) as the pipeline
  // relies on tensor level tiling.
  // TODO(thomasraoux): This could be improved by checking if the linalg op
  // would fail vectorization.
  if (!linalgOp || op->getNumResults() != 1 ||
      llvm::any_of(linalgOp.getIndexingMapsArray(),
                   [](AffineMap m) { return !m.isProjectedPermutation(); })) {
    vectorSize = 1;
  } else {
    passPipeline =
        IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  }

  // Set the inner most parallel loop to `lowerTs`.
  for (int64_t depth = numLoops; depth > 0; depth--) {
    if (partitionedLoopsSet.count(depth - 1)) {
      workgroupTileSizes[depth - 1] = workgroupSize[0] * vectorSize;
      break;
    }
  }
  if (linalgOp) {
    // Tile reduction dimension to 4 to allow doing load4 if the reduction size
    // is the most inner dimension.
    workgroupTileSizes.append(linalgOp.getNumReductionLoops(), 4);
  }
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  return setOpConfigAndEntryPointFnTranslation(entryPoint, op, tileSizes,
                                               passPipeline, workgroupSize);
}

/// Return the size of the given dimension in the linalg op.
// TODO: this should be part of LinalgOp interface, the equivalent member
// function currently only support the case where all the dimensions are static
// while we want to support dynamic shapes.
static Optional<int64_t> getLinalgDimSize(linalg::LinalgOp op, int64_t d) {
  for (auto map : llvm::enumerate(op.getIndexingMapsArray())) {
    for (auto dim : llvm::enumerate(map.value().getResults())) {
      auto expr = dim.value().dyn_cast<AffineDimExpr>();
      if (expr && expr.getPosition() == d) {
        auto type = op->getOperand(map.index()).getType().cast<ShapedType>();
        if (type.isDynamicDim(dim.index())) return llvm::None;
        return type.getDimSize(dim.index());
      }
    }
  }
  return llvm::None;
}

// Check if the given function contains an op that may require a broadcast of
// the reduced result.
static bool isFusedWithBroadcast(func::FuncOp entryPoint,
                                 linalg::LinalgOp reduce) {
  int64_t reducedRank =
      reduce->getResult(0).getType().cast<ShapedType>().getRank();
  bool hasBroadcast = false;
  entryPoint.walk([&](linalg::LinalgOp linalgOp) {
    if (reduce != linalgOp && linalgOp.getNumLoops() > reducedRank)
      hasBroadcast = true;
  });
  return hasBroadcast;
}

/// Set the configuration for reductions that can be mapped to warp reductions.
static LogicalResult setWarpReductionConfig(func::FuncOp entryPoint,
                                            linalg::LinalgOp op) {
  if (!isCudaTarget(entryPoint)) return failure();
  if (!isa<linalg::GenericOp>(op)) return failure();
  // TODO(thomasraoux): Enable dynamic shape.
  if (op.hasDynamicShape()) return failure();
  SmallVector<unsigned> reductionDims;
  op.getReductionDims(reductionDims);
  if (reductionDims.size() != 1 || reductionDims[0] != op.getNumLoops() - 1)
    return failure();
  if (op.getRegionOutputArgs().size() != 1) return failure();

  // Only support projected permutation, this could be extended to projected
  // permutated with broadcast.
  if (llvm::any_of(op.getInputOperands(), [&](OpOperand *input) {
        return !op.getMatchingIndexingMap(input).isProjectedPermutation();
      }))
    return failure();

  // Only single combiner operations are supported for now.
  SmallVector<Operation *, 4> combinerOps;
  if (!matchReduction(op.getRegionOutputArgs(), 0, combinerOps) ||
      combinerOps.size() != 1)
    return failure();
  Optional<int64_t> dimSize = getLinalgDimSize(op, reductionDims[0]);
  if (!dimSize || *dimSize % cudaWarpSize != 0) return failure();

  const Type elementType = op.getOutputOperand(0)
                               ->get()
                               .getType()
                               .cast<ShapedType>()
                               .getElementType();
  if (!elementType.isIntOrFloat()) return failure();
  // Reduction distribution only supports 32-bit types now.
  if (elementType.getIntOrFloatBitWidth() != 32) return failure();

  unsigned vectorSize = 4;
  while ((*dimSize / vectorSize) % cudaWarpSize != 0) vectorSize /= 2;

  // TODO: Add reduction tiling to handle larger reductions.
  const int64_t maxWorkgroupSize = 1024;
  int64_t groupSize = *dimSize / vectorSize;
  if (groupSize > maxWorkgroupSize) {
    groupSize = llvm::APIntOps::GreatestCommonDivisor(
                    {64, uint64_t(groupSize)}, {64, uint64_t(maxWorkgroupSize)})
                    .getZExtValue();
    // Workaround, the vector distribution doesn't handle cases where we fuse
    // the reduction with a consumer that needs to be tiled.
    // TODO(thomasraoux): remove the restriction once vector distribution is
    // improved.
    if (isFusedWithBroadcast(entryPoint, op)) return failure();

    // Current warp reduction pattern is a two step butterfly warp reduce.
    // First, do warp reductions along multiple warps.
    // Second, reduce results from multiple subgroups using single warp reduce.
    // The final warp reduce requires numWarp > cudaWarpSize to work.
    // TODO: Add flexible num steps of warp reduce to handle more tile/dist.
    if (groupSize / cudaWarpSize > cudaWarpSize) {
      return failure();
    }
  }
  std::array<int64_t, 3> workgroupSize = {groupSize, 1, 1};
  SmallVector<unsigned> partitionedLoops =
      cast<PartitionableLoopsInterface>(op.getOperation())
          .getPartitionableLoops(kNumMaxParallelDims);
  size_t numLoops = partitionedLoops.empty() ? 0 : partitionedLoops.back() + 1;
  // Tile all the parallel dimension to 1.
  SmallVector<int64_t, 4> workgroupTileSizes(numLoops, 1);
  SmallVector<int64_t, 4> reductionTileSizes(numLoops, 0);
  reductionTileSizes.push_back(groupSize * vectorSize);
  TileSizesListType tileSizes;
  tileSizes.emplace_back(std::move(workgroupTileSizes));  // Workgroup level
  tileSizes.emplace_back(std::move(reductionTileSizes));  // reduction level
  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, op, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUWarpReduction,
      workgroupSize);
  return success();
}

static bool hasTwoOrThreeLoopsInfo(linalg::LinalgOp linalgOp) {
  return linalgOp.getNumParallelLoops() >= 2 &&
         linalgOp.getNumParallelLoops() <= 3;
}

static LogicalResult setTransposeConfig(func::FuncOp entryPoint,
                                        linalg::LinalgOp linalgOp) {
  LinalgOpInfo opInfo(linalgOp, sharedMemTransposeFilter);

  // Checks preconditions for shared mem transpose.
  if (!opInfo.isTranspose() || opInfo.isDynamic() || opInfo.isReduction() ||
      !isa<linalg::GenericOp>(linalgOp) || !hasTwoOrThreeLoopsInfo(linalgOp)) {
    return failure();
  }

  ArrayRef<OpOperand *> transposedOperands = opInfo.getTransposeOperands();

  // Determine the fastest moving dimensions for the source/destination indices
  // of each transpose. These inform the tile sizes.
  int64_t outputFastestDim = linalgOp.getNumLoops() - 1;
  int64_t inputFastestDim =
      linalgOp.getMatchingIndexingMap(transposedOperands[0])
          .getDimPosition(outputFastestDim);
  // Ensure the other transposed operands match
  for (int i = 1; i < transposedOperands.size(); ++i) {
    if (inputFastestDim !=
        linalgOp.getMatchingIndexingMap(transposedOperands[i])
            .getDimPosition(outputFastestDim)) {
      return failure();
    }
  }

  int32_t tileM = 32;
  int32_t tileN = 32;
  TileSizesListType tileSizes;
  // Set all tile sizes to 1 except for fastest moving dimensions.
  SmallVector<int64_t> tileSizesTemp(linalgOp.getNumLoops(), 1);
  tileSizesTemp[outputFastestDim] = 32;
  tileSizesTemp[inputFastestDim] = 32;
  tileSizes.push_back(tileSizesTemp);

  // Check alignment with tile size for each transpose. Only the fastest moving
  // dims need to match the transpose tile.
  auto loopRanges = linalgOp.getStaticLoopRanges();
  if (loopRanges[outputFastestDim] % tileM != 0 ||
      loopRanges[inputFastestDim] % tileN != 0) {
    return failure();
  }

  // Workgroup size contains 8 warps. Configured with 8 threads on fastest
  // moving dimension so each thread can execute a vectorized copy of 4
  // contigious elements at a time from the 32 block.
  std::array<int64_t, 3> workgroupSize = {8, 32, 1};

  return setOpConfigAndEntryPointFnTranslation(
      entryPoint, linalgOp, tileSizes,
      IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUTransposeSharedMem,
      workgroupSize);
}

/// Decides the tiling and distribution parameters for one convolution
/// dimension. Returns true if we can succesfully deduce.
///
/// - `inputDim` is the size of the dimension to be distributed.
/// - `residualThreads` is the remaining threads we can distribute.
/// - `residualTilingFactor` indicates the remaining tiling scale factor.
/// - `wgDimSize` will be updated with the decided workgroup dimension size.
/// - `wgTileSize` will be updated with the decided workgroup tile size.
/// - `invoTileSize` will be updated with the decided invocation tile size.
static bool distributeToOneDim(const int64_t inputDim,
                               const bool isInnerMostDim,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               int64_t &wgDimSize, int64_t &wgTileSize) {
  const int64_t lb = isInnerMostDim ? 2 : 1;
  for (int64_t dim = residualThreads; dim >= lb; dim >>= 1) {
    int64_t chosenTileSize = 0;
    if (isInnerMostDim) {
      // Handle 4 elements per thread for the innermost dimension. We need
      // this for vectorized load.
      chosenTileSize = 4;
      if (inputDim % (dim * chosenTileSize) != 0) continue;
    } else {
      for (int64_t t = residualTilingFactor; t >= 1; t >>= 1)
        if (inputDim % (dim * t) == 0) {
          chosenTileSize = t;
          break;
        }
    }
    if (chosenTileSize) {
      wgDimSize = dim;
      wgTileSize = dim * chosenTileSize;
      residualThreads /= dim;
      residualTilingFactor /= chosenTileSize;
      return true;
    }
  }
  return false;
};

/// Decides the tiling and distribution parameters for two convolution window
/// dimensions to two workgroup dimensions as a square. Returns true if we can
/// succesfully deduce.
static bool distributeToSquare(const int64_t oh, const int64_t ow,
                               int64_t &residualThreads,
                               int64_t &residualTilingFactor,
                               MutableArrayRef<int64_t> wgDimSizes,
                               MutableArrayRef<int64_t> wgTileSizes) {
  assert(wgDimSizes.size() == 2 && wgTileSizes.size() == 2);

  const unsigned log2Threads = llvm::Log2_64(residualThreads);
  if (oh == ow && residualThreads != 1 && log2Threads % 2 == 0) {
    const int64_t yz = 1ll << (log2Threads / 2);

    int64_t chosenTileSize = 1ll << (llvm::Log2_64(residualTilingFactor) / 2);
    while (chosenTileSize >= 1 && ow % (yz * chosenTileSize) != 0) {
      chosenTileSize >>= 1;
    }

    if (chosenTileSize != 0) {
      wgDimSizes.front() = wgDimSizes.back() = yz;
      wgTileSizes.front() = wgTileSizes.back() = yz * chosenTileSize;
      return true;
    }
  }
  return false;
}

static LogicalResult setConvolutionConfig(linalg::LinalgOp linalgOp,
                                          const int64_t subgroupSize,
                                          const int64_t bestTilingFactor) {
  if (!isa<linalg::Conv2DNhwcHwcfOp, linalg::Conv2DNchwFchwOp>(linalgOp)) {
    return failure();
  }
  const bool isNCHW = isa<linalg::Conv2DNchwFchwOp>(*linalgOp);
  const bool isNHWC = isa<linalg::Conv2DNhwcHwcfOp>(*linalgOp);

  const int ohIndex = isNHWC ? 1 : 2;
  const int owIndex = isNHWC ? 2 : 3;
  const int ocIndex = isNHWC ? 3 : 1;

  Type inputType = linalgOp.getInputOperand(0)->get().getType();
  ArrayRef<int64_t> inputShape = inputType.cast<ShapedType>().getShape();
  Type outputType = linalgOp.getOutputOperand(0)->get().getType();
  ArrayRef<int64_t> outputShape = outputType.cast<ShapedType>().getShape();
  if (ShapedType::isDynamic(inputShape[3]) ||
      llvm::any_of(outputShape.drop_front(), ShapedType::isDynamic)) {
    return failure();
  }
  int64_t oh = outputShape[ohIndex], ow = outputShape[owIndex],
          oc = outputShape[ocIndex];

  // The core idea is to distribute the convolution dimensions to the workgroup
  // Z/Y/X dimensions, with each thread in a workgroup handling multiple vector
  // elements. We try to 1) utilize all threads in a subgroup, and 2) handle an
  // optimal tile size along each dimension.
  int64_t residualThreads = subgroupSize;
  int64_t residualTilingFactor = bestTilingFactor;

  SmallVector<int64_t, 3> workgroupSize(3, 1);  // (X, Y, Z)
  SmallVector<int64_t> workgroupTileSizes(4, 0);

  if (isNCHW) {
    // OW -> x, OH -> y, OC -> z
    if (!distributeToOneDim(ow, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]) ||
        !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[1],
                            workgroupTileSizes[2]) ||
        !distributeToOneDim(oc, /*isInnerMostDim=*/false, residualThreads,
                            residualTilingFactor, workgroupSize[2],
                            workgroupTileSizes[1])) {
      return failure();
    }
  } else {
    // OC -> x
    if (!distributeToOneDim(oc, /*isInnerMostDim=*/true, residualThreads,
                            residualTilingFactor, workgroupSize[0],
                            workgroupTileSizes[3]))
      return failure();

    // Deduce the configruation for the OW and OH dimension. Try to make them
    // even if possible given we typically have images with the same height
    // and width.
    const bool tileToSquare = distributeToSquare(
        oh, ow, residualThreads, residualTilingFactor,
        llvm::makeMutableArrayRef(workgroupSize).drop_front(),
        llvm::makeMutableArrayRef(workgroupTileSizes).drop_front().drop_back());

    // Otherwise treat OW and OH separately to allow them to have different
    // number of threads and tiling size.
    if (!tileToSquare) {
      if (!distributeToOneDim(ow, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[1],
                              workgroupTileSizes[2]) ||
          !distributeToOneDim(oh, /*isInnerMostDim=*/false, residualThreads,
                              residualTilingFactor, workgroupSize[2],
                              workgroupTileSizes[1])) {
        return failure();
      }
    }
  }
  auto pipeline = IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUVectorize;
  TileSizesListType tileSizes;
  // Add reduction tile sizes.
  if (isNCHW)
    workgroupTileSizes.append({4, 1, 1});
  else if (isNHWC)
    workgroupTileSizes.append({1, 1, 4});
  tileSizes.push_back(workgroupTileSizes);

  // Tile along OH by size 1 to enable downsizing 2-D convolution to 1-D.
  SmallVector<int64_t> windowTileSizes(4, 0);
  windowTileSizes[ohIndex] = 1;
  tileSizes.push_back(windowTileSizes);
  auto funcOp = linalgOp->getParentOfType<func::FuncOp>();
  return setOpConfigAndEntryPointFnTranslation(funcOp, linalgOp, tileSizes,
                                               pipeline, workgroupSize);
}

static LogicalResult setRootConfig(func::FuncOp entryPointFn,
                                   Operation *computeOp) {
  if (!clGPUCodegenTransformDialectTileSizes.empty()) {
    SmallVector<int64_t, 4> workgroupTileSizes(
        clGPUCodegenTransformDialectTileSizes.begin(),
        clGPUCodegenTransformDialectTileSizes.end());
    TileSizesListType tileSizes;
    tileSizes.emplace_back(std::move(workgroupTileSizes));
    auto config = IREE::Codegen::LoweringConfigAttr::get(
        computeOp->getContext(), tileSizes);
    setLoweringConfig(computeOp, config);
    return success();
  }
  if (IREE::Codegen::CompilationInfoAttr compilationInfo =
          getCompilationInfo(computeOp)) {
    // If the op already has a lowering config coming from the IR use this and
    // bypass the heuristic.
    return setUserConfig(entryPointFn, computeOp, compilationInfo);
  }
  if (auto linalgOp = dyn_cast<linalg::LinalgOp>(computeOp)) {
    if (linalg::isaContractionOpInterface(linalgOp) &&
        linalgOp.getNumParallelLoops() >= 2) {
      return setContractConfig(entryPointFn, linalgOp);
    }
    if (succeeded(setWarpReductionConfig(entryPointFn, linalgOp))) {
      return success();
    }
    if (succeeded(setConvolutionConfig(linalgOp, 32, 16))) {
      return success();
    }
    auto genericOp = dyn_cast<linalg::GenericOp>(computeOp);
    if (genericOp && succeeded(setTransposeConfig(entryPointFn, genericOp))) {
      return success();
    }
  }
  if (auto fftOp = dyn_cast<IREE::LinalgExt::FftOp>(computeOp)) {
    return setFftConfig(entryPointFn, fftOp);
  }
  if (auto sortOp = dyn_cast<IREE::LinalgExt::SortOp>(computeOp)) {
    return setSortConfig(entryPointFn, sortOp);
  }
  return setRootDefaultConfig(entryPointFn, computeOp);
}

namespace mlir {
namespace iree_compiler {

LogicalResult initGPULaunchConfig(ModuleOp moduleOp) {
  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    if (getTranslationInfo(exportOp)) continue;
    SmallVector<Operation *> computeOps;
    if (failed(getComputeOps(funcOp, computeOps))) {
      return funcOp.emitOpError("failed to get compute ops");
    }

    if (clGPUCodegenTransformDialectFileName.size() > 0) {
      auto translationInfo = IREE::Codegen::TranslationInfoAttr::get(
          moduleOp.getContext(), IREE::Codegen::DispatchLoweringPassPipeline::
                                     TransformDialectInterpreterCodegen);
      setTranslationInfo(funcOp, translationInfo);
      if (clGPUCodegenTransformDialectTileSizes.empty()) continue;
    }

    Operation *rootOperation = nullptr;
    // Find the root operation. linalg.generic and linalg.fill are not root
    // operations if there are other compute operations present.
    for (Operation *op : llvm::reverse(computeOps)) {
      if (!isa<linalg::GenericOp, linalg::FillOp>(op)) {
        rootOperation = op;
        break;
      }
      if (auto genericOp = dyn_cast<linalg::GenericOp>(op)) {
        // linalg.generic with `reduction` iterator types are roots as well.
        if (genericOp.getNumLoops() != genericOp.getNumParallelLoops()) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      for (Operation *op : llvm::reverse(computeOps)) {
        if (isa<linalg::GenericOp, linalg::FillOp>(op)) {
          rootOperation = op;
          break;
        }
      }
    }

    if (!rootOperation) {
      // setTranslationInfo(
      //    funcOp,
      //    IREE::Codegen::DispatchLoweringPassPipeline::LLVMGPUDistribute,
      //    {1, 1, 1});
      // continue;
      return funcOp.emitOpError("unable to find root operation");
    }
    if (failed(setRootConfig(funcOp, rootOperation))) continue;

    // Propogate the configuration to the other ops.
    // TODO(ravishankarm, thomasraoux): This is a very specific use (and
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
