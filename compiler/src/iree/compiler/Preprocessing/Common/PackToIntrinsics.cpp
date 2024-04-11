// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Preprocessing/Common/PassDetail.h"
#include "iree/compiler/Preprocessing/Common/Passes.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler::Preprocessing {

static void packContractionOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                              ArrayRef<GPUMatmulShapeType> intrinsics) {
  // TODO: Reorder dimension S.T lhs's index map will be as close to identity as
  // possible. this is done to ensure when we tile etc we don't get into a weird
  // vector.contract configuration. auto genericOp =
  // dyn_cast<linalg::GenericOp>(linalgOp.getOperation()); if (!genericOp)
  // {return; SetVector<unsigned> interchange; auto maps =
  // linalgOp.getIndexingMapsArray(); for (int i = 0; i <
  // maps[0].getNumResults(); i++) {
  //   llvm::outs()<<maps[0].getDimPosition(i)<<"\n";
  //   interchange.insert(maps[0].getDimPosition(i));
  // }
  // for (int j = 0; j < maps[1].getNumResults(); j++) {
  //   interchange.insert(maps[1].getDimPosition(j));
  // }
  // FailureOr<linalg::GenericOp> interchangeResult =
  // interchangeGenericOp(rewriter, genericOp,
  // SmallVector<unsigned>(interchange.begin(), interchange.end())); if
  // (failed(interchangeResult)) return; linalgOp = *interchangeResult;

  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);

  if (failed(contractionDims)) {
    llvm::outs() << "no contract!\n";
    return;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    llvm::outs() << "no mnk!\n";
    return;
  }

  // Naive handling by only looking into most inner dimensions.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // If none of the shape is dynamic, we'd fallback to using pad to intrinsics.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  if (!ShapedType::isDynamic(bounds[mDim]) &&
      !ShapedType::isDynamic(bounds[nDim]) &&
      !ShapedType::isDynamic(bounds[kDim])) {
    llvm::outs() << "no dyn!\n";
    return;
  }

  // Try to search for intrinsic with least amount of packing.
  SmallVector<std::pair<int64_t, int64_t>> dimsToPack;
  for (auto &intrinsic : intrinsics) {
    SmallVector<std::pair<int64_t, int64_t>> dimsToPackCandidates;
    if (bounds[mDim] % intrinsic.mSize != 0 ||
        ShapedType::isDynamic(bounds[mDim])) {
      dimsToPackCandidates.emplace_back(mDim, intrinsic.mSize);
    }

    if (bounds[nDim] % intrinsic.nSize != 0 ||
        ShapedType::isDynamic(bounds[nDim])) {
      dimsToPackCandidates.emplace_back(nDim, intrinsic.nSize);
    }

    if (bounds[kDim] % intrinsic.kSize != 0 ||
        ShapedType::isDynamic(bounds[kDim])) {
      dimsToPackCandidates.emplace_back(kDim, intrinsic.kSize);
    }
    if (dimsToPack.empty() || dimsToPackCandidates.size() < dimsToPack.size()) {
      dimsToPack = dimsToPackCandidates;
    }
  }

  // Cannot find intrinsic that matches.
  if (dimsToPack.empty()) {
    llvm::outs() << "no candidate!\n";
    return;
  }

  // Setting the pack-size of dimIdx using intrinsic size.
  SmallVector<OpFoldResult> adjustedPackedSizes(linalgOp.getNumLoops(),
                                                rewriter.getIndexAttr(0));
  for (auto [dimIdx, dimPackSize] : dimsToPack) {
    adjustedPackedSizes[dimIdx] = rewriter.getIndexAttr(dimPackSize);
  }

  // Calls helper function to pack operands, restructure to packed-linalgOp and
  // replace original linalgOp.
  FailureOr<linalg::PackResult> packResults =
      linalg::pack(rewriter, linalgOp, adjustedPackedSizes);
  if (failed(packResults)) {
    return;
  }
}

namespace {

class PackToIntrinsicsPass : public PackToIntrinsicsBase<PackToIntrinsicsPass> {
public:
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    auto funcOp = getOperation();
    ArrayAttr mmaKinds = nullptr;
    for (auto targetAttr :
         IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(funcOp)) {
      FailureOr<ArrayAttr> candidateMmaKinds =
          getSupportedMmaTypes(targetAttr.getConfiguration());
      if (succeeded(candidateMmaKinds)) {
        mmaKinds = *candidateMmaKinds;
        break;
      }
    }
    if (!mmaKinds)
      return;

    auto mmaAttrs =
        llvm::to_vector(mmaKinds.getAsRange<IREE::GPU::MmaInterfaceAttr>());
    SmallVector<GPUMatmulShapeType> intrinsics;
    intrinsics.reserve(mmaKinds.size());
    for (auto mma : mmaAttrs) {
      auto [mSize, nSize, kSize] = mma.getMNKShape();
      auto [aType, bType, cType] = mma.getABCElementTypes();
      intrinsics.emplace_back(mSize, nSize, kSize, aType, bType, cType);
    }

    SmallVector<linalg::LinalgOp> targetOps;
    funcOp->walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::BatchMatmulOp>(
              linalgOp.getOperation())) {
        targetOps.push_back(linalgOp);
      } else if (isa<linalg::GenericOp>(linalgOp) &&
                 succeeded(linalg::inferContractionDims(linalgOp))) {
        targetOps.push_back(linalgOp);
      }
    });

    IRRewriter rewriter(context);
    for (auto linalgOp : llvm::make_early_inc_range(targetOps)) {
      rewriter.setInsertionPoint(linalgOp);
      TypeSwitch<Operation *, void>(linalgOp.getOperation())
          .Case<linalg::BatchMatmulOp, linalg::GenericOp>([&](auto matmulOp) {
            packContractionOp(rewriter, linalgOp, intrinsics);
          })
          .Default([&](Operation *op) {});
    }
  }
};

} // namespace

std::unique_ptr<Pass> createPackToIntrinsicsPass() {
  return std::make_unique<PackToIntrinsicsPass>();
}

} // namespace mlir::iree_compiler::Preprocessing
