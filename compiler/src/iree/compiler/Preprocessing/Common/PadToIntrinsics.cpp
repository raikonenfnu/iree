// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cstdint>
#include <limits>
#include "iree/compiler/Codegen/Common/GPU/GPUHeuristics.h"
#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/IR/LinalgInterfaces.h"
#include "mlir/Dialect/Tensor/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler::Preprocessing {

#define GEN_PASS_DEF_PADTOINTRINSICS
#include "iree/compiler/Preprocessing/Common/Passes.h.inc" // IWYU pragma: export

namespace {

static Value getPaddedValue(RewriterBase &rewriter, Location loc,
                            Value padSource, ArrayRef<int64_t> padding) {
  auto sourceType = cast<RankedTensorType>(padSource.getType());
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        return std::get<0>(it) + std::get<1>(it);
      });
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  auto high = llvm::map_to_vector(padding, [&](int64_t v) -> OpFoldResult {
    return rewriter.getIndexAttr(v);
  });
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, high, paddingValue);
  return paddedResult;
}

/// Helper struct to encode origin of dims in linalgOp.
struct OriginOperandsDim {
  Value operand;
  int64_t dim;
};

std::optional<int64_t>
getOperandDim(SmallVector<OriginOperandsDim> &dimOriginOperands,
              Value operand) {
  std::optional<int64_t> operandDim;
  for (auto &originOperand : dimOriginOperands) {
    if (originOperand.operand == operand)
      operandDim = originOperand.dim;
  }
  return operandDim;
}

SmallVector<OpFoldResult>
getOperandPadding(RewriterBase &rewriter, Value operand,
                  SmallVector<SmallVector<OriginOperandsDim>> &dimsToOperandMap,
                  int64_t mDim, int64_t nDim, int64_t kDim,
                  OpFoldResult mPadding, OpFoldResult nPadding,
                  OpFoldResult kPadding) {
  auto operandType = operand.getType().dyn_cast<RankedTensorType>();
  if (!operandType)
    return {};
  SmallVector<OpFoldResult> operandPadding(operandType.getRank(),
                                           rewriter.getIndexAttr(0));
  std::optional<int64_t> operandMdim =
      getOperandDim(dimsToOperandMap[mDim], operand);
  std::optional<int64_t> operandNdim =
      getOperandDim(dimsToOperandMap[nDim], operand);
  std::optional<int64_t> operandKdim =
      getOperandDim(dimsToOperandMap[kDim], operand);
  if (operandMdim)
    operandPadding[operandMdim.value()] = mPadding;
  if (operandNdim)
    operandPadding[operandNdim.value()] = nPadding;
  if (operandKdim)
    operandPadding[operandKdim.value()] = kPadding;
  return operandPadding;
}

static Value padValue(RewriterBase &rewriter, Location loc, Value padSource,
                      ArrayRef<OpFoldResult> padding) {
  auto sourceType = padSource.getType().cast<RankedTensorType>();
  ArrayRef<int64_t> sourceShape = sourceType.getShape();
  auto paddedShape =
      llvm::map_to_vector(llvm::zip_equal(sourceShape, padding), [](auto it) {
        std::optional<int64_t> padInt = getConstantIntValue(std::get<1>(it));
        if (ShapedType::isDynamic(std::get<0>(it)) || !padInt) {
          return ShapedType::kDynamic;
        }
        return std::get<0>(it) + padInt.value();
      });
  auto paddedResultType =
      RankedTensorType::get(paddedShape, sourceType.getElementType());
  Value paddingValue = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getZeroAttr(sourceType.getElementType()));
  SmallVector<OpFoldResult> low(padding.size(), rewriter.getIndexAttr(0));
  Value paddedResult = rewriter.create<tensor::PadOp>(
      loc, paddedResultType, padSource, low, padding, paddingValue);
  return paddedResult;
}

static Value expandValue(RewriterBase &rewriter, Location loc,
                         Value expandSource, AffineMap &operandMap,
                         DenseMap<int64_t, int64_t> &dimsToExpand) {
  SetVector<int64_t> operandDimsToExpand;
  DenseMap<int64_t, int64_t> operandDimToExpandSize;
  for (auto [dimToExpand, sizeToExpand] : dimsToExpand) {
    std::optional<int64_t> maybeDim = operandMap.getResultPosition(
        getAffineDimExpr(dimToExpand, operandMap.getContext()));
    if (maybeDim) {
      operandDimsToExpand.insert(maybeDim.value());
      operandDimToExpandSize[maybeDim.value()] = sizeToExpand;
    }
  }
  if (operandDimsToExpand.empty()) {
    return expandSource;
  }
  // Expanded shape
  auto operandType = expandSource.getType().cast<RankedTensorType>();
  auto lhsShape = operandType.getShape();
  SmallVector<ReassociationIndices> reassoc;
  SmallVector<int64_t> expandedShape;
  int64_t reassocOffset = 0;
  for (int i = 0; i < operandType.getRank(); i++) {
    if (operandDimsToExpand.contains(i)) {
      expandedShape.append({lhsShape[i], operandDimToExpandSize[i]});
      reassoc.push_back(
          ReassociationIndices{reassocOffset + i, reassocOffset + i + 1});
      ++reassocOffset;
    } else {
      expandedShape.push_back(lhsShape[i]);
      reassoc.push_back(ReassociationIndices{reassocOffset + i});
    }
  }
  return rewriter.create<tensor::ExpandShapeOp>(
      loc, RankedTensorType::Builder(operandType).setShape(expandedShape),
      expandSource, reassoc);
}

static void padConvOp(RewriterBase &rewriter, linalg::LinalgOp linalgOp,
                      ArrayRef<GPUMatmulShapeType> intrinsics) {
  if (!isa<linalg::ConvolutionOpInterface>(*linalgOp)) {
    return;
  }
  // TODO: Handle other variants.
  if (!isa<linalg::Conv2DNhwcHwcfOp>(linalgOp))
    return;

  // Check that conv has met conditions to go down mfma.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  FailureOr<mlir::linalg::ConvolutionDimensions> convolutionDims =
      mlir::linalg::inferConvolutionDims(linalgOp);
  assert(succeeded(convolutionDims) && "Could not infer contraction dims");

  if (convolutionDims->outputChannel.size() != 1 ||
      convolutionDims->inputChannel.size() != 1 ||
      convolutionDims->filterLoop.size() < 1 ||
      convolutionDims->outputImage.size() < 1 ||
      convolutionDims->depth.size() != 0) {
    return;
  }

  auto isAllOnesList = [](ArrayRef<int64_t> list) {
    return llvm::all_of(list, [](int64_t i) { return i == 1; });
  };

  // TODO: Support non-unit strides/dilations.
  if (!isAllOnesList(convolutionDims->strides) ||
      !isAllOnesList(convolutionDims->dilations)) {
    return;
  }

  int64_t mDim = convolutionDims->outputImage.back();
  int64_t nDim = convolutionDims->outputChannel.front();
  // TODO: Support NCHW convolutions. This is just a matmul_transpose_a,
  // however the distribution patterns currently do not support that variant.
  if (mDim > nDim) {
    return;
  }
  int64_t kDim = convolutionDims->inputChannel.front();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  // TODO: Generalize to other dimensions.
  // Try to search for pad value and check only filter dimension is blocked.
  SmallVector<std::array<int64_t, 3>> mnkPaddingCandidates;
  for (const GPUMatmulShapeType &intrinsic : intrinsics) {
    std::optional<int64_t> mPadding, nPadding, kPadding;
    auto getPadding = [](int64_t value, int64_t padTo) {
      return llvm::divideCeil(value, padTo) * padTo - value;
    };

    if (mSize % intrinsic.mSize != 0) {
      mPadding = getPadding(mSize, intrinsic.mSize);
    }

    if (nSize % intrinsic.nSize != 0) {
      nPadding = getPadding(nSize, intrinsic.nSize);
    }

    if (kSize % intrinsic.kSize != 0) {
      kPadding = getPadding(kSize, intrinsic.kSize);
    }

    if (!mPadding && !nPadding && !kPadding) {
      // Some intrinsic matches. Nothing to do.
      return;
    }
    mnkPaddingCandidates.push_back(
        {mPadding.value_or(0), nPadding.value_or(0), kPadding.value_or(0)});
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }

  std::array<int64_t, 3> mnkPadding = mnkPaddingCandidates.front();

  Value newInput = linalgOp.getDpsInputOperand(0)->get();
  Value newFilter = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  Location loc = linalgOp.getLoc();
  int64_t mPadding = mnkPadding[0];
  int64_t nPadding = mnkPadding[1];
  int64_t kPadding = mnkPadding[2];
  if (mPadding != 0 || kPadding != 0) {
    // For NHWC, the m-padding is for W and k-padding is for C
    newInput =
        getPaddedValue(rewriter, loc, newInput, {0, 0, mPadding, kPadding});
  }
  if (nPadding != 0 || kPadding != 0) {
    // For HWCF, the n-padding is for F and k-padding is for C
    newFilter =
        getPaddedValue(rewriter, loc, newFilter, {0, 0, kPadding, nPadding});
  }
  if (mPadding != 0 || nPadding != 0) {
    // For output, the m-padding is for W and k-padding is for F
    newOuts =
        getPaddedValue(rewriter, loc, newOuts, {0, 0, mPadding, nPadding});
  }

  linalg::LinalgOp paddedConv2dOp =
      mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                  ArrayRef<Value>{newInput, newFilter, newOuts});
  // Extract slice.
  IntegerAttr zero = rewriter.getI64IntegerAttr(0);
  IntegerAttr one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(4, zero);
  SmallVector<OpFoldResult> strides(4, one);
  auto resultType = cast<RankedTensorType>(linalgOp->getResult(0).getType());
  ArrayRef<int64_t> resultShape = resultType.getShape();
  SmallVector<OpFoldResult> sizes = {rewriter.getIndexAttr(resultShape[0]),
                                     rewriter.getIndexAttr(resultShape[1]),
                                     rewriter.getIndexAttr(resultShape[2]),
                                     rewriter.getIndexAttr(resultShape[3])};
  Value extracted = rewriter.createOrFold<tensor::ExtractSliceOp>(
      loc, paddedConv2dOp->getResults()[0], offsets, sizes, strides);
  rewriter.replaceOp(linalgOp, extracted);
}

static void padContractionLikeOp(RewriterBase &rewriter,
                                 linalg::LinalgOp linalgOp,
                                 ArrayRef<GPUMatmulShapeType> intrinsics) {
  FailureOr<mlir::linalg::ContractionDimensions> contractionDims =
      mlir::linalg::inferContractionDims(linalgOp);

  if (failed(contractionDims)) {
    return;
  }

  if (contractionDims->k.size() < 1 || contractionDims->m.size() < 1 ||
      contractionDims->n.size() < 1) {
    return;
  }
  Location loc = linalgOp.getLoc();

  // Naive handling by only looking into most inner dimensions.
  int64_t mDim = contractionDims->m.back();
  int64_t nDim = contractionDims->n.back();
  int64_t kDim = contractionDims->k.back();

  // If none of the shape is dynamic, we'd fallback to using pad to intrinsics.
  SmallVector<int64_t, 4> bounds = linalgOp.getStaticLoopRanges();
  int64_t mSize = bounds[mDim];
  int64_t nSize = bounds[nDim];
  int64_t kSize = bounds[kDim];

  // Bail out on matvec-like cases.
  if (mSize == 1 || nSize == 1) {
    return;
  }

  // Obtain mapping from dims in linalgOp to originating dimensions.
  SmallVector<SmallVector<OriginOperandsDim>> dimsToOperandMap(bounds.size(),
                                                               {});
  SmallVector<OpOperand *> initOperands = llvm::to_vector(llvm::map_range(
      linalgOp.getDpsInitsMutable(), [](OpOperand &o) { return &o; }));
  SmallVector<OpOperand *> inputOperands = linalgOp.getDpsInputOperands();
  for (int64_t targetDim : {mDim, nDim, kDim}) {
    for (auto operand :
         llvm::concat<OpOperand *>(inputOperands, initOperands)) {
      auto operandMap = linalgOp.getMatchingIndexingMap(operand);
      std::optional<unsigned> maybeDim = operandMap.getResultPosition(
          getAffineDimExpr(targetDim, operandMap.getContext()));
      if (maybeDim)
        dimsToOperandMap[targetDim].push_back(
            {operand->get(), maybeDim.value()});
    }
  }

  SmallVector<DenseMap<int64_t, int64_t>> dimsToExpandCandidates;
  SmallVector<std::array<OpFoldResult, 3>> mnkPaddingCandidates;
  AffineExpr s0, s1; // problemSize, intrinsicSize
  bindSymbols(rewriter.getContext(), s0, s1);
  AffineExpr padByExpr = (s0).ceilDiv(s1) * s1 - s0;
  auto getPadding = [&](OpFoldResult value, int64_t padTo) {
    return affine::makeComposedFoldedAffineApply(
        rewriter, loc, padByExpr, {value, rewriter.getIndexAttr(padTo)});
  };
  OpFoldResult zeroExpr = rewriter.getIndexAttr(0);
  for (auto &intrinsic : intrinsics) {
    std::optional<OpFoldResult> mPadding, nPadding, kPadding;
    DenseMap<int64_t, int64_t> dimsToExpandCandidate;
    if (mSize % intrinsic.mSize != 0 || ShapedType::isDynamic(mSize)) {
      OriginOperandsDim originMdimOperand = dimsToOperandMap[mDim].front();
      OpFoldResult mSizeExpr = rewriter.getIndexAttr(mSize);
      if (ShapedType::isDynamic(mSize)) {
        mSizeExpr = rewriter
                        .create<tensor::DimOp>(loc, originMdimOperand.operand,
                                               originMdimOperand.dim)
                        .getResult();
      }
      mPadding = getPadding(mSizeExpr, intrinsic.mSize);
      if (!getConstantIntValue(mPadding.value())) {
        dimsToExpandCandidate[mDim] = intrinsic.nSize;
      }
    }

    if (nSize % intrinsic.nSize != 0 || ShapedType::isDynamic(nSize)) {
      OriginOperandsDim originNdimOperand = dimsToOperandMap[nDim].front();
      OpFoldResult nSizeExpr = rewriter.getIndexAttr(nSize);
      if (ShapedType::isDynamic(nSize)) {
        nSizeExpr = rewriter
                        .create<tensor::DimOp>(loc, originNdimOperand.operand,
                                               originNdimOperand.dim)
                        .getResult();
      }
      nPadding = getPadding(nSizeExpr, intrinsic.nSize);
      if (!getConstantIntValue(nPadding.value())) {
        dimsToExpandCandidate[nDim] = intrinsic.nSize;
      }
    }

    if (kSize % intrinsic.kSize != 0 || ShapedType::isDynamic(kSize)) {
      OriginOperandsDim originKdimOperand = dimsToOperandMap[kDim].front();
      OpFoldResult kSizeExpr = rewriter.getIndexAttr(kSize);
      if (ShapedType::isDynamic(kSize)) {
        kSizeExpr = rewriter
                        .create<tensor::DimOp>(loc, originKdimOperand.operand,
                                               originKdimOperand.dim)
                        .getResult();
      }
      kPadding = getPadding(kSizeExpr, intrinsic.kSize);
      if (!getConstantIntValue(kPadding.value())) {
        dimsToExpandCandidate[kDim] = intrinsic.kSize;
      }
    }

    if (!mPadding && !nPadding && !kPadding) {
      return;
    }
    mnkPaddingCandidates.push_back({mPadding.value_or(zeroExpr),
                                    nPadding.value_or(zeroExpr),
                                    kPadding.value_or(zeroExpr)});
    dimsToExpandCandidates.push_back(dimsToExpandCandidate);
  }
  if (mnkPaddingCandidates.empty()) {
    return;
  }
  std::array<OpFoldResult, 3> mnkPadding = mnkPaddingCandidates.front();
  DenseMap<int64_t, int64_t> dimsToExpand = dimsToExpandCandidates.front();

  OpFoldResult mPadding = mnkPadding[0];
  OpFoldResult nPadding = mnkPadding[1];
  OpFoldResult kPadding = mnkPadding[2];

  Value newLhs = linalgOp.getDpsInputOperand(0)->get();
  Value newRhs = linalgOp.getDpsInputOperand(1)->get();
  Value newOuts = linalgOp.getDpsInitOperand(0)->get();

  auto lhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(0));
  auto rhsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInputOperand(1));
  auto outsMap = linalgOp.getMatchingIndexingMap(linalgOp.getDpsInitOperand(0));

  SmallVector<OpFoldResult> lhsPadding =
      getOperandPadding(rewriter, newLhs, dimsToOperandMap, mDim, nDim, kDim,
                        mPadding, nPadding, kPadding);
  SmallVector<OpFoldResult> rhsPadding =
      getOperandPadding(rewriter, newRhs, dimsToOperandMap, mDim, nDim, kDim,
                        mPadding, nPadding, kPadding);
  SmallVector<OpFoldResult> outsPadding =
      getOperandPadding(rewriter, newOuts, dimsToOperandMap, mDim, nDim, kDim,
                        mPadding, nPadding, kPadding);
  if (lhsPadding.empty() || rhsPadding.empty() || outsPadding.empty()) {
    return;
  }
  newLhs = padValue(rewriter, loc, newLhs, lhsPadding);
  newRhs = padValue(rewriter, loc, newRhs, rhsPadding);
  newOuts = padValue(rewriter, loc, newOuts, outsPadding);

  auto paddedMatmulOp = mlir::clone(rewriter, linalgOp, {newOuts.getType()},
                                    ArrayRef<Value>{newLhs, newRhs, newOuts});
  Value paddedCompute = paddedMatmulOp->getResults()[0];

  // Expand dimensions if there are dynamic shapes.
  if (!dimsToExpand.empty()) {
    // Generating expanded indexing maps and iterator types.
    SmallVector<AffineMap> expandedMaps = linalgOp.getIndexingMapsArray();
    SmallVector<AffineMap> originalIterators = linalgOp.getIndexingMapsArray();
    SmallVector<utils::IteratorType> expandedIterators =
        linalgOp.getIteratorTypesArray();
    int expandOffset = 0;
    SmallVector<int64_t> dimsToExpandVec = llvm::to_vector(
        llvm::map_range(dimsToExpand, [](auto &dim) { return dim.first; }));
    llvm::sort(dimsToExpandVec);
    for (auto [expandIdx, expandDim] : llvm::enumerate(dimsToExpandVec)) {
      // Creating iterator type for newly expanded/dst dim from it's expans
      // source dim.
      int64_t expandSrcDim = expandDim + expandOffset;
      expandedIterators.insert(expandedIterators.begin() + expandSrcDim,
                               expandedIterators[expandSrcDim]);
      // Updating map of each operand to handle newly expanded/dst dim
      // based on the location of it's expand source dim.
      for (int operandIdx = 0; operandIdx < expandedMaps.size(); operandIdx++) {
        AffineMap &map = expandedMaps[operandIdx];
        int64_t expandSrcDim = expandDim + expandOffset;
        int64_t expandDstDim = expandSrcDim + 1;
        map = map.shiftDims(1, expandDstDim);
        std::optional<int64_t> maybeDim = map.getResultPosition(
            getAffineDimExpr(expandSrcDim, map.getContext()));
        if (!maybeDim)
          continue;
        map = map.insertResult(getAffineDimExpr(expandDstDim, map.getContext()),
                               maybeDim.value() + 1);
      }
      expandOffset++;
    }
    // Propagate to expand to operands
    newLhs = expandValue(rewriter, loc, newLhs, lhsMap, dimsToExpand);
    newRhs = expandValue(rewriter, loc, newRhs, rhsMap, dimsToExpand);
    newOuts = expandValue(rewriter, loc, newOuts, outsMap, dimsToExpand);
    // Create expanded contractionOp.
    auto expandedMatmulOp = rewriter.create<linalg::GenericOp>(
        loc, newOuts.getType(), ValueRange{newLhs, newRhs}, ValueRange{newOuts},
        expandedMaps, expandedIterators);
    expandedMatmulOp.getRegion().takeBody(linalgOp->getRegion(0));
    paddedCompute = expandedMatmulOp.getResults()[0];

    // Collapse back to non expanded shape if required.
    if (auto expandOutsOp =
            dyn_cast<tensor::ExpandShapeOp>(newOuts.getDefiningOp())) {
      paddedCompute = rewriter.create<tensor::CollapseShapeOp>(
          loc, expandOutsOp.getSrcType(), paddedCompute,
          expandOutsOp.getReassociationIndices());
    }
  }

  // extract slice.
  auto resultType = linalgOp->getResult(0).getType().cast<RankedTensorType>();
  auto resultShape = resultType.getShape();
  auto resultRank = resultType.getRank();
  auto zero = rewriter.getI64IntegerAttr(0);
  auto one = rewriter.getI64IntegerAttr(1);
  SmallVector<OpFoldResult> offsets(resultRank, zero), strides(resultRank, one),
      sizes;
  for (auto [dimIdx, dimSize] : llvm::enumerate(resultShape)) {
    if (ShapedType::isDynamic(dimSize))
      sizes.push_back(rewriter
                          .create<tensor::DimOp>(
                              loc, linalgOp.getDpsInitOperand(0)->get(), dimIdx)
                          .getResult());
    else
      sizes.push_back(rewriter.getIndexAttr(dimSize));
  }
  rewriter.replaceOpWithNewOp<tensor::ExtractSliceOp>(linalgOp, paddedCompute,
                                                      offsets, sizes, strides);
}

struct PadToIntrinsicsPass final
    : impl::PadToIntrinsicsBase<PadToIntrinsicsPass> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    Operation *rootOp = getOperation();
    ArrayAttr mmaKinds = nullptr;
    for (IREE::HAL::ExecutableTargetAttr targetAttr :
         IREE::HAL::DeviceTargetAttr::lookupExecutableTargets(rootOp)) {
      FailureOr<ArrayAttr> candidateMmaKinds =
          getSupportedMmaTypes(targetAttr.getConfiguration());
      if (succeeded(candidateMmaKinds)) {
        mmaKinds = *candidateMmaKinds;
        break;
      }
    }
    if (!mmaKinds)
      return;

    auto intrinsics = llvm::map_to_vector(
        mmaKinds.getAsRange<IREE::GPU::MMAAttr>(), [](IREE::GPU::MMAAttr mma) {
          auto [mSize, nSize, kSize] = mma.getMNKShape();
          auto [aType, bType, cType] = mma.getABCElementTypes();
          return GPUMatmulShapeType{mSize, nSize, kSize, aType, bType, cType};
        });

    SmallVector<linalg::LinalgOp> targetOps;
    rootOp->walk([&](linalg::LinalgOp linalgOp) {
      if (isa<linalg::Conv2DNhwcHwcfOp, linalg::BatchMatmulOp,
              linalg::BatchMatmulTransposeBOp, linalg::MatmulOp,
              linalg::MatmulTransposeBOp, linalg::MatmulOp, linalg::GenericOp>(
              linalgOp.getOperation()))
        targetOps.push_back(linalgOp);
    });

    IRRewriter rewriter(context);
    for (auto linalgOp : llvm::make_early_inc_range(targetOps)) {
      rewriter.setInsertionPoint(linalgOp);
      TypeSwitch<Operation *, void>(linalgOp.getOperation())
          .Case<linalg::Conv2DNhwcHwcfOp>(
              [&](auto convOp) { padConvOp(rewriter, linalgOp, intrinsics); })
          .Case<linalg::BatchMatmulOp, linalg::BatchMatmulTransposeBOp,
                linalg::MatmulOp, linalg::MatmulTransposeBOp,
                linalg::GenericOp>([&](auto matmulOp) {
            padContractionLikeOp(rewriter, linalgOp, intrinsics);
          })
          .Default([&](Operation *op) {});
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler::Preprocessing
