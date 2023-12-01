// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"

using namespace mlir::iree_compiler::IREE::VectorExt;
using namespace mlir::iree_compiler;
using namespace mlir;

static std::tuple<uint32_t, uint32_t, uint32_t>
getCanonicalDims(AMDCDNAGPULayoutProvider::MFMAType type) {
  switch (type) {
  case AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32:
    return {16, 16, 16};
  default:
    return {0, 0, 0};
  }
}

static SmallVector<uint32_t> getCanonicalShape(uint32_t M, uint32_t N,
                                               uint32_t K,
                                               ContractMatrixType matrixType,
                                               ContractType contractType) {
  SmallVector<uint32_t> shape;
  switch (matrixType) {
  case ContractMatrixType::A:
    shape = contractType == ContractType::MTM ? SmallVector<uint32_t>{K, M}
                                              : SmallVector<uint32_t>{M, K};
    break;
  case ContractMatrixType::B:
    shape = contractType == ContractType::MMT ? SmallVector<uint32_t>{N, K}
                                              : SmallVector<uint32_t>{K, N};
    break;
  default:
    shape = {M, N};
  }
  return shape;
}

static PerDimLayoutAttr createPerDimLayout(MLIRContext *ctx,
                                           ArrayRef<LayoutDimension> dims,
                                           ArrayRef<int64_t> shapes) {
  SmallVector<LayoutDimensionAttr> dimAttrs;
  for (auto dim : dims)
    dimAttrs.push_back(LayoutDimensionAttr::get(ctx, dim));
  return PerDimLayoutAttr::get(ctx, dimAttrs, shapes);
}

void AMDCDNAGPULayoutProvider::setAnchorOps() {
  MLIRContext *ctx = root->getContext();
  root->walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      uint32_t M, N, K;
      std::tie(M, N, K) = getCanonicalDims(mfmaType);
      auto setLayout = [&](Value value, ContractMatrixType matrixType,
                           int64_t numElements) {
        ArrayRef<int64_t> matrixShape =
            cast<VectorType>(value.getType()).getShape();
        SmallVector<uint32_t> canonicalShape =
            getCanonicalShape(M, N, K, matrixType, contractType);
        uint32_t batchRow = matrixShape[0] / canonicalShape[0];
        uint32_t batchCol = matrixShape[1] / canonicalShape[1];
        PerDimLayoutAttr rowLayout = createPerDimLayout(
            ctx, {LayoutDimension::BATCHX, LayoutDimension::LANEX},
            {batchRow, 16});
        auto colLayout = [&](int64_t numElements) -> PerDimLayoutAttr {
          return createPerDimLayout(ctx,
                                    {LayoutDimension::BATCHY,
                                     LayoutDimension::LANEY,
                                     LayoutDimension::VECTORX},
                                    {batchCol, 4, numElements});
        };
        LayoutAttr layout;
        if (matrixType == ContractMatrixType::A) {
          layout = LayoutAttr::get(ctx, {rowLayout, colLayout(numElements)});
        } else if (matrixType == ContractMatrixType::B) {
          layout = LayoutAttr::get(ctx, {colLayout(numElements), rowLayout});
        } else {
          layout = LayoutAttr::get(ctx, {colLayout(4), rowLayout});
        }
        analysis.setAnchor(value, layout);
      };
      setLayout(contractOp.getLhs(), ContractMatrixType::A, 4);
      setLayout(contractOp.getRhs(), ContractMatrixType::B, 4);
      setLayout(contractOp.getAcc(), ContractMatrixType::C, 4);
      setLayout(contractOp.getResult(), ContractMatrixType::D, 4);
    }
  });
}

SmallVector<int64_t>
AMDCDNAGPULayoutProvider::getDistributedShape(TypedValue<VectorType> value) {
  SmallVector<LayoutDimension> dims{LayoutDimension::BATCHX,
                                    LayoutDimension::BATCHY,
                                    LayoutDimension::VECTORX};
  auto layout = analysis.getLayout<LayoutAttr>(value);
  return layout.getSIMTVectorShape(dims);
}

SmallVector<AffineMap>
AMDCDNAGPULayoutProvider::getDistributedIndex(TypedValue<VectorType> val,
                                              ArrayRef<int64_t> iterate) {
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(val);
  MLIRContext *ctx = val.getContext();

  auto constructIndex = [&](int64_t dim) {
    // (distributedIndex)[threadX, threadY, threadZ]
    AffineExpr simdIndex, threadX, threadY, threadZ;
    bindDims(val.getContext(), simdIndex);
    bindSymbols(val.getContext(), threadX, threadY, threadZ);

    AffineExpr index = getAffineConstantExpr(0, ctx);
    AffineExpr indexScale = getAffineConstantExpr(1, ctx);

    // Get the dim layout for this dim.
    PerDimLayoutAttr dimLayout = layout.getDimLayout(dim);
    for (auto [label, shape, it] :
         llvm::zip(dimLayout.getLabels(), dimLayout.getShapes(), iterate)) {
      switch (label.getValue()) {
      case LayoutDimension::LANEX:
        index = index + indexScale * threadX;
        break;
      case LayoutDimension::LANEY:
        index = index + indexScale * threadY;
        break;
      case LayoutDimension::LANEZ:
        index = index + indexScale * threadZ;
        break;
      default:
        index = index + indexScale * getAffineConstantExpr(it, ctx);
        break;
      };
      indexScale = indexScale * getAffineConstantExpr(shape, ctx);
    }

    return AffineMap::get(1, 3, index + simdIndex);
  };

  SmallVector<AffineMap> maps;
  for (int64_t dim = 0; dim < val.getType().getRank(); ++dim) {
    maps.push_back(constructIndex(dim));
  }
  return maps;
}
