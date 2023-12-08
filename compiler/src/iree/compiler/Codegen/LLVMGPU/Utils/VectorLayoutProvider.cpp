// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "VectorLayoutProvider.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"

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

static SmallVector<int64_t> getCanonicalShape(int64_t M, int64_t N, int64_t K,
                                              ContractMatrixType matrixType,
                                              ContractType contractType) {
  SmallVector<int64_t> shape;
  switch (matrixType) {
  case ContractMatrixType::A:
    shape = contractType == ContractType::MTM ? SmallVector<int64_t>{K, M}
                                              : SmallVector<int64_t>{M, K};
    break;
  case ContractMatrixType::B:
    shape = contractType == ContractType::MMT ? SmallVector<int64_t>{N, K}
                                              : SmallVector<int64_t>{K, N};
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

LayoutAttr
AMDCDNAGPULayoutProvider::getCanonicalMFMALayout(TypedValue<VectorType> value,
                                                 ContractMatrixType matrixType,
                                                 int64_t numElements) {
  MLIRContext *ctx = root->getContext();

  // Get MMA matrix info.
  auto [M, N, K] = getCanonicalDims(mfmaType);
  ArrayRef<int64_t> matrixShape = value.getType().getShape();
  SmallVector<int64_t> canonicalShape =
      getCanonicalShape(M, N, K, matrixType, contractType);

  // Get batch sizes.
  int64_t batchRow = matrixShape[0] / canonicalShape[0];
  // If we load more elements, corresponds to smaller batch size.
  int64_t multiplier = numElements >= 4 ? numElements / 4 : 1;
  int64_t batchCol = matrixShape[1] / canonicalShape[1] / multiplier;

  auto rowLayout = [&](LayoutDimension batchDim,
                       int64_t batch) -> PerDimLayoutAttr {
    return createPerDimLayout(ctx, {batchDim, LayoutDimension::LANEX},
                              {batch, 16});
  };

  auto colLayout = [&](LayoutDimension batchDim, int64_t batch,
                       int64_t numElements) -> PerDimLayoutAttr {
    return createPerDimLayout(
        ctx, {batchDim, LayoutDimension::LANEY, LayoutDimension::VECTORX},
        {batch, 4, numElements});
  };

  // For MMT contract, the layout of B is same as layout of A.
  if (contractType == ContractType::MMT &&
      matrixType == ContractMatrixType::B) {
    matrixType = ContractMatrixType::A;
  }

  // Set layout based on matrix type.
  switch (matrixType) {
  case ContractMatrixType::A:
    return LayoutAttr::get(
        ctx, {rowLayout(LayoutDimension::BATCHX, batchRow),
              colLayout(LayoutDimension::BATCHY, batchCol, numElements)});
  case ContractMatrixType::B:
    return LayoutAttr::get(
        ctx, {colLayout(LayoutDimension::BATCHX, batchRow, numElements),
              rowLayout(LayoutDimension::BATCHY, batchCol)});
  default:
    return LayoutAttr::get(ctx,
                           {colLayout(LayoutDimension::BATCHX, batchRow, 4),
                            rowLayout(LayoutDimension::BATCHY, batchCol)});
  }
}

bool AMDCDNAGPULayoutProvider::hasCanonicalShape(ContractMatrixType matrixType,
                                                 ArrayRef<int64_t> shape) {
  SmallVector<int64_t> Ashape, Bshape, Cshape;
  switch (mfmaType) {
  case AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32:
    Ashape = Bshape = Cshape = {16, 16};
    break;
  default:
    return false;
  }

  switch (matrixType) {
  case ContractMatrixType::A:
    return Ashape == shape;
  case ContractMatrixType::B:
    return Bshape == shape;
  case ContractMatrixType::C:
  case ContractMatrixType::D:
    return Cshape == shape;
  default:
    return false;
  }
}
void AMDCDNAGPULayoutProvider::setAnchorOps() {
  root->walk([&](Operation *op) {
    TypeSwitch<Operation *, void>(op)
        .Case<vector::ContractionOp>([&](vector::ContractionOp contractOp) {
          LayoutAttr layoutA = getCanonicalMFMALayout(contractOp.getLhs(),
                                                      ContractMatrixType::A, 4);
          analysis.setAnchorForOperand(contractOp->getOpOperand(0), layoutA);
          LayoutAttr layoutB = getCanonicalMFMALayout(contractOp.getRhs(),
                                                      ContractMatrixType::B, 4);
          analysis.setAnchorForOperand(contractOp->getOpOperand(1), layoutB);
          // Do not allow scalar result.
          if (isa<VectorType>(contractOp.getAccType()) &&
              isa<VectorType>(contractOp.getResultType())) {
            auto acc = cast<TypedValue<VectorType>>(contractOp.getAcc());
            auto result = cast<TypedValue<VectorType>>(contractOp.getResult());
            LayoutAttr layoutC =
                getCanonicalMFMALayout(acc, ContractMatrixType::C, 1);
            analysis.setAnchorForOperand(contractOp->getOpOperand(2), layoutC);
            analysis.setAnchorForValue(result, layoutC);
          }
        })
        .Case<vector::TransferReadOp>([&](auto transferReadOp) {
          TypedValue<VectorType> vector = transferReadOp.getVector();
          for (Operation *user : vector.getUsers()) {
            if (auto contractOp = dyn_cast<vector::ContractionOp>(user)) {
              ArrayRef<int64_t> lhsShape =
                  cast<ShapedType>(contractOp.getLhs().getType()).getShape();
              ArrayRef<int64_t> rhsShape =
                  cast<ShapedType>(contractOp.getRhs().getType()).getShape();
              if (hasCanonicalShape(ContractMatrixType::A, lhsShape) ||
                  hasCanonicalShape(ContractMatrixType::B, rhsShape))
                return;
              if (vector == contractOp.getLhs()) {
                LayoutAttr layoutLhs =
                    getCanonicalMFMALayout(vector, ContractMatrixType::A, 8);
                analysis.setAnchorForValue(vector, layoutLhs);
              }
              if (vector == contractOp.getRhs()) {
                LayoutAttr layoutRhs =
                    getCanonicalMFMALayout(vector, ContractMatrixType::B, 8);
                analysis.setAnchorForValue(vector, layoutRhs);
              }
            }
          }
        })
        .Default([&](Operation *op) { return; });
  });
}

SmallVector<int64_t>
AMDCDNAGPULayoutProvider::getDistributedShape(TypedValue<VectorType> value) {
  auto layout = analysis.getLayout<LayoutAttr>(value);
  return layout.getSIMTVectorShape(simtLabels);
}

SmallVector<int64_t>
AMDCDNAGPULayoutProvider::getContractIndices(ContractMatrixType matrixType,
                                             int i, int j) {
  if (matrixType == ContractMatrixType::A) {
    switch (contractType) {
    case ContractType::MM:
    case ContractType::MMT:
      return SmallVector<int64_t>{i, j};
    case ContractType::MTM:
      return SmallVector<int64_t>{j, i};
    }
  }

  if (matrixType == ContractMatrixType::B) {
    switch (contractType) {
    case ContractType::MM:
    case ContractType::MTM:
      return SmallVector<int64_t>{i, j};
    case ContractType::MMT:
      return SmallVector<int64_t>{j, i};
    }
  }
  return SmallVector<int64_t>{i, j};
}

Value AMDCDNAGPULayoutProvider::computeMMA(Value a, Value b, Value c,
                                           Location loc, OpBuilder &rewriter) {
  uint32_t m, n, k, blks;
  if (mfmaType == AMDCDNAGPULayoutProvider::MFMAType::F16_16x16x16_F32) {
    m = n = k = 16;
  }
  blks = 1;
  return rewriter.create<amdgpu::MFMAOp>(loc, c.getType(), m, n, k, blks, a, b,
                                         c);
}

int64_t AMDCDNAGPULayoutProvider::getKDimension(int64_t rowBatch,
                                                int64_t colBatch) {
  if (contractType == ContractType::MTM) {
    return rowBatch;
  }
  return colBatch;
}

static LogicalResult
distributeContractionsToMFMA(RewriterBase &rewriter,
                             AMDCDNAGPULayoutProvider *provider,
                             vector::ContractionOp contractOp) {
  VectorLayoutAnalysis &analysis = provider->getAnalysis();
  TypedValue<VectorType> lhs = contractOp.getLhs();
  TypedValue<VectorType> rhs = contractOp.getRhs();
  Value accVal = contractOp.getAcc();
  if (!isa<VectorType>(accVal.getType()))
    return failure();
  TypedValue<VectorType> acc = cast<TypedValue<VectorType>>(accVal);
  Location loc = contractOp.getLoc();
  TypedValue<VectorType> result =
      cast<TypedValue<VectorType>>(contractOp.getResult());
  LayoutAttr layout = analysis.getLayout<LayoutAttr>(result);
  LayoutAttr lhsLayout = analysis.getLayout<LayoutAttr>(lhs);
  int K = provider->getKDimension(lhsLayout.getBatchDim(0),
                                  lhsLayout.getBatchDim(1));
  Type elementType = llvm::cast<ShapedType>(acc.getType()).getElementType();
  auto vectorType =
      VectorType::get(provider->getDistributedShape(result), elementType);
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  auto contractFn = [&](LayoutAttr::Iterator &iterator) {
    SmallVector<int64_t> simtIndices = layout.computeIteratorProjectedSIMTIndex(
        iterator, provider->getSIMTLabels(layout));
    Value dMatrix = rewriter.create<vector::ExtractOp>(
        loc, getDistributed(rewriter, acc, provider), simtIndices);
    for (int k = 0; k < K; k++) {
      Value aMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, lhs, provider),
          provider->getContractIndices(ContractMatrixType::A, simtIndices[0],
                                       k));
      Value bMatrix = rewriter.create<vector::ExtractOp>(
          loc, getDistributed(rewriter, rhs, provider),
          provider->getContractIndices(ContractMatrixType::B, k,
                                       simtIndices[1]));
      dMatrix = provider->computeMMA(aMatrix, bMatrix, dMatrix, loc, rewriter);
    }
    vector =
        rewriter.create<vector::InsertOp>(loc, dMatrix, vector, simtIndices);
  };
  LayoutAttr::Iterator iterator = layout.getBatchIterator();
  layout.map(contractFn, iterator);
  replaceOpWithDistributedValues(rewriter, contractOp, provider, vector);
  return success();
}

LogicalResult
AMDCDNAGPULayoutProvider::specializedDistribution(RewriterBase &rewriter,
                                                  Operation *op) {
  // Do specialized mfma distribution.
  if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
    return distributeContractionsToMFMA(rewriter, this, contractOp);
  }
  return failure();
}
