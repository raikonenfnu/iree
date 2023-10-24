#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUHWConfig.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "LLVMGPUHWConfig.h"

namespace mlir::iree_compiler {

static LLVMGPULayout createLayout(MatrixType type,
        LLVMGPULayout::layoutType layout,
        DenseMap<uint32_t, SmallVector<Dimension>> &vectorMapping,
        LLVMGPULayout::ContractType contractType,
        uint32_t maxTransferElements,
        std::function<Value(Value, Location, OpBuilder &)> encodeFn = nullptr,
        std::function<Value(Value, Location, OpBuilder &)> decodeFn = nullptr) {
  LLVMGPULayout newLayout(layout, vectorMapping);
  newLayout.contractType = contractType;
  newLayout.maxTransferElems = maxTransferElements;
  if (encodeFn) newLayout.encodeFn = encodeFn;
  if (decodeFn) newLayout.decodeFn = decodeFn;
  StringRef typeName;
  switch (type) {
    case MatrixType::A: typeName = "A"; break;
    case MatrixType::B: typeName = "B"; break;
    case MatrixType::C: typeName = "C"; break;
    case MatrixType::D: typeName = "D"; break;
  }
  newLayout.print(typeName);
  return newLayout;
}

SmallVector<int64_t> LLVMGPUHWConfig::getIndices(MatrixType matrixType, int i, int j) {
  if (matrixType == MatrixType::A) {
    switch (contractType) {
      case LLVMGPULayout::ContractType::MM:
      case LLVMGPULayout::ContractType::MMT:
        return SmallVector<int64_t>{i, j};
      case LLVMGPULayout::ContractType::MTM:
        return SmallVector<int64_t>{j, i};
    }
  }

  if (matrixType == MatrixType::B) {
    switch (contractType) {
      case LLVMGPULayout::ContractType::MM:
      case LLVMGPULayout::ContractType::MTM:
        return SmallVector<int64_t>{i, j};
      case LLVMGPULayout::ContractType::MMT:
        return SmallVector<int64_t>{j, i};
    }
  }
  return SmallVector<int64_t>{i, j};
}

LLVMGPULayout AMDWMMAConfig::getLayout(MatrixType matrixType, Value matrix) {
  auto type = llvm::cast<ShapedType>(matrix.getType());
  ArrayRef<int64_t> matrixShape = type.getShape();
  switch (wmmaType) {
    case WMMAType::F16_16X16X16_F16:
    case WMMAType::F32_16X16X16_F16:
        return createWMMAF16Layout(matrixType, matrixShape);
    default:
        return LLVMGPULayout();
  }
}

static bool hasF16Type(Value value) {
  ShapedType type = llvm::cast<ShapedType>(value.getType());
  return type.getElementType().isF16();
}

static bool hasF32Type(Value value) {
  ShapedType type = llvm::cast<ShapedType>(value.getType());
  return type.getElementType().isF32();
}

bool AMDWMMAConfig::verifyOperandTypes(Value a, Value b, Value c, Value d) {
  switch (wmmaType) {
    case WMMAType::F16_16X16X16_F16:
        return hasF16Type(a) && hasF16Type(b) && hasF16Type(c) && hasF16Type(d);
    case WMMAType::F32_16X16X16_F16:
        return hasF16Type(a) && hasF16Type(b) && hasF32Type(c) && hasF32Type(d);
  }
  return false;
}

bool LLVMGPUHWConfig::verifyContract(vector::ContractionOp contractOp) {
  // Set up the parallel/reduction structure in right form.
  using MapList = ArrayRef<ArrayRef<AffineExpr>>;
  auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
  AffineExpr m, n, k;
  bindDims(contractOp.getContext(), m, n, k);
  auto iteratorTypes = contractOp.getIteratorTypes().getValue();
  if (!(vector::isParallelIterator(iteratorTypes[0]) &&
        vector::isParallelIterator(iteratorTypes[1]) &&
        vector::isReductionIterator(iteratorTypes[2])))
    return false;
  SmallVector<AffineMap> maps = contractOp.getIndexingMapsArray();
  switch (contractType) {
    case LLVMGPULayout::ContractType::MTM:
      return maps == infer({{k, m}, {k, n}, {m, n}});
    case LLVMGPULayout::ContractType::MMT:
      return maps == infer({{m, k}, {n, k}, {m, n}});
    case LLVMGPULayout::ContractType::MM:
      return maps == infer({{m, k}, {k, n}, {m, n}});
    default:
      return false;
  }
}

LLVMGPULayout AMDWMMAConfig::createWMMAF16Layout(MatrixType matrixType, ArrayRef<int64_t> matrixShape) {
  uint32_t batchRow = matrixShape[0] / 16;
  uint32_t batchCol = matrixShape[1] / 16;
  LLVMGPULayout::layoutState colLayout, rowLayout;
  LLVMGPULayout::layoutType layout;
  DenseMap<uint32_t, SmallVector<Dimension>> vectorMapping{
      {0, {Dim::BATCHY}}, {1, {Dim::BATCHX}}, {2, {Dim::VECTORX}}};
  // Layout is specified in reverse here, starting from batch.
  colLayout[Dim::BATCHX] = batchCol;
  rowLayout[Dim::BATCHY] = batchRow;
  uint32_t maxTransferElements{1};
  if ((matrixType == MatrixType::A) || (matrixType == MatrixType::B)) {
    if (contractType == LLVMGPULayout::ContractType::MMT) {
      // B has a transposed layout, so should be the same as A
      if (matrixType == MatrixType::B) {
        matrixType = MatrixType::A;
      }
      // With mmt layout, we can transfer the maximum number of elements
      maxTransferElements = 8;
    }
    if (contractType == LLVMGPULayout::ContractType::MTM) {
      // A has a transposed layout, so should be the same as B
      if (matrixType == MatrixType::A) {
        matrixType = MatrixType::B;
      }
    }
    if (matrixType == MatrixType::A) {
      // Load A by row
      rowLayout[Dim::LANEX] = 16;
      colLayout[Dim::VECTORX] = 16;
    }
    if (matrixType == MatrixType::B) {
      // Load B by col
      colLayout[Dim::LANEX] = 16;
      rowLayout[Dim::VECTORX] = 16;
    }
    layout = {rowLayout, colLayout};
    return createLayout(matrixType, layout, vectorMapping, contractType, maxTransferElements);
  }
  // C and D have a different layout than A and B and require encode/decode functions
  colLayout[Dim::LANEX] = 16;
  if (warpSize == 32) {
    rowLayout[Dim::VECTORX] = 8;
  } else {
    rowLayout[Dim::VECTORX] = 4;
  }
  rowLayout[Dim::LANEY] = 2;
  layout = {rowLayout, colLayout};
  vectorMapping = {{0, {Dim::BATCHY}}, {1, {Dim::BATCHX}}, {2, {Dim::VECTORX}}};
  std::function<Value(Value, Location, OpBuilder &)> encodeFn, decodeFn;
  // Since only 8 values are produced, we need to broadcast to 16
  encodeFn = [&](Value vector, Location loc, OpBuilder &rewriter) {
    auto elementType = vector.getType().cast<VectorType>().getElementType();
    auto vectorType = VectorType::get({16}, elementType);
    Value result = rewriter.create<arith::ConstantOp>(loc, vectorType, rewriter.getZeroAttr(vectorType));
    for (int i = 0; i < 8; i++) {
      Value element = rewriter.create<vector::ExtractOp>(loc, vector, SmallVector<int64_t>{i});
      result = rewriter.create<vector::InsertOp>(loc, element, result, SmallVector<int64_t>{2 * i});
    }
    return result;
  };
  // We need to extract the correct 8 values from 16
  decodeFn = [&](Value vector, Location loc, OpBuilder &rewriter) {
    auto elementType = vector.getType().cast<VectorType>().getElementType();
    auto vectorType = VectorType::get({8}, elementType);
    Value result = rewriter.create<arith::ConstantOp>(loc, vectorType, rewriter.getZeroAttr(vectorType));
    for (int i = 0; i < 8; i++) {
      Value element = rewriter.create<vector::ExtractOp>(loc, vector, SmallVector<int64_t>{2 * i});
      result = rewriter.create<vector::InsertOp>(loc, element, result, SmallVector<int64_t>{i});
    }
    return result;
  };
  if (wmmaType == WMMAType::F32_16X16X16_F16) {
    encodeFn = decodeFn = nullptr;
  }
  // For C and D matrices, due to the non-unit stride we can only store one element at a time
  maxTransferElements = 1;
  return createLayout(matrixType, layout, vectorMapping, contractType, maxTransferElements, encodeFn, decodeFn);
}

Value AMDWMMAConfig::computeMMA(Value a, Value b, Value c, Location loc, OpBuilder &rewriter) {
  return rewriter.create<amdgpu::WMMAOp>(loc, c.getType(), a, b, c);
}

bool AMDMFMAConfig::verifyOperandTypes(Value a, Value b, Value c, Value d) {
  switch (mfmaType) {
    case MFMAType::F32_16X16X16_F16:
    case MFMAType::F32_32x32x8_F16:
        return hasF16Type(a) && hasF16Type(b) && hasF32Type(c) && hasF32Type(d);
  }
  return false;
}

LLVMGPULayout AMDMFMAConfig::getLayout(MatrixType matrixType, Value matrix) {
  auto type = llvm::cast<ShapedType>(matrix.getType());
  ArrayRef<int64_t> matrixShape = type.getShape();
  switch (mfmaType) {
    case MFMAType::F32_16X16X16_F16:
    case MFMAType::F32_32x32x8_F16:
        return createMFMALayout(matrixType, matrixShape);
    default:
        return LLVMGPULayout();
  }
}

LLVMGPULayout AMDMFMAConfig::getReadLayout(MatrixType matrixType,
                                           Value matrix) {
  auto type = llvm::cast<ShapedType>(matrix.getType());
  ArrayRef<int64_t> matrixShape = type.getShape();
  switch (mfmaType) {
    case MFMAType::F32_16X16X16_F16:
    case MFMAType::F32_32x32x8_F16:
        return createMFMALayout(matrixType, matrixShape, 2);
    default:
        return LLVMGPULayout();
  }
}

static std::tuple<uint32_t, uint32_t, uint32_t> getCanonicalDims(AMDMFMAConfig::MFMAType type) {
  switch (type) {
    case AMDMFMAConfig::MFMAType::F32_16X16X16_F16:
      return {16, 16, 16};
    case AMDMFMAConfig::MFMAType::F32_32x32x8_F16:
      return {32, 32, 8};
    default:
      return {0, 0, 0};
  }
}

static SmallVector<uint32_t> getCanonicalShape(uint32_t M, uint32_t N, uint32_t K, MatrixType matrixType,
                                               LLVMGPULayout::ContractType contractType) {
  SmallVector<uint32_t> shape;
  switch (matrixType) {
    case MatrixType::A:
      shape = contractType == LLVMGPULayout::ContractType::MTM ? SmallVector<uint32_t>{K, M}
                                                               : SmallVector<uint32_t>{M, K};
      break;
    case MatrixType::B:
      shape = contractType == LLVMGPULayout::ContractType::MMT ? SmallVector<uint32_t>{N, K}
                                                               : SmallVector<uint32_t>{K, N};
      break;
    default:
      shape = {M, N};
  }
  return shape;
}

// The multiplier allows for wider reads than what is possible just from the layout spec.
// Right now the multiplier is only applied to the A and B matrices.
LLVMGPULayout AMDMFMAConfig::createMFMALayout(MatrixType matrixType, ArrayRef<int64_t> matrixShape,
                                              int multiplier) {
  auto [M, N, K] = getCanonicalDims(mfmaType);
  SmallVector<uint32_t> canonicalShape = getCanonicalShape(M, N, K, matrixType, contractType);
  llvm::outs()<<"OG matrix shape"<<matrixShape[0]<<","<<matrixShape[1]<<"\n";
  llvm::outs()<<"canon matrix shape"<<canonicalShape[0]<<","<<canonicalShape[1]<<"\n\n";
  uint32_t batchRow = matrixShape[0] / canonicalShape[0];
  uint32_t batchCol = matrixShape[1] / canonicalShape[1];
  LLVMGPULayout::layoutType layout;
  DenseMap<uint32_t, SmallVector<Dimension>> vectorMapping{
      {0, {Dim::BATCHY}}, {1, {Dim::BATCHX}}, {2, {Dim::VECTORX}}};
  // Layout is specified in reverse here, starting from batch.
  uint32_t maxTransferElements{1};
  if (contractType == LLVMGPULayout::ContractType::MMT) {
    // B has a transposed layout, so should be the same as A
    if (matrixType == MatrixType::B) {
      matrixType = MatrixType::A;
    }
    if ((matrixType == MatrixType::A) || (matrixType == MatrixType::B)) {
      maxTransferElements = 4 * multiplier;
    }
  }
  LLVMGPULayout::layoutState colLayout, rowLayout;
  colLayout[Dim::BATCHX] = batchCol / multiplier;
  rowLayout[Dim::BATCHY] = batchRow;
  if (contractType == LLVMGPULayout::ContractType::MTM) {
    // A has a transposed layout, so should be the same as B
    if (matrixType == MatrixType::A) {
      matrixType = MatrixType::B;
    }
  }
  if (mfmaType == AMDMFMAConfig::MFMAType::F32_16X16X16_F16) {
    if (matrixType == MatrixType::A) {
      rowLayout[Dim::LANEX] = 16;
      colLayout[Dim::LANEY] = 4;
      colLayout[Dim::VECTORX] = maxTransferElements;
    } else if (matrixType == MatrixType::B) {
      colLayout[Dim::LANEX] = 16;
      rowLayout[Dim::LANEY] = 4;
      rowLayout[Dim::VECTORX] = maxTransferElements;
    } else {
      colLayout[Dim::LANEX] = 16;
      rowLayout[Dim::LANEY] = 4;
      rowLayout[Dim::VECTORX] = 4;
    }
  }
  if (mfmaType == AMDMFMAConfig::MFMAType::F32_32x32x8_F16) {
    if (matrixType == MatrixType::A) {
      rowLayout[Dim::LANEX] = 32;
      colLayout[Dim::LANEY] = 2;
      colLayout[Dim::VECTORX] = maxTransferElements;
    } else if (matrixType == MatrixType::B) {
      colLayout[Dim::LANEX] = 32;
      rowLayout[Dim::LANEY] = 2;
      rowLayout[Dim::VECTORX] = maxTransferElements;
    } else {
      colLayout[Dim::LANEX] = 32;
      rowLayout[Dim::VECTORY] = 4;
      rowLayout[Dim::LANEY] = 2;
      rowLayout[Dim::VECTORX] = 4;
      vectorMapping = {{0, {Dim::BATCHY}}, {1, {Dim::BATCHX}}, {2, {Dim::VECTORX, Dim::VECTORY}}};
    }
  }
  layout = {rowLayout, colLayout};
  return createLayout(matrixType, layout, vectorMapping, contractType, maxTransferElements);
}

Value AMDMFMAConfig::computeMMA(Value a, Value b, Value c, Location loc, OpBuilder &rewriter) {
  uint32_t m, n, k, blks;
  if (mfmaType == AMDMFMAConfig::MFMAType::F32_16X16X16_F16) {
    m = n = k = 16;
  } else if (mfmaType == AMDMFMAConfig::MFMAType::F32_32x32x8_F16) {
    m = n = 32;
    k = 8;
  }
  blks = 1;
  return rewriter.create<amdgpu::MFMAOp>(loc, c.getType(), m, n, k, blks, a, b, c);
}

}