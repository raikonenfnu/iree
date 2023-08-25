#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUHWConfig.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "LLVMGPUHWConfig.h"

namespace mlir::iree_compiler {

static LLVMGPULayout createLayout(MatrixType type,
        LLVMGPULayout::layoutType layout,
        DenseMap<uint32_t, SmallVector<Dimension>> &vectorMapping,
        LLVMGPULayout::ContractType contractType,
        std::function<Value(Value, Location, OpBuilder &)> encodeFn = nullptr,
        std::function<Value(Value, Location, OpBuilder &)> decodeFn = nullptr) {
  LLVMGPULayout newLayout(layout, vectorMapping);
  newLayout.contractType = contractType;
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

static bool isMatmulTransposeA(vector::ContractionOp contractOp) {
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
  return maps == infer({{k, m}, {k, n}, {m, n}});
}

bool AMDWMMAConfig::verifyContract(vector::ContractionOp contractOp) {
  return isMatmulTransposeA(contractOp);
}


LLVMGPULayout AMDWMMAConfig::createWMMAF16Layout(MatrixType matrixType, ArrayRef<int64_t> matrixShape) {
  uint32_t batchRow = matrixShape[0] / 16;
  uint32_t batchCol = matrixShape[1] / 16;
  LLVMGPULayout::layoutState colLayout, rowLayout;
  LLVMGPULayout::layoutType layout;
  LLVMGPULayout::ContractType contractType = LLVMGPULayout::ContractType::MTM;
  DenseMap<uint32_t, SmallVector<Dimension>> vectorMapping{
      {0, {Dim::BATCHY}}, {1, {Dim::BATCHX}}, {2, {Dim::VECTORX}}};
  // Layout is specified in reverse here, starting from batch.
  colLayout[Dim::BATCHX] = batchCol;
  rowLayout[Dim::BATCHY] = batchRow;
  if ((matrixType == MatrixType::A) || (matrixType == MatrixType::B)) {
    colLayout[Dim::LANEX] = 16;
    rowLayout[Dim::VECTORX] = 16;
    layout = {rowLayout, colLayout};
    return createLayout(matrixType, layout, vectorMapping, contractType);
  }
  colLayout[Dim::LANEX] = 16;
  rowLayout[Dim::LANEY] = 2;
  if (warpSize == 32) {
    rowLayout[Dim::VECTORX] = 8;
  } else {
    rowLayout[Dim::VECTORX] = 4;
  }
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
  return createLayout(matrixType, layout, vectorMapping, contractType, encodeFn, decodeFn);
}

}