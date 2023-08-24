#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPULayout.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir::iree_compiler {

using layoutMapType =
    llvm::DenseMap<Value, llvm::SmallVector<LLVMGPULayout, 2>>;

namespace {

static constexpr uint32_t kAMDWarpSize = 32;

static void
addToLayoutMap(StringRef type, Value value,
               const LLVMGPULayout::layoutType &layout,
               layoutMapType &layoutMap,
               DenseMap<uint32_t, SmallVector<Dimension>> &vectorMapping,
               LLVMGPULayout::ContractType contractType,
               std::function<Value(Value, Location, OpBuilder &)> encodeFn = nullptr,
               std::function<Value(Value, Location, OpBuilder &)> decodeFn = nullptr) {
  LLVMGPULayout newLayout(layout, vectorMapping);
  newLayout.contractType = contractType;
  if (encodeFn) newLayout.encodeFn = encodeFn;
  if (decodeFn) newLayout.decodeFn = decodeFn;
  if (layoutMap.contains(value)) {
    layoutMap[value].push_back(newLayout);
    return;
  }
  layoutMap[value] = {newLayout};
  newLayout.print(type);
}

static LLVMGPULayout &getLayout(layoutMapType &layoutMap, Value value) {
  assert(layoutMap.size() >= 1);
  // TODO: Figure out what to return when there are multiple values
  return layoutMap[value][0];
}

// Comes from here:
// https://www.amd.com/system/files/TechDocs/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf
// Currently only targets 16x16x16 f16.f16.f32 instruction.
static void createWMMALayout(Value matrix, StringRef type,
                             layoutMapType &layoutMap) {
  auto matrixType = llvm::cast<ShapedType>(matrix.getType());
  ArrayRef<int64_t> matrixShape = matrixType.getShape();
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
  if ((type == "aMatrix") || (type == "bMatrix")) {
    colLayout[Dim::LANEX] = 16;
    rowLayout[Dim::VECTORX] = 16;
    layout = {rowLayout, colLayout};
    addToLayoutMap(type, matrix, layout, layoutMap, vectorMapping, contractType);
    return;
  }
  colLayout[Dim::LANEX] = 16;
  rowLayout[Dim::LANEY] = 2;
  if (kAMDWarpSize == 32) {
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
  addToLayoutMap(type, matrix, layout, layoutMap, vectorMapping, contractType, encodeFn, decodeFn);
}

LogicalResult setAMDWMMALayouts(Value aMatrix, Value bMatrix, Value cMatrix,
                                Value dMatrix, layoutMapType &layoutMap) {
  createWMMALayout(aMatrix, "aMatrix", layoutMap);
  createWMMALayout(bMatrix, "bMatrix", layoutMap);
  createWMMALayout(cMatrix, "cMatrix", layoutMap);
  createWMMALayout(dMatrix, "dMatrix", layoutMap);
  return success();
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

static bool hasF16Type(Value value) {
  ShapedType type = dyn_cast<ShapedType>(value.getType());
  return type.getElementType().isF16();
}

LogicalResult setMMALayout(vector::ContractionOp contractOp,
                           layoutMapType &layoutMap, bool useAMDWMMA) {
  Value aMatrix = contractOp.getLhs();
  Value bMatrix = contractOp.getRhs();
  Value cMatrix = contractOp.getAcc();
  Value dMatrix = contractOp.getResult();
  if (!((hasF16Type(aMatrix)) && hasF16Type(bMatrix)))
    return failure();
  if (useAMDWMMA) {
    if (!isMatmulTransposeA(contractOp))
      return failure();
    return setAMDWMMALayouts(aMatrix, bMatrix, cMatrix, dMatrix, layoutMap);
  }
  return failure();
}

LogicalResult setLayouts(layoutMapType &layoutMap, func::FuncOp funcOp,
                         bool useAMDWMMA) {
  WalkResult result = funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (failed(setMMALayout(contractOp, layoutMap, useAMDWMMA)))
        return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult propagateLayouts(layoutMapType &layoutMap, func::FuncOp funcOp) {
  WalkResult result =
      funcOp.walk([&](Operation *op) { return WalkResult::advance(); });
  return result.wasInterrupted() ? failure() : success();
}

static SmallVector<Value> handlePermutations(SmallVector<Value> &indices,
                                             AffineMap permutationMap) {
  SmallVector<Value> newIndices{indices.begin(), indices.end()};
  int laneDim = 0;
  for (AffineExpr expr : permutationMap.getResults()) {
    auto dimExpr = expr.dyn_cast<AffineDimExpr>();
    if (!dimExpr)
      continue;
    unsigned pos = dimExpr.getPosition();
    newIndices[pos] = indices[laneDim++];
  }
  return newIndices;
}

// This function delinearizes thread-x into higher dimensions as required by the layout
static SmallVector<SmallVector<Value>> getLaneIds(LLVMGPULayout &layout,
                                                  Location loc, OpBuilder &rewriter) {
  Value threadX = rewriter.create<gpu::ThreadIdOp>(loc, gpu::Dimension::x);
  SmallVector<Dimension> laneOrder{Dim::LANEX, Dim::LANEY};
  SmallVector<SmallVector<Value>> laneIds;
  Value lastShape;
  for (auto id : laneOrder) {
    for (int i = 0; i < layout.layout.size(); i++) {
      laneIds.push_back({});
      for (auto [name, shape] : layout.layout[i]) {
        if (name != id) continue;
        if (name == Dim::LANEX) {
          Value shapeValue = rewriter.create<arith::ConstantIndexOp>(loc, shape);
          laneIds[i].push_back(rewriter.create<arith::RemUIOp>(loc, threadX, shapeValue));
          lastShape = shapeValue;
        }
        if (name == Dim::LANEY) {
          // By convention, laney follows lanex. See lane order defined above.
          assert((!laneIds.empty()) && "Laney defined without defining LaneX");
          laneIds[i].push_back(rewriter.create<arith::DivUIOp>(loc, threadX, lastShape));
        }
      }
    }
  }
  return laneIds;
}

static Value getOffset(int dim, LLVMGPULayout &layout,
                       LLVMGPULayout::IterationSpace::iteratorType &iterator,
                       SmallVector<Value> &laneIds,
                       Location loc, OpBuilder &rewriter) {
  return layout.substituteDimensions(
      layout.computeOffset(dim, iterator, layout.getLaneIds(dim), rewriter),
      laneIds, loc, rewriter);
}

static SmallVector<Value>
getIndices(LLVMGPULayout &layout,
           LLVMGPULayout::IterationSpace::iteratorType &iterator,
           SmallVector<Value> indices, AffineMap permutationMap, Location loc,
           OpBuilder &rewriter) {
  SmallVector<SmallVector<Value>> laneIds = getLaneIds(layout, loc, rewriter);
  Value rowOffset = getOffset(0, layout, iterator, laneIds[0], loc, rewriter);
  Value colOffset = getOffset(1, layout, iterator, laneIds[1], loc, rewriter);
  Value rowIndex = rewriter.create<arith::AddIOp>(loc, rowOffset, indices[0]);
  Value colIndex = rewriter.create<arith::AddIOp>(loc, colOffset, indices[1]);
  SmallVector<Value> newIndices{rowIndex, colIndex};
  newIndices = handlePermutations(newIndices, permutationMap);
  return newIndices;
}

static LogicalResult
distributeTransferReads(vector::TransferReadOp readOp, layoutMapType &layoutMap,
                        DenseMap<Value, Value> &valueMapping,
                        OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(readOp);
  Value result = readOp.getResult();
  if (!layoutMap.contains(result))
    return failure();
  auto layout = getLayout(layoutMap, result);
  Value source = readOp.getSource();
  Type elementType = llvm::cast<ShapedType>(source.getType()).getElementType();
  auto vectorType = VectorType::get(layout.getMappedVectorShape(), elementType);
  Location loc = readOp.getLoc();
  Value vector = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  // TODO: Switch to AMD specific load
  // TODO: Handle broadcasts
  auto loadFromMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        Value element = rewriter.create<memref::LoadOp>(
            loc, source,
            getIndices(layout, iterator, readOp.getIndices(),
                       readOp.getPermutationMap(), loc, rewriter));
        auto vectorType = VectorType::get({1}, elementType);
        Value broadcasted =
            rewriter.create<vector::BroadcastOp>(loc, vectorType, element);
        vector = rewriter.create<vector::InsertStridedSliceOp>(
            loc, broadcasted, vector, layout.getMappedVectorOffset(iterator),
            SmallVector<int64_t>{1});
      };
  auto rowColIterationSpace = layout.getCombinedIterationSpace();
  layout.map(loadFromMemref, rowColIterationSpace);
  valueMapping.try_emplace(result, vector);
  return success();
}

static LogicalResult distributeTransferWrites(
    vector::TransferWriteOp writeOp, layoutMapType &layoutMap,
    DenseMap<Value, Value> &valueMapping, OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(writeOp);
  Value vector = writeOp.getVector();
  if (!layoutMap.contains(vector) || !valueMapping.contains(vector))
    return failure();
  Value source = writeOp.getSource();
  auto layout = getLayout(layoutMap, vector);
  Location loc = writeOp.getLoc();
  auto storeToMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        Value result = rewriter.create<vector::ExtractOp>(
            loc, valueMapping.at(vector),
            layout.getMappedVectorOffset(iterator));
        rewriter.create<memref::StoreOp>(
            loc, result, source,
            getIndices(layout, iterator, writeOp.getIndices(),
                       writeOp.getPermutationMap(), loc, rewriter));
      };
  auto rowColIterationSpace = layout.getCombinedIterationSpace();
  layout.map(storeToMemref, rowColIterationSpace);
  return success();
}

template <typename T>
static bool hasKey(SmallVector<Value> &values, T &layoutMap) {
  for (Value value : values) {
    if (!layoutMap.contains(value))
      return false;
  }
  return true;
}

static LogicalResult distributeContracts(vector::ContractionOp contractOp,
                                         layoutMapType &layoutMap,
                                         DenseMap<Value, Value> &valueMapping,
                                         OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(contractOp);
  Value lhs = contractOp.getLhs();
  Value rhs = contractOp.getRhs();
  Value acc = contractOp.getAcc();
  SmallVector<Value> values{lhs, rhs, acc};
  if (!hasKey(values, layoutMap) || !hasKey(values, valueMapping))
    return failure();
  auto lhsLayout = getLayout(layoutMap, lhs);
  Value contractResult = contractOp.getResult();
  if (!layoutMap.contains(contractResult))
    return failure();
  auto resultLayout = getLayout(layoutMap, contractResult);
  Location loc = contractOp.getLoc();
  Type elementType = llvm::cast<ShapedType>(lhs.getType()).getElementType();
  auto vectorType =
      VectorType::get(resultLayout.getMappedVectorShape(), elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      loc, vectorType, rewriter.getZeroAttr(vectorType));
  int K{0};
  if (resultLayout.contractType == LLVMGPULayout::ContractType::MTM) {
    K = lhsLayout.getRowBatchDimension();
  } else {
    K = lhsLayout.getColBatchDimension();
  }
  auto createContract =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        SmallVector<int64_t> offset =
            resultLayout.getMappedVectorOffset(iterator);
        Value dMatrix = rewriter.create<vector::ExtractOp>(
            loc, valueMapping.at(acc), offset);
        if (resultLayout.encodeFn) {
          dMatrix = resultLayout.encodeFn(dMatrix, loc, rewriter);
        }
        for (int k = 0; k < K; k++) {
          Value aMatrix = rewriter.create<vector::ExtractOp>(
              loc, valueMapping.at(lhs), SmallVector<int64_t>{k, offset[0]});
          Value bMatrix = rewriter.create<vector::ExtractOp>(
              loc, valueMapping.at(rhs), SmallVector<int64_t>{k, offset[1]});
          dMatrix = rewriter.create<amdgpu::WMMAOp>(loc, dMatrix.getType(),
                                                    aMatrix, bMatrix, dMatrix);
        }
        if (resultLayout.decodeFn) {
          dMatrix = resultLayout.decodeFn(dMatrix, loc, rewriter);
        }
        result =
            rewriter.create<vector::InsertOp>(loc, dMatrix, result, offset);
      };
  auto batchIterationSpace = resultLayout.getBatchIterationSpace();
  resultLayout.map(createContract, batchIterationSpace);
  valueMapping.try_emplace(contractResult, result);
  return success();
}

static LogicalResult distributeConstants(arith::ConstantOp constantOp,
                                         layoutMapType &layoutMap,
                                         DenseMap<Value, Value> &valueMapping,
                                         OpBuilder &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(constantOp);
  Value constant = constantOp.getResult();
  // Allow non-layout constants to exist in the graph
  if (!layoutMap.count(constant))
    return success();
  auto attr = llvm::cast<DenseElementsAttr>(constantOp.getValue());
  // Only handle splat values for now
  if (!attr.isSplat())
    return failure();
  auto layout = getLayout(layoutMap, constant);
  Type elementType =
      llvm::cast<VectorType>(constant.getType()).getElementType();
  auto vectorType = VectorType::get(layout.getMappedVectorShape(), elementType);
  Value result = rewriter.create<arith::ConstantOp>(
      constantOp.getLoc(), vectorType,
      DenseElementsAttr::get(vectorType, attr.getSplatValue<APFloat>()));
  valueMapping.try_emplace(constant, result);
  return success();
}

LogicalResult doVectorDistribution(layoutMapType &layoutMap,
                                   SmallVector<Operation *> &operations,
                                   DenseMap<Value, Value> &valueMapping,
                                   RewriterBase &rewriter) {
  for (Operation *op : operations) {
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      if (failed(distributeTransferReads(readOp, layoutMap, valueMapping,
                                         rewriter)))
        return failure();
    }
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (failed(distributeContracts(contractOp, layoutMap, valueMapping,
                                     rewriter)))
        return failure();
    }
    if (auto writeOp = dyn_cast<vector::TransferWriteOp>(op)) {
      if (failed(distributeTransferWrites(writeOp, layoutMap, valueMapping,
                                          rewriter)))
        return failure();
    }
    if (auto constantOp = dyn_cast<arith::ConstantOp>(op)) {
      if (failed(distributeConstants(constantOp, layoutMap, valueMapping,
                                     rewriter)))
        return failure();
    }
  }
  return success();
}

LogicalResult eraseOps(SmallVector<Operation *> opsToErase,
                       RewriterBase &rewriter) {
  for (int i = opsToErase.size() - 1; i >= 0; i--) {
    if (isa<vector::TransferReadOp, vector::TransferWriteOp,
            vector::ContractionOp>(opsToErase[i])) {
      assert(opsToErase[i]->getUses().empty());
      rewriter.eraseOp(opsToErase[i]);
    }
  }
  return success();
}

static void collectOperations(Operation *rootOp,
                              SmallVectorImpl<Operation *> &opsToTraverse) {
  for (Region &region : rootOp->getRegions()) {
    for (Block &block : region.getBlocks()) {
      for (Operation &op : block.getOperations()) {
        opsToTraverse.push_back(&op);
        collectOperations(&op, opsToTraverse);
      }
    }
  }
}

} // namespace

LogicalResult convertVectorToGPUUsingLayout(RewriterBase &rewriter,
                                            func::FuncOp funcOp,
                                            bool useAMDWMMA) {
  layoutMapType layoutMap;
  SmallVector<Operation *> operationsToLower;
  collectOperations(funcOp, operationsToLower);
  if (failed(setLayouts(layoutMap, funcOp, useAMDWMMA)))
    return failure();
  if (failed(propagateLayouts(layoutMap, funcOp)))
    return failure();
  DenseMap<Value, Value> valueMapping;
  if (failed(doVectorDistribution(layoutMap, operationsToLower, valueMapping,
                                  rewriter)))
    return failure();
  if (failed(eraseOps(operationsToLower, rewriter)))
    return failure();
  return success();
}

} // namespace mlir::iree_compiler