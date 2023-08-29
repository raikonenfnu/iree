#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPULayout.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUHWConfig.h"
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

static void
addToLayoutMap(Value value, layoutMapType &layoutMap, LLVMGPULayout &newLayout) {
  if (layoutMap.contains(value)) {
    layoutMap[value].push_back(newLayout);
    return;
  }
  layoutMap[value] = {newLayout};
}

static LLVMGPULayout &getLayout(layoutMapType &layoutMap, Value value) {
  assert(layoutMap.size() >= 1);
  // TODO: Figure out what to return when there are multiple values
  return layoutMap[value][0];
}

LogicalResult setMMALayout(vector::ContractionOp contractOp,
                           layoutMapType &layoutMap, LLVMGPUHWConfig &hwConfig) {
  Value a = contractOp.getLhs();
  Value b = contractOp.getRhs();
  Value c = contractOp.getAcc();
  Value d = contractOp.getResult();
  if (!hwConfig.verifyOperandTypes(a, b, c, d) || !hwConfig.verifyContract(contractOp)) {
    return failure();
  }
  SmallVector<Value> values{a, b, c, d};
  SmallVector<MatrixType> names{MatrixType::A, MatrixType::B, MatrixType::C, MatrixType::D};
  for (auto [name, value] : llvm::zip(names, values)) {
    auto layout = hwConfig.getLayout(name, value);
    addToLayoutMap(value, layoutMap, layout);
  }
  return success();
}

LogicalResult setLayouts(layoutMapType &layoutMap, func::FuncOp funcOp,
                         LLVMGPUHWConfig &hwConfig) {
  WalkResult result = funcOp.walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (failed(setMMALayout(contractOp, layoutMap, hwConfig)))
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
  // Load 4 elements at a time
  uint32_t stride{4};
  auto loadFromMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        auto vectorType = VectorType::get({stride}, elementType);
        Value sgprOffset = rewriter.create<arith::ConstantOp>(loc, rewriter.getI32IntegerAttr(0));
        auto indices = getIndices(layout, iterator, readOp.getIndices(),
                       readOp.getPermutationMap(), loc, rewriter);
        // Bitcast to integer
        for (int i = 0; i < indices.size(); i++)
          indices[i] = rewriter.create<arith::IndexCastOp>(loc, rewriter.getI32Type(), indices[i]);
        Value element = rewriter.create<amdgpu::RawBufferLoadOp>(loc, vectorType, source,
            indices, rewriter.getBoolAttr(false), rewriter.getI32IntegerAttr(0),
            sgprOffset);
        vector = rewriter.create<vector::InsertStridedSliceOp>(
            loc, element, vector, layout.getMappedVectorOffset(iterator),
            SmallVector<int64_t>{1});
      };
  auto rowColIterationSpace = layout.getVectorStridedCombinedIterationSpace(stride);
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
  std::function<SmallVector<int64_t>(int, int)> aMatrixIndices, bMatrixIndices;
  if (resultLayout.contractType == LLVMGPULayout::ContractType::MTM) {
    K = lhsLayout.getRowBatchDimension();
    // A matrix indices = (k, i)
    aMatrixIndices = [](int i, int k) -> SmallVector<int64_t> { return {k, i}; };
    // B matrix indices = (k, j)
    bMatrixIndices = [](int k, int j) -> SmallVector<int64_t> { return {k, j}; };
  } else {
    K = lhsLayout.getColBatchDimension();
    // A matrix indices = (i, k)
    aMatrixIndices = [](int i, int k) -> SmallVector<int64_t> { return {i, k}; };
    // B matrix indices = (j, k)
    bMatrixIndices = [](int k, int j) -> SmallVector<int64_t> { return {j, k}; };
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
              loc, valueMapping.at(lhs), aMatrixIndices(offset[0], k));
          Value bMatrix = rewriter.create<vector::ExtractOp>(
              loc, valueMapping.at(rhs), bMatrixIndices(k, offset[1]));
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
  // TODO: Use fold extf pattern before switching to mixed precision
  AMDWMMAConfig hwConfig(AMDWMMAConfig::WMMAType::F16_16X16X16_F16, 32);
  SmallVector<Operation *> operationsToLower;
  collectOperations(funcOp, operationsToLower);
  if (failed(setLayouts(layoutMap, funcOp, hwConfig)))
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