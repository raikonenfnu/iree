#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPULayout.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUHWConfig.h"
#include "llvm/ADT/DenseMap.h"
#include "mlir/Dialect/AMDGPU/IR/AMDGPUDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/Pass.h"

#define DEBUG_TYPE "iree-llvmgpu-layout-vector-to-gpu"

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

static void propagateLayoutToFor(scf::ForOp forOp,
                                 layoutMapType &layoutMap) {
  for (auto argIndex : llvm::enumerate(forOp.getRegionIterArgs())) {
    BlockArgument &arg = argIndex.value();
    if (!layoutMap.count(arg))
      continue;
    OpOperand &operand = forOp.getOpOperandForRegionIterArg(arg);
    Value result = forOp.getResult(argIndex.index());
    LLVMGPULayout newLayout = getLayout(layoutMap, arg);
    addToLayoutMap(operand.get(), layoutMap, newLayout);
    addToLayoutMap(result, layoutMap, newLayout);
    newLayout.print("for operand/result");
  }
}

LogicalResult propagateLayouts(layoutMapType &layoutMap, func::FuncOp funcOp) {
  WalkResult result = funcOp.walk([&](Operation *op) {
   if (auto forOp = dyn_cast<scf::ForOp>(op)) {
    propagateLayoutToFor(forOp, layoutMap);
   }
   return WalkResult::advance();
  });
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
                        LLVMGPUHWConfig &hwConfig,
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
  std::function<void(LLVMGPULayout::IterationSpace::iteratorType &)> loadFromMemref;
  LLVMGPULayout::IterationSpace rowColIterationSpace;
  // TODO: Determine number of elements to load from layout
  uint32_t numElements{1};
  // For MMT, load 8 elements at a time
  if (hwConfig.contractType == LLVMGPULayout::ContractType::MMT) {
    numElements = 8;
  }
  // For MM, load 8 elements at a time only for A matrix
  if (hwConfig.contractType == LLVMGPULayout::ContractType::MM) {
    for (Operation *user : result.getUsers()) {
      if (auto contractOp = dyn_cast<vector::ContractionOp>(user)) {
        if (result == contractOp.getLhs()) {
          numElements = 8;
          break;
        }
      }
    }
  }
  if ((numElements > 1) && layout.supportsVectorLoadsStores(numElements)) {
    loadFromMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        auto vectorType = VectorType::get({numElements}, elementType);
        auto indices = getIndices(layout, iterator, readOp.getIndices(),
                       readOp.getPermutationMap(), loc, rewriter);
        Value element = rewriter.create<vector::LoadOp>(loc, vectorType, source,
            indices);
        vector = rewriter.create<vector::InsertStridedSliceOp>(
            loc, element, vector, layout.getMappedVectorOffset(iterator),
            SmallVector<int64_t>{1});
    };
    rowColIterationSpace = layout.getVectorStridedCombinedIterationSpace(numElements);
  } else {
    loadFromMemref =
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
    rowColIterationSpace = layout.getCombinedIterationSpace();
  }
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
  std::function<void(LLVMGPULayout::IterationSpace::iteratorType &)> storeToMemref;
  LLVMGPULayout::IterationSpace rowColIterationSpace;
  // TODO: Determine number of elements to load from layout
  uint32_t numElements{1};
  if ((numElements > 1) && layout.supportsVectorLoadsStores(numElements)) {
    storeToMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        SmallVector<int64_t> offsets = layout.getMappedVectorOffset(iterator);
        SmallVector<int64_t> strides(offsets.size(), 1);
        SmallVector<int64_t> shapes(offsets.size(), 1);
        shapes[shapes.size() - 1] = numElements;
        Value result = rewriter.create<vector::ExtractStridedSliceOp>(
            loc, valueMapping.at(vector), offsets, shapes, strides);
        result = rewriter.create<vector::ExtractOp>(loc, result, SmallVector<int64_t>(offsets.size() - 1, 0));
        auto indices = getIndices(layout, iterator, writeOp.getIndices(),
                       writeOp.getPermutationMap(), loc, rewriter);
        rewriter.create<vector::StoreOp>(loc, result, source, indices);
    };
    rowColIterationSpace = layout.getVectorStridedCombinedIterationSpace(numElements);
  } else {
    storeToMemref =
      [&](LLVMGPULayout::IterationSpace::iteratorType &iterator) {
        Value result = rewriter.create<vector::ExtractOp>(
            loc, valueMapping.at(vector),
            layout.getMappedVectorOffset(iterator));
        rewriter.create<memref::StoreOp>(
            loc, result, source,
            getIndices(layout, iterator, writeOp.getIndices(),
                       writeOp.getPermutationMap(), loc, rewriter));
    };
    rowColIterationSpace = layout.getCombinedIterationSpace();
  }
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
                                         LLVMGPUHWConfig &hwConfig,
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
              loc, valueMapping.at(lhs), hwConfig.getIndices(MatrixType::A, offset[0], k));
          Value bMatrix = rewriter.create<vector::ExtractOp>(
              loc, valueMapping.at(rhs), hwConfig.getIndices(MatrixType::B, k, offset[1]));
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

static void replaceForOpWithNewSignature(RewriterBase &rewriter,
                                         scf::ForOp loop,
                                         ValueRange newIterOperands,
                                         layoutMapType &layoutMap,
                                         DenseMap<Value, Value> &valueMapping) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(loop);

  // Create a new loop before the existing one, with the extra operands.
  // We will be using dummy values instead of the old operands
  // only for those operands that are being distributed
  SmallVector<Value> newOperands;
  auto operands = llvm::to_vector(loop.getIterOperands());
  for (auto operand : operands) {
    if (!layoutMap.count(operand)) {
      newOperands.push_back(operand);
      continue;
    }
    Value zero = rewriter.create<arith::ConstantOp>(
        loop.getLoc(), rewriter.getZeroAttr(operand.getType()));
    newOperands.push_back(zero);
  }

  newOperands.append(newIterOperands.begin(), newIterOperands.end());
  scf::ForOp newLoop = rewriter.create<scf::ForOp>(
      loop.getLoc(), loop.getLowerBound(), loop.getUpperBound(), loop.getStep(),
      newOperands);
  newLoop.getBody()->erase();

  newLoop.getLoopBody().getBlocks().splice(
      newLoop.getLoopBody().getBlocks().begin(),
      loop.getLoopBody().getBlocks());
  for (Value operand : newIterOperands)
    newLoop.getBody()->addArgument(operand.getType(), operand.getLoc());

  // Replace old results and propagate layouts
  int numOldResults = loop.getNumResults();
  for (auto it : llvm::zip(loop.getResults(),
                           newLoop.getResults().take_front(numOldResults))) {
    if (layoutMap.count(std::get<0>(it))) {
      addToLayoutMap(std::get<1>(it), layoutMap, getLayout(layoutMap, std::get<0>(it)));
    }
    rewriter.replaceAllUsesWith(std::get<0>(it), std::get<1>(it));
  }

  // Propagate layout + mapping from old to new block args and results
  auto bbArgs = newLoop.getRegionIterArgs();
  auto results = newLoop.getResults();
  for (int i = 0; i < numOldResults; i++) {
    if (layoutMap.count(bbArgs[i])) {
      addToLayoutMap(bbArgs[i + numOldResults], layoutMap, getLayout(layoutMap, bbArgs[i]));
    }
    valueMapping.try_emplace(bbArgs[i], bbArgs[i + numOldResults]);
    if (layoutMap.count(results[i])) {
      addToLayoutMap(results[i + numOldResults], layoutMap, getLayout(layoutMap, results[i]));
    }
    valueMapping.try_emplace(results[i], results[i + numOldResults]);
  }

  return;
}

static LogicalResult distributeFor(scf::ForOp forOp,
                                   layoutMapType &layoutMap,
                                   DenseMap<Value, Value> &valueMapping,
                                   RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(forOp);

  SmallVector<Value> newOperands;
  for (const auto &operand : llvm::enumerate(forOp.getIterOperands())) {
    if (!valueMapping.count(operand.value())) {
      continue;
    }
    newOperands.push_back(valueMapping.at(operand.value()));
  }
  replaceForOpWithNewSignature(rewriter, forOp, newOperands, layoutMap,
                               valueMapping);
  return success();
}

static LogicalResult distributeYield(scf::YieldOp yieldOp,
                                     layoutMapType &layoutMap,
                                     DenseMap<Value, Value> &valueMapping,
                                     RewriterBase &rewriter) {
  OpBuilder::InsertionGuard g(rewriter);
  rewriter.setInsertionPoint(yieldOp);

  // Update yield op with additional operand
  auto loop = cast<scf::ForOp>(yieldOp->getParentOp());
  auto yieldOperands = llvm::to_vector(yieldOp.getOperands());
  for (const auto &operand : llvm::enumerate(yieldOp.getOperands())) {
    if (!valueMapping.count(operand.value()))
      continue;
    // Replace the yield of old value with the for op argument to make it easier
    // to remove the dead code.
    yieldOperands[operand.index()] = loop.getIterOperands()[operand.index()];
    yieldOperands.push_back(valueMapping.at(operand.value()));
  }
  rewriter.create<scf::YieldOp>(yieldOp.getLoc(), yieldOperands);
  return success();
}

LogicalResult doVectorDistribution(layoutMapType &layoutMap,
                                   SmallVector<Operation *> &operations,
                                   DenseMap<Value, Value> &valueMapping,
                                   LLVMGPUHWConfig &hwConfig,
                                   RewriterBase &rewriter) {
  for (Operation *op : operations) {
    LLVM_DEBUG({
      llvm::dbgs() << "Distributing op = ";
      op->dump();
    });
    if (auto readOp = dyn_cast<vector::TransferReadOp>(op)) {
      if (failed(distributeTransferReads(readOp, layoutMap, valueMapping, hwConfig,
                                         rewriter)))
        return failure();
    }
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {
      if (failed(distributeContracts(contractOp, layoutMap, valueMapping, hwConfig,
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
    if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      if (failed(distributeFor(forOp, layoutMap, valueMapping, rewriter)))
        return failure();
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(op)) {
      if (failed(distributeYield(yieldOp, layoutMap, valueMapping, rewriter)))
        return failure();
    }
  }
  return success();
}

LogicalResult eraseOps(SmallVector<Operation *> opsToErase,
                       RewriterBase &rewriter) {
  for (int i = opsToErase.size() - 1; i >= 0; i--) {
    if (isa<vector::TransferReadOp, vector::TransferWriteOp,
            vector::ContractionOp, scf::ForOp, scf::YieldOp>(opsToErase[i])) {
      LLVM_DEBUG({
        llvm::dbgs() << "Erasing op = ";
        opsToErase[i]->dump();
        if (!opsToErase[i]->getUses().empty()) {
          llvm::dbgs() << "\n Uses = \n";
          for (OpOperand &use : opsToErase[i]->getUses())
            use.getOwner()->dump();
        }
      });
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
  AMDWMMAConfig hwConfig(AMDWMMAConfig::WMMAType::F16_16X16X16_F16,
                         LLVMGPULayout::ContractType::MTM, 32);
  SmallVector<Operation *> operationsToLower;
  collectOperations(funcOp, operationsToLower);
  if (failed(setLayouts(layoutMap, funcOp, hwConfig)))
    return failure();
  if (failed(propagateLayouts(layoutMap, funcOp)))
    return failure();
  DenseMap<Value, Value> valueMapping;
  if (failed(doVectorDistribution(layoutMap, operationsToLower, valueMapping,
                                  hwConfig, rewriter)))
    return failure();
  if (failed(eraseOps(operationsToLower, rewriter)))
    return failure();
  return success();
}

} // namespace mlir::iree_compiler