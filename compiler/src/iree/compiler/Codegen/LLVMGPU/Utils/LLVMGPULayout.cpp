#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPULayout.h"
#include <iostream>
#include "LLVMGPULayout.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define DEBUG_TYPE "iree-llvmgpu-layout"

using namespace llvm;

namespace mlir::iree_compiler {

static StringRef dimensionToString(Dimension dim) {
  switch (dim) {
    case Dim::BATCHX: return "batchx";
    case Dim::BATCHY: return "batchy";
    case Dim::LANEX: return "lanex";
    case Dim::LANEY: return "laney";
    case Dim::LANEZ: return "lanez";
    case Dim::VECTORX: return "vectorx";
    case Dim::VECTORY: return "vectory";
    case Dim::VECTORZ: return "vectorz";
    default: return "unknown";
  }
}

void LLVMGPULayout::print(StringRef str) {
  LLVM_DEBUG({
    llvm::dbgs() << str << " = \n";
    uint32_t i = 0;
    for (auto &state : layout) {
      llvm::dbgs() << "(Tensor dim " << i++ << ") [ ";
      for (auto &[name, shape] : state) {
        llvm::dbgs() << dimensionToString(name) << " : " << shape << " ";
      }
      llvm::dbgs() << " ] \n";
    }
  });
}

int32_t LLVMGPULayout::getDimension(int dim, Dimension name) {
  if (layout[dim].contains(name))
    return layout[dim][name];
  return -1;
}

int32_t LLVMGPULayout::getRowDimension(Dimension name) {
  return getDimension(0, name);
}

int32_t LLVMGPULayout::getColDimension(Dimension name) {
  return getDimension(1, name);
}

static bool isBatchDimension(Dimension name) {
  return ((name == Dim::BATCHX) || (name == Dim::BATCHY));
}

static bool isLaneDimension(Dimension name) {
  return ((name == Dim::LANEX) || (name == Dim::LANEY) || (name == Dim::LANEZ));
}

static bool isVectorDimension(Dimension name) {
  return ((name == Dim::VECTORX) || (name == Dim::VECTORY) || (name == Dim::VECTORZ));
}

int32_t LLVMGPULayout::getBatchDimension(int dim) {
  for (auto [name, shape] : layout[dim]) {
    if (isBatchDimension(name))
      return shape;
  }
  return -1;
}

int32_t LLVMGPULayout::getColBatchDimension() {
  return getBatchDimension(1);
}

int32_t LLVMGPULayout::getRowBatchDimension() {
  return getBatchDimension(0);
}

DenseSet<Dimension> LLVMGPULayout::getLaneIds(int dim) {
  assert(layout.size() > dim);
  DenseSet<Dimension> laneIds;
  for (auto [name, shape] : layout[dim]) {
    if (isLaneDimension(name))
      laneIds.insert(name);
  }
  return laneIds;
}

LLVMGPULayout::IterationSpace
LLVMGPULayout::getIterationSpace(uint32_t tensorDim,
                                 std::function<bool(Dimension)> filter) {
  LLVMGPULayout::IterationSpace iterationSpace;
  for (auto [label, shape] : layout[tensorDim]) {
    if (filter && !filter(label)) continue;
    LLVMGPULayout::Iterator iterator(0, shape);
    iterationSpace.iterators[label] = iterator;
  }
  return iterationSpace;
}

LLVMGPULayout::IterationSpace LLVMGPULayout::getCombinedIterationSpace() {
  assert(layout.size() == 2);
  auto isNotLaneDimension = [&](Dimension name) { return !isLaneDimension(name); };
  auto rowIterationSpace = getIterationSpace(0, isNotLaneDimension);
  auto colIterationSpace = getIterationSpace(1, isNotLaneDimension);
  return rowIterationSpace.combine(colIterationSpace);
}

static LLVMGPULayout::IterationSpace createVectorStridedIterationSpace(LLVMGPULayout::IterationSpace iterationSpace,
  uint32_t stride) {
  LLVMGPULayout::IterationSpace newIterationSpace;
  for (auto [name, iterator] : iterationSpace.iterators) {
    if (isVectorDimension(name)) {
      newIterationSpace.iterators[name] = LLVMGPULayout::Iterator(iterator.begin, iterator.end, stride);
    } else {
      newIterationSpace.iterators[name] = iterator;
    }
  }
  return newIterationSpace;
}

LLVMGPULayout::IterationSpace LLVMGPULayout::getVectorStridedCombinedIterationSpace(uint32_t stride) {
  assert(layout.size() == 2);
  auto isNotLaneDimension = [&](Dimension name) { return !isLaneDimension(name); };
  auto rowIterationSpace = getIterationSpace(0, isNotLaneDimension);
  rowIterationSpace = createVectorStridedIterationSpace(rowIterationSpace, stride);
  auto colIterationSpace = getIterationSpace(1, isNotLaneDimension);
  colIterationSpace = createVectorStridedIterationSpace(colIterationSpace, stride);
  return rowIterationSpace.combine(colIterationSpace);
}

LLVMGPULayout::IterationSpace LLVMGPULayout::getBatchIterationSpace() {
  assert(layout.size() == 2);
  auto batchRowIterationSpace = getIterationSpace(0, isBatchDimension);
  auto batchColIterationSpace = getIterationSpace(1, isBatchDimension);
  return batchRowIterationSpace.combine(batchColIterationSpace);
}

void LLVMGPULayout::map(
    std::function<void(LLVMGPULayout::IterationSpace::iteratorType &)> function,
    IterationSpace &iterationSpace) {
  do {
    function(iterationSpace.iterators);
    iterationSpace.print();
  } while (!iterationSpace.next());
}

SmallVector<int64_t> LLVMGPULayout::getMappedVectorShape() {
  SmallVector<int64_t> shape(vectorMapping.size(), 1);
  for (int i = 0; i < vectorMapping.size(); i++) {
    for (auto label : vectorMapping[i]) {
      for (auto layoutState : layout) {
        for (auto [name, size] : layoutState) {
          if (name == label) {
            shape[i] *= size;
          }
        }
      }
    }
  }
  return shape;
}

SmallVector<int64_t>
LLVMGPULayout::getMappedVectorOffset(IterationSpace::iteratorType &iterator) {
  SmallVector<int64_t> offset(iterator.size(), 0);
  for (int i = 0; i < vectorMapping.size(); i++) {
    for (auto label : vectorMapping[i]) {
      for (auto layoutState : layout) {
        for (auto [name, size] : layoutState) {
          if ((name == label) && (iterator.contains(name))) {
            offset[i] = iterator[name].current + offset[i] * size;
          }
        }
      }
    }
  }
  return offset;
}

// Moves the iterator forward.
// Returns true if iterator is at the end of iteration space.
// Returns false otherwise.
bool LLVMGPULayout::Iterator::next() {
  current += stride;
  bool done = current >= end;
  if (done)
    current = 0;
  return done;
}

LLVMGPULayout::IterationSpace LLVMGPULayout::IterationSpace::combine(
    const LLVMGPULayout::IterationSpace &newSpace) {
  LLVMGPULayout::IterationSpace newIterationSpace;
  for (auto [name, iterator] : iterators) {
    newIterationSpace.iterators[name] = iterator;
  }
  for (auto [name, iterator] : newSpace.iterators) {
    newIterationSpace.iterators[name] = iterator;
  }
  return newIterationSpace;
}

// Moves the iterator forward.
// Returns true if iterators are at the end of the iteration spaces.
// Returns false otherwise.
bool LLVMGPULayout::IterationSpace::next() {
  bool done{true};
  for (auto &[key, iterator] : llvm::reverse(iterators)) {
    if (!iterator.next()) {
      done = false;
      break;
    }
  }
  return done;
}

void LLVMGPULayout::IterationSpace::print() {
  LLVM_DEBUG({
    for (auto [key, iterator] : iterators) {
      llvm::dbgs() << dimensionToString(key) << " = " << iterator.current << " / " << iterator.end
                   << "\n";
    }
    llvm::dbgs() << "====================\n";
  });
}

AffineExpr LLVMGPULayout::computeOffset(
    uint32_t tensorDim, LLVMGPULayout::IterationSpace::iteratorType &iterator,
    const DenseSet<Dimension> &layoutDims, OpBuilder &builder) {
  assert(tensorDim < layout.size());
  SmallVector<AffineExpr> dims(layoutDims.size());
  bindDimsList(builder.getContext(), MutableArrayRef{dims});
  AffineExpr offset = builder.getAffineConstantExpr(0);
  AffineExpr stride = builder.getAffineConstantExpr(1);
  int i = 0;
  for (const auto &[name, shape] : layout[tensorDim]) {
    if (layoutDims.contains(name)) {
      offset = offset + stride * dims[i++];
      stride = stride * builder.getAffineConstantExpr(shape);
      continue;
    }
    if (!iterator.contains(name))
      continue;
    offset =
        offset + stride * builder.getAffineConstantExpr(iterator[name].current);
    stride = stride * builder.getAffineConstantExpr(shape);
  }
  return offset;
}

Value LLVMGPULayout::substituteDimensions(AffineExpr expr,
                                          SmallVector<Value> &dims,
                                          Location loc, OpBuilder &builder) {
  return builder.create<affine::AffineApplyOp>(loc, expr, dims);
}

} // namespace mlir::iree_compiler