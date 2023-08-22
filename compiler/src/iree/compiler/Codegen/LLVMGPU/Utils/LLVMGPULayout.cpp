#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPULayout.h"
#include <iostream>
#include "LLVMGPULayout.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"

#define DEBUG_TYPE "iree-llvmgpu-layout"

using namespace llvm;

namespace mlir::iree_compiler {

void LLVMGPULayout::print(StringRef str) {
  LLVM_DEBUG({
    llvm::dbgs() << str << " = \n";
    uint32_t i = 0;
    for (auto &state : layout) {
      llvm::dbgs() << "(Tensor dim " << i++ << ") [ ";
      for (auto &[name, shape] : state) {
        llvm::dbgs() << name << " : " << shape << " ";
      }
      llvm::dbgs() << " ] \n";
    }
  });
}

int32_t LLVMGPULayout::getDimension(int dim, llvm::StringRef name) {
  if (layout[0].contains(name))
    return layout[0][name];
  return -1;
}

int32_t LLVMGPULayout::getRowDimension(llvm::StringRef name) {
  return getDimension(0, name);
}

int32_t LLVMGPULayout::getColDimension(llvm::StringRef name) {
  return getDimension(1, name);
}

int32_t LLVMGPULayout::getColBatchDimension() {
  for (auto [name, shape] : layout[1]) {
    if (name.starts_with_insensitive("batch"))
      return shape;
  }
  return -1;
}

DenseSet<StringRef> LLVMGPULayout::getLaneIds(int dim) {
  assert(layout.size() > dim);
  DenseSet<StringRef> laneIds;
  for (auto [name, shape] : layout[dim]) {
    if (name.starts_with_insensitive("lane"))
      laneIds.insert(name);
  }
  return laneIds;
}

LLVMGPULayout::IterationSpace
LLVMGPULayout::getIterationSpace(uint32_t tensorDim,
                                 DenseSet<llvm::StringRef> dims) {
  LLVMGPULayout::IterationSpace iterationSpace;
  for (auto [label, shape] : layout[tensorDim]) {
    if (!dims.empty() && !dims.contains(label))
      continue;
    LLVMGPULayout::Iterator iterator(0, shape);
    iterationSpace.iterators[label] = iterator;
  }
  return iterationSpace;
}

LLVMGPULayout::IterationSpace LLVMGPULayout::getCombinedIterationSpace() {
  assert(layout.size() == 2);
  auto rowIterationSpace = getIterationSpace(0);
  auto colIterationSpace = getIterationSpace(1);
  return rowIterationSpace.combine(colIterationSpace);
}

LLVMGPULayout::IterationSpace LLVMGPULayout::getBatchIterationSpace() {
  assert(layout.size() == 2);
  auto batchRowIterationSpace = getIterationSpace(0, {"batchy"});
  auto batchColIterationSpace = getIterationSpace(1, {"batchx"});
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
  current += 1;
  bool done = current == end;
  if (done)
    current = 0;
  return done;
}

LLVMGPULayout::IterationSpace LLVMGPULayout::IterationSpace::combine(
    const LLVMGPULayout::IterationSpace &newSpace) {
  LLVMGPULayout::IterationSpace newIterationSpace;
  for (auto [name, iterator] : iterators) {
    // Ignore lane dims
    if (name.starts_with_insensitive("lane"))
      continue;
    newIterationSpace.iterators[name] = iterator;
  }
  for (auto [name, iterator] : newSpace.iterators) {
    // Ignore lane dims
    if (name.starts_with_insensitive("lane"))
      continue;
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
      llvm::dbgs() << key << " = " << iterator.current << " / " << iterator.end
                   << "\n";
    }
    llvm::dbgs() << "====================\n";
  });
}

AffineExpr LLVMGPULayout::computeOffset(
    uint32_t tensorDim, LLVMGPULayout::IterationSpace::iteratorType &iterator,
    const DenseSet<StringRef> &layoutDims, OpBuilder &builder) {
  assert(tensorDim < layout.size());
  SmallVector<AffineExpr> dims(layoutDims.size());
  bindDimsList(builder.getContext(), MutableArrayRef{dims});
  AffineExpr offset = builder.getAffineConstantExpr(0);
  AffineExpr stride = builder.getAffineConstantExpr(1);
  int i = 0;
  for (const auto &[name, shape] : layout[tensorDim]) {
    if (layoutDims.contains(name)) {
      offset = offset + stride * dims[i++];
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