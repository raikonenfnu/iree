
#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPULAYOUT_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPULAYOUT_H_

#include <cstddef>
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "mlir/IR/Builders.h"

namespace mlir::iree_compiler {

// List of all possible dimensions that can be used in the layout.
// Add new values here.
namespace Dim {
  static constexpr uint32_t BATCHX = 0;
  static constexpr uint32_t BATCHY = 1;
  static constexpr uint32_t LANEX = 2;
  static constexpr uint32_t LANEY = 3;
  static constexpr uint32_t LANEZ = 4;
  static constexpr uint32_t VECTORX = 5;
  static constexpr uint32_t VECTORY = 6;
  static constexpr uint32_t VECTORZ = 7;
};
using Dimension = uint32_t;

/// This struct can be used to represent the layout of data
/// as a high-dimensional vector. The layout can be used
/// for vector distribution.
struct LLVMGPULayout {
  using layoutState = llvm::SmallMapVector<Dimension, uint32_t, 8>;
  using layoutType = llvm::SmallVector<layoutState, 2>;
  LLVMGPULayout(const layoutType &layout,
                DenseMap<uint32_t, SmallVector<Dimension>> &vectorMapping,
                Operation *source = nullptr)
      : layout(layout), vectorMapping(vectorMapping), source(source) {}

  void print(llvm::StringRef str);

  // Iterator-class that is used to represent the induction variable
  // for a single dimension.
  struct Iterator {
    Iterator() {}
    Iterator(uint32_t begin, uint32_t end) : begin(begin), end(end) {}
    bool next();
    uint32_t begin{0}, end{0}, current{0};
  };

  // Every layout defines an iteration space. This struct represents that
  // iteration space and provides access to the iterators for each dimension.
  struct IterationSpace {
    using iteratorType = llvm::SmallMapVector<Dimension, Iterator, 8>;
    IterationSpace combine(const IterationSpace &newSpace);
    bool next();
    iteratorType iterators;
    void print();
  };

  int32_t getDimension(int dim, Dimension name);
  int32_t getRowDimension(Dimension name);
  int32_t getColDimension(Dimension name);
  int32_t getColBatchDimension();
  DenseSet<Dimension> getLaneIds(int dim);

  IterationSpace getIterationSpace(uint32_t tensorDim,
                                   std::function<bool(Dimension)> filter = nullptr);

  // Returns the column iteration space nested inside the row iteration space.
  IterationSpace getCombinedIterationSpace();

  // Returns the iteration space spanned by the batch dimensions of the rows and
  // cols.
  IterationSpace getBatchIterationSpace();

  // Indexing Utilities
  AffineExpr computeOffset(uint32_t tensorDim,
                           IterationSpace::iteratorType &state,
                           const llvm::DenseSet<Dimension> &layoutDims,
                           OpBuilder &builder);
  Value substituteDimensions(AffineExpr expr, SmallVector<Value> &dims,
                             Location loc, OpBuilder &builder);

  // Map functions
  void map(std::function<void(IterationSpace::iteratorType &)> function,
           IterationSpace &iterationSpace);

  // Shape functions
  SmallVector<int64_t> getMappedVectorShape();
  SmallVector<int64_t>
  getMappedVectorOffset(IterationSpace::iteratorType &iterator);

  layoutType layout;
  // Where this layout is being derived from. If null, layout comes from
  // operator definition.
  Operation *source;
  // Mapping of vector index to label(s)
  DenseMap<uint32_t, SmallVector<Dimension>> vectorMapping;

  std::function<Value(Value, Location, OpBuilder &)> encodeFn{nullptr};
  std::function<Value(Value, Location, OpBuilder &)> decodeFn{nullptr};

};

} // namespace mlir::iree_compiler

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_LLVMGPULAYOUT_H_