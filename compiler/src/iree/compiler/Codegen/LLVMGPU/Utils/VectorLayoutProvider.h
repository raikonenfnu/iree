// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

class LayoutProvider {
public:
  LayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : analysis(analysis), root(root) {}

  virtual ~LayoutProvider() = default;

  /// Set the anchor ops in the analysis rooted on the root operation.
  virtual void setAnchorOps() = 0;

  // /// Given a Value of type VectorType, return the distributed shape of the
  // /// value, based on it's layout in the analysis.
  // virtual SmallVector<int64_t>
  // getDistributedShape(TypedValue<VectorType> val) = 0;

  // /// Given a value, iterate over all elements assigned to a single thread in
  // /// distribution, for that particular value.
  // virtual void
  // forAllElementsInThread(TypedValue<VectorType> val,
  //                        std::function<void(ArrayRef<int64_t>)> callback) =
  //                        0;

  // /// Given a index of an element in a single thread, get the index of this
  // /// thread in the distributed layout, i.e. parameterized by thread
  // /// indexes.
  // /// TODO: What should be the return value?
  // virtual void getDistributedIndex(ArrayRef<int64_t> index) = 0;

  // /// Given an operation, do specialized distribution for it. Return true if
  // /// the operation if a specialized distribution is done.
  // /// Return false if the operation is not specialized.
  // virtual bool specializedDistribution(Operation *op) = 0;

protected:
  VectorLayoutAnalysis &analysis;
  Operation *root;
};

class AMDGPULayoutProvider : public LayoutProvider {
public:
  AMDGPULayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : LayoutProvider(analysis, root) {}

  virtual void setAnchorOps() override;
};

// class NVIDIAGPULayoutProvider : public LayoutProvider {
// public:
//   NVIDIAGPULayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
//       : LayoutProvider(analysis, root) {}
//
//   virtual void setAnchorOps() override;
// };

}; // namespace iree_compiler
}; // namespace mlir

#endif // IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_VECTORLAYOUTPROVIDER_H_
