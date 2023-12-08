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
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/AffineMap.h"

namespace mlir {
namespace iree_compiler {

enum class ContractMatrixType { A, B, C, D };

// Adding support for just these contract types for now
enum class ContractType { MMT, MTM, MM };

class LayoutProvider {
public:
  LayoutProvider(VectorLayoutAnalysis &analysis, Operation *root)
      : analysis(analysis), root(root) {}

  virtual ~LayoutProvider() = default;

  VectorLayoutAnalysis &getAnalysis() { return analysis; }

  /// Set the anchor ops in the analysis rooted on the root operation.
  virtual void setAnchorOps() = 0;

  /// Given a Value of type VectorType, return the distributed shape of the
  /// value, based on it's layout in the analysis.
  virtual SmallVector<int64_t>
  getDistributedShape(TypedValue<VectorType> val) = 0;

  virtual SmallVector<IREE::VectorExt::LayoutDimension>
  getSIMTLabels(IREE::VectorExt::LayoutAttr layout) = 0;

  virtual IREE::VectorExt::LayoutDimension getInnerMostVecDim() const = 0;

  virtual int64_t getLoadWidth(TypedValue<VectorType> val,
                               ArrayRef<int64_t> iterate) {
    // Convervative choice.
    return 1;
  }

  virtual int64_t getStoreWidth(TypedValue<VectorType> val,
                                ArrayRef<int64_t> iterate) {
    // Convervative choice.
    return 1;
  }

  /// Given an operation, do specialized distribution for it. Return true if
  /// the operation if a specialized distribution is done.
  /// Return false if the operation is not specialized.
  virtual LogicalResult specializedDistribution(RewriterBase &rewriter,
                                                Operation *op) {
    // No specialization by default.
    return failure();
  }

  virtual SmallVector<Value> getThreadGrid(RewriterBase &rewriter) = 0;

protected:
  VectorLayoutAnalysis &analysis;
  Operation *root;
}; // namespace iree_compiler

// This is specific for MI-series GPUs.
class AMDCDNAGPULayoutProvider : public LayoutProvider {
public:
  // Format is INPUTTYPE_MxNxK_OUTPUTTYPE.
  enum class MFMAType {
    F16_16x16x16_F32,
  };
  // Default format for MFMA is MMT.
  // The mfmaType is a parameter that can be tuned.
  AMDCDNAGPULayoutProvider(VectorLayoutAnalysis &analysis, Operation *root,
                           MFMAType mfmaType = MFMAType::F16_16x16x16_F32,
                           ContractType contractType = ContractType::MMT)
      : LayoutProvider(analysis, root), mfmaType(mfmaType),
        contractType(contractType) {}

  virtual void setAnchorOps() override;

  /// Given a Value of type VectorType, return the distributed shape of the
  /// value, based on it's layout in the analysis.
  virtual SmallVector<int64_t>
  getDistributedShape(TypedValue<VectorType> val) override;

  virtual SmallVector<IREE::VectorExt::LayoutDimension>
  getSIMTLabels(IREE::VectorExt::LayoutAttr layout) override {
    // Intersect the labels of layout and simtLabels.
    SmallVector<bool> layoutLabels(simtLabels.size());
    for (auto layout : layout.getLayouts()) {
      for (auto label : layout.getLabels()) {
        // Check if this label is in simtLabels.
        auto *it =
            std::find(simtLabels.begin(), simtLabels.end(), label.getValue());
        if (it != simtLabels.end()) {
          layoutLabels[std::distance(simtLabels.begin(), it)] = true;
        }
      }
    }

    SmallVector<IREE::VectorExt::LayoutDimension> result;
    for (int i = 0; i < simtLabels.size(); ++i) {
      if (layoutLabels[i]) {
        result.push_back(simtLabels[i]);
      }
    }

    return result;
  }

  virtual LogicalResult specializedDistribution(RewriterBase &rewriter,
                                                Operation *op) override;

  SmallVector<int64_t> getContractIndices(ContractMatrixType matrixType, int i,
                                          int j);

  Value computeMMA(Value a, Value b, Value c, Location loc,
                   OpBuilder &rewriter);

  int64_t getKDimension(int64_t M, int64_t K);

  IREE::VectorExt::LayoutAttr
  getCanonicalMFMALayout(TypedValue<VectorType> value,
                         ContractMatrixType matrixType, int64_t numElements);

  bool hasCanonicalShape(ContractMatrixType matrixType,
                         ArrayRef<int64_t> shape);

  IREE::VectorExt::LayoutDimension getInnerMostVecDim() const override {
    for (auto label : llvm::reverse(simtLabels)) {
      switch (label) {
      case IREE::VectorExt::LayoutDimension::VECTORX:
      case IREE::VectorExt::LayoutDimension::VECTORY:
      case IREE::VectorExt::LayoutDimension::VECTORZ:
        return label;
      default:
        break;
      }
    }
    llvm_unreachable("layout has no vector dimensions!");
  }

  SmallVector<Value> getThreadGrid(RewriterBase &rewriter) override {
    // We want a grid of 4 X 16, delinearized over the theadX.
    SmallVector<Value> threadGrid(2);
    Value threadX =
        rewriter.create<gpu::ThreadIdOp>(root->getLoc(), gpu::Dimension::x);
    // threadGrid[0] = threadX % 16
    Value const16 = rewriter.create<arith::ConstantIndexOp>(root->getLoc(), 16);
    threadGrid[0] =
        rewriter.create<arith::RemUIOp>(root->getLoc(), threadX, const16);
    // threadGrid[1] = threadX / 16
    threadGrid[1] =
        rewriter.create<arith::DivUIOp>(root->getLoc(), threadX, const16);
    return threadGrid;
  }

private:
  MFMAType mfmaType;
  ContractType contractType;
  SmallVector<IREE::VectorExt::LayoutDimension> simtLabels{
      IREE::VectorExt::LayoutDimension::BATCHX,
      IREE::VectorExt::LayoutDimension::BATCHY,
      IREE::VectorExt::LayoutDimension::VECTORX};
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
