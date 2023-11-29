// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree-dialects/Dialect/VectorExt/IR/VectorExtOps.h"
#include "iree/compiler/Codegen/Common/VectorLayoutAnalysis.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/LLVMGPUUtils.h"
#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/NVGPU/IR/NVGPUDialect.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Matchers.h"

using namespace mlir::iree_compiler::IREE::VectorExt;

namespace mlir::iree_compiler {

namespace {

class VectorDistribution {
public:
  VectorDistribution(func::FuncOp root, RewriterBase &rewriter,
                     LayoutProvider *provider, VectorLayoutAnalysis &analysis)
      : root(root), analysis(analysis), provider(provider), rewriter(rewriter) {
    provider->setAnchorOps();
    if (failed(analysis.run()))
      return;

    // Anotate ops for debugging for now.
    root->walk([&](Operation *op) {
      if (op->getNumResults() != 1)
        return;
      Attribute layout = analysis.getLayout<Attribute>(op->getResult(0));
      if (!layout)
        return;
      op->setAttr("layout", layout);
    });
  }

  void distribute() {
    // TODO: We are returning for now, to just see the affect of layout
    // analysis. Later, this should be removed.
    return;

    root->walk([&](Operation *op) {
      rewriter.setInsertionPoint(op);

      // if (provider->specializedDistribution(op)) {
      //   return;
      // }

      TypeSwitch<Operation *, void>(op)
          .Case<vector::ContractionOp>([&](auto contractOp) {
            distributeContractions(contractOp);
            return;
          })
          .Case<vector::TransferReadOp>([&](auto transferReadOp) {
            distributeTransferReads(transferReadOp);
            return;
          })
          .Case<vector::TransferWriteOp>([&](auto transferWriteOp) {
            distributeTransferWrites(transferWriteOp);
          })
          .Case<arith::ConstantOp>([&](auto constantOp) {
            distributeConstants(constantOp);
            return;
          })
          .Default([&](auto op) {});
    });
  }

private:
  void distributeTransferWrites(vector::TransferWriteOp transferWriteOp) {}

  void distributeTransferReads(vector::TransferReadOp transferReadOp) {}

  void distributeContractions(vector::ContractionOp contractionOp) {}

  void distributeConstants(arith::ConstantOp constantOp) {}

  func::FuncOp root;
  VectorLayoutAnalysis &analysis;
  LayoutProvider *provider;
  RewriterBase &rewriter;
  IRMapping simdToSimt;
}; // namespace

} // namespace

void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp) {
  VectorLayoutAnalysis analysis(funcOp);
  AMDGPULayoutProvider layoutProvider(analysis, funcOp);
  VectorDistribution distribution(funcOp, rewriter, &layoutProvider, analysis);
  distribution.distribute();
}

} // namespace mlir::iree_compiler
