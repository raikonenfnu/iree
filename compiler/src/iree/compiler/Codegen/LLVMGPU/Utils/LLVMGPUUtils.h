// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_
#define IREE_COMPILER_CODEGEN_LLVMGPU_UTILS_LLVMGPUUTILS_H_

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::iree_compiler {

/// Helper to convert copy to shared memory to async copy. This creates groups
/// of consecutive copies and emit wait operation right after.
void createAsyncGroups(RewriterBase &rewriter, func::FuncOp funcOp,
                       bool useMMASync);

/// Function to do layout analysis and distribution.
void doLayoutAnalysisAndDistribution(RewriterBase &rewriter,
                                     func::FuncOp funcOp);

TypedValue<VectorType> getDistributed(RewriterBase &rewriter,
                                      TypedValue<VectorType> value,
                                      LayoutProvider *provider);

void replaceOpWithDistributedValues(RewriterBase &rewriter, Operation *op,
                                    LayoutProvider *provider,
                                    ValueRange values);

template <typename OpTy, typename... Args>
OpTy replaceOpWithNewDistributedOp(LayoutProvider *provider,
                                   RewriterBase &rewriter, Operation *op,
                                   Args &&...args) {
  auto newOp = rewriter.create<OpTy>(op->getLoc(), std::forward<Args>(args)...);
  replaceOpWithDistributedValues(rewriter, op, provider, newOp->getResults());
  return newOp;
}

/// Function to do layout analysis and distribution. This is a rewrite of
/// doLayoutAnalysisAndDistribution.
void distributeVectors(RewriterBase &rewriter, func::FuncOp funcOp);

/// Function to reorder transposes and elementwise ops.
void reorderTranspose(RewriterBase &rewriter, func::FuncOp funcOp);

/// Look for allocs in shared memory space with overlapping liveness,
/// group them, and then pack all the allocations in each group into one i8
/// alloc.
///
/// Also adds barriers to make sure we are done writing/reading
/// from the previous alias group before starting a new one.
void packSharedMemoryAlloc(func::FuncOp funcOp);

} // namespace mlir::iree_compiler

#endif
