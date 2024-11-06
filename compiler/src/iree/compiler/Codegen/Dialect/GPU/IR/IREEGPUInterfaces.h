// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_
#define IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUEnums.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtInterfaces.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir::iree_compiler::IREE::GPU {

// Partial nested layout for an MMA intrinsic's matrix input/output inside
// a single subgroup.
struct MMASingleSubgroupLayout {
  llvm::SmallVector<int64_t, 2> outer;
  llvm::SmallVector<int64_t, 2> thread;
  llvm::SmallVector<int64_t, 2> tstrides;
  llvm::SmallVector<int64_t, 2> element;
};

} // namespace mlir::iree_compiler::IREE::GPU

// clang-format off
#define GET_ATTRDEF_CLASSES
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUInterfaces.h.inc"
// clang-format on

#endif // IREE_COMPILER_CODEGEN_DIALECT_GPU_IREEGPUINTERFACES_H_
