// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_SPIRVTOWGSL_H_
#define IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_SPIRVTOWGSL_H_

#include <string>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// Compiles SPIR-V into WebGPU Shading Language (WGSL) source code.
// Returns llvm::None on failure.
llvm::Optional<std::string> compileSPIRVToWGSL(
    llvm::ArrayRef<uint32_t> spvBinary);

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_HAL_TARGET_WEBGPU_SPIRVTOWGSL_H_
