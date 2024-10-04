// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_CODEGEN_TRITON_PASSES
#define IREE_CODEGEN_TRITON_PASSES

include "mlir/Pass/PassBase.td"

//------------------------------------------------------------------------------
// TRITON Passes (keep alphabetical)
//------------------------------------------------------------------------------

def TritonLowerExecutableTargetPass :
    InterfacePass<"iree-triton-lower-executable-target", "mlir::FunctionOpInterface"> {
  let summary = "Perform lowering of executable target using one of the IREE::HAL::DispatchLoweringPassPipeline";
}

#endif // IREE_CODEGEN_TRITON_PASSES
