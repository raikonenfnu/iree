// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_DIALECT_NOD_PASSES
#define IREE_DIALECT_NOD_PASSES

include "mlir/Pass/PassBase.td"

def LinalgToNod :
    Pass<"iree-linalg-to-nod", "FuncOp"> {
  let summary = "Lower Linalg Matmul Ops into Custom Nod C++ Ops";
  let constructor = "mlir::iree_compiler::Nod::createLinalgToNodPass()";
}

#endif  // IREE_DIALECT_NOD_PASSES
