// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests the printing/parsing of the nod dialect ops.
// This doesn't have much meaning here as we don't define any nod printers or
// parsers but does serve as a reference for the op usage.

// RUN: iree-opt -split-input-file %s | iree-opt -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !nod.message) {
  %c1_i32 = constant 1 : i32
  // CHECK: "nod.print"(%arg0, %c1_i32) : (!nod.message, i32) -> ()
  "nod.print"(%arg0, %c1_i32) : (!nod.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !nod.message) -> !nod.message {
  // CHECK: %0 = "nod.reverse"(%arg0) : (!nod.message) -> !nod.message
  %0 = "nod.reverse"(%arg0) : (!nod.message) -> !nod.message
  return %0 : !nod.message
}

// -----

// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !nod.message {
  // CHECK: %0 = "nod.get_unique_message"() : () -> !nod.message
  %0 = "nod.get_unique_message"() : () -> !nod.message
  return %0 : !nod.message
}
