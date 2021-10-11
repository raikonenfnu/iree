// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Prints the %message provided reversed %count times using the native
// implementation of the "nod.print" op.
//
// See nod_modules/dialect/nod_ops.td for the op definitions and
// nod_modules/dialect/nod.imports.mlir for the import definitions.
func @reverseAndPrint(%message : !nod.message, %count : i32) -> !nod.message
    attributes { iree.module.export, iree.abi.none } {
  %c1 = constant 1 : i32
  %0 = "nod.get_unique_message"() : () -> !nod.message
  "nod.print"(%0, %c1) : (!nod.message, i32) -> ()
  %1 = call @reverse(%message) : (!nod.message) -> !nod.message
  "nod.print"(%1, %count) : (!nod.message, i32) -> ()
  return %1 : !nod.message
}

// Reverses a message. Just an example to show intra-module calls.
func @reverse(%message : !nod.message) -> !nod.message {
  %0 = "nod.reverse"(%message) : (!nod.message) -> !nod.message
  return %0 : !nod.message
}

// Prints the provided tensor to by first converting it to a message.
func @printTensor(%tensor : tensor<2x4xf32>) -> !nod.message
    attributes { iree.module.export, iree.abi.none } {
  %0 = "nod.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !nod.message
  %c1 = constant 1 : i32
  "nod.print"(%0, %c1) : (!nod.message, i32) -> ()
  return %0 : !nod.message
}

// Round-trips a tensor through a message.
func @roundTripTensor(%tensor : tensor<2x4xf32>) -> !nod.message
    attributes { iree.module.export, iree.abi.none } {
  %0 = "nod.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !nod.message
  %1 = "nod.message_to_tensor"(%0) : (!nod.message) -> tensor<2x4xf32>
  %2 = "nod.tensor_to_message"(%1) : (tensor<2x4xf32>) -> !nod.message
  %c1 = constant 1 : i32
  "nod.print"(%2, %c1) : (!nod.message, i32) -> ()
  return %0 : !nod.message
}
