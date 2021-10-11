// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Tests the (automatic) conversion from the nod dialect to the VM dialect.
// Depending on whether any manual conversion is performed this may get complex,
// such as when versioning imports or performing optimizations.

// RUN: iree-opt %s -iree-convert-to-hal -iree-shape-expand-function-ranked-shape-dims -iree-vm-conversion -split-input-file | IreeFileCheck %s

// CHECK-LABEL: @tensorToMessage
func @tensorToMessage(%tensor : tensor<2x4xf32>) {
  //  CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  //  CHECK-DAG: [[ENCODING:%.+]] = vm.const.i32 1 : i32
  //  CHECK-DAG: [[DIM0:%.+]] = vm.const.i32 2 : i32
  //  CHECK-DAG: [[DIM1:%.+]] = vm.const.i32 4 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(
  // CHECK-SAME:     %arg0, [[TYPE]], [[ENCODING]], [
  // CHECK-SAME:       [[DIM0]], [[DIM1]]
  // CHECK-SAME:     ])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @nod.buffer_to_message([[VIEW]]) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!nod.message>
  %0 = "nod.tensor_to_message"(%tensor) : (tensor<2x4xf32>) -> !nod.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @nod.print([[MSG]]
  "nod.print"(%0, %c1) : (!nod.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage
func @dynamicTensorToMessage(%arg0 : tensor<?x?xf32>, %arg1 : index, %arg2 : index) {
  //  CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  //  CHECK-DAG: [[ENCODING:%.+]] = vm.const.i32 1 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(
  // CHECK-SAME:     %arg0, [[TYPE]], [[ENCODING]], [%arg1, %arg2])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @nod.buffer_to_message([[VIEW]]) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!nod.message>
  %shape = shapex.make_ranked_shape %arg1, %arg2 : (index, index) -> !shapex.ranked_shape<[?, ?]>
  %shaped_tensor = shapex.tie_shape %arg0, %shape : tensor<?x?xf32>, !shapex.ranked_shape<[?, ?]>
  %0 = "nod.tensor_to_message"(%shaped_tensor) : (tensor<?x?xf32>) -> !nod.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @nod.print([[MSG]]
  "nod.print"(%0, %c1) : (!nod.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @dynamicTensorToMessage2
func @dynamicTensorToMessage2(%arg0 : tensor<?x?xf32>, %arg1: !shapex.ranked_shape<[?, ?]> {iree.reflection = {}}) {
  //  CHECK-DAG: [[TYPE:%.+]] = vm.const.i32 50331680 : i32
  //  CHECK-DAG: [[ENCODING:%.+]] = vm.const.i32 1 : i32
  // CHECK-NEXT: [[VIEW:%.+]] = vm.call.variadic @hal.buffer_view.create(
  // CHECK-SAME:     %arg0, [[TYPE]], [[ENCODING]], [%arg1, %arg2])
  // CHECK-NEXT: [[MSG:%.+]] = vm.call @nod.buffer_to_message([[VIEW]]) {nosideeffects} : (!vm.ref<!hal.buffer_view>) -> !vm.ref<!nod.message>
  %shaped_tensor = shapex.tie_shape %arg0, %arg1 : tensor<?x?xf32>, !shapex.ranked_shape<[?, ?]>
  %0 = "nod.tensor_to_message"(%shaped_tensor) : (tensor<?x?xf32>) -> !nod.message
  %c1 = constant 1 : i32
  // CHECK: vm.call @nod.print([[MSG]]
  "nod.print"(%0, %c1) : (!nod.message, i32) -> ()
  return
}

// -----

// CHECK-LABEL: @messageToTensor
func @messageToTensor(%arg0 : !nod.message) -> tensor<2x4xf32> {
  // CHECK: [[VIEW:%.+]] = vm.call @nod.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!nod.message>) -> !vm.ref<!hal.buffer_view>
  %0 = "nod.message_to_tensor"(%arg0) : (!nod.message) -> tensor<2x4xf32>
  // CHECK-NEXT: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]])
  // CHECK-NEXT: vm.return [[BUFFER]]
  return %0 : tensor<2x4xf32>
}

// -----

// CHECK-LABEL: @messageToTensorReturnDim
func @messageToTensorReturnDim(%arg0 : !nod.message) -> index {
  %0 = "nod.message_to_tensor"(%arg0) : (!nod.message) -> tensor<?x4xf32>
  %c0 = constant 0 : index
  %1 = tensor.dim %0, %c0 : tensor<?x4xf32>
  // CHECK: [[VIEW:%.+]] = vm.call @nod.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!nod.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]])
  // CHECK: %{{.*}} = vm.const.i32.zero
  // CHECK: [[ZERO:%.+]] = vm.const.i32.zero
  // CHECK: [[DIM:%.+]] = vm.call @hal.buffer_view.dim([[VIEW]], [[ZERO]])
  // CHECK: vm.return [[DIM]]
  return %1 : index
}

// -----

// CHECK-LABEL: @messageToTensorReturnRank
func @messageToTensorReturnRank(%arg0 : !nod.message) -> index {
  %0 = "nod.message_to_tensor"(%arg0) : (!nod.message) -> tensor<*xf32>
  %1 = rank %0 : tensor<*xf32>
  // CHECK-DAG: [[VIEW:%.+]] = vm.call @nod.message_to_buffer(%arg0) {nosideeffects} : (!vm.ref<!nod.message>) -> !vm.ref<!hal.buffer_view>
  // CHECK-DAG: [[BUFFER:%.+]] = vm.call @hal.buffer_view.buffer([[VIEW]])
  // CHECK-DAG: [[RANK:%.+]] = vm.call @hal.buffer_view.rank([[VIEW]])
  // CHECK: vm.return [[RANK]]
  return %1 : index
}

// -----

// CHECK-LABEL: @printOp
func @printOp(%arg0 : !nod.message) {
  %c1_i32 = constant 1 : i32
  // CHECK: vm.call @nod.print(%arg0, %c1) : (!vm.ref<!nod.message>, i32) -> ()
  "nod.print"(%arg0, %c1_i32) : (!nod.message, i32) -> ()
  return
}

// CHECK: vm.import @nod.print

// -----

// CHECK-LABEL: @reverseOp
func @reverseOp(%arg0 : !nod.message) -> !nod.message {
  // CHECK: %ref = vm.call @nod.reverse(%arg0) {nosideeffects} : (!vm.ref<!nod.message>) -> !vm.ref<!nod.message>
  %0 = "nod.reverse"(%arg0) : (!nod.message) -> !nod.message
  return %0 : !nod.message
}

// CHECK: vm.import @nod.reverse

// -----

// CHECK: vm.import @nod.get_unique_message
// CHECK-LABEL: @getUniqueMessageOp
func @getUniqueMessageOp() -> !nod.message {
  // CHECK: %ref = vm.call @nod.get_unique_message() {nosideeffects} : () -> !vm.ref<!nod.message>
  %0 = "nod.get_unique_message"() : () -> !nod.message
  return %0 : !nod.message
}

