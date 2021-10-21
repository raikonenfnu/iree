// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Describes a nod module implemented in native code.
// The imports are used for mapping higher-level IR types to the VM ABI and for
// attaching additional attributes for compiler optimization.
//
// Each import defined here has a matching function exported from the native
// module (nod_modules/native_module.cc). In most cases an op in the source
// dialect will map directly to an import here, though it's possible for
// versioning and overrides to cause M:N mappings.
vm.module @nod {

// Formats the tensor using the IREE buffer printer to have a shape/type and
// the contents as a string.
vm.import @buffer_to_message(
  %buffer_view : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!nod.message>
attributes {nosideeffects}

// Parses the message containing a IREE buffer parser-formatted tensor.
vm.import @message_to_buffer(
  %message : !vm.ref<!nod.message>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Formats the tensor using the IREE buffer printer to have a shape/type and
// the contents as a string.
vm.import @matmul_buffer(
  %lhs_buffer_view : !vm.ref<!hal.buffer_view>,
  %rhs_buffer_view : !vm.ref<!hal.buffer_view>,
  %out_buffer_view : !vm.ref<!hal.buffer_view>
) -> !vm.ref<!hal.buffer_view>
attributes {nosideeffects}

// Prints the %message provided %count times.
// Maps to the IREE::Nod::PrintOp.
vm.import @print(
  %message : !vm.ref<!nod.message>,
  %count : i32
)

// Returns the message with its characters reversed.
// Maps to the IREE::Nod::ReverseOp.
vm.import @reverse(
  %message : !vm.ref<!nod.message>
) -> !vm.ref<!nod.message>
attributes {nosideeffects}

// Returns a per-context unique message.
// Maps to the IREE::Nod::GetUniqueMessageOp.
vm.import @get_unique_message() -> !vm.ref<!nod.message>
attributes {nosideeffects}

}  // vm.module
