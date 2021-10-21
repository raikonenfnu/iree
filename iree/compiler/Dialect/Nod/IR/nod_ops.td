// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_SAMPLES_NOD_MODULES_DIALECT_NOD_OPS
#define IREE_SAMPLES_NOD_MODULES_DIALECT_NOD_OPS

include "iree/compiler/Dialect/HAL/IR/HALBase.td"
include "iree/compiler/Dialect/Util/IR/UtilBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def NOD_Dialect : Dialect {
  let name = "nod";
  let cppNamespace = "::mlir::iree_compiler::IREE::Nod";

  let summary = [{
    A nod dialect demonstrating nod ops and implementations.
  }];
  let description = [{
    The ops in this dialect are lowered to vm.imports as described in the
    `nod_modules/dialect/nod.imports.mlir` file and the implementations
    are provided by the `nod_modules/native_module.cc` file.
  }];
}

def NOD_Message : DialectType<
    NOD_Dialect,
    CPred<"$_self.isa<IREE::Nod::MessageType>()">,
    "message"> {
  let description = [{
    A string message that can be printed. These types will be reference counted
    at runtime and must derive from iree_vm_ref_t but otherwise can be whatever
    the dialect wants.
  }];
}

def NOD_TensorToMessageOp : Op<NOD_Dialect, "tensor_to_message"> {
  let summary = [{formats tensor contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.

    This demonstrates handling tensor->buffer conversion through the HAL layer.
    This op (tensor_to_message) is only used prior to translation and instances
    of it will be converted to the lower-level HAL-based buffer_to_message op.
  }];

  let arguments = (ins
    AnyTensor:$operand
  );
  let results = (outs
    NOD_Message:$result
  );
}

def NOD_BufferToMessageOp : Op<NOD_Dialect, "buffer_to_message"> {
  let summary = [{formats buffer contents as a message}];
  let description = [{
    Formats the tensor using the IREE buffer printer to have a shape/type and
    the contents as a string.
  }];

  let arguments = (ins
    HAL_BufferView:$buffer_view
  );
  let results = (outs
    NOD_Message:$result
  );
}

def NOD_MessageToTensorOp : Op<NOD_Dialect, "message_to_tensor"> {
  let summary = [{parses message contents as a tensor}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    NOD_Message:$message
  );
  let results = (outs
    AnyTensor:$result
  );
}

def NOD_MessageToBufferOp : Op<NOD_Dialect, "message_to_buffer"> {
  let summary = [{parses message contents as a buffer}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    NOD_Message:$message
  );
  let results = (outs
    HAL_BufferView:$result
  );
}

def NOD_MatmulTensorOp : Op<NOD_Dialect, "matmul_tensor"> {
  let summary = [{parses message contents as a tensor}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    AnyTensor:$lhs_operand,
    AnyTensor:$rhs_operand,
    AnyTensor:$out_operand
  );
  let results = (outs
    AnyTensor:$result
  );
}

def NOD_MatmulBufferOp : Op<NOD_Dialect, "matmul_buffer"> {
  let summary = [{parses message contents as a buffer}];
  let description = [{
    Parses the message containing a IREE buffer parser-formatted tensor.
  }];

  let arguments = (ins
    HAL_BufferView:$lhs_buffer_view,
    HAL_BufferView:$rhs_buffer_view,
    HAL_BufferView:$out_buffer_view
  );
  let results = (outs
    HAL_BufferView:$result
  );
}

def NOD_PrintOp : Op<NOD_Dialect, "print"> {
  let summary = [{prints a message zero or more times}];
  let description = [{
    Prints the input message zero or more times with a newline inbetween.

    This demonstrates passing arguments from the application (the original
    message).
  }];

  let arguments = (ins
    NOD_Message:$message,
    I32:$count
  );
}

def NOD_ReverseOp : Op<NOD_Dialect, "reverse", [NoSideEffect]> {
  let summary = [{reverses the characters in the given the message}];
  let description = [{
    Reverses the message characters and returns the new value.

    This demonstrates allocating new values from within the op implementation
    (the reversed result). Note that this op (as well as the vm.import) are
    marked with NoSideEffect/nosideeffect to enable additional compiler
    optimization.
  }];

  let arguments = (ins
    NOD_Message:$operand
  );
  let results = (outs
    NOD_Message:$result
  );
}

def NOD_GetUniqueMessageOp : Op<NOD_Dialect, "get_unique_message", [NoSideEffect]> {
  let summary = [{returns a per-context unique message}];
  let description = [{
    Returns a unique message allocated per context using the module.

    This demonstrates context-specific global storage. Mutation methods could
    exist that update the global storage as desired.
  }];

  let results = (outs
    NOD_Message:$result
  );
}

#endif  // IREE_SAMPLES_NOD_MODULES_DIALECT_NOD_OPS
