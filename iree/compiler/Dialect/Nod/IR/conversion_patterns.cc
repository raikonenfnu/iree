// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/Nod/IR/conversion_patterns.h"

#include "iree/compiler/Dialect/HAL/Conversion/ConversionTarget.h"
#include "iree/compiler/Dialect/VM/Conversion/ImportUtils.h"
#include "iree/compiler/Dialect/Nod/IR/nod_dialect.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Nod {

void populateNodToHALPatterns(MLIRContext *context,
                                 OwningRewritePatternList &patterns,
                                 TypeConverter &typeConverter) {
  // We can use the HAL conversion handler for this tensor->buffer conversion
  // as we just want the simple form. If we wanted to perform additional
  // verification or have a specific use case (such as a place where only the
  // buffer is required and the shape is not) we could add our own.
  patterns.insert<HALOpConversion<TensorToMessageOp, BufferToMessageOp>>(
      context, typeConverter);
  patterns.insert<HALOpConversion<MessageToTensorOp, MessageToBufferOp>>(
      context, typeConverter);
}

void populateNodToVMPatterns(MLIRContext *context,
                                SymbolTable &importSymbols,
                                OwningRewritePatternList &patterns,
                                TypeConverter &typeConverter) {
  // We can use the VM conversion handler for all of these as they are simple
  // 1:1 mappings. More complex mappings can provide their own conversions
  // (such as the HAL dialect does).
  patterns.insert<VMImportOpConversion<IREE::Nod::BufferToMessageOp>>(
      context, importSymbols, typeConverter, "nod.buffer_to_message");
  patterns.insert<VMImportOpConversion<IREE::Nod::MessageToBufferOp>>(
      context, importSymbols, typeConverter, "nod.message_to_buffer");
  patterns.insert<VMImportOpConversion<IREE::Nod::PrintOp>>(
      context, importSymbols, typeConverter, "nod.print");
  patterns.insert<VMImportOpConversion<IREE::Nod::ReverseOp>>(
      context, importSymbols, typeConverter, "nod.reverse");
  patterns.insert<VMImportOpConversion<IREE::Nod::GetUniqueMessageOp>>(
      context, importSymbols, typeConverter, "nod.get_unique_message");
}

}  // namespace Nod
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
