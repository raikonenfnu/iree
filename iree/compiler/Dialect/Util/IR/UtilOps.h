// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_
#define IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_

#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"

#define GET_OP_CLASSES
#include "iree/compiler/Dialect/Util/IR/UtilOps.h.inc"  // IWYU pragma: export

namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Utils
//===----------------------------------------------------------------------===//

// Returns the dynamic size of the value at |index|.
Value findValueSizeInList(unsigned index, ValueRange values, ValueRange sizes);

//===----------------------------------------------------------------------===//
// custom<SymbolVisibility>($sym_visibility)
//===----------------------------------------------------------------------===//
// some.op custom<SymbolVisibility>($sym_visibility) $sym_name
// ->
// some.op @foo
// some.op private @foo

ParseResult parseSymbolVisibility(OpAsmParser &parser,
                                  StringAttr &symVisibilityAttr);
void printSymbolVisibility(OpAsmPrinter &p, Operation *op,
                           StringAttr symVisibilityAttr);

//===----------------------------------------------------------------------===//
// custom<TypeOrAttr>($type, $attr)
//===----------------------------------------------------------------------===//
// some.op custom<TypeOrAttr>($type, $attr)
// ->
// some.op : i32
// some.op = 42 : i32
// some.op : i32 = 42 : index

ParseResult parseTypeOrAttr(OpAsmParser &parser, TypeAttr &typeAttr,
                            Attribute &attr);
void printTypeOrAttr(OpAsmPrinter &p, Operation *op, TypeAttr type,
                     Attribute attr);

//===----------------------------------------------------------------------===//
// custom<TypeAlias>($encoding_type, $storage_type)
//===----------------------------------------------------------------------===//
// some.op custom<TypeAlias>($encoding_type, $storage_type)
// ->
// some.op tensor<4xf32>
// some.op tensor<4xf32> as tensor<2xf64>
// some.op tensor<4xf32> as tensor<?xf32>{...}

ParseResult parseTypeAlias(OpAsmParser &parser, TypeAttr &encodingTypeAttr,
                           Type &storageType);
void printTypeAlias(OpAsmPrinter &p, Operation *op, TypeAttr encodingTypeAttr,
                    Type storageType);

//===----------------------------------------------------------------------===//
// custom<SizeAwareType>
//===----------------------------------------------------------------------===//
// type{%size}

ParseResult parseSizeAwareType(OpAsmParser &parser, Type &type,
                               OpAsmParser::OperandType &size);
void printSizeAwareType(OpAsmPrinter &p, Operation *op, Type type, Value size);

//===----------------------------------------------------------------------===//
// custom<SizeAwareTypeList>
//===----------------------------------------------------------------------===//
// (type{%size0}, type, type{%size1})

ParseResult parseSizeAwareTypeList(
    OpAsmParser &parser, SmallVectorImpl<Type> &types,
    SmallVectorImpl<OpAsmParser::OperandType> &sizes);
void printSizeAwareTypeList(OpAsmPrinter &p, Operation *op, TypeRange types,
                            OperandRange sizes);
ParseResult parseSizeAwareTypeList(
    OpAsmParser &parser, SmallVectorImpl<Type> &types0,
    SmallVectorImpl<Type> &types1,
    SmallVectorImpl<OpAsmParser::OperandType> &sizes);
void printSizeAwareTypeList(OpAsmPrinter &p, Operation *op, TypeRange types0,
                            TypeRange types1, OperandRange sizes);

//===----------------------------------------------------------------------===//
// custom<ShapedTiedResult>
//===----------------------------------------------------------------------===//
// type{%dim0, %dim1}
// %arg0 as type{%dim0}

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims);
inline ParseResult parseShapedTiedResult(OpAsmParser &parser, Type &resultType,
                                         OpAsmParser::OperandType &resultDim) {
  SmallVector<OpAsmParser::OperandType, 1> resultDims;
  if (failed(parseShapedTiedResult(parser, resultType, resultDims))) {
    return failure();
  }
  assert(resultDims.size() == 1 && "requires one dim");
  resultDim = std::move(resultDims.front());
  return success();
}
void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims);

ParseResult parseShapedTiedResult(
    OpAsmParser &parser, Type &resultType,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands);
void printShapedTiedResult(OpAsmPrinter &p, Operation *op, Type resultType,
                           ValueRange resultDims, ArrayAttr tiedOperands);

inline ParseResult parseShapedTiedResult(OpAsmParser &parser, Type &resultType,
                                         OpAsmParser::OperandType &resultDim,
                                         ArrayAttr &tiedOperands) {
  SmallVector<OpAsmParser::OperandType> resultDims;
  if (failed(parseShapedTiedResult(parser, resultType, resultDims,
                                   tiedOperands))) {
    return failure();
  }
  assert(resultDims.size() == 1 && "requires one dim");
  resultDim = std::move(resultDims.front());
  return success();
}
inline void printShapedTiedResult(OpAsmPrinter &p, Operation *op,
                                  Type resultType, Value resultDim,
                                  ArrayAttr tiedOperands) {
  printShapedTiedResult(p, op, resultType, ValueRange{resultDim}, tiedOperands);
}

ParseResult parseShapedResultList(
    OpAsmParser &parser, ArrayRef<OpAsmParser::OperandType> operands,
    TypeRange operandTypes, ArrayRef<OpAsmParser::OperandType> operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands);
void printShapedResultList(OpAsmPrinter &p, Operation *op, ValueRange operands,
                           TypeRange operandTypes, ValueRange operandDims,
                           TypeRange resultTypes, ValueRange resultDims,
                           ArrayAttr tiedOperands);

//===----------------------------------------------------------------------===//
// custom<ShapedFunctionType>
//===----------------------------------------------------------------------===//
// (type, type{%dim0, %dim1}, type) -> (type{%dim2}, %operand4)

ParseResult parseShapedFunctionType(
    OpAsmParser &parser, ArrayRef<OpAsmParser::OperandType> operands,
    SmallVectorImpl<Type> &operandTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &operandDims,
    SmallVectorImpl<Type> &resultTypes,
    SmallVectorImpl<OpAsmParser::OperandType> &resultDims,
    ArrayAttr &tiedOperands);
void printShapedFunctionType(OpAsmPrinter &p, Operation *op,
                             ValueRange operands, TypeRange operandTypes,
                             OperandRange operandDims, TypeRange resultTypes,
                             OperandRange resultDims, ArrayAttr tiedOperands);

}  // namespace iree_compiler
}  // namespace mlir

#endif  // IREE_COMPILER_DIALECT_UTIL_IR_UTILOPS_H_
