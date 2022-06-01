// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <memory>
#include <utility>

#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilTypes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace Util {
namespace {

// Converts from |SourceType| to |TargetType|.
template <typename SourceType, typename TargetType>
struct PrimitiveTypeConverter : public TypeConverter {
  explicit PrimitiveTypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([&](SourceType type) -> Type {
      if (!isSourceType(type)) return type;
      return getTargetType(type);
    });
    addConversion([&](RankedTensorType type) {
      return RankedTensorType::get(type.getShape(),
                                   convertType(type.getElementType()),
                                   type.getEncoding());
    });
    addConversion([&](IREE::Util::PtrType ptrType) {
      return IREE::Util::PtrType::get(convertType(ptrType.getTargetType()));
    });
  }

  virtual ~PrimitiveTypeConverter() = default;

  // Returns true if |type| matches the expected source type.
  // Subclasses can override to restrict their conversion to specific subtypes.
  virtual bool isSourceType(SourceType type) { return true; }

  // Returns the newly converted type of |type|.
  // Subclasses can override to pass additional type parameters.
  virtual Type getTargetType(SourceType type) = 0;
};

// Returns |oldAttr| converted to its new type via |typeConverter|, if needed.
static Attribute convertAttribute(Location loc, Attribute oldAttr,
                                  TypeConverter &typeConverter) {
  // Type attributes get their nested type converted.
  if (auto oldTypeAttr = oldAttr.dyn_cast<TypeAttr>()) {
    return TypeAttr::get(typeConverter.convertType(oldTypeAttr.getValue()));
  }

  // Convert the attribute type - if it's the same then it's already legal.
  auto oldType = oldAttr.getType();
  auto newType = typeConverter.convertType(oldType);
  if (oldType == newType) return oldAttr;

  if (auto intAttr = oldAttr.dyn_cast<IntegerAttr>()) {
    APInt value = intAttr.getValue();
    if (newType.isSignedInteger()) {
      value = value.truncSSat(newType.getIntOrFloatBitWidth());
    } else if (newType.isUnsignedInteger()) {
      value = value.truncUSat(newType.getIntOrFloatBitWidth());
    } else {
      value = value.trunc(newType.getIntOrFloatBitWidth());
    }
    return IntegerAttr::get(newType, value);
  } else if (auto floatAttr = oldAttr.dyn_cast<FloatAttr>()) {
    auto newFloatType = newType.cast<FloatType>();
    APFloat value = floatAttr.getValue();
    bool losesInfo = false;
    value.convert(newFloatType.getFloatSemantics(), APFloat::rmTowardZero,
                  &losesInfo);
    return FloatAttr::get(newType, value);
  } else if (auto splatAttr = oldAttr.dyn_cast<SplatElementsAttr>()) {
    // NOTE: splats are also dense but this way we avoid needing to convert the
    // same splat value N times.
    return SplatElementsAttr::get(
        newType.cast<ShapedType>(),
        convertAttribute(loc, splatAttr.getSplatValue<Attribute>(),
                         typeConverter));
  } else if (auto denseAttr = oldAttr.dyn_cast<DenseIntElementsAttr>()) {
    auto newElementType = newType.cast<ShapedType>().getElementType();
    auto newElementBitWidth = newElementType.getIntOrFloatBitWidth();
    if (newElementType.isSignedInteger()) {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.truncSSat(newElementBitWidth);
      });
    } else if (newElementType.isUnsignedInteger()) {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.truncUSat(newElementBitWidth);
      });
    } else {
      return denseAttr.mapValues(newElementType, [&](APInt src) {
        return src.trunc(newElementBitWidth);
      });
    }
  } else if (auto denseAttr = oldAttr.dyn_cast<DenseFPElementsAttr>()) {
    auto newElementType =
        newType.cast<ShapedType>().getElementType().cast<FloatType>();
    const auto &newFloatSemantics = newElementType.getFloatSemantics();
    return denseAttr.mapValues(newElementType, [&](APFloat src) {
      bool losesInfo = false;
      src.convert(newFloatSemantics, APFloat::rmTowardZero, &losesInfo);
      return src.bitcastToAPInt();
    });
  }

  return oldAttr;
}

// Tries to completely convert a generic Operation.
// This will process attributes, result types, and nested regions.
struct GenericTypeConversionPattern : public ConversionPattern {
  GenericTypeConversionPattern(MLIRContext *context,
                               TypeConverter &typeConverter)
      : ConversionPattern(typeConverter, MatchAnyOpTypeTag(), 0, context) {}
  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Convert attributes only if this is a constant-like op.
    // This is because some ops use typed attributes for structural information
    // - like linalg ops using i64 for dimension indices - and if we converted
    // them all the ops would become invalid. This may still be too broad,
    // though, if some constant ops include attributes with both the type we
    // want to convert and structural information in the same type.
    llvm::SmallVector<NamedAttribute> newAttrs;
    if (op->hasTrait<OpTrait::ConstantLike>() ||
        isa<IREE::Util::GlobalOp>(op)) {
      for (auto attr : op->getAttrs()) {
        auto newAttr = convertAttribute(op->getLoc(), attr.getValue(),
                                        *getTypeConverter());
        newAttrs.push_back(NamedAttribute(attr.getName(), newAttr));
      }
    } else {
      newAttrs.append(op->getAttrs().begin(), op->getAttrs().end());
    }

    llvm::SmallVector<Type, 4> newResults;
    (void)getTypeConverter()->convertTypes(op->getResultTypes(), newResults);

    OperationState state(op->getLoc(), op->getName().getStringRef(), operands,
                         newResults, newAttrs, op->getSuccessors());

    for (Region &r : op->getRegions()) {
      Region *newRegion = state.addRegion();
      rewriter.inlineRegionBefore(r, *newRegion, newRegion->begin());
      TypeConverter::SignatureConversion result(newRegion->getNumArguments());
      (void)getTypeConverter()->convertSignatureArgs(
          newRegion->getArgumentTypes(), result);
      rewriter.applySignatureConversion(newRegion, result);
    }

    Operation *newOp = rewriter.create(state);
    rewriter.replaceOp(op, newOp->getResults());
    return success();
  }
};

template <typename OpTy, typename TypeTy,
          typename OperandToResultWidthLegalityRelation>
struct ConvertTypeSensitiveArithCastOp : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = this->getTypeConverter()
                          ->convertType(op.getResult().getType())
                          .template cast<TypeTy>();
    auto operandType = this->getTypeConverter()
                           ->convertType(op.getOperand().getType())
                           .template cast<TypeTy>();
    // If post-conversion, the types would be equal, then the op becomes a
    // no-op. Note that the op does not itself allow such a configuration, so we
    // have to catch this before creating the new op.
    if (resultType == operandType) {
      rewriter.replaceOp(op, adaptor.getOperands()[0]);
      return success();
    }
    // If after conversion the op becomes invalid, but not same-type (which we
    // can fold above), then bail out.
    // TODO: In some cases, we can repair the situation here, but for integer
    // truncation, we don't know whether we should invert with signed or
    // unsigned extension.
    if (!OperandToResultWidthLegalityRelation()(operandType.getWidth(),
                                                resultType.getWidth())) {
      return rewriter.notifyMatchFailure(op, "invalid width combination");
    }
    rewriter.replaceOpWithNewOp<OpTy>(op, resultType, op.getOperand());
    return success();
  }
};

template <typename T, typename Converter>
struct ConvertTypesPass : public PassWrapper<T, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    MLIRContext *context = &this->getContext();
    RewritePatternSet patterns(context);
    patterns.insert<GenericTypeConversionPattern>(context, typeConverter);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::TruncFOp, FloatType,
                                                    std::greater<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtFOp, FloatType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<
        arith::TruncIOp, IntegerType, std::less<unsigned>>>(typeConverter,
                                                            context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtUIOp, IntegerType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    patterns.insert<ConvertTypeSensitiveArithCastOp<arith::ExtSIOp, IntegerType,
                                                    std::less<unsigned>>>(
        typeConverter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(
        patterns, typeConverter);
    ConversionTarget target(*context);

    // Operations are legal if they don't contain any illegal type.
    target.markUnknownOpDynamicallyLegal([&](Operation *op) {
      if (auto globalOp = dyn_cast<IREE::Util::GlobalOp>(op)) {
        return typeConverter.isLegal(globalOp.type());
      } else if (auto funcOp = dyn_cast<func::FuncOp>(op)) {
        for (Type type : funcOp.getFunctionType().getInputs()) {
          if (!typeConverter.isLegal(type)) return false;
        }
        for (Type type : funcOp.getFunctionType().getResults()) {
          if (!typeConverter.isLegal(type)) return false;
        }
      }
      for (Type type : op->getResultTypes()) {
        if (!typeConverter.isLegal(type)) return false;
      }
      for (Type type : op->getOperandTypes()) {
        if (!typeConverter.isLegal(type)) return false;
      }
      for (auto &region : op->getRegions()) {
        if (!typeConverter.isLegal(&region)) return false;
      }
      return true;
    });

    // Note that this will fail if we can't convert any types.
    if (failed(applyFullConversion(this->getOperation(), target,
                                   std::move(patterns)))) {
      return this->signalPassFailure();
    }
  }

  Converter typeConverter;
};


template <typename T, typename Converter>
struct SignAwareDemotePass : public PassWrapper<T, OperationPass<mlir::ModuleOp>> {
  void runOnOperation() override {
    // MLIRContext *context = &this->getContext();
    auto moduleOp =  this->getOperation();
    // OpBuilder builder(context);
    // OpBuilder builder(moduleOp.getBodyRegion());
    for (const auto &op : moduleOp.getOps()) {
        if(!isa<func::FuncOp>(op)) continue;
        auto funcOp = dyn_cast<func::FuncOp>(op);
        // if(!op) continue;
        funcOp.walk([&](Operation* nestedOp) {
          // llvm::outs()<<"op:"<<nestedOp->getName()<<"\n";
          //if(something) return WalkResult::skip();
          if(checkSignedness(nestedOp)){
            OpBuilder builder(nestedOp->getParentRegion());
            demoteSignedOp(nestedOp, builder);
          }
        });
    }
  }

  bool checkSignedness(Operation *op) {
    if(isa<arith::CmpIOp>(op)) {
      bool isSigned = op->getAttr("predicate").dyn_cast<mlir::arith::CmpIPredicateAttr>().getValue() < mlir::arith::CmpIPredicate::ult ? true : false;
      return isSigned;
    }
    if(isa<arith::SelectOp>(op)) {
      for (auto childOp : op->getUsers()) {
        bool childOpSign = checkSignedness(childOp);
        if(childOpSign) {
          return true;
        }
      }
    }
    return false;
  }

  void demoteSignedOp(Operation* op, OpBuilder& builder) {
      // llvm::SmallVector<Value, 4> opOperands;
      // llvm::SmallVector<NamedAttribute, 1> opAttrs;
      // auto resultTypes = op.getResultTypes();
      for (uint operandIdx = 0; operandIdx < op->getNumOperands(); operandIdx++) {
        Value operand = op->getOperand(operandIdx);
        if(!operand.getDefiningOp()) continue;
        auto oldType = operand.getType();
        auto newType = typeConverter.convertType(oldType);
        if(oldType == newType) continue;
        auto constOp = dyn_cast<arith::ConstantOp>(operand.getDefiningOp());
        if(!constOp) continue;
        auto intAttr = constOp->getAttr("value").dyn_cast<IntegerAttr>();
        APInt value = intAttr.getValue();
        APInt newValue = value.truncSSat(newType.getIntOrFloatBitWidth());
        if(newValue == value) continue;
        auto newConstantOp =  builder.create<arith::ConstantOp>(constOp->getLoc(), IntegerAttr::get(oldType, newValue));
        op->setOperand(operandIdx, newConstantOp.getResult());
      }
    }

  Converter typeConverter;
};

}  // namespace

namespace {
struct DemoteI64ToI32Converter
    : public PrimitiveTypeConverter<IntegerType, IntegerType> {
  bool isSourceType(IntegerType type) override { return type.isInteger(64); }
  Type getTargetType(IntegerType type) override {
    return IntegerType::get(type.getContext(), 32, type.getSignedness());
  }
};
struct DemoteI64ToI32Pass
    : public ConvertTypesPass<DemoteI64ToI32Pass, DemoteI64ToI32Converter> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteI64ToI32Pass)

  StringRef getArgument() const override {
    return "iree-util-demote-i64-to-i32";
  }
  StringRef getDescription() const override {
    return "Demotes i64 types to i32 types.";
  }
};

struct SignednessPrepI64ToI32Pass
    : public SignAwareDemotePass<DemoteI64ToI32Pass, DemoteI64ToI32Converter> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteI64ToI32Pass)

  StringRef getArgument() const override {
    return "iree-util-prep-demote-i64-to-i32";
  }
  StringRef getDescription() const override {
    return "Preprocess Demotion i64 types to i32 types.";
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteI64ToI32Pass() {
  return std::make_unique<DemoteI64ToI32Pass>();
}
static PassRegistration<DemoteI64ToI32Pass> demoteI64ToI32Pass;

std::unique_ptr<OperationPass<mlir::ModuleOp>> createSignednessPrepI64ToI32Pass() {
  return std::make_unique<SignednessPrepI64ToI32Pass>();
}
static PassRegistration<SignednessPrepI64ToI32Pass> signedPrepDemoteI64ToI32Pass;

namespace {
struct DemoteF32ToF16Converter
    : public PrimitiveTypeConverter<Float32Type, Float16Type> {
  Type getTargetType(Float32Type type) override {
    return Float16Type::get(type.getContext());
  }
};
struct DemoteF32ToF16Pass
    : public ConvertTypesPass<DemoteF32ToF16Pass, DemoteF32ToF16Converter> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteF32ToF16Pass)

  StringRef getArgument() const override {
    return "iree-util-demote-f32-to-f16";
  }
  StringRef getDescription() const override {
    return "Demotes f32 types to f16 types.";
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF32ToF16Pass() {
  return std::make_unique<DemoteF32ToF16Pass>();
}
static PassRegistration<DemoteF32ToF16Pass> demoteF32ToF16Pass;

namespace {
struct PromoteF16ToF32Converter
    : public PrimitiveTypeConverter<Float16Type, Float32Type> {
  Type getTargetType(Float16Type type) override {
    return Float32Type::get(type.getContext());
  }
};
struct PromoteF16ToF32Pass
    : public ConvertTypesPass<PromoteF16ToF32Pass, PromoteF16ToF32Converter> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PromoteF16ToF32Pass)

  StringRef getArgument() const override {
    return "iree-util-promote-f16-to-f32";
  }
  StringRef getDescription() const override {
    return "Promotes f16 types to f32 types.";
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createPromoteF16ToF32Pass() {
  return std::make_unique<PromoteF16ToF32Pass>();
}
static PassRegistration<PromoteF16ToF32Pass> promoteF16ToF32Pass;

namespace {
struct DemoteF64ToF32Converter
    : public PrimitiveTypeConverter<Float64Type, Float32Type> {
  Type getTargetType(Float64Type type) override {
    return Float32Type::get(type.getContext());
  }
};
struct DemoteF64ToF32Pass
    : public ConvertTypesPass<DemoteF64ToF32Pass, DemoteF64ToF32Converter> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DemoteF64ToF32Pass)

  StringRef getArgument() const override {
    return "iree-util-demote-f64-to-f32";
  }
  StringRef getDescription() const override {
    return "Demotes f64 types to f32 types.";
  }
};
}  // namespace

std::unique_ptr<OperationPass<mlir::ModuleOp>> createDemoteF64ToF32Pass() {
  return std::make_unique<DemoteF64ToF32Pass>();
}
static PassRegistration<DemoteF64ToF32Pass> demoteF64ToF32Pass;

}  // namespace Util
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
