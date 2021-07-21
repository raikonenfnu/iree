// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- XLAToLinalgOnTensors.cpp - Pass to convert XLA to Linalg on tensors-===//
//
// Pass to convert from XLA to linalg on tensers. Uses the patterns from
// tensorflow/compiler/mlir/xla/transforms/legalize_to_linalg.cc along with
// some IREE specific patterns.
//
//===----------------------------------------------------------------------===//
#include <memory>

#include "iree/compiler/Dialect/Flow/IR/FlowOps.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeDialect.h"
#include "iree/compiler/Dialect/Shape/IR/ShapeOps.h"
#include "iree/compiler/InputConversion/MHLO/ConvertMHLOToFlow.h"
#include "iree/compiler/InputConversion/MHLO/PassDetail.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Rewriters.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Complex/IR/Complex.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {
namespace {

//===----------------------------------------------------------------------===//
// mhlo.concatenate conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Converts mhlo.concatenate operation to extract_slice ops + insert_slice ops.
struct ConcatenateOpConversion
    : public OpConversionPattern<mhlo::ConcatenateOp> {
  using OpConversionPattern<mhlo::ConcatenateOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::ConcatenateOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    auto resultType = this->typeConverter->convertType(op.getResult().getType())
                          .dyn_cast<RankedTensorType>();
    if (!resultType || !resultType.hasStaticShape()) {
      return rewriter.notifyMatchFailure(op,
                                         "expected static shape for output");
    }

    Location loc = op.getLoc();
    int dim = op.dimension();
    int rank = resultType.getRank();
    SmallVector<Value, 3> offsets, sizes, strides;
    for (int i = 0; i < rank; ++i) {
      offsets.push_back(rewriter.create<ConstantIndexOp>(loc, 0));
      sizes.push_back(rewriter.create<tensor::DimOp>(loc, args[0], i));
      strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      auto size = rewriter.create<tensor::DimOp>(loc, arg, dim);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.create<tensor::DimOp>(loc, arg, dim);
      result = rewriter.create<tensor::InsertSliceOp>(loc, arg, result, offsets,
                                                      sizes, strides);
      accBound = rewriter.create<AddIOp>(loc, accBound, sizes[dim]);
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};
}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.fft conversion patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Creats coefficients based on DFT definition, see
/// https://en.wikipedia.org/wiki/Discrete_Fourier_transform
Value getDFTMatmulCoeff(OpBuilder b, Location loc, RankedTensorType matrixType,
                        bool isRealPart) {
  // scale = 2 * pi / N
  double scale = 2 * M_PI / matrixType.getDimSize(0);

  SmallVector<Attribute> values;
  assert(matrixType.getRank() == 2 && "expected 2D matrix");
  for (auto i : llvm::seq<unsigned>(0, matrixType.getDimSize(0))) {
    for (auto j : llvm::seq<unsigned>(0, matrixType.getDimSize(1))) {
      double v = scale * i * j;
      if (isRealPart) {
        v = cos(v);
      } else {
        v = -sin(v);
      }
      values.push_back(b.getF32FloatAttr(v));
    }
  }
  return b.create<ConstantOp>(loc, matrixType,
                              DenseFPElementsAttr::get(matrixType, values));
}

Value createLinalgMatmulOnTensors(OpBuilder b, Location loc,
                                  RankedTensorType resultType, Value lhs,
                                  Value rhs) {
  Value zero =
      b.create<ConstantOp>(loc, b.getZeroAttr(resultType.getElementType()));
  auto initTensor = b.create<linalg::InitTensorOp>(
      loc, /*dyn_size=*/ValueRange{}, resultType.getShape(),
      resultType.getElementType());
  Value zeroTensor =
      b.create<linalg::FillOp>(loc, zero, initTensor).getResult(0);

  switch (lhs.getType().cast<RankedTensorType>().getRank()) {
    case 1:
      return b
          .create<linalg::VecmatOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    case 2:
      return b
          .create<linalg::MatmulOp>(loc, TypeRange{resultType},
                                    ValueRange{lhs, rhs},
                                    ValueRange{zeroTensor})
          .getResult(0);
    default:
      llvm_unreachable("unhandled matmul type");
  }
}

/// Converts mhlo.fft operation to Linalg ops.
struct FftOpConversion : public OpConversionPattern<mhlo::FftOp> {
  using OpConversionPattern<mhlo::FftOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::FftOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const override {
    if (op.fft_type() != "RFFT") {
      return rewriter.notifyMatchFailure(op,
                                         "non RFFT types are supported yet");
    }

    mhlo::FftOpAdaptor adaptor(args);
    auto inputType = adaptor.operand().getType().dyn_cast<RankedTensorType>();
    if (!inputType || !inputType.hasStaticShape() || inputType.getRank() > 2) {
      return rewriter.notifyMatchFailure(op, "only static 1D or 2D dft ops");
    }

    int rank = inputType.getRank();
    int n = inputType.getDimSize(rank - 1);
    int fftLength =
        op.fft_length().getSplatValue().cast<IntegerAttr>().getInt() / 2 + 1;

    Location loc = op.getLoc();
    auto matrixType =
        RankedTensorType::get({n, fftLength}, inputType.getElementType());
    auto resultType =
        RankedTensorType::get(op.getType().cast<RankedTensorType>().getShape(),
                              inputType.getElementType());

    auto realMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/true);
    auto real = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), realMatrix);

    auto imagMatrix =
        getDFTMatmulCoeff(rewriter, loc, matrixType, /*isRealPart=*/false);
    auto imag = createLinalgMatmulOnTensors(rewriter, loc, resultType,
                                            adaptor.operand(), imagMatrix);

    // Pack the results back to mhlo::ComplexOp.
    rewriter.replaceOpWithNewOp<mhlo::ComplexOp>(op, op.getType(), real, imag);
    return success();
  }
};
}  // namespace

struct ConvertMHLOToLinalgOnTensorsPass
    : public ConvertMHLOToLinalgOnTensorsBase<
          ConvertMHLOToLinalgOnTensorsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Flow::FlowDialect, linalg::LinalgDialect,
                    mhlo::MhloDialect, ShapeDialect, math::MathDialect,
                    memref::MemRefDialect, complex::ComplexDialect>();
  }

  void runOnOperation() override {
    OwningRewritePatternList patterns(&getContext());
    MLIRContext *context = &getContext();

    auto typeConverter = mhlo::createHloToLinalgSignedIntegerConverter();
    // NOTE: not using corresponding setupMHLOToFlowPatterns because the entire
    // MHLO dialects are marked illegal by this pass.
    // TODO: Collapse/rework all of these patterns once the consolidation
    // lands. There is little reason to have these so spread out.
    populateMHLOToFlowPatterns(context, patterns);
    chlo::PopulateDecomposeChloPatterns(context, &patterns);
    populateMHLOBroadcastingToLinalgPatterns(context, *typeConverter, patterns);
    populateMHLOToLinalgOnTensorsConversionPatterns(context, *typeConverter,
                                                    patterns);
    populateMHLOComplexToRealPatterns(context, *typeConverter, patterns);

    ConversionTarget target(getContext());
    target.addIllegalDialect<chlo::HloClientDialect>();
    target.addIllegalDialect<mhlo::MhloDialect>();

    // Let the rest fall through.
    target.markUnknownOpDynamicallyLegal([](Operation *) { return true; });

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

/// Convert mhlo.constant op into std.const.
struct ConstOpConversion : public OpConversionPattern<mhlo::ConstOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      mhlo::ConstOp op, ArrayRef<Value> /*operands*/,
      ConversionPatternRewriter &rewriter) const override {
    auto valueAttr = op.value();
    Type oldElType = valueAttr.getType().getElementType();
    Type newElType = this->typeConverter->convertType(oldElType);
    ElementsAttr newValueAttr = valueAttr;
    if (newElType != oldElType) {
      // Values don't change, just their reported type.
      newValueAttr = valueAttr.mapValues(
          newElType, [](const APInt &oldEl) { return oldEl; });
    }
    rewriter.replaceOpWithNewOp<ConstantOp>(op, newValueAttr);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// mhlo.RngUniformOp conversion patterns.
//===----------------------------------------------------------------------===//

namespace {
/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> GetParallelAndReductionIterators(
    unsigned nLoops, unsigned nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
}

SmallVector<StringRef, 3> GetNParallelLoopsAttrs(unsigned nParallelLoops) {
  return GetParallelAndReductionIterators(nParallelLoops, 0);
}

// temp1 = (cast(I32, index(D.m)) + seed) * multiplier + increment
// temp2 = (cast(I32, index(D.n)) + temp1) * multiplier + increment
// ...
// tempN = (cast(I32, index(D.(last_dim))) + tempNmin1) * multiplier + increment
Value lcg_helper(OpBuilder& b, Location loc, Value ind, Value update,
                 Value multiplier, Value increment) {
  Value add = b.create<AddIOp>(loc, ind, update);
  Value mult = b.create<MulIOp>(loc, add, mult);
  Value inc = b.create<AddIOp>(loc, mult, increment);
  return inc;
}

struct RngUniformConversion : public OpConversionPattern<mhlo::RngUniformOp> {
  using OpConversionPattern<mhlo::RngUniformOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::RngUniformOp op, ArrayRef<Value> args,
      ConversionPatternRewriter& rewriter) const final {
    // Get Target Rank
    Type int32Type = IntegerType::get(op.getContext(), /*width=*/32);
    auto loc = op.getLoc();
    auto targetTy = op->getResult(0).getType().dyn_cast<ShapedType>();
    auto targetRank = targetTy.getRank();
    // Start Init Tensor
    Value output;
    Value initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, ArrayRef<Value>({}), targetTy.getShape(),
        targetTy.getElementType());
    auto result_ty = this->typeConverter->convertType(op.getResult().getType())
                         .cast<ShapedType>();
    // Create Indexing Map from target's rank
    SmallVector<AffineMap, 3> indexing_maps;
    indexing_maps.push_back(AffineMap::get(targetRank, /*symbolCount=*/0,
                                           SmallVector<AffineExpr>({}),
                                           rewriter.getContext()));
    indexing_maps.push_back(AffineMap::get(targetRank, /*symbolCount=*/0,
                                           SmallVector<AffineExpr>({}),
                                           rewriter.getContext()));
    SmallVector<AffineExpr> output_exprs;
    for (int i = 0; i < targetRank; i++) {
      output_exprs.push_back(rewriter.getAffineDimExpr(i));
    }
    indexing_maps.push_back(AffineMap::get(
        targetRank, /*symbolCount=*/0, output_exprs, rewriter.getContext()));
    // Create Region according to https://reviews.llvm.org/D101364 random
    // algorithm and
    // https://github.com/llvm/llvm-project/blob/main/mlir/include/mlir/Dialect/Linalg/IR/LinalgNamedStructuredOps.yaml
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        op.getLoc(), /*resultTensors=*/result_ty,
        /*inputs=*/ValueRange{args[0], args[1]},
        /*outputs=*/initTensor, indexing_maps,
        GetNParallelLoopsAttrs(targetRank),
        [&](OpBuilder& b, Location loc, ValueRange args) {
          // Use the output arg, so some update values won't be init value
          // again.
          llvm::SmallVector<Value> update_vec = {
              b.create<ConstantOp>(loc, b.getI32IntegerAttr(0))};
          Value multiplier =
              b.create<ConstantOp>(loc, b.getI32IntegerAttr(1103515245));
          Value increment_step =
              b.create<ConstantOp>(loc, b.getI32IntegerAttr(12345));
          // temp1 = (cast(I32, index(D.m)) + seed) * multiplier + increment
          // temp2 = (cast(I32, index(D.n)) + temp1) * multiplier + increment
          // ...
          // tempN = (cast(I32, index(D.(last_dim))) + tempNmin1) * multiplier +
          // increment
          for (int rank_it = 0; rank_it < targetRank; rank_it++) {
            Value update = update_vec.back();
            Value ind = b.create<linalg::IndexOp>(loc, rank_it);
            Value cast_ind = b.create<IndexCastOp>(loc, int32Type, ind);
            Value add_res = b.create<AddIOp>(loc, cast_ind, update);
            Value mult_res = b.create<MulIOp>(loc, add_res, multiplier);
            Value inc = b.create<AddIOp>(loc, mult_res, increment_step);
            update_vec.push_back(inc);
          }
          // Scaling:(max - min) * const(F64, 2.3283064e-10)
          Value epsilon = b.create<ConstantOp>(
              loc, b.getFloatAttr(args[0].getType(), 2.3283063999999999E-10));
          Value range = b.create<SubFOp>(loc, args[1], args[0]);
          Value scale = b.create<MulFOp>(loc, range, epsilon);
          // Res = cast(T, cast(F64, tempN) * scaling + min)
          auto scaleFloatType = scale.getType().dyn_cast<FloatType>();
          Value update_cast =
              b.create<UIToFPOp>(loc, scaleFloatType, update_vec.back());
          Value scale_update = b.create<MulFOp>(loc, update_cast, scale);
          Value res = b.create<AddFOp>(loc, scale_update, args[0]);
          // todo(raikonenfnu): if result type is int set it to the correct
          // width
          b.create<linalg::YieldOp>(loc, res);
        });
    rewriter.replaceOp(op, linalg_op.getResults());
    return success();
  }
};

}  // namespace

}  // namespace

void populateMHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, typeConverter, &patterns);
  // TODO(#5809): Drop ConcatenateOp lowering in favor of the upstream version
  //              then remove the PatternBenefit here
  patterns
      .insert<ConstOpConversion, ConcatenateOpConversion,
              RngUniformConversion, FftOpConversion>(
          typeConverter, context, PatternBenefit(1000));
}

std::unique_ptr<OperationPass<FuncOp>> createMHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertMHLOToLinalgOnTensorsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
