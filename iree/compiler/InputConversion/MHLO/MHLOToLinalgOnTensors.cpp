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
#include <set>
#include <unordered_map>

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
/// Converts mhlo.concatenate operation to subtensor ops + subtensor_insert ops.
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
      sizes.push_back(rewriter.create<memref::DimOp>(loc, args[0], i));
      strides.push_back(rewriter.create<ConstantIndexOp>(loc, 1));
    }
    Value resultDimSize = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      auto size = rewriter.create<memref::DimOp>(loc, arg, dim);
      resultDimSize = rewriter.create<AddIOp>(loc, resultDimSize, size);
    }
    sizes[dim] = resultDimSize;
    auto initTensor = rewriter.create<linalg::InitTensorOp>(
        loc, resultType.getShape(), resultType.getElementType());
    auto zeroAttr = rewriter.getZeroAttr(resultType.getElementType());
    Value zero = rewriter.create<ConstantOp>(loc, zeroAttr);
    Value result =
        rewriter.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

    Value accBound = rewriter.create<ConstantIndexOp>(loc, 0);
    for (auto arg : args) {
      offsets[dim] = accBound;
      sizes[dim] = rewriter.create<memref::DimOp>(loc, arg, dim);
      result = rewriter.create<SubTensorInsertOp>(loc, arg, result, offsets,
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
      b.create<linalg::FillOp>(loc, initTensor, zero).getResult(0);

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

}  // namespace

//===----------------------------------------------------------------------===//
// mhlo.Einsum conversion patterns.
//===----------------------------------------------------------------------===//

namespace {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> GetLoopsAttrs(std::set<char> input_ind,
                                        std::set<char> reduction_dims) {
  SmallVector<StringRef, 3> res;
  for (char dim : input_ind) {
    if (reduction_dims.find(dim) == reduction_dims.end())
      res.push_back(getParallelIteratorTypeName());
    else
      res.push_back(getReductionIteratorTypeName());
  }
  return res;
}

Value GetInitTensor(OpBuilder &b, Location loc, ShapedType type,
                    ArrayRef<Value> dyn_sizes) {
  return b.create<linalg::InitTensorOp>(loc, dyn_sizes, type.getShape(),
                                        type.getElementType());
}

SmallVector<Value, 2> ExtractDynamicSizes(
    OpBuilder &b, Location loc, Value tensorA, Value tensorB,
    std::vector<std::pair<int, size_t>> dim_map) {
  auto tensorA_type = tensorA.getType().dyn_cast<RankedTensorType>();
  auto tensorB_type = tensorB.getType().dyn_cast<RankedTensorType>();
  if (!tensorA_type && !tensorB_type) return {};
  auto tensorA_shape = tensorA_type.getShape();
  auto tensorB_shape = tensorB_type.getShape();
  SmallVector<Value, 2> dyn_sizes;
  for (auto ind : dim_map) {
    auto arg_number = ind.first;
    auto dim_arg_index = ind.second;
    Value tensor;
    ArrayRef<int64_t> tensor_shape;
    if (arg_number == 0) {
      tensor = tensorA;
      tensor_shape = tensorA_shape;
    } else if (arg_number == 1) {
      tensor = tensorB;
      tensor_shape = tensorB_shape;
    }

    if (tensor_shape[dim_arg_index] != ShapedType::kDynamicSize) continue;
    dyn_sizes.push_back(b.create<memref::DimOp>(loc, tensor, dim_arg_index));
  }
  return dyn_sizes;
}

// Adds indices/axes that are missing from output set
std::set<char> findSummationAxes(std::set<char> input_set,
                                 std::set<char> output_set) {
  std::set<char> summation_axes;
  for (auto ind : input_set) {
    if (output_set.find(ind) == output_set.end()) summation_axes.insert(ind);
  }
  return summation_axes;
}

SmallVector<AffineExpr, 4> getExprFromConfig(
    std::set<char> input_set, std::string loop_dims,
    std::unordered_map<char, AffineExpr> char_affine_dim_umap) {
  SmallVector<AffineExpr, 4> OutputExpr;
  for (char const &dim : loop_dims) {
    OutputExpr.push_back(char_affine_dim_umap[dim]);
  }
  return OutputExpr;
}

std::vector<std::pair<int, size_t>> findDimMapping(std::string inputA_ind,
                                                   std::string inputB_ind,
                                                   std::string output_loop) {
  std::vector<std::pair<int, size_t>> dim_map;
  for (char const &ind : output_loop) {
    int arg_number = 0;
    auto pos = inputA_ind.find(ind);
    if (pos == std::string::npos) {
      arg_number = 1;
      pos = inputB_ind.find(ind);
    }
    dim_map.emplace_back(arg_number, pos);
  }
  return dim_map;
}

std::string rangeStr(size_t number) {
  std::string range_str;
  for (size_t i = 0; i < number; i++) range_str.append(std::to_string(i));
  return range_str;
}

// Convert mhlo.einsum op into linalg.generic
class EinsumToLinalgConverter : public OpConversionPattern<mhlo::EinsumOp> {
 public:
  using OpConversionPattern<mhlo::EinsumOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      mhlo::EinsumOp op, ArrayRef<Value> args,
      ConversionPatternRewriter &rewriter) const final {
    // Find maximum rank / number of loops.
    auto get_rank = [](Value v) {
      return v.getType().cast<ShapedType>().getRank();
    };
    auto is_scalar = [&](Value v) { return get_rank(v) == 0; };
    auto einsum_config = op.einsum_config().str();

    // Setting Up Commonly Used Symbols as strings
    std::string arrow = "->";
    std::string comma = ",";
    std::string ellipsis = "...";

    // With the assumption of binary Input Operand and Single Output
    // Get the inputs and output operands' indices
    // einsum_config = "inputA_loop,inputB_loop->output_loop"
    std::size_t pos_arrow = einsum_config.find(arrow);
    std::size_t pos_comma = einsum_config.find(comma);

    std::string inputA_loop = einsum_config.substr(0, pos_comma);
    std::string inputB_loop = einsum_config.substr(
        pos_comma + comma.size(), pos_arrow - (pos_comma + comma.size()));
    std::string output_loop = einsum_config.substr(pos_arrow + arrow.size());

    // Check for Invalid Configs
    // 1.Check that there is only maximum 2 inputs
    // 2.Check that there is only maximum 1 output
    // 3.Check that there is 1 arrow
    if (inputB_loop.find(comma) != std::string::npos ||
        output_loop.find(comma) != std::string::npos ||
        output_loop.find(arrow) != std::string::npos) {
      llvm::errs() << "Invalid Einsum Config!\n";
      return failure();
    }

    // Find result type, if on tensors.
    Optional<ShapedType> result_ty;
    result_ty = this->typeConverter->convertType(op->getResultTypes().front())
                    .template dyn_cast<ShapedType>();

    // Check result type compatibility.
    if (!result_ty || !result_ty->hasRank() ||
        !(result_ty->getElementType().isSignlessIntOrFloat() ||
          result_ty->getElementType().isa<ComplexType>())) {
      return rewriter.notifyMatchFailure(
          op, "mismatched operand/result types or iterator count");
    }

    // Find elilipsis in loop and replace with dim index
    // Find number of dimensions/ranks elipsis represent and replace with
    // chars(0-9)
    std::vector<std::string *> all_loops = {&inputA_loop, &inputB_loop,
                                            &output_loop};
    std::vector<int> batch_rank_vec;
    size_t max_operand_sz = 3;  // 2 Inputs + 1 output operand
    for (auto loop : llvm::enumerate(all_loops)) {
      size_t preElip = loop.value()->find(ellipsis);
      if (preElip != std::string::npos) {
        size_t postElip = loop.value()->size() - (preElip + ellipsis.size());
        size_t operand_rank = loop.index() < (max_operand_sz - 1)
                                  ? get_rank(args[loop.index()])
                                  : result_ty->getRank();
        size_t batch_rank = operand_rank - (preElip + postElip);
        if (batch_rank > 10) {
          llvm::errs() << "Maximum batch dimension that ellipsis(\"...\") can "
                          "represent is 10!\n";
          return failure();
        }
        std::string batch_str = rangeStr(batch_rank);
        loop.value()->replace(preElip, ellipsis.size(), batch_str);
        // llvm::outs()<<loop.value()<<"\n";
        batch_rank_vec.push_back(batch_rank);
      }
    }
    // Check that all ellipsis represent same rank of batch
    if (batch_rank_vec.size() > 1) {
      if (!std::equal(batch_rank_vec.begin() + 1, batch_rank_vec.end(),
                      batch_rank_vec.begin())) {
        llvm::errs() << "Invalid Elipsis(ellipsis) within Einsum config!\n";
        return failure();
      }
    }

    // Find all unique indices in the input and output
    std::set<char> input_ind;
    std::set<char> output_ind;
    for (auto indA : inputA_loop) input_ind.insert(indA);
    for (auto indB : inputB_loop) input_ind.insert(indB);
    for (auto indOut : output_loop) output_ind.insert(indOut);

    // 2.Check for contraction/summation indices
    std::set<char> reduction_axe = findSummationAxes(input_ind, output_ind);

    // Find input/output values and types.
    auto loc = op.getLoc();
    ValueRange inputs = args;

    // Setting up Initial buffer for Output Tensor
    Value output;
    auto dim_map = findDimMapping(inputA_loop, inputB_loop, output_loop);
    auto dyn_sizes =
        ExtractDynamicSizes(rewriter, loc, args.front(), args.back(), dim_map);
    output = GetInitTensor(rewriter, loc, *result_ty, dyn_sizes);

    // Create indexing maps.
    int64_t nloops = input_ind.size();
    std::unordered_map<char, AffineExpr> char_affine_dim_umap;
    for (auto it : llvm::enumerate(input_ind)) {
      char_affine_dim_umap[it.value()] = rewriter.getAffineDimExpr(it.index());
    }
    SmallVector<AffineMap, 4> maps;
    std::vector<std::string> input_loops = {inputA_loop, inputB_loop};
    for (auto it : llvm::enumerate(inputs)) {
      AffineMap scalar_map = AffineMap::get(nloops, 0, rewriter.getContext());
      auto inputExprs = getExprFromConfig(input_ind, input_loops[it.index()],
                                          char_affine_dim_umap);
      AffineMap multidim_map =
          AffineMap::get(nloops, 0, inputExprs, rewriter.getContext());
      maps.push_back(is_scalar(it.value()) ? scalar_map : multidim_map);
    }
    auto inputExprs =
        getExprFromConfig(input_ind, output_loop, char_affine_dim_umap);
    AffineMap multidim_map =
        AffineMap::get(nloops, 0, inputExprs, rewriter.getContext());
    maps.push_back(multidim_map);

    // Creating Regions to be put into linalg.generic
    auto reduction_region = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                                ValueRange args) {
      auto MulOp =
          nestedBuilder.create<mlir::MulFOp>(nestedLoc, args[0], args[1]);
      auto AddOp = nestedBuilder.create<mlir::AddFOp>(nestedLoc, args[2],
                                                      MulOp.getResult());
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, AddOp.getResult());
    };

    auto pointwise_region = [&](OpBuilder &nestedBuilder, Location nestedLoc,
                                ValueRange args) {
      auto MulOp =
          nestedBuilder.create<mlir::MulFOp>(nestedLoc, args[0], args[1]);
      nestedBuilder.create<linalg::YieldOp>(nestedLoc, MulOp.getResult());
    };

    // Selecting different region based on whether or not einsum has
    // contraction/summation axes
    llvm::function_ref<void(OpBuilder &, Location, ValueRange)> region_used;
    bool hasReduction = reduction_axe.size() > 0;
    if (hasReduction)
      region_used = reduction_region;
    else
      region_used = pointwise_region;

    // Build `linalg.generic` op.
    auto linalg_op = rewriter.create<linalg::GenericOp>(
        loc, result_ty ? *result_ty : TypeRange{}, inputs, output, maps,
        GetLoopsAttrs(input_ind, reduction_axe), region_used);
    rewriter.replaceOp(op, linalg_op->getResults());
    return success();
  }
};
}  // namespace

void populateMHLOToLinalgOnTensorsConversionPatterns(
    MLIRContext *context, TypeConverter &typeConverter,
    OwningRewritePatternList &patterns) {
  mhlo::populateHLOToLinalgConversionPattern(context, typeConverter, &patterns);
  // TODO(#5809): Drop ConcatenateOp lowering in favor of the upstream version
  //              then remove the PatternBenefit here
  patterns.insert<ConstOpConversion, ConcatenateOpConversion, FftOpConversion,
                  EinsumToLinalgConverter>(typeConverter, context,
                                           PatternBenefit(1000));
}

std::unique_ptr<OperationPass<FuncOp>> createMHLOToLinalgOnTensorsPass() {
  return std::make_unique<ConvertMHLOToLinalgOnTensorsPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
