// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree_tf_compiler/TF/Passes.h"
#include "mlir-hlo/Dialect/mhlo/IR/chlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/IR/hlo_ops.h"
#include "mlir-hlo/Dialect/mhlo/transforms/rewriters.h"
#include "mlir/Dialect/Linalg/IR/LinalgOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_device.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_executor.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_ops.h"
#include "tensorflow/compiler/mlir/tensorflow/ir/tf_saved_model.h"
#include "tensorflow/compiler/mlir/tensorflow/transforms/lower_tf.h"
#include "tensorflow/compiler/mlir/xla/transforms/passes.h"

namespace mlir {
namespace iree_integrations {
namespace TF {

namespace {

/// Returns an ArrayAttr that contains `nLoops` attributes. All the attributes
/// are "parallel" except the last `nReduction` elements, where are "reduction"
/// attributes.
SmallVector<StringRef, 3> getParallelAndReductionIterators(int nLoops,
                                                           int nReduction) {
  SmallVector<StringRef, 3> res(nLoops - nReduction,
                                getParallelIteratorTypeName());
  res.append(nReduction, getReductionIteratorTypeName());
  return res;
}

SmallVector<StringRef, 3> getNParallelLoopsAttrs(int nParallelLoops) {
  return getParallelAndReductionIterators(nParallelLoops, 0);
}

// Holds a static extent or Value for dynamic extents.
class ExtentOrValue {
 public:
  ExtentOrValue() {}
  ExtentOrValue(int64_t extent) : extent(extent) {}
  ExtentOrValue(Value value) : value(value) {}

  bool isExtent() { return !value; }
  bool isUnitExtent() { return isExtent() && getExtent() == 1; }
  int64_t getExtent() {
    assert(isExtent());
    return extent;
  }
  Value getValue() {
    assert(!isExtent());
    return value;
  }

  Value convertToValue(OpBuilder &builder, Location loc) {
    if (!isExtent()) return getValue();
    return builder.create<ConstantIndexOp>(loc, getExtent());
  }

 private:
  int64_t extent;
  Value value;
};

Value broadcast(OpBuilder &builder, Location loc, Value operand,
                SmallVectorImpl<ExtentOrValue> &resultExtents,
                SmallVectorImpl<bool> &isExpansion) {
  auto operandType = operand.getType().cast<RankedTensorType>();
  SmallVector<int64_t> resultShape;
  SmallVector<Value> dynDims;
  for (ExtentOrValue &dim : resultExtents) {
    if (dim.isExtent()) {
      resultShape.push_back(dim.getExtent());
    } else {
      resultShape.push_back(-1);
      dynDims.push_back(dim.getValue());
    }
  }

  // Traverse the right aligned operand dimensions and form expressions.
  // We keep 1-dims in place instead of reshaping them away, relying on the
  // DropUnitDims pass to run later.
  SmallVector<AffineExpr> dimExprs;
  dimExprs.reserve(operandType.getRank());
  for (int i = resultExtents.size() - operandType.getRank();
       i < resultExtents.size(); ++i) {
    if (isExpansion[i]) {
      dimExprs.push_back(builder.getAffineConstantExpr(0));
    } else {
      dimExprs.push_back(builder.getAffineDimExpr(i));
    }
  }

  int nloops = resultExtents.size();
  Value init = builder.create<linalg::InitTensorOp>(
      loc, dynDims, resultShape, operandType.getElementType());
  auto generic = builder.create<linalg::GenericOp>(
      loc, TypeRange{init.getType()}, ValueRange{operand},
      /*outputBuffers=*/ValueRange{init},
      llvm::makeArrayRef({
          AffineMap::get(/*dimCount=*/nloops, /*symbolCount=*/0, dimExprs,
                         builder.getContext()),
          builder.getMultiDimIdentityMap(nloops),
      }),
      getNParallelLoopsAttrs(nloops),
      [&](OpBuilder &nestedBuilder, Location nestedLoc, ValueRange args) {
        nestedBuilder.create<linalg::YieldOp>(loc, *args.begin());
      });
  return generic.getResult(0);
}

Optional<ExtentOrValue> computeResultExtent(OpBuilder &builder, Location loc,
                                            ExtentOrValue &lhsDim,
                                            ExtentOrValue &rhsDim,
                                            bool &isLhsExpansion,
                                            bool &isRhsExpansion) {
  if (lhsDim.isExtent() && rhsDim.isExtent()) {
    // Both are static. Just check.
    if (lhsDim.getExtent() != rhsDim.getExtent() &&
        !(lhsDim.getExtent() == 1 || rhsDim.getExtent() == 1)) {
      // Statically illegal.
      emitError(loc) << "cannot broadcast extents of differing size unless "
                        "if one of them is 1 (got "
                     << lhsDim.getExtent() << ", " << rhsDim.getExtent() << ")";
      return llvm::None;
    }

    // Static expansions.
    if (lhsDim.isUnitExtent() && rhsDim.isUnitExtent()) {
      // For the fully static case, we can trivially check the 1-equality,
      // and know we are not expanding.
      isLhsExpansion = false;
      isRhsExpansion = false;
    } else {
      // Otherwise, mark the dim as expanding if it is 1.
      isLhsExpansion = lhsDim.isUnitExtent();
      isRhsExpansion = rhsDim.isUnitExtent();
    }
    return ExtentOrValue(std::max(lhsDim.getExtent(), rhsDim.getExtent()));
  }

  // At least one of them is dynamic.
  // Branch on whether one of them is a static-1, which is the only case
  // we allow for dynamic expansion.
  if (lhsDim.isUnitExtent() || rhsDim.isUnitExtent()) {
    if (lhsDim.isUnitExtent()) {
      isLhsExpansion = true;
      isRhsExpansion = false;
      return rhsDim;
    } else {
      isLhsExpansion = false;
      isRhsExpansion = true;
      return lhsDim;
    }
  }

  // At least one is dynamic and neither are a static 1.
  // In this case, we do not allow either to be an expanding dim and
  // error if this is the case at runtime.
  isLhsExpansion = false;
  isRhsExpansion = false;
  Value lhsExtentValue = lhsDim.convertToValue(builder, loc);
  Value rhsExtentValue = rhsDim.convertToValue(builder, loc);

  Value isEqual = builder.create<CmpIOp>(loc, CmpIPredicate::eq, lhsExtentValue,
                                         rhsExtentValue);
  builder.create<AssertOp>(
      loc, isEqual,
      builder.getStringAttr("mismatched dynamic broadcast extents"));

  // Here, if one of them is static, that has to be the result extent
  // (because we checked the error condition above).
  if (lhsDim.isExtent()) {
    return ExtentOrValue(lhsDim.getExtent());
  } else if (rhsDim.isExtent()) {
    return ExtentOrValue(rhsDim.getExtent());
  }

  // Both are dynamic. Compute the max.
  Value lhsIsGreater = builder.create<CmpIOp>(loc, CmpIPredicate::sge,
                                              lhsExtentValue, rhsExtentValue);
  Value resultExtent = builder.create<SelectOp>(loc, lhsIsGreater,
                                                lhsExtentValue, rhsExtentValue);
  return ExtentOrValue(resultExtent);
}

void padExtents(SmallVectorImpl<ExtentOrValue> &extents, int size) {
  for (int i = 0; i < size; ++i) {
    extents.push_back({1});
  }
}

void appendExtents(OpBuilder &builder, Location loc,
                   SmallVectorImpl<ExtentOrValue> &extents, Value v,
                   RankedTensorType t) {
  for (int i = 0; i < t.getRank(); ++i) {
    if (t.isDynamicDim(i)) {
      // Emit a dim op.
      Value dim = builder.create<memref::DimOp>(loc, v, i);
      extents.push_back(dim);
    } else {
      // Static dim.
      extents.push_back({t.getDimSize(i)});
    }
  }
}

template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertRankedBroadcastBinaryOp : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    // Only rewrite for statically determinable non-broadcasting cases.
    typename ChloOpTy::Adaptor transformed(operands);
    Value lhs = transformed.lhs();
    Value rhs = transformed.rhs();
    auto lhsType = lhs.getType().template dyn_cast<RankedTensorType>();
    auto rhsType = rhs.getType().template dyn_cast<RankedTensorType>();
    if (!lhsType || !rhsType)
      return rewriter.notifyMatchFailure(op, "not ranked tensors");

    // Extract the original extents.
    SmallVector<ExtentOrValue> lhsOrigExtents;
    lhsOrigExtents.reserve(lhsType.getRank());
    appendExtents(rewriter, loc, lhsOrigExtents, lhs, lhsType);
    SmallVector<ExtentOrValue> rhsOrigExtents;
    rhsOrigExtents.reserve(rhsType.getRank());
    appendExtents(rewriter, loc, rhsOrigExtents, rhs, rhsType);

    // Left pad with 1-extents to the result rank.
    int resultRank = std::max(lhsType.getRank(), rhsType.getRank());
    SmallVector<ExtentOrValue> lhsBcastExtents;
    lhsBcastExtents.reserve(resultRank);
    SmallVector<ExtentOrValue> rhsBcastExtents;
    rhsBcastExtents.reserve(resultRank);
    padExtents(lhsBcastExtents, resultRank - lhsType.getRank());
    lhsBcastExtents.append(lhsOrigExtents);
    padExtents(rhsBcastExtents, resultRank - rhsType.getRank());
    rhsBcastExtents.append(rhsOrigExtents);

    // Compute the result extents.
    SmallVector<ExtentOrValue> resultExtents(resultRank);
    SmallVector<bool> isLhsExpansion(resultRank);
    SmallVector<bool> isRhsExpansion(resultRank);
    bool lhsNeedsBroadcast = resultRank != lhsType.getRank();
    bool rhsNeedsBroadcast = resultRank != rhsType.getRank();
    for (int i = 0; i < resultRank; i++) {
      auto resultExtent = computeResultExtent(
          rewriter, loc, lhsBcastExtents[i], rhsBcastExtents[i],
          isLhsExpansion[i], isRhsExpansion[i]);
      if (!resultExtent)
        return rewriter.notifyMatchFailure(op,
                                           "could not compute result extent");
      resultExtents[i] = *resultExtent;
      if (isLhsExpansion[i]) lhsNeedsBroadcast = true;
      if (isRhsExpansion[i]) rhsNeedsBroadcast = true;
    }

    // Broadcast the operands.
    Value lhsBcast =
        lhsNeedsBroadcast
            ? broadcast(rewriter, loc, lhs, resultExtents, isLhsExpansion)
            : lhs;
    Value rhsBcast =
        rhsNeedsBroadcast
            ? broadcast(rewriter, loc, rhs, resultExtents, isRhsExpansion)
            : rhs;

    rewriter.replaceOpWithNewOp<HloOpTy>(op, lhsBcast, rhsBcast);
    return success();
  }
};

// Converts binary ops that statically are determined to not broadcast directly
// to the corresponding mhlo non-broadcasting op.
template <typename ChloOpTy, typename HloOpTy, typename Adaptor>
struct ConvertTrivialNonBroadcastBinaryOp
    : public OpConversionPattern<ChloOpTy> {
  using OpConversionPattern<ChloOpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      ChloOpTy op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    // Only rewrite for statically determinable non-broadcasting cases.
    typename ChloOpTy::Adaptor transformed(operands);
    auto lhs_type =
        transformed.lhs().getType().template dyn_cast<RankedTensorType>();
    auto rhs_type =
        transformed.rhs().getType().template dyn_cast<RankedTensorType>();
    if (!lhs_type || !rhs_type) return failure();

    // Requires rank broadcast.
    if (lhs_type.getRank() != rhs_type.getRank()) return failure();
    // Any dynamic dimension may require broadcasting and requires more
    // analysis.
    if (!lhs_type.hasStaticShape() || !rhs_type.hasStaticShape())
      return failure();

    for (auto extents : llvm::zip(lhs_type.getShape(), rhs_type.getShape())) {
      auto lhs_extent = std::get<0>(extents);
      auto rhs_extent = std::get<1>(extents);
      if (lhs_extent != rhs_extent) {
        return failure();
      }
    }

    rewriter.replaceOp(op, {Adaptor::CreateOp(op, op.getResult().getType(),
                                              operands, rewriter)});
    return success();
  }
};

template <typename FromOpTy, typename ToOpTy>
struct HloNaryElementwiseAdaptor {
  static ToOpTy CreateOp(FromOpTy from_op, Type result_type,
                         ValueRange broadcasted_operands, OpBuilder &builder) {
    return builder.create<ToOpTy>(from_op.getLoc(), result_type,
                                  broadcasted_operands);
  }
};

// Populate a pattern for each Broadcasting CHlo op. This requires the pattern
// to take a ChloOpTy, NonBroadcastingOpTy, and an Adaptor as templated values.
template <template <typename, typename, typename> class Pattern,
          typename... ConstructorArgs>
void PopulateForBroadcastingBinaryOp(MLIRContext *context,
                                     OwningRewritePatternList *patterns,
                                     ConstructorArgs &&...args) {
#define POPULATE_BCAST(ChloOp, HloOp)                                    \
  patterns->insert<                                                      \
      Pattern<ChloOp, HloOp, HloNaryElementwiseAdaptor<ChloOp, HloOp>>>( \
      context, args...);

  POPULATE_BCAST(chlo::BroadcastAddOp, mhlo::AddOp);
  POPULATE_BCAST(chlo::BroadcastAndOp, mhlo::AndOp);
  POPULATE_BCAST(chlo::BroadcastAtan2Op, mhlo::Atan2Op);
  POPULATE_BCAST(chlo::BroadcastComplexOp, mhlo::ComplexOp);
  POPULATE_BCAST(chlo::BroadcastDivOp, mhlo::DivOp);
  POPULATE_BCAST(chlo::BroadcastMaxOp, mhlo::MaxOp);
  POPULATE_BCAST(chlo::BroadcastMinOp, mhlo::MinOp);
  POPULATE_BCAST(chlo::BroadcastMulOp, mhlo::MulOp);
  POPULATE_BCAST(chlo::BroadcastOrOp, mhlo::OrOp);
  // POPULATE_BCAST(chlo::BroadcastPolygammaOp, PolygammaOp);
  POPULATE_BCAST(chlo::BroadcastPowOp, mhlo::PowOp);
  POPULATE_BCAST(chlo::BroadcastRemOp, mhlo::RemOp);
  POPULATE_BCAST(chlo::BroadcastShiftLeftOp, mhlo::ShiftLeftOp);
  POPULATE_BCAST(chlo::BroadcastShiftRightArithmeticOp,
                 mhlo::ShiftRightArithmeticOp);
  POPULATE_BCAST(chlo::BroadcastShiftRightLogicalOp, mhlo::ShiftRightLogicalOp);
  POPULATE_BCAST(chlo::BroadcastSubOp, mhlo::SubOp);
  POPULATE_BCAST(chlo::BroadcastXorOp, mhlo::XorOp);
  // POPULATE_BCAST(chlo::BroadcastZetaOp, ZetaOp);
}

void PopulateChloBroadcastingPatterns(MLIRContext *context,
                                      OwningRewritePatternList *patterns) {
  // Instantiate conversion templates for conforming binary elementwise ops
  // that do not have different dtypes between operands and results and do
  // not have special attributes that need to be preserved.
  PopulateForBroadcastingBinaryOp<ConvertTrivialNonBroadcastBinaryOp>(
      context, patterns, 10);
  PopulateForBroadcastingBinaryOp<ConvertRankedBroadcastBinaryOp>(context,
                                                                  patterns, 5);

  // PopulateForBroadcastingBinaryOp<ConvertRankedDynamicBroadcastBinaryOp>(
  //     context, patterns, 5);
  // patterns->insert<ConvertSelectOp>(context);
}
}  // namespace

// This is a customized version of the TF to XLA lowering in:
//    tensorflow/compiler/mlir/xla/transforms/legalize_tf.cc
// It does not require the same number of options as we can hardcode as the pass
// the IREE requires.
class ConvertToMHLOPass : public PassWrapper<ConvertToMHLOPass, FunctionPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::TF::TensorFlowDialect,
                    mlir::tf_executor::TensorFlowExecutorDialect,
                    mlir::tf_device::TensorFlowDeviceDialect,
                    mlir::tf_saved_model::TensorFlowSavedModelDialect,
                    chlo::HloClientDialect, mhlo::MhloDialect,
                    shape::ShapeDialect, StandardOpsDialect>();
  }

 public:
  ConvertToMHLOPass() = default;
  ConvertToMHLOPass(const ConvertToMHLOPass &) {}

  void runOnFunction() override {
    auto op = getFunction();
    MLIRContext *context = op.getContext();

    // Lower TF Patterns must be separate from canonocalization patterns as
    // they are sometimes inversions of eachother.
    OwningRewritePatternList lowerTfPatterns(&getContext());
    mlir::TF::PopulateLoweringTFPatterns(context, &lowerTfPatterns);

    OwningRewritePatternList canonicalizePatterns(&getContext());
    for (auto *op : context->getRegisteredOperations()) {
      op->getCanonicalizationPatterns(canonicalizePatterns, context);
    }

    OwningRewritePatternList patterns(&getContext());
    // Note that the `OperationConverter` orders patterns lexicographically by:
    // 1) Ascending legalization depth (i.e., minimum number of patterns
    // necessary to arrive at conversion target).
    // 2) Descending pattern benefit.
    // 3) Order of patterns in `OwningRewritePatternList`.

    // Add TF->HLO legalization patterns.
    mhlo::PopulateLegalizeTfPatterns(context, &patterns);

    // TF::PopulateLoweringTFPatterns(context, &patterns);

    // Populate with CHLO->HLO lowerings to account for TF ops legalized to
    // CHLO first.
    // chlo::PopulateLegalizeChloToHloPatterns(context, &patterns);
    PopulateChloBroadcastingPatterns(context, &patterns);

    // ConstantLike op is convenient to create splat constants, but is
    // canonicalized to plain HLO constant if statically shaped. Add the
    // canonicalization pattern to pattern list to enable multi-hop lowering.
    chlo::ConstantLikeOp::getCanonicalizationPatterns(patterns, context);

    ConversionTarget target(*context);
    target.addIllegalDialect<chlo::HloClientDialect>();
    target.addLegalDialect<linalg::LinalgDialect>();
    target.addLegalDialect<mhlo::MhloDialect>();
    target.addLegalDialect<mlir::StandardOpsDialect>();
    target.addLegalDialect<shape::ShapeDialect>();
    target.addLegalDialect<tensor::TensorDialect>();
    target.addLegalOp<mlir::CallOp>();
    target.addLegalOp<mlir::tensor::CastOp>();
    target.addLegalOp<mlir::memref::DimOp>();

    // TODO(suderman): Enable logicistic op for lowering once the op is
    // supported in IREE. Also, remove the numerically unstable ConvertSigmoidOp
    // pattern in the legalize-tf pass.
    target.addIllegalOp<mhlo::LogisticOp>();

    DenseSet<Operation *> prevUnconvertedOps;
    DenseSet<Operation *> unconvertedOps;

    FrozenRewritePatternSet frozenPatterns(std::move(patterns));
    FrozenRewritePatternSet frozenCanonicalizePatterns(
        std::move(canonicalizePatterns));
    FrozenRewritePatternSet frozenTfPatterns(std::move(lowerTfPatterns));
    while (true) {
      if (failed(
              applyPatternsAndFoldGreedily(op, frozenCanonicalizePatterns))) {
        return signalPassFailure();
      }

      if (failed(applyPatternsAndFoldGreedily(op, frozenTfPatterns))) {
        return signalPassFailure();
      }

      if (failed(applyPartialConversion(op, target, frozenPatterns,
                                        &unconvertedOps))) {
        return signalPassFailure();
      }

      if (prevUnconvertedOps == unconvertedOps) break;
      prevUnconvertedOps = std::move(unconvertedOps);
    }
  }

 private:
  Option<bool> allow_partial_conversion_{
      *this, "allow-partial-conversion",
      llvm::cl::desc("Allow operations that can't be legalized."),
      llvm::cl::init(false)};
  Option<bool> legalize_chlo_{
      *this, "legalize-chlo",
      llvm::cl::desc(
          "Also legalizes intermediate chlo ops to hlo (default true)"),
      llvm::cl::init(false)};
  Option<bool> use_tf2xla_fallback_{
      *this, "use-tf2xla-fallback",
      llvm::cl::desc(
          "Also use TF2XLA fallback for legalization (default false)"),
      llvm::cl::init(false)};
  Option<std::string> device_type_{
      *this, "device-type",
      llvm::cl::desc(
          "The device type used by TF2XLA fallback. Must be specified if "
          "use-tf2xla-fallback is true, otherwise not used."),
      llvm::cl::init("INVALID_DEVICE_TYPE")};
};

std::unique_ptr<FunctionPass> createConvertToMHLOPass() {
  return std::make_unique<ConvertToMHLOPass>();
}

static PassRegistration<ConvertToMHLOPass> pass(
    "iree-tf-convert-to-mhlo",
    "Converts from TensorFlow to the XLA MHLO dialect");

}  // namespace TF
}  // namespace iree_integrations
}  // namespace mlir
