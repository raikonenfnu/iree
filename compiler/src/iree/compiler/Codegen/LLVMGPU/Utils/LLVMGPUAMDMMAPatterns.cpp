
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Region.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

namespace mlir::iree_compiler {

namespace {
// Transform contract into (k, m)x(k, n)x(m, n) form so that it can be converted
// to WMMA matmul.
struct PrepareContractToAMDGPUWMMA
    : public OpRewritePattern<vector::ContractionOp> {
  using OpRewritePattern<vector::ContractionOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ContractionOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value lhs = op.getLhs(), rhs = op.getRhs(), res = op.getAcc();

    // Set up the parallel/reduction structure in right form.
    using MapList = ArrayRef<ArrayRef<AffineExpr>>;
    auto infer = [](MapList m) { return AffineMap::inferFromExprList(m); };
    AffineExpr m, n, k;
    bindDims(rewriter.getContext(), m, n, k);
    static constexpr std::array<int64_t, 2> perm = {1, 0};
    auto iteratorTypes = op.getIteratorTypes().getValue();
    SmallVector<AffineMap, 4> maps = op.getIndexingMapsArray();
    if (!(vector::isParallelIterator(iteratorTypes[0]) &&
          vector::isParallelIterator(iteratorTypes[1]) &&
          vector::isReductionIterator(iteratorTypes[2])))
      return rewriter.notifyMatchFailure(op, "not a gemm contraction");
    //
    // Two outer parallel, one inner reduction (matmat flavor).
    //
    if (maps == infer({{k, m}, {k, n}, {m, n}}))
      return rewriter.notifyMatchFailure(op, "contraction already prepared");
    if (maps == infer({{m, k}, {k, n}, {m, n}})) {
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{m, k}, {n, k}, {m, n}})) {
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{k, m}, {k, n}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{k, m}, {n, k}, {n, m}})) {
      std::swap(rhs, lhs);
      rhs = rewriter.create<vector::TransposeOp>(loc, rhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      std::swap(lhs, rhs);
      lhs = rewriter.create<vector::TransposeOp>(loc, lhs, perm);
    } else if (maps == infer({{m, k}, {k, n}, {n, m}})) {
      std::swap(lhs, rhs);
    } else {
      // TODO: llvm_unreachable ?
      return rewriter.notifyMatchFailure(op, "unexpected contraction case");
    }
    rewriter.replaceOpWithNewOp<vector::ContractionOp>(
        op, lhs, rhs, res,
        rewriter.getAffineMapArrayAttr(infer({{k, m}, {k, n}, {m, n}})),
        op.getIteratorTypes());
    return success();
  }
};

// Fold transpose op into the transfer read op. amdgpu wmma op only supports
// column-, row-, and row-major layout for matrixA, matrixB, and matrixC,
// respectively. We can fold the transpose operation when loading the data from
// Shared Memory to registers.
struct CombineTransferReadOpTranspose final
    : public OpRewritePattern<vector::TransposeOp> {
  using OpRewritePattern<vector::TransposeOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::TransposeOp op,
                                PatternRewriter &rewriter) const override {
    // Look through integer extend ops.
    Value source = op.getVector();
    Type resultType = op.getType();
    Operation *extOp;
    if ((extOp = source.getDefiningOp<arith::ExtSIOp>()) ||
        (extOp = source.getDefiningOp<arith::ExtUIOp>())) {
      source = extOp->getOperand(0);
      resultType =
          VectorType::get(cast<VectorType>(resultType).getShape(),
                          cast<VectorType>(source.getType()).getElementType());
    }

    auto transferReadOp = source.getDefiningOp<vector::TransferReadOp>();
    if (!transferReadOp)
      return rewriter.notifyMatchFailure(op, "no transfer read");

    // TODO: support 0-d corner case.
    if (transferReadOp.getTransferRank() == 0)
      return rewriter.notifyMatchFailure(op, "0-D transfer read");

    if (transferReadOp.getMask() || transferReadOp.hasOutOfBoundsDim())
      return rewriter.notifyMatchFailure(op, "not inbounds transfer read");

    SmallVector<int64_t, 2> perm;
    op.getTransp(perm);
    SmallVector<unsigned, 2> permU;
    for (int64_t o : perm)
      permU.push_back(unsigned(o));
    AffineMap permutationMap =
        AffineMap::getPermutationMap(permU, op.getContext());
    AffineMap newMap =
        permutationMap.compose(transferReadOp.getPermutationMap());

    auto loc = op.getLoc();
    Value result =
        rewriter
            .create<vector::TransferReadOp>(
                loc, resultType, transferReadOp.getSource(),
                transferReadOp.getIndices(), AffineMapAttr::get(newMap),
                transferReadOp.getPadding(), transferReadOp.getMask(),
                transferReadOp.getInBoundsAttr())
            .getResult();

    // Fuse through the integer extend op.
    if (extOp) {
      if (isa<arith::ExtSIOp>(extOp))
        result = rewriter.create<arith::ExtSIOp>(loc, op.getType(), result)
                     .getResult();
      else
        result = rewriter.create<arith::ExtUIOp>(loc, op.getType(), result)
                     .getResult();
    }

    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void populatePrepareVectorToAMDMMAPatterns(RewritePatternSet &patterns,
                                           bool useMfma) {
  patterns.add<PrepareContractToAMDGPUWMMA, CombineTransferReadOpTranspose>(
      patterns.getContext());
}

} // namespace mlir::iree_compiler
