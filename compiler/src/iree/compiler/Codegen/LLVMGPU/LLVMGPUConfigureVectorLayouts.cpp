// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <algorithm>

#include "iree/compiler/Codegen/Dialect/Codegen/IR/IREECodegenAttrs.h"
#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "iree/compiler/Codegen/Dialect/VectorExt/IR/VectorExtDialect.h"
#include "iree/compiler/Codegen/LLVMGPU/Passes.h"
#include "iree/compiler/Codegen/Utils/GPUUtils.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"

#define DEBUG_TYPE "iree-llvmgpu-configure-vector-layouts"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_LLVMGPUCONFIGUREVECTORLAYOUTSPASS
#include "iree/compiler/Codegen/LLVMGPU/Passes.h.inc"

namespace {

// Sets a layout anchor for reads from global memory.
// The layout this generates is approximately the following:
//
// #layout = #iree_vector_ext.nested_layout<
//    subgroup_tile = [1, ..., 1]
//    batch_tile    = [<remaining undistributed elements>]
//    outer_tile    = [1, ..., 1]
//    thread_tile   = [<greedy from innermost memref dim>]
//    element_tile  = [1, ..., 128/element_bitwidth, ..., 1]
//            innermost_memref_dimension ^^^^^^
//
// (All orders are the same)
//    *_order = [<broadcasted_dims>, <transfer_permutation>]>
//
// So for the following transfer_read with 64 threads:
//  vector.transfer_read ... : memref<16x256xf16>, vector<16x32xf16>
//
// We use the following layout:
// #layout = #iree_vector_ext.nested_layout<
//    subgroup_tile = [1, 1]
//    batch_tile    = [1, 1]
//    outer_tile    = [1, 1]
//    thread_tile   = [16, 4]
//    element_tile  = [1, 8]
LogicalResult setTransferReadAnchor(ArrayRef<int64_t> workgroupSize,
                                    RewriterBase &rewriter,
                                    vector::TransferReadOp transfer) {
  MLIRContext *context = rewriter.getContext();

  // Get the forward slice of the transfer to approximate whether it will take
  // the layout of a contraction instead. Transfer_read ops used directly by a
  // contraction (i.e. without a copy to shared memory in between) should take
  // the layout of the contraction op. This is common for cases where the
  // initial values of the accumulator in a linalg.matmul is read from memory
  // instead of just being a zerofill.
  ForwardSliceOptions forwardOptions;
  forwardOptions.filter = [&](Operation *op) -> bool {
    return llvm::any_of(op->getResultTypes(), llvm::IsaPred<VectorType>);
  };
  BackwardSliceOptions backwardOptions;
  backwardOptions.filter = [&](Operation *op) -> bool {
    return llvm::any_of(op->getOperandTypes(), llvm::IsaPred<VectorType>);
  };
  SetVector<Operation *> slice =
      getSlice(transfer, backwardOptions, forwardOptions);

  if (llvm::any_of(slice, llvm::IsaPred<vector::ContractionOp>)) {
    llvm::outs()<<"CONTRACTION:"<<*transfer<<"\n";
    return success();
  }

  // Shared memory loads are expected to take the layout of the contraction.
  auto sourceMemRefType = dyn_cast<MemRefType>(transfer.getSource().getType());
  if (!sourceMemRefType || hasSharedMemoryAddressSpace(sourceMemRefType)) {
    llvm::outs()<<"MEMREF:"<<*transfer<<"\n";
    return success();
  }

  // Take on layout of broadcast.
  if (transfer->hasOneUse() &&
      dyn_cast<vector::BroadcastOp>(*transfer->getUsers().begin())) {
    llvm::outs()<<"TRANSFER:"<<*transfer<<"\n";
    return success();
  }

  // TODO: Support masking.
  if (transfer.getMask()) {
    llvm::outs()<<"MASK!\n";
    transfer->emitOpError(
        "Anchoring on transfer_read with masks is not yet implemented.");
    return failure();
  }

  int64_t bitWidth = IREE::Util::getTypeBitWidth(
      getElementTypeOrSelf(transfer.getVectorType()));
  if (!llvm::isPowerOf2_64(bitWidth) || bitWidth > 128) {
    llvm::outs()<<"NO POWER OF 2!\n";
    transfer->emitOpError(
        "Anchoring on transfer_read with element type of bitwidth " +
        std::to_string(bitWidth) + " is not yet implemented");
    return failure();
  }
  int64_t numElementTile = 128 / bitWidth;
  int64_t flatNumElements =
      ShapedType::getNumElements(transfer.getVectorType().getShape());
  int64_t flatNumThreads = ShapedType::getNumElements(workgroupSize);
  if (flatNumElements % flatNumThreads != 0) {
    llvm::outs()<<"NON DIVISIBLE!\n";
    transfer->emitOpError()
        << "Anchoring on transfer_read with unsupported number of elements "
           "(not divisible by workgroup size)"
        << ", number of elements: " << flatNumElements
        << ", workgroup size: " << flatNumThreads;
    return failure();
  }
  numElementTile = std::min(numElementTile, flatNumElements / flatNumThreads);

  AffineMap transferMap = transfer.getPermutationMap();
  if (transferMap.getNumDims() == 0) {
    llvm::outs()<<"NUM DIM 0!\n";
    transfer->emitOpError("Anchoring on transfer_read with zero-rank "
                          "permutation map is not supported.");
    return failure();
  }

  // Select the innermost dim of the memref as the contiguous dim to load
  // from.
  int64_t transferRank = transfer.getVectorType().getRank();
  std::optional<unsigned> maybeDim = transferMap.getResultPosition(
      getAffineDimExpr(transferMap.getNumDims() - 1, context));
  int64_t distXDim = maybeDim ? *maybeDim : transferRank - 1;

  ArrayRef<int64_t> vectorShape = transfer.getVectorType().getShape();

  // Limit the maximum inner vector read width to the innermost contiguous
  // dimension. We could try to be clever and extend this to adjacent
  // dimensions in cases where the innermost read vector dimension is small,
  // but that requires comparing memref strides and is uncommon. For now
  // prioritize warp contiguity over 128-bit read granularity.
  numElementTile = std::min(numElementTile, vectorShape[distXDim]);

  llvm::SetVector<unsigned> vectorDimDistributionOrder;
  // Get the order in which to distribute vector dimensions to threads, going
  // from innermost to outermost memref dimension. It's important to note
  // that this heuristic only applies to matrix multiplication cases where
  // we are promoting the operands of a contraction to shared memory and we
  // have no producers fused with the matmul. In general there is no universal
  // way to set an anchoring layout for reads without doing an analysis of how
  // the read values are used.
  for (int i = transferMap.getNumDims() - 1; i >= 0; --i) {
    std::optional<unsigned> maybeDim =
        transferMap.getResultPosition(getAffineDimExpr(i, context));
    if (maybeDim) {
      vectorDimDistributionOrder.insert(*maybeDim);
    }
  }
  // Add all remaining (broadcasted) dimensions
  for (auto dim : llvm::seq(static_cast<int64_t>(0), transferRank)) {
    if (!vectorDimDistributionOrder.contains(dim))
      vectorDimDistributionOrder.insert(dim);
  }

  int64_t residualThreads = flatNumThreads;
  int64_t residualElements = numElementTile;

  SmallVector<int64_t> order(vectorDimDistributionOrder.rbegin(),
                             vectorDimDistributionOrder.rend());

  // Distribute all threads in the workgroup to the "threads" dimension,
  // meaning subgroup counts is unit here, even though the read is being
  // distributed to multiple subgroups. This is in an attempt to do a
  // workgroup contiguous load.
  SmallVector<int64_t> subgroupCounts(transferRank, 1);
  SmallVector<int64_t> batchSizes(transferRank, 1);
  SmallVector<int64_t> outerSizes(transferRank, 1);
  SmallVector<int64_t> threadCounts(transferRank, 1);
  SmallVector<int64_t> elementSizes(transferRank, 1);

  SmallVector<int64_t> subgroupStrides(transferRank, 1);
  SmallVector<int64_t> threadStrides(transferRank, 1);

  int64_t currStrides = 1;
  for (auto dim : llvm::reverse(order)) {
    int64_t vectorSize = vectorShape[dim];
    // Set the element count for the innermost vector dimension.
    if (residualElements != 1) {
      elementSizes[dim] = residualElements;
      vectorSize /= residualElements;
      residualElements = 1;
    }

    assert((residualThreads % vectorSize == 0 ||
            vectorSize % residualThreads == 0) &&
           "dividing threads to incompatible vector");
    if (residualThreads <= vectorSize) {
      vectorSize /= residualThreads;
      threadCounts[dim] = residualThreads;
      threadStrides[dim] = currStrides;
      currStrides *= residualThreads;
      residualThreads = 1;
    } else {
      residualThreads /= vectorSize;
      threadCounts[dim] = vectorSize;
      threadStrides[dim] = currStrides;
      currStrides *= vectorSize;
      vectorSize = 1;
    }

    batchSizes[dim] = vectorSize;
  }

  auto layout = IREE::VectorExt::NestedLayoutAttr::get(
      context, subgroupCounts, batchSizes, outerSizes, threadCounts,
      elementSizes, subgroupStrides, threadStrides);

  Location loc = transfer.getLoc();
  rewriter.setInsertionPointAfter(transfer);
  auto toLayout = rewriter.create<IREE::VectorExt::ToLayoutOp>(
      loc, transfer.getResult(), layout);
  rewriter.replaceAllUsesExcept(transfer, toLayout.getResult(), toLayout);

  return success();
}

struct LLVMGPUConfigureVectorLayoutsPass final
    : impl::LLVMGPUConfigureVectorLayoutsPassBase<
          LLVMGPUConfigureVectorLayoutsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::VectorExt::IREEVectorExtDialect>();
    registry.insert<vector::VectorDialect>();
  }

  void runOnOperation() override {
    auto func = getOperation();

    std::array<int64_t, 3> workgroupSize;
    if (func->hasAttr("workgroup_size")) {
      auto tmpSizes =
          llvm::cast<ArrayAttr>(func->getAttr("workgroup_size")).getValue();
      for (auto [i, size] : llvm::enumerate(tmpSizes)) {
        workgroupSize[i] = llvm::cast<IntegerAttr>(size).getInt();
      }
    } else {
      std::optional<SmallVector<int64_t>> maybeWorkgroupSize =
          getWorkgroupSize(func);
      if (!maybeWorkgroupSize) {
        func->emitOpError()
            << "unable to query workgroup_size information from entry point";
        return signalPassFailure();
      }
      for (auto [index, value] : llvm::enumerate(maybeWorkgroupSize.value())) {
        workgroupSize[index] = value;
      }
      for (auto index : llvm::seq<size_t>(maybeWorkgroupSize->size(), 3)) {
        workgroupSize[index] = 1;
      }
    }

    llvm::StringLiteral scheduleAttrName =
        IREE::GPU::MMAScheduleAttr::getMnemonic();
    auto scheduleAttr =
        func->getAttrOfType<IREE::GPU::MMAScheduleAttr>(scheduleAttrName);
    if (!scheduleAttr) {
      DictionaryAttr configDict = getTranslationInfo(func).getConfiguration();
      scheduleAttr = dyn_cast_or_null<IREE::GPU::MMAScheduleAttr>(
          configDict.get(scheduleAttrName));
    }

    // Vector layout option setter aimed at contractions. Currently this only
    // sets anchors for two types of operations; vector.contract and
    // vector.transfer_read from non-shared memory. The assumption in this case
    // is that all IR input to this pass has a leaf rooted on a transfer_read or
    // includes a contraction in the program slice, meaning all operations
    // should receive layouts. Layout setting for other problems like reductions
    // is TODO.
    SmallVector<vector::TransferReadOp> reads;

    func->walk([&](Operation *op) {
      llvm::TypeSwitch<Operation *>(op).Case(
          [&](vector::TransferReadOp transfer) { reads.push_back(transfer); });
    });

    IRRewriter rewriter(func);
    llvm::outs()<<"NUM of transfer_read:"<<reads.size()<<"\n";
    for (vector::TransferReadOp read : reads) {
      // llvm::outs()<<"READ:"<<read<<"\n";
      if (failed(setTransferReadAnchor(workgroupSize, rewriter, read))) {
        return signalPassFailure();
      }
      // llvm::outs()<<"READ After:"<<read<<"\n";
    }
  }
};
} // namespace
} // namespace mlir::iree_compiler
