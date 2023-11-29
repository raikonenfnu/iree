// Copyright 2023 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMGPU/Utils/VectorLayoutProvider.h"

using namespace mlir::iree_compiler::IREE::VectorExt;
using namespace mlir::iree_compiler;
using namespace mlir;

void AMDGPULayoutProvider::setAnchorOps() {
  MLIRContext *ctx = root->getContext();

  // TODO: This is a fake layout, just to test the analysis.
  root->walk([&](Operation *op) {
    if (auto contractOp = dyn_cast<vector::ContractionOp>(op)) {

      auto setLayout = [&](TypedValue<VectorType> val) {
        // Get shape of value.
        ArrayRef<int64_t> shape = val.getType().getShape();
        assert(shape.size() == 2 && "Only support 2D contraction for now.");

        SmallVector<LayoutDimensionAttr> labelsRow = {
            LayoutDimensionAttr::get(ctx, LayoutDimension::VECTORY)};
        SmallVector<int64_t> sizesRow = {shape[1]};
        PerDimLayoutAttr row = PerDimLayoutAttr::get(ctx, labelsRow, sizesRow);

        SmallVector<LayoutDimensionAttr> labelsCol = {
            LayoutDimensionAttr::get(ctx, LayoutDimension::VECTORY)};
        SmallVector<int64_t> sizesCol = {shape[0]};
        PerDimLayoutAttr col = PerDimLayoutAttr::get(ctx, labelsCol, sizesCol);

        SmallVector<PerDimLayoutAttr> layouts = {row, col};
        LayoutAttr layout = LayoutAttr::get(ctx, layouts);

        analysis.setAnchor(val, layout);
      };

      setLayout(contractOp.getLhs());
      setLayout(contractOp.getRhs());

      Value result = contractOp.getResult();
      assert(isa<VectorType>(result.getType()) &&
             "Only support vector result for now.");
      TypedValue<VectorType> resultVal = cast<TypedValue<VectorType>>(result);
      setLayout(resultVal);
    }
  });
}
