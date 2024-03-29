// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "compiler/plugins/target/ROCM/ROCMTargetFeatures.h"

#include "iree/compiler/Codegen/Dialect/GPU/IR/IREEGPUAttrs.h"
#include "llvm/ADT/StringSwitch.h"

namespace mlir::iree_compiler::IREE::HAL {

template <typename MMAAttr, typename MMAIntrinsic>
static ArrayAttr getMmaArrayAttr(MLIRContext *context,
                                 ArrayRef<MMAIntrinsic> types) {
  SmallVector<Attribute> attrs(types.size(), MMAAttr());
  for (auto [idx, type] : llvm::enumerate(types)) {
    attrs[idx] = MMAAttr::get(context, type);
  }
  return ArrayAttr::get(context, attrs);
}

ArrayAttr getROCMSupportedMmaAttrs(MLIRContext *context, StringRef targetArch) {
  if (targetArch == "gfx940" || targetArch == "gfx942") { // MI300A/X
    return getMmaArrayAttr<IREE::GPU::MFMAAttr, IREE::GPU::MFMAIntrinsic>(
        context, {IREE::GPU::MFMAIntrinsic::MFMA_F16_16x16x16_F32,
                  IREE::GPU::MFMAIntrinsic::MFMA_F16_32x32x8_F32});
  } else if (targetArch == "gfx90a") { // MI210
    return getMmaArrayAttr<IREE::GPU::MFMAAttr, IREE::GPU::MFMAIntrinsic>(
        context, {IREE::GPU::MFMAIntrinsic::MFMA_F16_16x16x16_F32,
                  IREE::GPU::MFMAIntrinsic::MFMA_F16_32x32x8_F32});
  } else if (targetArch == "gfx1100") { // RDNA3
    return getMmaArrayAttr<IREE::GPU::WMMAAttr, IREE::GPU::WMMAIntrinsic>(
        context, {IREE::GPU::WMMAIntrinsic::WMMA_F16_16x16x16_F16,
                  IREE::GPU::WMMAIntrinsic::WMMA_F16_16x16x16_F16});
  }
  return ArrayAttr();
}

} // namespace mlir::iree_compiler::IREE::HAL
