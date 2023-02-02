// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===- CovertToSPIRVPass.cpp - Performs the final SPIR-V conversion -------===//
//
// This file implements a pass to perform the final conversion to SPIR-V.
// This pass converts remaining interface ops into SPIR-V global variables,
// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
// corresponding SPIR-V ops.
//
//===----------------------------------------------------------------------===//

#include <tuple>

#include "iree/compiler/Codegen/Dialect/LoweringConfig.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/SPIRV/Utils.h"
#include "iree/compiler/Codegen/Utils/MarkerUtils.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "iree/compiler/Dialect/Util/IR/UtilOps.h"
#include "llvm/ADT/DenseMapInfo.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/FormatVariadic.h"
#include "mlir/Conversion/ArithToSPIRV/ArithToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/Transforms/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVTypes.h"
#include "mlir/Dialect/SPIRV/Transforms/SPIRVConversion.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"
#include "iree/compiler/Dialect/Stream/IR/StreamDialect.h"
#include "iree/compiler/Dialect/Stream/IR/StreamOps.h"
#include "iree/compiler/Dialect/Stream/IR/StreamTypes.h"


namespace mlir {
namespace iree_compiler {

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

/// A pass to perform the SPIR-V conversion.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
class SPIRVCombineDispatchPass : public SPIRVCombineDispatchBase<SPIRVCombineDispatchPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  void runOnOperation() override;
};
}  // namespace

void SPIRVCombineDispatchPass::runOnOperation() {
  llvm::outs()<<"testing combine dispatcher!\n";
  // MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();
  auto AncestormoduleOp = moduleOp->getParentOfType<mlir::ModuleOp>();
  // Build dependency graph.
  // go down topology.
  /*
  parent_dispatch = null.
  for cur_dispatch in dispatchOps:
    for read in cur_dispatch.read_resources:
      if read in parent_write_resources:
       remove read
      else:
       add read into combined_read.
    combine_dispatches(parent_dispatch, cur_dispatch)
    write_map[resource] -> sourceDispatchOp.
  */
  SymbolTable symbolTable(AncestormoduleOp);
  MLIRContext *context = &getContext();
  // If need multiple executeOp, use AncestormoduleOp->walk([&](IREE::Stream::CmdDispatchOp dispatchOp)
  for (auto funcOp : AncestormoduleOp.getOps<func::FuncOp>()) {
    for (auto executeOp : funcOp.getOps<IREE::Stream::CmdExecuteOp>()) {
      IREE::Stream::CmdDispatchOp prevDispatchOp;
      llvm::SmallSetVector<Value, 4> writeResources;
      for (auto dispatchOp : llvm::make_early_inc_range(executeOp.getOps<IREE::Stream::CmdDispatchOp>())) {
        llvm::outs()<<"dispatch:"<<dispatchOp<<"\n";
        size_t resourceCount = dispatchOp.getResources().size();
        auto resourceAccessesAttrs = dispatchOp.getResourceAccesses().getValue();
        auto resourceSizes = dispatchOp.getResourceSizes();
        auto resourceOffsets = dispatchOp.getResourceOffsets();
        auto resourceLengths = dispatchOp.getResourceLengths();
        auto operands = dispatchOp.getResources();
        auto exportName = dispatchOp.getEntryPointAttr().getRootReference().getValue().split("::").first;
        auto fn_name = StringAttr::get(context, exportName);
        auto exportOp = symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableOp>(
            AncestormoduleOp, fn_name);
        if(!exportOp) continue;
        llvm::outs()<<"export:"<<exportName<<" : "<<exportOp.getSymName()<<"\n";
        for (unsigned i = 0; i < resourceCount; ++i) {
          auto resourceAccessAttr =
              resourceAccessesAttrs[i]
                  .cast<IREE::Stream::ResourceAccessBitfieldAttr>();
          auto resourceAccess = static_cast<IREE::Stream::ResourceAccessBitfield>(
              resourceAccessAttr.getInt());
          if (resourceAccess == IREE::Stream::ResourceAccessBitfield::Read) {
            if (writeResources.count(operands[i]) != 0 && operands[i].getLifetime() == IREE::Stream::Lifetime::Transient) {
              // TODO: check with resourceOffset and resourceLength to make sure it's exactly the same.
              eliminatedResource.insert(operands[i]);
            } else {
              newReadResource.insert(operands[i]);
            }
          } else if (resourceAccess == IREE::Stream::ResourceAccessBitfield::Write) {
            newWriteResources.insert(operands[i]);
          }
          /*
          0.Create list of newReadResources + newWriteResources
          1.Looking at eliminatedResource, combine the function regions:
          2. Create new funcOp using all reads and all writes. (look at the fusion code in llvm-project)
            copy  block of first funcOp region, copy block of SecondFuncOp region,
            find eliminatedResource part in secondOpRegion and replace use with the returning variable of eliminatedResource of first funcOp.
          */
          // combineDispatchOps(prevDispatchOp, dispatchOp, eliminatedResource, newWriteResources, newReadResources);
          llvm::outs()<<"operand:"<<operands[i]<<"with access"<<resourceAccess<<","<<resourceSizes[i]<<","<<resourceOffsets[i]<<","<<resourceLengths[i]<<"\n";
        }
        prevDispatchOp = dispatchOp;
      }
    }
  }
  // for (auto funcOp : llvm::make_early_inc_range(
  //           AncestormoduleOp.getOps<func::FuncOp>())) {
  //   for (auto dispatchOp : llvm::make_early_inc_range(
  //             funcOp.getOps<IREE::Stream::CmdDispatchOp>())) {
  // // AncestormoduleOp->walk([&](IREE::Stream::CmdDispatchOp dispatchOp) {
  //   llvm::outs()<<"dispatch:"<<dispatchOp<<"\n";
  //   size_t resourceCount = dispatchOp.getResources().size();
  //   auto resourceAccessesAttrs = dispatchOp.getResourceAccesses().getValue();
  //   auto resourceSizes = dispatchOp.getResourceSizes();
  //   auto resourceOffsets = dispatchOp.getResourceOffsets();
  //   auto resourceLengths = dispatchOp.getResourceLengths();
  //   auto operands = dispatchOp.getResources();
  //   auto exportName = dispatchOp.getEntryPointAttr().getRootReference().getValue().split("::").first;
  //   auto fn_name = StringAttr::get(context, exportName);
  //   auto exportOp = symbolTable.lookupNearestSymbolFrom<IREE::HAL::ExecutableOp>(
  //       dispatchOp, fn_name);
  //   llvm::outs()<<"export:"<<exportName<<" : "<<exportOp<<"\n";
  //   for (unsigned i = 0; i < resourceCount; ++i) {
  //     auto resourceAccessAttr =
  //         resourceAccessesAttrs[i]
  //             .cast<IREE::Stream::ResourceAccessBitfieldAttr>();
  //     auto resourceAccess = static_cast<IREE::Stream::ResourceAccessBitfield>(
  //         resourceAccessAttr.getInt());
  //     llvm::outs()<<"operand:"<<operands[i]<<"with access"<<resourceAccess<<","<<resourceSizes[i]<<","<<resourceOffsets[i]<<","<<resourceLengths[i]<<"\n";
  //   }
  //   prevDispatchOp = dispatchOp;
  //   llvm::outs()<<"\n\n";
  // }
  // }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createSPIRVCombineDispatchPass() {
  return std::make_unique<SPIRVCombineDispatchPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
