// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/KernelDispatch.h"
#include "iree/compiler/Codegen/PassDetail.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Codegen/Utils/Utils.h"
#include "iree/compiler/Dialect/HAL/IR/HALDialect.h"
#include "iree/compiler/Dialect/HAL/IR/HALOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/Passes.h"

namespace mlir {
namespace iree_compiler {

namespace {
/// Lowers an hal.executable.variant operation to scalar/native-vector
/// code. Invokes different compilation pipeline to
/// - first lower to scalar/native-vector code
/// - then convert to LLVM dialect.
/// In due course this could be used to generate code for all backends.
class NodHardwareOptimizationPass
    : public NodHardwareOptimizationBase<
          NodHardwareOptimizationPass> {
 public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::HAL::HALDialect, linalg::LinalgDialect,
                    LLVM::LLVMDialect, vector::VectorDialect>();
  }

  NodHardwareOptimizationPass(){}
  NodHardwareOptimizationPass(
      const NodHardwareOptimizationPass &pass) {}

  void runOnOperation() override;

};
}  // namespace


void NodHardwareOptimizationPass::runOnOperation() {
    MLIRContext *context = &getContext();
    auto funcOp = getOperation();
    /*
    Final Goal: Get LLVMCPULowerExecutableTarget to use PE_IDs instead of HAL
     - TODO(raikonenfnu): Set maximum number of workgroup to 64 => size of PE GRIDS
     - TODO(raikonenfnu): Look for HAL Operations below:
      %workgroup_id_x = hal.interface.workgroup.id[0] : index
      %workgroup_count_x = hal.interface.workgroup.count[0] : index

     - TODO(raikonenfnu): Replace hal.interface.workgroup.id flattened pe_id
     - TODO(raikonenfnu): Replace hal.interface.workgroup.count by finding number of workgroup
     actually used workloadperworkgroup = workload/workgroup.count
          -> Might need to slice as an alternative LLVMCPULowerExecutablTargetPass
    */
}

std::unique_ptr<OperationPass<IREE::HAL::ExecutableVariantOp>>
createNodHardwareOptimizationPass() {
  return std::make_unique<NodHardwareOptimizationPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
