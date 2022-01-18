// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"

#include <mutex>

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/rocm_executable_def_builder.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Target/TargetMachine.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

static llvm::cl::opt<std::string> clROCMTargetChip(
    "iree-rocm-target-chip", llvm::cl::desc("ROCm target Chip"),
    llvm::cl::init("gfx908"));

static llvm::cl::opt<bool> clROCMLinkBC(
    "iree-rocm-link-bc",
    llvm::cl::desc("Whether to try Linking to AMD Bitcodes"),
    llvm::cl::init(false));

static llvm::cl::opt<std::string> clROCMBitcodeDir(
    "iree-rocm-bc-dir", llvm::cl::desc("Directory of ROCM Bitcode"),
    llvm::cl::init("/opt/rocm/amdgcn/bitcode"));

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

static std::string translateModuleToISA(llvm::Module &module,
                                        llvm::TargetMachine &targetMachine) {
  std::string targetISA;
  {
    llvm::raw_string_ostream stream(targetISA);
    llvm::buffer_ostream pstream(stream);
    llvm::legacy::PassManager codegenPasses;
    targetMachine.addPassesToEmitFile(codegenPasses, pstream, nullptr,
                                      llvm::CGFT_ObjectFile);
    codegenPasses.run(module);
  }
  return targetISA;
}

class ROCMTargetBackend final : public TargetBackend {
 public:
  std::string name() const override { return "rocm"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    mlir::registerLLVMDialectTranslation(registry);
    mlir::registerROCDLDialectTranslation(registry);
    registry.insert<IREE::Codegen::IREECodegenDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;
    configItems.emplace_back(b.getIdentifier("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildLLVMGPUTransformPassPipeline(passManager, true);
  }

  LogicalResult serializeExecutable(
      iree_compiler::IREE::HAL::ExecutableVariantOp variantOp,
      OpBuilder &executableBuilder) override {
    // Perform the translation in a separate context to avoid any
    // multi-threading issues.
    llvm::LLVMContext context;

    // We name our files after the executable name so that they are easy to
    // track both during compilation (logs/artifacts/etc), as outputs (final
    // intermediate code/binary files), and at runtime (loaded
    // libraries/symbols/etc).
    auto libraryName =
        variantOp->getParentOfType<iree_compiler::IREE::HAL::ExecutableOp>()
            .getName()
            .str();

    ModuleOp innerModuleOp = variantOp.getInnerModule();

    // Remove all the functions that are not part of the ROCM kernel.
    // TODO: Find a better solution to handle this.
    auto illegalFuncOps = llvm::to_vector<4>(innerModuleOp.getOps<FuncOp>());
    for (auto funcOp : illegalFuncOps) {
      funcOp.erase();
    }

    auto llvmModule =
        mlir::translateModuleToLLVMIR(innerModuleOp, context, libraryName);
    if (!llvmModule) {
      return variantOp.emitError() << "failed to translate the MLIR LLVM "
                                      "dialect to the native llvm::Module";
    }

    // Collect all the entry point names.
    llvm::StringMap<IREE::HAL::ExecutableEntryPointOp> entryPointOps;
    for (auto op : variantOp.getOps<IREE::HAL::ExecutableEntryPointOp>()) {
      entryPointOps[op.sym_name()] = op;
    }
    std::vector<std::array<int32_t, 3>> workgroupSizes;
    for (auto func : innerModuleOp.getOps<LLVM::LLVMFuncOp>()) {
      int32_t flatWgSize = 1;
      auto *llvmFunc = llvmModule->getFunction(func.getName());
      if (llvmFunc->isDeclaration()) continue;
      std::array<int32_t, 3> workgroup_size;
      auto entryPointOp = entryPointOps[func.getName()];
      if (Optional<ArrayAttr> workgroupSizeAttr =
              entryPointOp.workgroup_size()) {
        for (auto it : llvm::enumerate(workgroupSizeAttr.getValue())) {
          workgroup_size[it.index()] = it.value().cast<IntegerAttr>().getInt();
          flatWgSize *= it.value().cast<IntegerAttr>().getInt();
        }
      } else {
        workgroup_size = {1, 1, 1};
      }
      workgroupSizes.push_back(workgroup_size);
      // For GPU kernels,
      // 1. Insert AMDGPU_KERNEL calling convention.
      // 2. Insert amdgpu-flat-workgroup-size(1, 256) attribute.
      // 3. Insert amdgpu-implicitarg-num-bytes=56 (which must be set on OpenCL
      // and HIP kernels per Clang)
      llvmFunc->setCallingConv(llvm::CallingConv::AMDGPU_KERNEL);
      std::string wgSizeRange = std::string("1, ") + std::to_string(flatWgSize);
      llvmFunc->addFnAttr("amdgpu-flat-work-group-size", wgSizeRange);
    }

    std::unique_ptr<llvm::TargetMachine> targetMachine;
    {
      llvm::Triple triple("amdgcn--amdhsa-amdgiz");
      std::string targetChip = clROCMTargetChip;
      std::string error;
      const llvm::Target *target =
          llvm::TargetRegistry::lookupTarget("", triple, error);
      if (target == nullptr) {
        return variantOp.emitError() << "cannot initialize target triple";
      }
      targetMachine.reset(
          target->createTargetMachine(triple.str(), targetChip, {}, {}, {}));
      if (targetMachine == nullptr) {
        return variantOp.emitError() << "cannot initialize target machine";
      }
    }

    llvmModule->setDataLayout(targetMachine->createDataLayout());

    iree_compiler::FlatbufferBuilder builder;
    iree_ROCMExecutableDef_start_as_root(builder);

    // Link module to Device Library
    if (clROCMLinkBC) {
      LinkROCDLIfNecessary(llvmModule.get(), clROCMTargetChip,
                           clROCMBitcodeDir);
    }

    // Serialize hsaco kernel into the binary that we will embed in the
    // final flatbuffer.
    std::string targetISA = translateModuleToISA(*llvmModule, *targetMachine);
    std::string targetHSACO = createHsaco(targetISA, libraryName);
    auto hsacoRef = flatbuffers_uint8_vec_create(
        builder, reinterpret_cast<const uint8_t *>(targetHSACO.c_str()),
        targetHSACO.size());

    auto entryPointNames = llvm::to_vector<8>(llvm::map_range(
        variantOp.getBlock()
            .getOps<iree_compiler::IREE::HAL::ExecutableEntryPointOp>(),
        [&](auto op) { return op.getName(); }));
    auto entryPointsRef = builder.createStringVec(entryPointNames);

    iree_ROCMBlockSizeDef_vec_start(builder);
    auto blockSizes = workgroupSizes.begin();
    for (int i = 0, e = entryPointNames.size(); i < e; ++i) {
      iree_ROCMBlockSizeDef_vec_push_create(builder, (*blockSizes)[0],
                                            (*blockSizes)[1], (*blockSizes)[2]);
      ++blockSizes;
    }
    auto blockSizesRef = iree_ROCMBlockSizeDef_vec_end(builder);

    iree_ROCMExecutableDef_entry_points_add(builder, entryPointsRef);
    iree_ROCMExecutableDef_block_sizes_add(builder, blockSizesRef);
    iree_ROCMExecutableDef_hsaco_image_add(builder, hsacoRef);
    iree_ROCMExecutableDef_end_as_root(builder);

    // Add the binary data to the target executable.
    executableBuilder.create<iree_compiler::IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.sym_name(),
        variantOp.target().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));

    return success();
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(getExecutableTarget(context));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("rocm"), b.getStringAttr("rocm-hsaco-fb"),
        configAttr);
  }

};

void registerROCMTargetBackends() {
  // #hal.device.target<"rocm", ...
  // #hal.executable.target<"rocm", ...
  static iree_compiler::IREE::HAL::TargetBackendRegistration registration(
      "rocm", [=]() {
        LLVMInitializeAMDGPUTarget();
        LLVMInitializeAMDGPUTargetMC();
        LLVMInitializeAMDGPUTargetInfo();
        LLVMInitializeAMDGPUAsmPrinter();
        return std::make_shared<ROCMTargetBackend>();
      });
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
