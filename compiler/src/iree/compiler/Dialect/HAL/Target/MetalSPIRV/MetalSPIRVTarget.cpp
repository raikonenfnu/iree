// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/MetalSPIRVTarget.h"

#include "iree/compiler/Codegen/Dialect/IREECodegenDialect.h"
#include "iree/compiler/Codegen/Passes.h"
#include "iree/compiler/Dialect/HAL/Target/MetalSPIRV/SPIRVToMSL.h"
#include "iree/compiler/Dialect/HAL/Target/TargetRegistry.h"
#include "iree/compiler/Utils/FlatbufferUtils.h"
#include "iree/schemas/metal_executable_def_builder.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVDialect.h"
#include "mlir/Dialect/SPIRV/IR/SPIRVOps.h"
#include "mlir/Dialect/SPIRV/IR/TargetAndABI.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/SPIRV/Serialization.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

// TODO(antiagainst): provide a proper target environment for Metal.
static spirv::TargetEnvAttr getMetalTargetEnv(MLIRContext *context) {
  auto triple = spirv::VerCapExtAttr::get(
      spirv::Version::V_1_0, {spirv::Capability::Shader},
      {spirv::Extension::SPV_KHR_storage_buffer_storage_class}, context);
  return spirv::TargetEnvAttr::get(triple, spirv::Vendor::Unknown,
                                   spirv::DeviceType::Unknown,
                                   spirv::TargetEnvAttr::kUnknownDeviceID,
                                   spirv::getDefaultResourceLimits(context));
}

class MetalSPIRVTargetBackend : public TargetBackend {
 public:
  MetalSPIRVTargetBackend() = default;

  // NOTE: we could vary this based on the options such as 'metal-v2'.
  std::string name() const override { return "metal"; }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::Codegen::IREECodegenDialect, spirv::SPIRVDialect,
                    gpu::GPUDialect>();
  }

  IREE::HAL::DeviceTargetAttr getDefaultDeviceTarget(
      MLIRContext *context) const override {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    // Indicates that the runtime HAL driver operates only in the legacy
    // synchronous mode.
    configItems.emplace_back(b.getStringAttr("legacy_sync"), b.getUnitAttr());

    configItems.emplace_back(b.getStringAttr("executable_targets"),
                             getExecutableTargets(context));

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::DeviceTargetAttr::get(
        context, b.getStringAttr(deviceID()), configAttr);
  }

  void buildTranslationPassPipeline(OpPassManager &passManager) override {
    buildSPIRVCodegenPassPipeline(passManager, /*enableFastMath=*/false,
                                  /*use64bitIndex=*/false);
  }

  LogicalResult serializeExecutable(const SerializationOptions &options,
                                    IREE::HAL::ExecutableVariantOp variantOp,
                                    OpBuilder &executableBuilder) override {
    ModuleOp innerModuleOp = variantOp.getInnerModule();
    auto spvModuleOp = *innerModuleOp.getOps<spirv::ModuleOp>().begin();

    // The runtime use ordinals instead of names but Metal requires function
    // names for constructing pipeline states. Get an ordered list of the entry
    // point names.
    SmallVector<StringRef, 8> entryPointNames;
    spvModuleOp.walk([&](spirv::EntryPointOp exportOp) {
      entryPointNames.push_back(exportOp.getFn());
    });

    // 1. Serialize the spirv::ModuleOp into binary format.
    SmallVector<uint32_t, 0> spvBinary;
    if (failed(spirv::serialize(spvModuleOp, spvBinary))) {
      return variantOp.emitError() << "failed to serialize spirv.module";
    }
    if (!options.dumpIntermediatesPath.empty()) {
      dumpDataToPath<uint32_t>(options.dumpIntermediatesPath,
                               options.dumpBaseName, variantOp.getName(),
                               ".spv", spvBinary);
    }

    // 2. Cross compile SPIR-V to MSL source code.
    llvm::SmallVector<MetalShader, 2> mslShaders;
    for (const auto &entryPoint : entryPointNames) {
      llvm::Optional<MetalShader> mslShader = crossCompileSPIRVToMSL(
          // We can use ArrayRef here given spvBinary reserves 0 bytes on stack.
          llvm::makeArrayRef(spvBinary.data(), spvBinary.size()), entryPoint);
      if (!mslShader) {
        return variantOp.emitError()
               << "failed to cross compile SPIR-V to Metal shader";
      }
      mslShaders.push_back(std::move(*mslShader));
    }

    // 3. Compile MSL to MTLLibrary.
    // TODO(antiagainst): provide the option to compile the shaders into a
    // library and embed in the FlatBuffer. Metal provides APIs for compiling
    // shader sources into a MTLLibrary at run-time, but does not provie
    // a way to serialize the generated MTLLibrary. The only way available is
    // to use command-line tools like `metal` and `metallib`. Likely we need
    // to invoke them in C++.

    if (!options.dumpBinariesPath.empty()) {
      for (auto shader : llvm::enumerate(mslShaders)) {
        dumpDataToPath(
            options.dumpBinariesPath, options.dumpBaseName,
            (variantOp.getName() + std::to_string(shader.index())).str(),
            ".msl", shader.value().source);
      }
    }

    // 4. Pack the MTLLibrary and metadata into a FlatBuffer.
    FlatbufferBuilder builder;
    iree_MetalExecutableDef_start_as_root(builder);

    auto shaderSourcesRef = builder.createStringVec(llvm::map_range(
        mslShaders, [&](const MetalShader &shader) { return shader.source; }));

    iree_MetalThreadgroupSize_vec_start(builder);
    for (auto &shader : mslShaders) {
      iree_MetalThreadgroupSize_vec_push_create(
          builder, shader.threadgroupSize.x, shader.threadgroupSize.y,
          shader.threadgroupSize.z);
    }
    auto threadgroupSizesRef = iree_MetalThreadgroupSize_vec_end(builder);

    auto entryPointNamesRef = builder.createStringVec(entryPointNames);

    iree_MetalExecutableDef_entry_points_add(builder, entryPointNamesRef);
    iree_MetalExecutableDef_threadgroup_sizes_add(builder, threadgroupSizesRef);
    iree_MetalExecutableDef_shader_sources_add(builder, shaderSourcesRef);
    iree_MetalExecutableDef_end_as_root(builder);

    // 5. Add the binary data to the target executable.
    auto binaryOp = executableBuilder.create<IREE::HAL::ExecutableBinaryOp>(
        variantOp.getLoc(), variantOp.getSymName(),
        variantOp.getTarget().getFormat(),
        builder.getBufferAttr(executableBuilder.getContext()));
    binaryOp.setMimeTypeAttr(
        executableBuilder.getStringAttr("application/x-flatbuffers"));

    return success();
  }

 private:
  ArrayAttr getExecutableTargets(MLIRContext *context) const {
    SmallVector<Attribute> targetAttrs;
    // If we had multiple target environments we would generate one target attr
    // per environment, with each setting its own environment attribute.
    targetAttrs.push_back(
        getExecutableTarget(context, getMetalTargetEnv(context)));
    return ArrayAttr::get(context, targetAttrs);
  }

  IREE::HAL::ExecutableTargetAttr getExecutableTarget(
      MLIRContext *context, spirv::TargetEnvAttr targetEnv) const {
    Builder b(context);
    SmallVector<NamedAttribute> configItems;

    configItems.emplace_back(b.getStringAttr(spirv::getTargetEnvAttrName()),
                             targetEnv);

    auto configAttr = b.getDictionaryAttr(configItems);
    return IREE::HAL::ExecutableTargetAttr::get(
        context, b.getStringAttr("metal"), b.getStringAttr("metal-msl-fb"),
        configAttr);
  }
};

void registerMetalSPIRVTargetBackends() {
  auto backendFactory = [=]() {
    return std::make_shared<MetalSPIRVTargetBackend>();
  };
  // #hal.device.target<"metal", ...
  static TargetBackendRegistration registration0("metal", backendFactory);
  // #hal.executable.target<"metal-spirv", ...
  static TargetBackendRegistration registration1("metal-spirv", backendFactory);
}

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
