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
#include "mlir/Conversion/ArithmeticToSPIRV/ArithmeticToSPIRV.h"
#include "mlir/Conversion/ControlFlowToSPIRV/ControlFlowToSPIRVPass.h"
#include "mlir/Conversion/FuncToSPIRV/FuncToSPIRV.h"
#include "mlir/Conversion/GPUToSPIRV/GPUToSPIRV.h"
#include "mlir/Conversion/MathToSPIRV/MathToSPIRV.h"
#include "mlir/Conversion/MemRefToSPIRV/MemRefToSPIRV.h"
#include "mlir/Conversion/SCFToSPIRV/SCFToSPIRV.h"
#include "mlir/Conversion/TensorToSPIRV/TensorToSPIRV.h"
#include "mlir/Conversion/TosaToArith/TosaToArith.h"
#include "mlir/Conversion/VectorToSPIRV/VectorToSPIRV.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace iree_compiler {
namespace {
//===----------------------------------------------------------------------===//
// Resource utilities
//===----------------------------------------------------------------------===//

/// Map from hal.interface.binding.subspan ops to their corresponding
/// spv.GlobalVariable ops.
using InterfaceResourceMap =
    llvm::DenseMap<Operation *, spirv::GlobalVariableOp>;

/// Creates a resource evariable of the given `type` at the beginning of
/// `moduleOp`'s block via `symbolTable` and bind it to `set` and `binding`.
spirv::GlobalVariableOp createResourceVariable(Location loc, Type type,
                                               unsigned set, unsigned binding,
                                               bool alias, ModuleOp moduleOp,
                                               SymbolTable *symbolTable) {
  std::string name = llvm::formatv("__resource_var_{0}_{1}_", set, binding);
  OpBuilder builder(moduleOp.getContext());
  auto variable =
      builder.create<spirv::GlobalVariableOp>(loc, type, name, set, binding);
  if (alias) variable->setAttr("aliased", builder.getUnitAttr());
  symbolTable->insert(variable, moduleOp.getBody()->begin());
  return variable;
}

/// Returns the (set, binding) pair for the given interface op.
std::pair<int32_t, int32_t> getInterfaceSetAndBinding(
    IREE::HAL::InterfaceBindingSubspanOp op) {
  return {op.set().getSExtValue(), op.binding().getSExtValue()};
}

/// Scans all hal.interface.binding.subspan ops in `module`, creates their
/// corresponding spv.GlobalVariables when needed, and returns the map.
/// The created variables need to have their types fixed later.
InterfaceResourceMap createResourceVariables(mlir::ModuleOp module) {
  SymbolTable symbolTable(module);
  InterfaceResourceMap interfaceToResourceVars;

  auto fns = llvm::to_vector<1>(module.getOps<func::FuncOp>());
  for (func::FuncOp func : llvm::reverse(fns)) {
    // Collect all interface ops and their (set, binding) pairs in this
    // function. Use SmallVector here for a deterministic order.
    SmallVector<IREE::HAL::InterfaceBindingSubspanOp, 8> subspanOps;
    SmallVector<std::pair<uint32_t, uint32_t>, 8> setBindings;

    // Use a map to see if we have different types for one (set, binding) pair,
    // which will require creating multiple SPIR-V global variables.
    llvm::DenseMap<std::pair<uint32_t, uint32_t>, llvm::DenseSet<Type>>
        setBindingTypes;

    func.walk([&](Operation *op) {
      auto subspanOp = dyn_cast<IREE::HAL::InterfaceBindingSubspanOp>(op);
      if (!subspanOp || subspanOp.use_empty()) return;
      subspanOps.emplace_back(subspanOp);
      setBindings.emplace_back(getInterfaceSetAndBinding(subspanOp));
      setBindingTypes[setBindings.back()].insert(subspanOp.getType());
    });

    // Keep track of created SPIR-V global variables. This allows us to
    // deduplicate when possible to reduce generated SPIR-V blob size.
    llvm::DenseMap<std::tuple<uint32_t, uint32_t, Type>,
                   spirv::GlobalVariableOp>
        resourceVars;

    for (int i = subspanOps.size() - 1; i >= 0; --i) {
      auto subspanOp = subspanOps[i];
      const auto &setBinding = setBindings[i];

      auto key = std::make_tuple(setBinding.first, setBinding.second,
                                 subspanOp.getType());
      auto var = resourceVars.lookup(key);
      if (!var) {
        // If we have multiple SPIR-V global variables bound to the same (set,
        // binding) pair and they are used in the same function, those variables
        // need to have alias decoration.
        bool alias = setBindingTypes[setBindings[i]].size() > 1;

        // We are using the interface op's type for creating the global
        // variable. It's fine. The correctness boundary is the pass.
        // We will fix it up during conversion so it won't leak.
        var = createResourceVariable(subspanOp.getLoc(), subspanOp.getType(),
                                     setBinding.first, setBinding.second, alias,
                                     module, &symbolTable);
        resourceVars[key] = var;
      }

      interfaceToResourceVars[subspanOp] = var;
    }
  }

  return interfaceToResourceVars;
}

}  // namespace

//===----------------------------------------------------------------------===//
// Conversion patterns
//===----------------------------------------------------------------------===//

namespace {
/// A pattern to convert hal.interface.constant.load into a sequence of SPIR-V
/// ops to load from a global variable representing the push constant storage.
struct HALInterfaceLoadConstantConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceConstantLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceConstantLoadOp loadOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    // TODO(#1519): this conversion should look up the entry point information
    // to get the total push constant count.
    auto variantOp = loadOp->getParentOfType<IREE::HAL::ExecutableVariantOp>();
    auto exportOps =
        llvm::to_vector<1>(variantOp.getOps<IREE::HAL::ExecutableExportOp>());
    assert(exportOps.size() == 1);
    auto layoutAttr = exportOps.front().layout();

    uint64_t elementCount = layoutAttr.getPushConstants();
    unsigned index = loadOp.index().getZExtValue();

    // The following function generates SPIR-V ops with i32 types. So it does
    // type "conversion" (index -> i32) implicitly.
    auto i32Type = rewriter.getIntegerType(32);
    auto value = spirv::getPushConstantValue(loadOp, elementCount, index,
                                             i32Type, rewriter);

    rewriter.replaceOp(loadOp, value);
    return success();
  }
};

/// A pattern to convert hal.interface.workgroup.id/count into corresponding
/// SPIR-V Builtin ops.
template <typename InterfaceOpTy, spirv::BuiltIn builtin>
struct HALInterfaceWorkgroupIdAndCountConverter final
    : public OpConversionPattern<InterfaceOpTy> {
  using OpConversionPattern<InterfaceOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(
      InterfaceOpTy op, typename InterfaceOpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    int32_t index = static_cast<int32_t>(op.dimension().getSExtValue());
    auto i32Type = rewriter.getIntegerType(32);
    Value spirvBuiltin =
        spirv::getBuiltinVariableValue(op, builtin, i32Type, rewriter);
    rewriter.replaceOpWithNewOp<spirv::CompositeExtractOp>(
        op, i32Type, spirvBuiltin, rewriter.getI32ArrayAttr({index}));
    return success();
  }
};

using SetBinding = std::pair<APInt, APInt>;

/// Convention with the HAL side to pass kernel arguments.
/// The bindings are ordered based on binding set and binding index then
/// compressed and mapped to dense set of arguments.
/// This function looks at the symbols and return the mapping between
/// InterfaceBindingOp and kernel argument index.
/// For instance if the kernel has (set, bindings) A(0, 1), B(1, 5), C(0, 6) it
/// will return the mapping [A, 0], [C, 1], [B, 2]
static llvm::SmallDenseMap<SetBinding, size_t> getKernelArgMapping(
    Operation *funcOp) {
  llvm::SetVector<SetBinding> usedBindingSet;
  funcOp->walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
    usedBindingSet.insert(SetBinding(subspanOp.set(), subspanOp.binding()));
  });
  auto sparseBindings = usedBindingSet.takeVector();
  std::sort(sparseBindings.begin(), sparseBindings.end(),
            [](SetBinding lhs, SetBinding rhs) {
              if (lhs.first == rhs.first) return lhs.second.ult(rhs.second);
              return lhs.first.ult(rhs.first);
            });
  llvm::SmallDenseMap<SetBinding, size_t> mapBindingArgIndex;
  for (auto binding : llvm::enumerate(sparseBindings)) {
    mapBindingArgIndex[binding.value()] = binding.index();
  }
  return mapBindingArgIndex;
}

// TODO(raikonenfnu): Modify subspan conversion to replace subspanOp, by the
// function argument.

/// A pattern to convert hal.interface.binding.subspan into a sequence of SPIR-V
/// ops to get the address to a global variable representing the resource
/// buffer.
struct HALInterfaceBindingSubspanConverter final
    : public OpConversionPattern<IREE::HAL::InterfaceBindingSubspanOp> {
  HALInterfaceBindingSubspanConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      const InterfaceResourceMap &interfaceToResourceVars, const bool hasKernelCapabilty,
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit),
        interfaceToResourceVars(interfaceToResourceVars),
        hasKernelCapabilty(hasKernelCapabilty) {}

  LogicalResult matchAndRewrite(
      IREE::HAL::InterfaceBindingSubspanOp subspanOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (subspanOp.use_empty()) {
      rewriter.eraseOp(subspanOp);
      return success();
    }

    Type resultType = subspanOp.getOperation()->getResult(0).getType();
    Type convertedType = this->getTypeConverter()->convertType(resultType);
    if (!convertedType) {
      return subspanOp.emitError()
             << "failed to convert SPIR-V type: " << resultType;
    }

    if(hasKernelCapabilty) {
      // Bail until nested under an SPV::FuncOp.
      auto spirvFuncOp = subspanOp.getOperation()->getParentOfType<spirv::FuncOp>();
      auto argMapping = getKernelArgMapping(spirvFuncOp);
      size_t argIndex = argMapping.lookup(SetBinding(subspanOp.set(),
                                           subspanOp.binding()));
      if(argIndex >= argMapping.size()) return failure();
      if(argIndex >= spirvFuncOp.getNumArguments()) return failure();
      auto argValue = spirvFuncOp.getArgument(argIndex);
      rewriter.replaceOp(subspanOp, argValue);
      llvm::outs()<<"ocl print:"<<"\n"<<spirvFuncOp<<"\n";
      return success();
    }
    auto varOp = interfaceToResourceVars.lookup(subspanOp);
    // Fix up the variable's type.
    varOp.typeAttr(TypeAttr::get(convertedType));
    auto newspirvFuncOp = subspanOp.getOperation()->getParentOfType<spirv::FuncOp>();
    rewriter.replaceOpWithNewOp<spirv::AddressOfOp>(subspanOp, varOp);
    llvm::outs()<<"vulkanpritn:"<<"\n"<<newspirvFuncOp<<"\n";
    return success();
  }

 private:
  const InterfaceResourceMap &interfaceToResourceVars;
  const bool hasKernelCapabilty;
};

// TODO(raikonenfnu): Add ConvertFunc op to chang spv.Func to have the correct
// function signature with the subspans.
// Used for kernel style spirv. Moves subspan to function arguments. 
// struct FuncOpToSPVConverter final
//     : public OpConversionPattern<func::FuncOp> {
//   FuncOpToSPVConverter(
//       TypeConverter &typeConverter, MLIRContext *context,
//       PatternBenefit benefit = 1)
//       : OpConversionPattern(typeConverter, context, benefit) {}

//   LogicalResult matchAndRewrite(
//       func::FuncOp funcOp, OpAdaptor adaptor,
//       ConversionPatternRewriter &rewriter) const override {
//     llvm::outs()<<"Trying to get spirv funcop!\n";
//     FunctionType fnType = funcOp.getFunctionType();
//     (void)fnType;
//     if (!funcOp.isPublic()) return failure();

//     // illegal FuncOp must have 0 inputs.
//     assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

//     TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
//     auto argMapping = getKernelArgMapping(funcOp);
//     // There may be dead symbols, we pick i32 pointer as default argument type.
//     SmallVector<Type, 8> newInputTypes(argMapping.size(), rewriter.getI32Type());
//     funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
//       Type subspanType = subspanOp.getOperation()->getResult(0).getType();
//       newInputTypes[argMapping[SetBinding(subspanOp.set(),
//                                            subspanOp.binding())]] = subspanType;
//     });
//     // As a convention with HAL, push constants are appended as kernel arguments
//     // after all the binding inputs.
//     uint64_t numConstants = 0;
//     funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp constantOp) {
//       numConstants =
//           std::max(constantOp.index().getZExtValue() + 1, numConstants);
//     });
//     newInputTypes.resize(argMapping.size() + numConstants,
//                           rewriter.getI32Type());
//     if (!newInputTypes.empty()) signatureConverter.addInputs(newInputTypes);

//     auto newFuncType = FunctionType::get(rewriter.getContext(), newInputTypes,
//                             /*resultTypes=*/{});
//     auto newFuncOp = rewriter.create<func::FuncOp>(
//         funcOp.getLoc(), funcOp.getName(), newFuncType);

//     // Copy all of funcOp's operations into newFuncOp's body and perform region
//     // type conversion.
//     rewriter.inlineRegionBefore(funcOp.getBody(), newFuncOp.getBody(),
//                                 newFuncOp.end());
//     if (failed(rewriter.convertRegionTypes(&newFuncOp.getBody(), *typeConverter,
//                                            &signatureConverter))) {
//       return failure();
//     }
//     llvm::outs()<<"new FuncOp:"<<newFuncOp<<"\n";
//     rewriter.eraseOp(funcOp);
//     return success();
//   }
// };
struct FuncOpToSPVConverter final
    : public OpConversionPattern<func::FuncOp> {
  FuncOpToSPVConverter(
      TypeConverter &typeConverter, MLIRContext *context,
      PatternBenefit benefit = 1)
      : OpConversionPattern(typeConverter, context, benefit) {}

  LogicalResult matchAndRewrite(
      func::FuncOp funcOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    llvm::outs()<<"Trying to get spirv funcop!\n";
    FunctionType fnType = funcOp.getFunctionType();
    (void)fnType;
    if (!funcOp.isPublic()) return failure();

    // illegal FuncOp must have 0 inputs.
    assert(fnType.getNumInputs() == 0 && fnType.getNumResults() == 0);

    TypeConverter::SignatureConversion signatureConverter(/*numOrigInputs=*/0);
    auto argMapping = getKernelArgMapping(funcOp);
    // There may be dead symbols, we pick i32 pointer as default argument type.
    SmallVector<Type, 8> spirvInputTypes(
        argMapping.size(), spirv::PointerType::get(rewriter.getI32Type(), spirv::StorageClass::CrossWorkgroup));
    funcOp.walk([&](IREE::HAL::InterfaceBindingSubspanOp subspanOp) {
      Type subspanType = subspanOp.getOperation()->getResult(0).getType();
      Type convertedSpirvType = this->getTypeConverter()->convertType(subspanType);
      Type inputConvertedSpirvType = spirv::PointerType::get(convertedSpirvType.dyn_cast<spirv::PointerType>().getPointeeType(), spirv::StorageClass::CrossWorkgroup);
      spirvInputTypes[argMapping[SetBinding(subspanOp.set(),
                                           subspanOp.binding())]] = inputConvertedSpirvType;
    });
    // As a convention with HAL, push constants are appended as kernel arguments
    // after all the binding inputs.
    uint64_t numConstants = 0;
    funcOp.walk([&](IREE::HAL::InterfaceConstantLoadOp constantOp) {
      numConstants =
          std::max(constantOp.index().getZExtValue() + 1, numConstants);
    });
    spirvInputTypes.resize(argMapping.size() + numConstants,
                          rewriter.getI32Type());
    if (!spirvInputTypes.empty()) signatureConverter.addInputs(spirvInputTypes);

    auto spirvFuncType = FunctionType::get(rewriter.getContext(), spirvInputTypes,
                            /*resultTypes=*/{});
    auto spirvFuncOp = rewriter.create<spirv::FuncOp>(
        funcOp.getLoc(), funcOp.getName(), spirvFuncType, spirv::FunctionControl::None);

    // Copy all of funcOp's operations into spirvFuncOp's body and perform region
    // type conversion.
    rewriter.inlineRegionBefore(funcOp.getBody(), spirvFuncOp.getBody(),
                                spirvFuncOp.end());
    if (failed(rewriter.convertRegionTypes(&spirvFuncOp.getBody(), *typeConverter,
                                           &signatureConverter))) {
      return failure();
    }
    llvm::outs()<<"Original FuncOp"<<funcOp<<"\n";
    llvm::outs()<<"new FuncOp"<<spirvFuncOp<<"\n";
    llvm::outs()<<"OG num of arg"<<spirvFuncOp.getNumArguments()<<"\n";
    rewriter.eraseOp(funcOp);
    return success();
  }
};

/// Pattern to lower operations that become a no-ops at this level.
template <typename OpTy>
struct FoldAsNoOp final : public OpConversionPattern<OpTy> {
  using OpConversionPattern<OpTy>::OpConversionPattern;
  LogicalResult matchAndRewrite(
      OpTy op, typename OpTy::Adaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOp(op, adaptor.getOperands());
    return success();
  }
};

/// Removes unrealized_conversion_cast ops introduced during progressive
/// lowering when possible.
struct RemoveIdentityConversionCast final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
  using OpConversionPattern::OpConversionPattern;
  LogicalResult matchAndRewrite(
      UnrealizedConversionCastOp op, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 1 && op->getNumResults() == 1 &&
        adaptor.getOperands().front().getType() ==
            op->getResultTypes().front()) {
      rewriter.replaceOp(op, adaptor.getOperands());
      return success();
    }

    return failure();
  }
};

//===----------------------------------------------------------------------===//
// Conversion pass
//===----------------------------------------------------------------------===//

/// A pass to perform the SPIR-V conversion.
///
/// This pass converts remaining interface ops into SPIR-V global variables,
/// GPU processor ID ops into SPIR-V global variables, loop/standard ops into
/// corresponding SPIR-V ops.
struct ConvertToSPIRVPass : public ConvertToSPIRVBase<ConvertToSPIRVPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<spirv::SPIRVDialect>();
  }

  ConvertToSPIRVPass() {}
  ConvertToSPIRVPass(const ConvertToSPIRVPass &pass) {}

  void runOnOperation() override;
};
}  // namespace

void ConvertToSPIRVPass::runOnOperation() {
  MLIRContext *context = &getContext();
  ModuleOp moduleOp = getOperation();

  llvm::StringMap<IREE::HAL::ExecutableExportOp> exportOps =
      getAllEntryPoints(moduleOp);
  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    auto exportOp = exportOps.lookup(funcOp.getName());
    if (!exportOp) continue;
    // TODO(ravishankarm): This needs to be removed after ConvertToGPU is
    // deprecated. All passes must set the `workgroup_size` on the
    // `hal.executable.export` directly and not on the function.
    if (funcOp->hasAttr(spirv::getEntryPointABIAttrName())) continue;
    SmallVector<int64_t> workgroupSize = getWorkgroupSize(exportOp);
    if (workgroupSize.empty()) {
      exportOp.emitOpError(
          "expected workgroup_size attribute to be set for SPIR-V lowering");
      return signalPassFailure();
    }
    auto workgroupSize32 = llvm::to_vector<4>(llvm::map_range(
        workgroupSize, [](int64_t v) { return static_cast<int32_t>(v); }));
    funcOp->setAttr(spirv::getEntryPointABIAttrName(),
                    spirv::getEntryPointABIAttr(workgroupSize32, context));
  }

  spirv::TargetEnvAttr targetAttr = getSPIRVTargetEnvAttr(moduleOp);
  moduleOp->setAttr(spirv::getTargetEnvAttrName(), targetAttr);
  SPIRVTypeConverter typeConverter(targetAttr);
  RewritePatternSet patterns(&getContext());
  ScfToSPIRVContext scfToSPIRVContext;

  bool hasKernelCapabilty = false;
  for (auto capabiltiy : targetAttr.getCapabilities()) {
    if(capabiltiy == spirv::Capability::Kernel) {
      hasKernelCapabilty = true;
    }
  }

  // Pull in GPU patterns to convert processor ID ops and loop ops.
  populateGPUToSPIRVPatterns(typeConverter, patterns);

  // Pull in SCF patterns to convert control flow ops.
  populateSCFToSPIRVPatterns(typeConverter, scfToSPIRVContext, patterns);

  // Use the default 64-bit lowering for TOSA's ApplyScale operator:
  //   This lowering widens integer types to 64-bit an performs the non-fused
  //   operations, specifically multiply, add, and shift. Bit-widening
  //   is used to guarantee higher-order bits are not truncated during the
  //   multiply or add.
  //
  // TODO(antiagainst): Use a lowering that uses specific SPIRV intrinsics.
  tosa::populateTosaRescaleToArithConversionPatterns(&patterns);

  // Pull in MemRef patterns to convert load/store ops.
  populateMemRefToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard/math patterns to convert arithmetic ops and others.
  arith::populateArithmeticToSPIRVPatterns(typeConverter, patterns);
  populateFuncToSPIRVPatterns(typeConverter, patterns);
  populateMathToSPIRVPatterns(typeConverter, patterns);

  // Pull in standard patterns to convert tensor operations to SPIR-V. These are
  // primarily used to handle tensor-type constants and contain a
  // threshold. Only those constants that are below the threshold are converted
  // to SPIR-V. In IREE we want to control this threshold at Flow level. So set
  // this value arbitrarily high to make sure that everything within a dispatch
  // region is converted.
  mlir::populateTensorToSPIRVPatterns(
      typeConverter, std::numeric_limits<int64_t>::max() / 8, patterns);

  // Pull in vector patterns to convert vector ops.
  mlir::populateVectorToSPIRVPatterns(typeConverter, patterns);

  // Pull in builtin func to spv.func conversion.
  populateBuiltinFuncToSPIRVPatterns(typeConverter, patterns);

  // Add IREE HAL interface op conversions.
  patterns.insert<
      HALInterfaceLoadConstantConverter,
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupIDOp, spirv::BuiltIn::WorkgroupId>,
      HALInterfaceWorkgroupIdAndCountConverter<
          IREE::HAL::InterfaceWorkgroupCountOp, spirv::BuiltIn::NumWorkgroups>>(
      typeConverter, context);

  if(hasKernelCapabilty) {
    patterns.insert<FuncOpToSPVConverter>(typeConverter, context);
  }
  // Performs a prelimiary step to analyze all hal.interface.binding.subspan ops
  // and create spv.GlobalVariables.
  auto interfaceToResourceVars = createResourceVariables(moduleOp);
  // For using use them in conversion.
  patterns.insert<HALInterfaceBindingSubspanConverter>(typeConverter, context,
                                                       interfaceToResourceVars, hasKernelCapabilty);

  /// Fold certain operations as no-ops:
  /// - linalg.reshape becomes a no-op since all memrefs are linearized in
  ///   SPIR-V.
  /// - tensor_to_memref can become a no-op since tensors are lowered to
  ///   !spv.array.
  /// - unrealized_conversion_cast with the same source and target type.
  patterns.insert<
      FoldAsNoOp<memref::CollapseShapeOp>, FoldAsNoOp<memref::ExpandShapeOp>,
      FoldAsNoOp<bufferization::ToMemrefOp>, RemoveIdentityConversionCast>(
      typeConverter, context);

  std::unique_ptr<ConversionTarget> target =
      SPIRVConversionTarget::get(targetAttr);
  // Disallow all other ops.
  target->markUnknownOpDynamicallyLegal([](Operation *) { return false; });

  SmallVector<func::FuncOp, 1> functions;
  for (func::FuncOp fn : moduleOp.getOps<func::FuncOp>()) {
    if (!fn.isPublic()) continue;
    functions.push_back(fn);
  }

  FrozenRewritePatternSet frozenPatterns(std::move(patterns));
  for (func::FuncOp fn : functions) {
    if (failed(applyFullConversion(fn, *target, frozenPatterns))) {
      return signalPassFailure();
    }
  }

  // Collect all SPIR-V ops into a spv.module.
  spirv::AddressingModel addressingModel = spirv::AddressingModel::Logical;
  spirv::MemoryModel memoryModel = spirv::MemoryModel::GLSL450;
  if(hasKernelCapabilty) {
    addressingModel = spirv::AddressingModel::Physical32;
    memoryModel = spirv::MemoryModel::OpenCL;
  }
  auto builder = OpBuilder::atBlockBegin(moduleOp.getBody());
  auto spvModule = builder.create<spirv::ModuleOp>(
      moduleOp.getLoc(), addressingModel, memoryModel);
  Block *body = spvModule.getBody();
  Dialect *spvDialect = spvModule->getDialect();
  for (Operation &op : llvm::make_early_inc_range(*moduleOp.getBody())) {
    // Skip the newly created spv.module itself.
    if (&op == spvModule) continue;
    if (op.getDialect() == spvDialect) op.moveBefore(body, body->end());
  }
}

//===----------------------------------------------------------------------===//
// Pass entry point and registration
//===----------------------------------------------------------------------===//

std::unique_ptr<OperationPass<ModuleOp>> createConvertToSPIRVPass() {
  return std::make_unique<ConvertToSPIRVPass>();
}

}  // namespace iree_compiler
}  // namespace mlir
