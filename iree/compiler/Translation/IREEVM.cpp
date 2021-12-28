// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Translation/IREEVM.h"

#include "iree/compiler/Bindings/Native/Transforms/Passes.h"
#include "iree/compiler/Bindings/TFLite/Transforms/Passes.h"
#include "iree/compiler/ConstEval/Passes.h"
#include "iree/compiler/Dialect/Flow/Transforms/Passes.h"
#include "iree/compiler/Dialect/HAL/Transforms/Passes.h"
#include "iree/compiler/Dialect/Stream/Transforms/Passes.h"
#include "iree/compiler/Dialect/Util/Transforms/Passes.h"
#include "iree/compiler/Dialect/VM/Target/Bytecode/TranslationFlags.h"
#include "iree/compiler/Dialect/VM/Transforms/Passes.h"
#include "iree/compiler/InputConversion/Common/Passes.h"
#include "iree/compiler/InputConversion/MHLO/Passes.h"
#include "iree/compiler/InputConversion/TOSA/Passes.h"
#include "iree/compiler/Utils/TracingUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Translation.h"

#ifdef IREE_HAVE_EMITC_DIALECT
#include "iree/compiler/Dialect/VM/Target/C/CModuleTarget.h"
#include "iree/compiler/Dialect/VM/Target/C/TranslationFlags.h"
#endif  // IREE_HAVE_EMITC_DIALECT

namespace mlir {
namespace iree_compiler {

static BindingOptions getBindingOptionsFromFlags() {
  static llvm::cl::OptionCategory bindingOptionsCategory(
      "IREE translation binding support options");

  static llvm::cl::opt<bool> *bindingsNativeFlag = new llvm::cl::opt<bool>{
      "iree-native-bindings-support",
      llvm::cl::desc(
          "Include runtime support for native IREE ABI-compatible bindings"),
      llvm::cl::init(true), llvm::cl::cat(bindingOptionsCategory)};

  static llvm::cl::opt<bool> *bindingsTFLiteFlag = new llvm::cl::opt<bool>{
      "iree-tflite-bindings-support",
      llvm::cl::desc(
          "Include runtime support for the IREE TFLite compatibility bindings"),
      llvm::cl::init(false), llvm::cl::cat(bindingOptionsCategory)};

  BindingOptions bindingOptions;
  bindingOptions.native = *bindingsNativeFlag;
  bindingOptions.tflite = *bindingsTFLiteFlag;
  return bindingOptions;
}

static InputDialectOptions getInputDialectOptionsFromFlags() {
  static llvm::cl::OptionCategory inputDialectOptions(
      "IREE options for controlling the input transformations to apply");

  static llvm::cl::opt<InputDialectOptions::Type> *typeFlag =
      new llvm::cl::opt<InputDialectOptions::Type>{
          "iree-input-type", llvm::cl::desc("IREE input type"),
          llvm::cl::values(
              clEnumValN(InputDialectOptions::Type::none, "none",
                         "No input dialect transformation"),
              clEnumValN(InputDialectOptions::Type::tosa, "tosa",
                         "Legalize from TOSA ops"),
              clEnumValN(InputDialectOptions::Type::mhlo, "mhlo",
                         "Legalize from MHLO ops"),
              clEnumValN(
                  InputDialectOptions::Type::xla, "xla",
                  "Legalize from MHLO ops (with XLA cleanup preprocessing)")),
          llvm::cl::init(InputDialectOptions::Type::none),
          llvm::cl::cat(inputDialectOptions)};

  InputDialectOptions options;
  options.type = *typeFlag;
  return options;
}

static HighLevelOptimizationOptions getHighLevelOptimizationOptionsFromFlags() {
  static llvm::cl::OptionCategory category(
      "IREE options for controlling high level optimizations");

  static llvm::cl::opt<bool> *constEval = new llvm::cl::opt<bool>{
      "iree-const-eval",
      llvm::cl::desc("Enables eager evaluation of constants using the full "
                     "compiler and runtime"),
      llvm::cl::init(false), llvm::cl::cat(category)};
  static llvm::cl::opt<bool> *constExprHoisting = new llvm::cl::opt<bool>{
      "iree-const-expr-hoisting",
      llvm::cl::desc(
          "Hoists the results of latent constant expressions into immutable "
          "global initializers for evaluation at program load"),
      llvm::cl::init(false), llvm::cl::cat(category)};

  HighLevelOptimizationOptions options;
  options.constEval = *constEval;
  options.constExprHoisting = *constExprHoisting;
  return options;
}

void buildIREEVMTransformPassPipeline(
    BindingOptions bindingOptions, InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions, OpPassManager &passManager) {
  // Input pipelines can result in changes to the exported functions and types
  // and must run before generating bindings.
  // After input processing, there should only be IREE legal types in
  // signatures.
  switch (inputOptions.type) {
    case InputDialectOptions::Type::none:
      break;
    case InputDialectOptions::Type::tosa:
      buildTOSAInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::mhlo:
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
    case InputDialectOptions::Type::xla:
      MHLO::buildXLACleanupPassPipeline(passManager);
      MHLO::buildMHLOInputConversionPassPipeline(passManager);
      break;
  }
  buildCommonInputConversionPassPipeline(passManager);

  // Now that inputs are legalized, generate wrapper for entry functions.
  if (bindingOptions.native) {
    IREE::ABI::buildTransformPassPipeline(passManager);
  }
  if (bindingOptions.tflite) {
    IREE::TFLite::buildTransformPassPipeline(passManager);
  }

  IREE::Flow::TransformOptions flowOptions;
  flowOptions.constExprHoisting =
      highLevelOptimizationOptions.constExprHoisting;
  if (highLevelOptimizationOptions.constEval) {
    flowOptions.buildConstEvalPassPipeline = [](OpPassManager &passManager) {
      passManager.addPass(ConstEval::createJitGlobalsPass());
    };
  }

  IREE::Flow::buildFlowTransformPassPipeline(passManager, flowOptions);
  IREE::Stream::TransformOptions streamOptions;
  IREE::Stream::buildStreamTransformPassPipeline(passManager, streamOptions);
  IREE::HAL::buildHALTransformPassPipeline(passManager, executableOptions);
  IREE::VM::buildVMTransformPassPipeline(passManager, targetOptions);
  passManager.addPass(IREE::Util::createDropCompilerHintsPass());
}

void buildDefaultIREEVMTransformPassPipeline(OpPassManager &passManager) {
  buildIREEVMTransformPassPipeline(
      getBindingOptionsFromFlags(), getInputDialectOptionsFromFlags(),
      getHighLevelOptimizationOptionsFromFlags(),
      IREE::HAL::getTargetOptionsFromFlags(),
      IREE::VM::getTargetOptionsFromFlags(), passManager);
}

void registerIREEVMTransformPassPipeline() {
  PassPipelineRegistration<> transformPassPipeline(
      "iree-transformation-pipeline",
      "Runs the full IREE input to VM transformation pipeline",
      [](OpPassManager &passManager) {
        buildDefaultIREEVMTransformPassPipeline(passManager);
      });
}

// Converts from our source to a vm.module in canonical form.
// After this completes we have a non-bytecode-specific vm.module that we
// could lower to other forms (LLVM IR, C, etc).
static LogicalResult translateFromMLIRToVM(
    ModuleOp moduleOp, BindingOptions bindingOptions,
    InputDialectOptions inputOptions,
    HighLevelOptimizationOptions highLevelOptimizationOptions,
    IREE::HAL::TargetOptions executableOptions,
    IREE::VM::TargetOptions targetOptions) {
  PassManager passManager(moduleOp.getContext());
  mlir::applyPassManagerCLOptions(passManager);
  mlir::applyDefaultTimingPassManagerCLOptions(passManager);
  passManager.addInstrumentation(std::make_unique<PassTracing>());
  buildIREEVMTransformPassPipeline(
      bindingOptions, inputOptions, highLevelOptimizationOptions,
      executableOptions, targetOptions, passManager);
  if (failed(passManager.run(moduleOp))) {
    return moduleOp.emitError() << "conversion from source -> vm failed";
  }
  return success();
}

// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM bytecode module for loading at runtime.
//
// See iree/schemas/bytecode_module_def.fbs for the description of the
// serialized module format.
//
// Exposed via the --iree-mlir-to-vm-bytecode-module translation.
static LogicalResult translateFromMLIRToVMBytecodeModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = getBindingOptionsFromFlags();
  auto inputOptions = getInputDialectOptionsFromFlags();
  auto highLevelOptimizationOptions =
      getHighLevelOptimizationOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto bytecodeTargetOptions = IREE::VM::getBytecodeTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions, inputOptions,
                                      highLevelOptimizationOptions,
                                      halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  return translateModuleToBytecode(moduleOp, bytecodeTargetOptions, output);
}

#ifdef IREE_HAVE_EMITC_DIALECT
// Translates an MLIR module containing a set of supported IREE input dialects
// to an IREE VM C module.
//
// Exposed via the --iree-mlir-to-vm-c-module translation.
static LogicalResult translateFromMLIRToVMCModuleWithFlags(
    ModuleOp moduleOp, llvm::raw_ostream &output) {
  mlir::registerPassManagerCLOptions();
  auto bindingOptions = getBindingOptionsFromFlags();
  auto inputOptions = getInputDialectOptionsFromFlags();
  auto highLevelOptimizationOptions =
      getHighLevelOptimizationOptionsFromFlags();
  auto halTargetOptions = IREE::HAL::getTargetOptionsFromFlags();
  auto vmTargetOptions = IREE::VM::getTargetOptionsFromFlags();
  auto cTargetOptions = IREE::VM::getCTargetOptionsFromFlags();
  auto result = translateFromMLIRToVM(moduleOp, bindingOptions, inputOptions,
                                      highLevelOptimizationOptions,
                                      halTargetOptions, vmTargetOptions);
  if (failed(result)) {
    return result;
  }
  // Serialize to c code.
  return mlir::iree_compiler::IREE::VM::translateModuleToC(
      moduleOp, cTargetOptions, output);
}
#endif  // IREE_HAVE_EMITC_DIALECT

void registerIREEVMTranslationFlags() {
  getBindingOptionsFromFlags();
  getInputDialectOptionsFromFlags();
  getHighLevelOptimizationOptionsFromFlags();
}

void registerIREEVMTranslation() {
  registerIREEVMTranslationFlags();
  TranslateFromMLIRRegistration toVMBytecodeModuleWithFlags(
      "iree-mlir-to-vm-bytecode-module",
      translateFromMLIRToVMBytecodeModuleWithFlags);

#ifdef IREE_HAVE_EMITC_DIALECT
  TranslateFromMLIRRegistration toVMCModuleWithFlags(
      "iree-mlir-to-vm-c-module", translateFromMLIRToVMCModuleWithFlags);
#endif  // IREE_HAVE_EMITC_DIALECT
}

}  // namespace iree_compiler
}  // namespace mlir
