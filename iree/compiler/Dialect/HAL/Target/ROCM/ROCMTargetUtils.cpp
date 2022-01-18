// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===========Link LLVM Module to ROCDL Start===================/
// Inspiration of code from this section comes from TF Kernel Gen Project
// https://github.com/tensorflow/tensorflow/blob/master/tensorflow/compiler/xla/service/gpu/llvm_gpu_backend/gpu_backend_lib.cc

bool CouldNeedDeviceBitcode(const llvm::Module &module) {
  for (const llvm::Function &function : module.functions()) {
    // The list of prefixes should be in sync with library functions used in
    // target_util.cc.
    if (!function.isIntrinsic() && function.isDeclaration() &&
        (function.getName().startswith("__ocml_") ||
         function.getName().startswith("__ockl_"))) {
      return true;
    }
  }
  return false;
}

static void DieWithSMDiagnosticError(llvm::SMDiagnostic *diagnostic) {
  llvm::WithColor::error(llvm::errs())
      << diagnostic->getFilename().str() << ":" << diagnostic->getLineNo()
      << ":" << diagnostic->getColumnNo() << ": "
      << diagnostic->getMessage().str();
}

std::unique_ptr<llvm::Module> LoadIRModule(const std::string &filename,
                                           llvm::LLVMContext *llvm_context) {
  llvm::SMDiagnostic diagnostic_err;
  std::unique_ptr<llvm::Module> module(
      llvm::parseIRFile(llvm::StringRef(filename.data(), filename.size()),
                        diagnostic_err, *llvm_context));

  if (module == nullptr) {
    DieWithSMDiagnosticError(&diagnostic_err);
  }

  return module;
}

LogicalResult LinkWithBitcodeVector(
    llvm::Module *module, const std::vector<std::string> &bitcode_path_vector) {
  llvm::Linker linker(*module);

  for (auto &bitcode_path : bitcode_path_vector) {
    if (!(llvm::sys::fs::exists(bitcode_path))) {
      llvm::WithColor::error(llvm::errs())
          << "bitcode module is required by this HLO module but was "
             "not found at "
          << bitcode_path;
      return failure();
    }
    std::unique_ptr<llvm::Module> bitcode_module =
        LoadIRModule(bitcode_path, &module->getContext());
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcode_module->setDataLayout(module->getDataLayout());
    if (linker.linkInModule(
            std::move(bitcode_module), llvm::Linker::Flags::LinkOnlyNeeded,
            [](llvm::Module &M, const llvm::StringSet<> &GVS) {
              llvm::internalizeModule(M, [&GVS](const llvm::GlobalValue &GV) {
                return !GV.hasName() || (GVS.count(GV.getName()) == 0);
              });
            })) {
      llvm::WithColor::error(llvm::errs()) << "Link Bitcode error.\n";
      return failure();
    }
  }
  return success();
}

static std::vector<std::string> GetROCDLPaths(std::string targetChip,
                                              std::string bitCodeDir) {
  // AMDGPU bitcodes.
  int lenOfChipPrefix = 3;
  std::string chipId = targetChip.substr(lenOfChipPrefix);
  std::string chip_isa_bc = "oclc_isa_version_" + chipId + ".bc";
  static const std::vector<std::string> rocdl_filenames(
      {"hc.bc", "opencl.bc", "ocml.bc", "ockl.bc", "oclc_finite_only_off.bc",
       "oclc_daz_opt_off.bc", "oclc_correctly_rounded_sqrt_on.bc",
       "oclc_unsafe_math_off.bc", "oclc_wavefrontsize64_on.bc", chip_isa_bc});

  // Construct full path to ROCDL bitcode libraries.
  std::vector<std::string> result;
  std::string app = "/";
  for (auto &filename : rocdl_filenames) {
    result.push_back(bitCodeDir + app + filename);
  }
  return result;
}

// Links ROCm-Device-Libs into the given module if the module needs it.
void LinkROCDLIfNecessary(llvm::Module *module, std::string targetChip,
                          std::string bitCodeDir) {
  if (!HAL::CouldNeedDeviceBitcode(*module)) {
    return;
  }
  if (!succeeded(HAL::LinkWithBitcodeVector(
          module, GetROCDLPaths(targetChip, bitCodeDir)))) {
    llvm::WithColor::error(llvm::errs()) << "Fail to Link ROCDL.\n";
  };
}

//===========Link LLVM Module to ROCDL End===================/

//=====================Create HSACO Begin=============//
// Link object file using ld.lld lnker to generate code object
// Inspiration from this section comes from LLVM-PROJECT-MLIR by
// ROCmSoftwarePlatform
// https://github.com/ROCmSoftwarePlatform/llvm-project-mlir/blob/miopen-dialect/mlir/lib/ExecutionEngine/ROCm/BackendUtils.cpp
std::string createHsaco(const std::string isa, StringRef name) {
  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  std::error_code ec = llvm::sys::fs::createTemporaryFile(
      "kernel", "o", tempIsaBinaryFd, tempIsaBinaryFilename);
  if (ec) {
    llvm::WithColor::error(llvm::errs(), name)
        << "temporary file for ISA binary creation error.\n";
    return {};
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << isa;
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  ec = llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                          tempHsacoFilename);
  if (ec) {
    llvm::WithColor::error(llvm::errs(), name)
        << "temporary file for HSA code object creation error.\n";
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  // Invoke lld. Expect a true return value from lld.
  // Searching for LLD
  std::string lld_program;
  std::string toolName = "ld.lld";
  if (llvm::sys::fs::exists(toolName)) {
    llvm::SmallString<256> absolutePath(toolName);
    llvm::sys::fs::make_absolute(absolutePath);
    lld_program = std::string(absolutePath);
  } else {
    // Next search the environment path.
    if (auto result = llvm::sys::Process::FindInEnvPath("PATH", toolName)) {
      lld_program = std::string(*result);
    } else {
    }
  }
  if (lld_program.empty()) {
    llvm::WithColor::error(llvm::errs(), name)
        << "unable to find ld.lld in PATH\n";
    return {};
  }
  // Setting Up LLD Args
  std::vector<llvm::StringRef> lld_args{
      llvm::StringRef("ld.lld"),   llvm::StringRef("-flavor"),
      llvm::StringRef("gnu"),      llvm::StringRef("-shared"),
      tempIsaBinaryFilename.str(), llvm::StringRef("-o"),
      tempHsacoFilename.str(),
  };

  // Executing LLD
  std::string error_message;
  int lld_result = llvm::sys::ExecuteAndWait(
      lld_program, llvm::ArrayRef<llvm::StringRef>(lld_args), llvm::None, {}, 5,
      0, &error_message);
  if (lld_result) {
    llvm::WithColor::error(llvm::errs(), name)
        << "ld.lld execute fail:" << error_message
        << "Error Code:" << lld_result << "\n";
    return {};
  }

  // Load the HSA code object.
  auto hsacoFile = mlir::openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    llvm::WithColor::error(llvm::errs(), name)
        << "read HSA code object from temp file error.\n";
    return {};
  }
  std::string strHSACO(hsacoFile->getBuffer().begin(),
                       hsacoFile->getBuffer().end());
  return strHSACO;
}
//==============Create HSACO End=============//

}  // namespace HAL
}  // namespace IREE
}  // namespace iree_compiler
}  // namespace mlir
