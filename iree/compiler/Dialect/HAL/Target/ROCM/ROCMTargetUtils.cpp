// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "iree/compiler/Dialect/HAL/Target/ROCM/ROCMTarget.h"
#include "iree/compiler/Dialect/HAL/Target/ROCM/rocm_ockl.h"
#include "iree/compiler/Dialect/HAL/Target/ROCM/rocm_ocml.h"
#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/Linker/Linker.h"
#include "llvm/Support/Program.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "mlir/Support/LogicalResult.h"

// MLIR Support Files Header
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/SourceMgr.h"
#include "mlir/Support/FileUtilities.h"

namespace mlir {
namespace iree_compiler {
namespace IREE {
namespace HAL {

//===========Link LLVM Module to ROCDL Start===================/
// Inspiration of code from this section comes from IREE CUDA Backend

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

LogicalResult linkModule(llvm::Module &module) {
  llvm::Linker linker(module);
  std::vector<llvm::MemoryBufferRef> bitcode_ref_vector;
  bitcode_ref_vector.emplace_back(
      llvm::StringRef(rocm_ocml_create()->data, rocm_ocml_create()->size),
      "ocml bitcode");
  bitcode_ref_vector.emplace_back(
      llvm::StringRef(rocm_ockl_create()->data, rocm_ockl_create()->size),
      "ockl bitcode");
  for (auto &bitcode_ref : bitcode_ref_vector) {
    std::unique_ptr<llvm::Module> bitcode_module = std::move(
        llvm::parseBitcodeFile(bitcode_ref, module.getContext()).get());
    // Ignore the data layout of the module we're importing. This avoids a
    // warning from the linker.
    bitcode_module->setDataLayout(module.getDataLayout());
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

// Links ROCm-Device-Libs into the given module if the module needs it.
void LinkROCDLIfNecessary(llvm::Module *module) {
  if (!HAL::CouldNeedDeviceBitcode(*module)) {
    return;
  }

  if (!succeeded(HAL::linkModule(*module))) {
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
