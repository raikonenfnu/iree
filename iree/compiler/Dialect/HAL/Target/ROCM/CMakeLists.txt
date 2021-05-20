if(NOT "${IREE_TARGET_BACKEND_ROCM}")
  return()
endif()

iree_add_all_subdirs()

iree_c_embed_data(
  NAME
    rocm_ockl
  SRCS
    "../../../../../../third_party/rocm/amdgcn/ockl.bc"
  C_FILE_OUTPUT
    "rocm_ockl.c"
  H_FILE_OUTPUT
    "rocm_ockl.h"
  FLATTEN
  PUBLIC
)

iree_c_embed_data(
  NAME
    rocm_ocml
  SRCS
    "../../../../../../third_party/rocm/amdgcn/ocml.bc"
  C_FILE_OUTPUT
    "rocm_ocml.c"
  H_FILE_OUTPUT
    "rocm_ocml.h"
  FLATTEN
  PUBLIC
)

iree_cc_library(
  NAME
    ROCM
  HDRS
    "ROCMTarget.h"
  SRCS
    "ROCMTarget.cpp"
    "ROCMTargetUtils.cpp"
  DEPS
    ::rocm_ockl
    ::rocm_ocml
    LLVMAMDGPUCodeGen
    LLVMCore
    LLVMSupport
    LLVMTarget
    MLIRLLVMIR
    MLIRLLVMToLLVMIRTranslation
    MLIRPass
    MLIRROCDLIR
    MLIRROCDLToLLVMIRTranslation
    MLIRSupport
    MLIRTargetLLVMIRExport
    iree::base::internal::flatcc
    iree::compiler::Conversion::LinalgToLLVMGPU
    iree::compiler::Dialect::HAL::Target
    iree::compiler::Utils
    iree::schemas::rocm_executable_def_c_fbs
  PUBLIC
)

### BAZEL_TO_CMAKE_PRESERVES_ALL_CONTENT_BELOW_THIS_LINE ###
