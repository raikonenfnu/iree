# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER})
  return()
endif()

# Set iree-translate binary
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

# Setup args for iree-transalte
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-target-triple=riscv64-pc-linux-elf")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-target-cpu=generic-rv64")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-target-abi=lp64d")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-target-cpu-features=+m,+a,+f,+d,+experimental-v")
list(APPEND _TRANSLATE_ARGS "-riscv-v-vector-bits-min=128")
list(APPEND _TRANSLATE_ARGS "-riscv-v-fixed-length-vector-lmul-max=8")
list(APPEND _TRANSLATE_ARGS "--iree-llvm-loop-vectorization")
list(APPEND _TRANSLATE_ARGS "--iree-llvm-slp-vectorization")
list(APPEND _TRANSLATE_ARGS "-iree-vm-bytecode-source-listing=foo.mlir")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=riscv_mul.o")
list(APPEND _TRANSLATE_ARGS "--print-ir-after-all")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/riscv_mul.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "riscv_mul.vmfb")
list(APPEND _TRANSLATE_ARGS "2>")
list(APPEND _TRANSLATE_ARGS "debug.mlir")


set(CUDA_TRANSLATE_ARGS)
list(APPEND CUDA_TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND CUDA_TRANSLATE_ARGS "-iree-hal-target-backends=cuda")
list(APPEND CUDA_TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND CUDA_TRANSLATE_ARGS "-iree-llvm-static-library-output-path=cuda_mul.o")
list(APPEND CUDA_TRANSLATE_ARGS "--print-ir-after-all")
list(APPEND CUDA_TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/riscv_mul.mlir")
list(APPEND CUDA_TRANSLATE_ARGS "-o")
list(APPEND CUDA_TRANSLATE_ARGS "cuda_mul.vmfb")
list(APPEND CUDA_TRANSLATE_ARGS "2>")
list(APPEND CUDA_TRANSLATE_ARGS "cuda_debug.mlir")


# Custom command for iree-translate to generate static library and bytecode
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_mul.h
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_mul.o
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_mul.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "riscv_mul.mlir"
)

iree_c_embed_data(
  NAME
    riscv_mul_c
  IDENTIFIER
    iree_samples_static_library_riscv_mul
  GENERATED_SRCS
    riscv_mul.vmfb
  C_FILE_OUTPUT
    riscv_mul_c.c
  H_FILE_OUTPUT
    riscv_mul_c.h
  FLATTEN
  PUBLIC
)


add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_mul.h
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_mul.o
    ${CMAKE_CURRENT_BINARY_DIR}/cuda_mul.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${CUDA_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "riscv_mul.mlir"
)
iree_c_embed_data(
  NAME
    cuda_mul_c
  IDENTIFIER
    iree_samples_static_library_cuda_mul
  GENERATED_SRCS
    cuda_mul.vmfb
  C_FILE_OUTPUT
    cuda_mul_c.c
  H_FILE_OUTPUT
    cuda_mul_c.h
  FLATTEN
  PUBLIC
)