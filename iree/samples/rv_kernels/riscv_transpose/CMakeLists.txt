# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(${IREE_HAL_DRIVER_DYLIB} AND (DEFINED IREE_HOST_BINARY_ROOT))

# Set iree-translate binary
set(_TRANSLATE_TOOL_EXECUTABLE ${IREE_HOST_BINARY_ROOT}/bin/iree-translate)

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
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=riscv_transpose.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/riscv_transpose.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "riscv_transpose.vmfb")

# Custom command for iree-translate to generate static library and bytecode
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_transpose.h
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_transpose.o
    ${CMAKE_CURRENT_BINARY_DIR}/riscv_transpose.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "riscv_transpose.mlir"
)

# # Tell cmake about the riscv_transpose library so it will link it
add_library(riscv_transpose
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/riscv_transpose.o)

SET_TARGET_PROPERTIES(
  riscv_transpose
  PROPERTIES
  LINKER_LANGUAGE C)

# Note: If you're cross compiling the riscv_transpose for a different backend, you'll
# need to run iree-translate manually to produce the desired '.vmfb' and '.h/.o'
# files. Substitute the 'riscv_transpose' dependency in iree_cc_binary() below with
# your own static library and the `riscv_transpose.vmfb` in the iree_c_embed_data()
# rule. You can use paths to files, i.e. 'path/to/generated/output.vmfb' in
# place of CMake targets.

# Generate the embed data with the bytecode module
iree_c_embed_data(
  NAME
    riscv_transpose_c
  IDENTIFIER
    iree_samples_static_library_riscv_transpose
  GENERATED_SRCS
    riscv_transpose.vmfb
  C_FILE_OUTPUT
    riscv_transpose_c.c
  H_FILE_OUTPUT
    riscv_transpose_c.h
  FLATTEN
  PUBLIC
)

iree_cc_binary(
NAME
  riscv_transpose_exec
SRCS
  "riscv_transpose_demo.c"
DEPS
  ::riscv_transpose_c
  iree::hal::local::loaders::static_library_loader
  iree::hal::local::sync_driver
  iree::base
  iree::hal
  iree::hal::local
  iree::modules::hal
  iree::vm
  iree::vm::bytecode_module
  riscv_transpose
)

endif()