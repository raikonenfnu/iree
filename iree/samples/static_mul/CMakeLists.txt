# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}
  OR NOT ${IREE_HAL_DRIVER_DYLIB}
  OR NOT ${IREE_BUILD_COMPILER})
  return()
endif()

# Set iree-translate binary.
set(_TRANSLATE_TOOL_EXECUTABLE $<TARGET_FILE:iree_tools_iree-translate>)

# Setup args for iree-translate.
set(_TRANSLATE_ARGS)
list(APPEND _TRANSLATE_ARGS "-iree-mlir-to-vm-bytecode-module")
list(APPEND _TRANSLATE_ARGS "-iree-hal-target-backends=dylib-llvm-aot")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-link-static")
list(APPEND _TRANSLATE_ARGS "-iree-llvm-static-library-output-path=linalg_mul.o")
list(APPEND _TRANSLATE_ARGS "${CMAKE_CURRENT_SOURCE_DIR}/linalg_mul.mlir")
list(APPEND _TRANSLATE_ARGS "-o")
list(APPEND _TRANSLATE_ARGS "linalg_mul.vmfb")

# Custom command for iree-translate to generate static library and bytecode.
add_custom_command(
  OUTPUT
    ${CMAKE_CURRENT_BINARY_DIR}/linalg_mul.h
    ${CMAKE_CURRENT_BINARY_DIR}/linalg_mul.o
    ${CMAKE_CURRENT_BINARY_DIR}/linalg_mul.vmfb
  COMMAND ${_TRANSLATE_TOOL_EXECUTABLE} ${_TRANSLATE_ARGS}
  DEPENDS ${_TRANSLATE_TOOL_EXECUTABLE} "linalg_mul.mlir"
)

# Tell cmake about the linalg_mul library so it will link it.
add_library(linalg_mul
  STATIC
  ${CMAKE_CURRENT_BINARY_DIR}/linalg_mul.o)

SET_TARGET_PROPERTIES(
  linalg_mul
  PROPERTIES
  LINKER_LANGUAGE C)

# Note: If you're cross compiling the linalg_mul for a different backend, you'll
# need to run iree-translate manually to produce the desired '.vmfb' and '.h/.o'
# files. Substitute the 'linalg_mul' dependency in iree_cc_binary() below with
# your own static library and the `linalg_mul.vmfb` in the iree_c_embed_data()
# rule. You can use paths to files, i.e. 'path/to/generated/output.vmfb' in
# place of CMake targets.

# Generate the embed data with the bytecode module
iree_c_embed_data(
  NAME
    linalg_mul_c
  IDENTIFIER
    iree_samples_static_library_linalg_mul
  GENERATED_SRCS
    linalg_mul.vmfb
  C_FILE_OUTPUT
    linalg_mul_c.c
  H_FILE_OUTPUT
    linalg_mul_c.h
  FLATTEN
  PUBLIC
)

iree_cc_binary(
NAME
  static_mul
SRCS
  "static_library_demo.c"
DEPS
  ::linalg_mul_c
  iree::runtime
  iree::hal::local::loaders::static_library_loader
  iree::hal::local::task_driver
  iree::task::api
  linalg_mul
)
