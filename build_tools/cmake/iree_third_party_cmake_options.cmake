# Copyright 2021 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

macro(iree_set_benchmark_cmake_options)
  set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
  set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_cpuinfo_cmake_options)
  set(CPUINFO_BUILD_TOOLS ON CACHE BOOL "" FORCE)

  set(CPUINFO_BUILD_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(CPUINFO_BUILD_UNIT_TESTS OFF CACHE BOOL "" FORCE)
  set(CPUINFO_BUILD_MOCK_TESTS OFF CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_googletest_cmake_options)
  set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_llvm_cmake_options)
  set(LLVM_INCLUDE_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_TESTS OFF CACHE BOOL "" FORCE)
  set(LLVM_INCLUDE_BENCHMARKS OFF CACHE BOOL "" FORCE)
  set(LLVM_APPEND_VC_REV OFF CACHE BOOL "" FORCE)
  set(LLVM_ENABLE_IDE ON CACHE BOOL "" FORCE)

  # TODO(ataei): Use optional build time targets selection for LLVMAOT.
  set(LLVM_TARGETS_TO_BUILD "WebAssembly;X86;ARM;AArch64;RISCV;NVPTX;AMDGPU"
      CACHE STRING "" FORCE)

  set(LLVM_ENABLE_PROJECTS "mlir;lld" CACHE STRING "" FORCE)
  set(LLVM_ENABLE_BINDINGS OFF CACHE BOOL "" FORCE)
endmacro()

macro(iree_add_llvm_external_project name identifier location)
  message(STATUS "Adding LLVM external project ${name} (${identifier}) -> ${location}")
  if(NOT EXISTS "${location}/CMakeLists.txt")
    message(FATAL_ERROR "External project location ${location} is not valid")
  endif()
  list(APPEND LLVM_EXTERNAL_PROJECTS ${name})
  list(REMOVE_DUPLICATES LLVM_EXTERNAL_PROJECTS)
  set(LLVM_EXTERNAL_${identifier}_SOURCE_DIR ${location} CACHE STRING "" FORCE)
  set(LLVM_EXTERNAL_PROJECTS ${LLVM_EXTERNAL_PROJECTS} CACHE STRING "" FORCE)
endmacro()

macro(iree_set_spirv_headers_cmake_options)
  set(SPIRV_HEADERS_SKIP_EXAMPLES ON CACHE BOOL "" FORCE)
  set(SPIRV_HEADERS_SKIP_INSTALL ON CACHE BOOL "" FORCE)
endmacro()

macro(iree_set_spirv_cross_cmake_options)
  set(SPIRV_CROSS_ENABLE_MSL ON CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_GLSL ON CACHE BOOL "" FORCE) # Required to enable MSL

  set(SPIRV_CROSS_EXCEPTIONS_TO_ASSERTIONS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_CLI OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_SKIP_INSTALL ON CACHE BOOL "" FORCE)

  set(SPIRV_CROSS_ENABLE_HLSL OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_CPP OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_REFLECT OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_C_API OFF CACHE BOOL "" FORCE)
  set(SPIRV_CROSS_ENABLE_UTIL OFF CACHE BOOL "" FORCE)
endmacro()
