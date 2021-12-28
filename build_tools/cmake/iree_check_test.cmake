# Copyright 2020 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

include(CMakeParseArguments)

# Helper for iree_check_test and iree_trace_runner_test.
# Just a thin wrapper around iree_bytecode_module, passing it some
# common flags, including the appropriate --iree-llvm-target-triple in the
# Android case.
function(iree_bytecode_module_for_iree_check_test_and_friends)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "MODULE_NAME;SRC;TARGET_BACKEND;OPT_TOOL;MODULE_FILE_NAME"
    "FLAGS;OPT_FLAGS"
    ${ARGN}
  )

  if(ANDROID)
    # Android's CMake toolchain defines some variables that we can use to infer
    # the appropriate target triple from the configured settings:
    # https://developer.android.com/ndk/guides/cmake#android_platform
    #
    # In typical CMake fashion, the various strings are pretty fuzzy and can
    # have multiple values like "latest", "android-25"/"25"/"android-N-MR1".
    #
    # From looking at the toolchain file, ANDROID_PLATFORM_LEVEL seems like it
    # should pretty consistently be just a number we can use for target triple.
    set(_TARGET_TRIPLE "aarch64-none-linux-android${ANDROID_PLATFORM_LEVEL}")
    list(APPEND _RULE_FLAGS "--iree-llvm-target-triple=${_TARGET_TRIPLE}")
  endif()

  iree_bytecode_module(
    NAME
      "${_RULE_MODULE_NAME}"
    MODULE_FILE_NAME
      "${_RULE_MODULE_FILE_NAME}"
    SRC
      "${_RULE_SRC}"
    FLAGS
      "-iree-mlir-to-vm-bytecode-module"
      "-mlir-print-op-on-diagnostic=false"
      "--iree-hal-target-backends=${_RULE_TARGET_BACKEND}"
      ${_RULE_FLAGS}
    OPT_TOOL
      ${_RULE_OPT_TOOL}
    OPT_FLAGS
      ${_RULE_OPT_FLAGS}
    TESTONLY
  )
endfunction()

# iree_check_test()
#
# Creates a test using iree-check-module for the specified source file.
#
# Mirrors the bzl rule of the same name.
#
# Parameters:
#   NAME: Name of the target
#   SRC: mlir source file to be compiled to an IREE module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode
#       translation and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to iree-check-module. The driver
#       and input file are passed automatically.
#   LABELS: Additional labels to apply to the test. The package path and
#       "driver=${DRIVER}" are added automatically.
#   OPT_TOOL: Defaulting to iree-opt. Tool used to preprocess the source files
#       if OPT_FLAGS is specified.
#   OPT_FLAGS: If specified, source files are preprocessed with OPT_TOOL with
#       these flags.
#   MODULE_FILE_NAME: Optional, specifies the absolute path to the filename
#       to use for the generated IREE module (.vmfb).
function(iree_check_test)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Check tests require (by way of iree_bytecode_module) some tools.
  #
  # On the host, we can either build the tools directly, if IREE_BUILD_COMPILER
  # is enabled, or reuse the tools from an existing build (or binary release).
  #
  # In some configurations (e.g. when cross compiling for Android), we can't
  # always build the tools and may depend on them from a host build.
  #
  # For now we enable check tests:
  #   On the host if IREE_BUILD_COMPILER is set
  #   Always when cross compiling (assuming host tools exist)
  #
  # In the future, we should probably add some orthogonal options that give
  # more control (such as using tools from a binary release in a runtime-only
  # host build, or skipping check tests in an Android build).
  # TODO(#4662): add flexible configurable options that cover more uses
  if(NOT IREE_BUILD_COMPILER AND NOT CMAKE_CROSSCOMPILING)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;SRC;TARGET_BACKEND;DRIVER;OPT_TOOL;MODULE_FILE_NAME"
    "COMPILER_FLAGS;RUNNER_ARGS;LABELS;OPT_FLAGS"
    ${ARGN}
  )

  iree_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  set(_MODULE_NAME "${_RULE_NAME}_module")

  if(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_RULE_MODULE_FILE_NAME}")
  else(DEFINED _RULE_MODULE_FILE_NAME)
    set(_MODULE_FILE_NAME "${_MODULE_NAME}.vmfb")
  endif(DEFINED _RULE_MODULE_FILE_NAME)

  iree_bytecode_module_for_iree_check_test_and_friends(
    MODULE_NAME
      "${_MODULE_NAME}"
    MODULE_FILE_NAME
      "${_MODULE_FILE_NAME}"
    SRC
      "${_RULE_SRC}"
    TARGET_BACKEND
      "${_RULE_TARGET_BACKEND}"
    FLAGS
      ${_RULE_COMPILER_FLAGS}
    OPT_TOOL
      ${_RULE_OPT_TOOL}
    OPT_FLAGS
      ${_RULE_OPT_FLAGS}
  )

  # iree_bytecode_module does not define a target, only a custom command.
  # We need to create a target that depends on the command to ensure the
  # module gets built.
  # TODO(b/146898896): Do this in iree_bytecode_module and avoid having to
  # reach into the internals.
  set(_MODULE_TARGET_NAME "${_NAME}_module")
  add_custom_target(
    "${_MODULE_TARGET_NAME}"
     DEPENDS
       "${_MODULE_FILE_NAME}"
  )

  set(_RUNNER_TARGET "iree_tools_iree-check-module")

  # A target specifically for the test. We could combine this with the above,
  # but we want that one to get pulled into iree_bytecode_module.
  add_custom_target("${_NAME}" ALL)
  add_dependencies(
    "${_NAME}"
    "${_MODULE_TARGET_NAME}"
    "${_RUNNER_TARGET}"
  )

  iree_run_binary_test(
    NAME
      "${_RULE_NAME}"
    DRIVER
      "${_RULE_DRIVER}"
    TEST_BINARY
      "${_RUNNER_TARGET}"
    TEST_INPUT_FILE_ARG
      "${_MODULE_FILE_NAME}"
    ARGS
      ${_RULE_RUNNER_ARGS}
    LABELS
      ${_RULE_LABELS}
  )
endfunction()

# iree_check_single_backend_test_suite()
#
# Creates a test suite of iree-check-module tests for a single backend/driver pair.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source file.
# Parameters:
#   NAME: name of the generated test suite.
#   SRCS: source mlir files containing the module.
#   TARGET_BACKEND: target backend to compile for.
#   DRIVER: driver to run the module with.
#   COMPILER_FLAGS: additional flags to pass to the compiler. Bytecode
#       translation and backend flags are passed automatically.
#   RUNNER_ARGS: additional args to pass to the underlying iree-check-module
#       tests. The driver and input file are passed automatically. To use
#       different args per test, create a separate suite or iree_check_test.
#   LABELS: Additional labels to apply to the generated tests. The package path is
#       added automatically.
#   OPT_TOOL: Defaulting to iree-opt. Tool used to preprocess the source files
#       if OPT_FLAGS is specified.
#   OPT_FLAGS: If specified, source files are preprocessed with OPT_TOOL with
#       these flags.
function(iree_check_single_backend_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  # Note: we could check IREE_BUILD_COMPILER here, but cross compilation makes
  # that a little tricky. Instead, we let iree_check_test handle the checks,
  # meaning this function may run some configuration but generate no targets.

  cmake_parse_arguments(
    _RULE
    ""
    "NAME;TARGET_BACKEND;DRIVER;OPT_TOOL"
    "SRCS;COMPILER_FLAGS;RUNNER_ARGS;LABELS;OPT_FLAGS"
    ${ARGN}
  )

  # Omit tests for which the specified driver or target backend is not enabled.
  # This overlaps with directory exclusions and other filtering mechanisms.
  string(TOUPPER ${_RULE_DRIVER} _UPPERCASE_DRIVER)
  string(REPLACE "-" "_" _NORMALIZED_DRIVER ${_UPPERCASE_DRIVER})
  if(NOT DEFINED IREE_HAL_DRIVER_${_NORMALIZED_DRIVER})
    message(SEND_ERROR "Unknown driver '${_RULE_DRIVER}'. Check IREE_HAL_DRIVER_* options.")
  endif()
  if(NOT IREE_HAL_DRIVER_${_NORMALIZED_DRIVER})
    return()
  endif()
  string(TOUPPER ${_RULE_TARGET_BACKEND} _UPPERCASE_TARGET_BACKEND)
  string(REPLACE "-" "_" _NORMALIZED_TARGET_BACKEND ${_UPPERCASE_TARGET_BACKEND})
  if(NOT DEFINED IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
    message(SEND_ERROR "Unknown backend '${_RULE_TARGET_BACKEND}'. Check IREE_TARGET_BACKEND_* options.")
  endif()
  if(DEFINED IREE_HOST_BINARY_ROOT)
    # If we're not building the host tools from source under this configuration,
    # such as when cross compiling, then we can't easily check for which
    # compiler target backends are enabled. Just assume all are enabled and only
    # rely on the runtime HAL driver check above for filtering.
  else()
    # We are building the host tools, so check enabled compiler target backends.
    if(NOT IREE_TARGET_BACKEND_${_NORMALIZED_TARGET_BACKEND})
      return()
    endif()
  endif()

  foreach(_SRC IN LISTS _RULE_SRCS)
    set(_TEST_NAME "${_RULE_NAME}_${_SRC}")
    iree_check_test(
      NAME
        ${_TEST_NAME}
      SRC
        ${_SRC}
      TARGET_BACKEND
        ${_RULE_TARGET_BACKEND}
      DRIVER
        ${_RULE_DRIVER}
      COMPILER_FLAGS
        ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      OPT_TOOL
        ${_RULE_OPT_TOOL}
      OPT_FLAGS
        ${_RULE_OPT_FLAGS}
    )
  endforeach()
endfunction()


# iree_check_test_suite()
#
# Creates a test suite of iree-check-module tests.
#
# Mirrors the bzl rule of the same name.
#
# One test is generated per source and backend/driver pair.
# Parameters:
#   NAME: name of the generated test suite.
#   SRCS: source mlir files containing the module.
#   TARGET_BACKENDS: backends to compile the module for. These form pairs with
#       the DRIVERS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   DRIVERS: drivers to run the module with. These form pairs with the
#       TARGET_BACKENDS argument (due to cmake limitations they are separate list
#       arguments). The lengths must exactly match. If no backends or drivers are
#       specified, a test will be generated for every supported pair.
#   RUNNER_ARGS: additional args to pass to the underlying iree-check-module tests. The
#       driver and input file are passed automatically. To use different args per
#       test, create a separate suite or iree_check_test.
#   LABELS: Additional labels to apply to the generated tests. The package path is
#       added automatically.
#   OPT_TOOL: Defaulting to iree-opt. Tool used to preprocess the source files
#       if OPT_FLAGS is specified.
#   OPT_FLAGS: If specified, source files are preprocessed with OPT_TOOL with
#       these flags.
function(iree_check_test_suite)
  if(NOT IREE_BUILD_TESTS)
    return()
  endif()

  cmake_parse_arguments(
    _RULE
    ""
    "NAME"
    "SRCS;TARGET_BACKENDS;DRIVERS;RUNNER_ARGS;LABELS"
    ${ARGN}
  )

  if(NOT DEFINED _RULE_TARGET_BACKENDS AND NOT DEFINED _RULE_DRIVERS)
    set(_RULE_TARGET_BACKENDS "vmvx" "vulkan-spirv" "dylib-llvm-aot")
    set(_RULE_DRIVERS "vmvx" "vulkan" "dylib")
  endif()

  list(LENGTH _RULE_TARGET_BACKENDS _TARGET_BACKEND_COUNT)
  list(LENGTH _RULE_DRIVERS _DRIVER_COUNT)

  if(NOT _TARGET_BACKEND_COUNT EQUAL _DRIVER_COUNT)
    message(SEND_ERROR
        "TARGET_BACKENDS count ${_TARGET_BACKEND_COUNT} does not match DRIVERS count ${_DRIVER_COUNT}")
  endif()

  math(EXPR _MAX_INDEX "${_TARGET_BACKEND_COUNT} - 1")
  foreach(_INDEX RANGE "${_MAX_INDEX}")
    list(GET _RULE_TARGET_BACKENDS ${_INDEX} _TARGET_BACKEND)
    list(GET _RULE_DRIVERS ${_INDEX} _DRIVER)
    set(_SUITE_NAME "${_RULE_NAME}_${_TARGET_BACKEND}_${_DRIVER}")
    iree_check_single_backend_test_suite(
      NAME
        ${_SUITE_NAME}
      SRCS
        ${_RULE_SRCS}
      TARGET_BACKEND
        ${_TARGET_BACKEND}
      DRIVER
        ${_DRIVER}
      COMPILER_FLAGS
        ${_RULE_COMPILER_FLAGS}
      RUNNER_ARGS
        ${_RULE_RUNNER_ARGS}
      LABELS
        ${_RULE_LABELS}
      OPT_TOOL
        ${_RULE_OPT_TOOL}
      OPT_FLAGS
        ${_RULE_OPT_FLAGS}
    )
  endforeach()
endfunction()
