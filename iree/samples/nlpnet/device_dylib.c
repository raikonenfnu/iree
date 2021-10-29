// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of setting up the the dylib driver.

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/local/executable_loader.h"
#include "iree/hal/local/loaders/embedded_library_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/task/api.h"

// Compiled module embedded here to avoid file IO:
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module_dylib_arm_64_c.h"
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module_dylib_riscv_64_c.h"
#include "iree/samples/simple_embedding/simple_embedding_test_bytecode_module_dylib_x86_64_c.h"

iree_status_t create_sample_device(iree_hal_device_t** device) {
  // Set paramters for the device created in the next step.
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  iree_hal_executable_loader_t* loader = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_embedded_library_loader_create(
      iree_hal_executable_import_provider_null(), iree_allocator_system(),
      &loader));

  iree_task_executor_t* executor = NULL;
  iree_status_t status =
      iree_task_executor_create_from_flags(iree_allocator_system(), &executor);

  iree_string_view_t identifier = iree_make_cstring_view("dylib");
  if (iree_status_is_ok(status)) {
    // Create the device.
    status = iree_hal_task_device_create(identifier, &params, executor,
                                         /*loader_count=*/1, &loader,
                                         iree_allocator_system(), device);
  }
  iree_task_executor_release(executor);
  iree_hal_executable_loader_release(loader);
  return status;
}

const iree_const_byte_span_t load_bytecode_module_data() {
#if IREE_ARCH_X86_64
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_dylib_x86_64_create();
#elif IREE_ARCH_RISCV_64
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_dylib_riscv_64_create();
#elif IREE_ARCH_ARM_64
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_simple_embedding_test_module_dylib_arm_64_create();
#else
#error "Unsupported platform."
#endif
  return iree_make_const_byte_span(module_file_toc->data,
                                   module_file_toc->size);
}
