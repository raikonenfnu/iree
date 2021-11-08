// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// A example of static library loading in IREE. See the README.md for more info.
// Note: this demo requires artifacts from iree-translate before it will run.

#include <stdio.h>

#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/hal/local/task_device.h"
#include "iree/modules/hal/module.h"
#include "iree/runtime/api.h"
#include "iree/samples/static_mul/linalg_mul_c.h"
#include "iree/task/api.h"
#include "iree/vm/bytecode_module.h"

// Compiled static library module here to avoid IO:
#include "iree/samples/static_mul/linalg_mul.h"

// A function to create the HAL device from the different backend targets.
// The HAL device is returned based on the implementation, and it must be
// released by the caller.
iree_status_t create_device_with_static_loader(iree_hal_device_t** device) {
  iree_status_t status = iree_ok_status();

  // Set paramters for the device created in the next step.
  iree_hal_task_device_params_t params;
  iree_hal_task_device_params_initialize(&params);

  // Load the statically embedded library
  const iree_hal_executable_library_header_t** static_library =
      forward_dispatch_0_library_query(
          IREE_HAL_EXECUTABLE_LIBRARY_LATEST_VERSION, /*reserved=*/NULL);
  const iree_hal_executable_library_header_t** libraries[1] = {static_library};

  iree_hal_executable_loader_t* library_loader = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_static_library_loader_create(
        IREE_ARRAYSIZE(libraries), libraries,
        iree_hal_executable_import_provider_null(), iree_allocator_system(),
        &library_loader);
  }

  iree_task_executor_t* executor = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_task_executor_create_from_flags(iree_allocator_system(),
                                                  &executor);
  }

  // Create the device and release the executor and loader afterwards.
  if (iree_status_is_ok(status)) {
    status = iree_hal_task_device_create(iree_make_cstring_view("dylib"),
                                         &params, executor, 1, &library_loader,
                                         iree_allocator_system(), device);
  }
  iree_task_executor_release(executor);
  iree_hal_executable_loader_release(library_loader);

  return status;
}

iree_status_t Run() {
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());
  iree_status_t status = iree_ok_status();

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));

  // Create dylib device with static loader.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = create_device_with_static_loader(&device);
  }

  iree_vm_module_t* hal_module = NULL;
  if (iree_status_is_ok(status)) {
    status =
        iree_hal_module_create(device, iree_allocator_system(), &hal_module);
  }

  // Load bytecode module from the embedded data. Append to the session.
  const struct iree_file_toc_t* module_file_toc =
      iree_samples_static_library_linalg_mul_create();
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  iree_vm_module_t* bytecode_module = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_vm_bytecode_module_create(module_data, iree_allocator_null(),
                                            iree_allocator_system(),
                                            &bytecode_module);
  }

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, IREE_VM_CONTEXT_FLAG_NONE, &modules[0], IREE_ARRAYSIZE(modules), iree_allocator_system(),
      &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function call.
  const char kMainFunctionName[] = "module.forward";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // Populate initial values for 4 * 2 = 8.
  const int kElementCount = 4;
  iree_hal_dim_t shape[1] = {kElementCount};
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;
  iree_hal_buffer_view_t* arg2_buffer_view = NULL;
  float kFloat4[] = {4.0f, 4.0f, 4.0f, 4.0f};
  float kFloat2[] = {2.0f, 2.0f, 2.0f, 2.0f};
  float kFloat0[] = {0.0f, 0.0f, 0.0f, 0.0f};

  iree_hal_memory_type_t input_memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_hal_device_allocator(device), shape, IREE_ARRAYSIZE(shape),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        input_memory_type, IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)kFloat4,
                                  sizeof(float) * kElementCount),
        &arg0_buffer_view);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_hal_device_allocator(device), shape, IREE_ARRAYSIZE(shape),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        input_memory_type, IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)kFloat2,
                                  sizeof(float) * kElementCount),
        &arg1_buffer_view);
  }

  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_wrap_heap_buffer(
        iree_hal_device_allocator(device), shape, IREE_ARRAYSIZE(shape),
        IREE_HAL_ELEMENT_TYPE_FLOAT_32, IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        input_memory_type, IREE_HAL_MEMORY_ACCESS_ALL, IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_byte_span((void*)kFloat0,
                                  sizeof(float) * kElementCount), iree_allocator_null(),
        &arg2_buffer_view);
  }


  // Queue buffer views for input.
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/3, iree_allocator_system(), &inputs),
                       "can't allocate input vm list");

  iree_vm_ref_t arg0_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg0_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg0_buffer_view_ref));

  iree_vm_ref_t arg1_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg1_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg1_buffer_view_ref));

  iree_vm_ref_t arg2_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg2_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg2_buffer_view_ref));

  // Prepare outputs list to accept the results from the invocation.
  // The output vm list is allocated statically.
  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/0, iree_allocator_system(), &outputs),
                       "can't allocate output vm list");

  // Synchronously invoke the function.
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context, main_function, IREE_VM_INVOCATION_FLAG_NONE,
                                      /*policy=*/NULL, inputs, outputs,
                                      iree_allocator_system()));

  if (iree_status_is_ok(status)) {
    for (iree_host_size_t i = 0; i < kElementCount; ++i) {
      printf("%f,",kFloat0[i]);
      if (kFloat0[i] != 8.0f) {
        status = iree_make_status(IREE_STATUS_UNKNOWN, "result mismatches");
      }
    }
  }

  // Cleanup session and instance.
  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return status;
}


int main() {
  const iree_status_t result = Run();
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_free(result);
    return -1;
  }
  printf("static_library_run passed\n");
  return 0;
}
