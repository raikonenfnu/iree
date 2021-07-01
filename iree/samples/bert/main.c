// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdio.h>

#include "iree/runtime/api.h"

iree_status_t predict(iree_runtime_session_t* session,
                            iree_hal_buffer_view_t** out_buffer_view) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.predict"), &call));

// [input_word_ids, input_mask, segment_ids]
  iree_hal_buffer_view_t* arg0 = NULL;
  iree_hal_buffer_view_t* arg1 = NULL;
  iree_hal_buffer_view_t* arg2 = NULL;
  const int kBatchSize = 1;
  const int kSequenceLength = 512;
  const iree_hal_dim_t arg0_shape[2] = {kBatchSize, kSequenceLength};
  const int values_length = kBatchSize*kSequenceLength;
  int values[values_length];
  int max_num = 5;
  for (int i = 0; i < values_length; i++) {
    values[i] = i%max_num;
  }

  // TODO(scotttodd): use iree_hal_buffer_view_wrap_or_clone_heap_buffer
  //   * debugging some apparent memory corruption with the stack-local value
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg2);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg2);
  }
  iree_hal_buffer_view_release(arg0);
  iree_hal_buffer_view_release(arg1);
  iree_hal_buffer_view_release(arg2);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, out_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}

iree_status_t learn(iree_runtime_session_t* session,
                            iree_hal_buffer_view_t** out_buffer_view) {
  iree_runtime_call_t call;
  IREE_RETURN_IF_ERROR(iree_runtime_call_initialize_by_name(
      session, iree_make_cstring_view("module.learn"), &call));

// [input_word_ids, input_mask, segment_ids]
  iree_hal_buffer_view_t* arg0 = NULL;
  iree_hal_buffer_view_t* arg1 = NULL;
  iree_hal_buffer_view_t* arg2 = NULL;
  iree_hal_buffer_view_t* arg3 = NULL;
  const int kBatchSize = 1;
  const int kSequenceLength = 512;

  // Filling in values to be used for input_word_ids, input_mask, segment_ids
  const iree_hal_dim_t arg0_shape[2] = {kBatchSize, kSequenceLength};
  const int values_length = kBatchSize*kSequenceLength;
  int values[values_length];
  int max_num = 5;
  for (int i = 0; i < values_length; i++) {
    values[i] = i%max_num;
  }

  // Filling in values to be used for label
  const iree_hal_dim_t label_shape[1] = {kBatchSize};
  const int label_length = kBatchSize;
  int label[label_length];
  for (int i = 0; i < label_length; i++) {
    label[i] = i%max_num;
  }

  // TODO(scotttodd): use iree_hal_buffer_view_wrap_or_clone_heap_buffer
  //   * debugging some apparent memory corruption with the stack-local value
  iree_status_t status = iree_ok_status();
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), arg0_shape,
        IREE_ARRAYSIZE(arg0_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)values, sizeof(int) * values_length),
        &arg2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_hal_buffer_view_clone_heap_buffer(
        iree_runtime_session_device_allocator(session), label_shape,
        IREE_ARRAYSIZE(label_shape), IREE_HAL_ELEMENT_TYPE_SINT_32,
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE,
        IREE_HAL_BUFFER_USAGE_ALL,
        iree_make_const_byte_span((void*)label, sizeof(int) * label_length),
        &arg3);
  }

  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg1);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg2);
  }
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg3);
  }
  iree_hal_buffer_view_release(arg0);
  iree_hal_buffer_view_release(arg1);
  iree_hal_buffer_view_release(arg2);
  iree_hal_buffer_view_release(arg3);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
  }

  if (iree_status_is_ok(status)) {
    status =
        iree_runtime_call_outputs_pop_front_buffer_view(&call, out_buffer_view);
  }

  iree_runtime_call_deinitialize(&call);
  return status;
}


iree_status_t run_sample(iree_string_view_t bytecode_module_path,
                         iree_string_view_t driver_name) {
  iree_status_t status = iree_ok_status();

  //===-------------------------------------------------------------------===//
  // Instance configuration (this should be shared across sessions).
  fprintf(stdout, "Configuring IREE runtime instance and '%s' device\n",
          driver_name.data);
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(IREE_API_VERSION_LATEST,
                                           &instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t* instance = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_create(&instance_options,
                                          iree_allocator_system(), &instance);
  }
  // TODO(#5724): move device selection into the compiled modules.
  iree_hal_device_t* device = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_instance_try_create_default_device(
        instance, driver_name, &device);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Session configuration (one per loaded module to hold module state).
  fprintf(stdout, "Creating IREE runtime session\n");
  iree_runtime_session_options_t session_options;
  iree_runtime_session_options_initialize(&session_options);
  iree_runtime_session_t* session = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
  }
  iree_hal_device_release(device);

  fprintf(stdout, "Loading bytecode module at '%s'\n",
          bytecode_module_path.data);
  if (iree_status_is_ok(status)) {
    status = iree_runtime_session_append_bytecode_module_from_file(
        session, bytecode_module_path.data);
  }
  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Call functions to manipulate the counter
  fprintf(stdout, "Calling functions\n\n");

  // Example of prediction
  // if (iree_status_is_ok(status)) {
  //   iree_hal_buffer_view_t* result_buffer_view = NULL;
  //   status = predict(session, &result_buffer_view);
  //   if (iree_status_is_ok(status)) {
  //     fprintf(stdout, "Predict Result: ");
  //     status = iree_hal_buffer_view_fprint(stdout, result_buffer_view,
  //                                          /*max_element_count=*/4096);
  //     fprintf(stdout, "\n");
  //   }
  //   iree_hal_buffer_view_release(result_buffer_view);
  // }

  // Example of Training
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_view_t* result_buffer_view = NULL;
    status = learn(session, &result_buffer_view);
    if (iree_status_is_ok(status)) {
      fprintf(stdout, "Training Loss: ");
      status = iree_hal_buffer_view_fprint(stdout, result_buffer_view,
                                           /*max_element_count=*/4096);
      fprintf(stdout, "\n");
    }
    iree_hal_buffer_view_release(result_buffer_view);
  }


  //===-------------------------------------------------------------------===//

  //===-------------------------------------------------------------------===//
  // Cleanup.
  iree_runtime_session_release(session);
  iree_runtime_instance_release(instance);
  //===-------------------------------------------------------------------===//

  return status;
}

int main(int argc, char** argv) {
  if (argc != 3) {
    fprintf(
        stderr,
        "Usage: bert-model </path/to/counter.vmfb> <driver_name>\n");
    fprintf(stderr, "  (See the README for this sample for details)\n ");
    return -1;
  }

  iree_string_view_t bytecode_module_path = iree_make_cstring_view(argv[1]);
  iree_string_view_t driver_name = iree_make_cstring_view(argv[2]);

  iree_status_t result = run_sample(bytecode_module_path, driver_name);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  fprintf(stdout, "\nSuccess!\n");
  return 0;
}
