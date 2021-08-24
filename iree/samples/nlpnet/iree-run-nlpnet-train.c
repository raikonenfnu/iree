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

// This sample uses iree/tools/utils/image_util to load a hand-written image
// as an iree_hal_buffer_view_t then passes it to the bytecode module built
// from nlpnet.mlir on the dylib-llvm-aot backend.

#include <float.h>
#include <stdio.h>

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/api.h"
#include "iree/vm/bytecode_module.h"

#include "iree/tools/utils/image_util.h"
#include "iree/samples/nlpnet/nlpnet_bytecode_module_c.h"

extern iree_status_t create_sample_device(iree_hal_device_t** device);

iree_status_t Run(const iree_string_view_t image_path) {
  IREE_RETURN_IF_ERROR(iree_hal_module_register_types());

  iree_vm_instance_t* instance = NULL;
  IREE_RETURN_IF_ERROR(
      iree_vm_instance_create(iree_allocator_system(), &instance));

  iree_hal_device_t* device = NULL;
  IREE_RETURN_IF_ERROR(create_sample_device(&device), "create device");
  iree_vm_module_t* hal_module = NULL;
  IREE_RETURN_IF_ERROR(
      iree_hal_module_create(device, iree_allocator_system(), &hal_module));

  const struct iree_file_toc_t* module_file_toc = nlpnet_bytecode_module_c_create();

  iree_vm_module_t* bytecode_module = NULL;
  iree_const_byte_span_t module_data =
      iree_make_const_byte_span(module_file_toc->data, module_file_toc->size);
  IREE_RETURN_IF_ERROR(iree_vm_bytecode_module_create(
      module_data, iree_allocator_null(), iree_allocator_system(),
      &bytecode_module));

  // Allocate a context that will hold the module state across invocations.
  iree_vm_context_t* context = NULL;
  iree_vm_module_t* modules[] = {hal_module, bytecode_module};
  IREE_RETURN_IF_ERROR(iree_vm_context_create_with_modules(
      instance, &modules[0], IREE_ARRAYSIZE(modules), iree_allocator_system(),
      &context));
  iree_vm_module_release(hal_module);
  iree_vm_module_release(bytecode_module);

  // Lookup the entry point function.
  // Note that we use the synchronous variant which operates on pure type/shape
  // erased buffers.
  const char kMainFunctionName[] = "module.learn";
  iree_vm_function_t main_function;
  IREE_RETURN_IF_ERROR(iree_vm_context_resolve_function(
      context, iree_make_cstring_view(kMainFunctionName), &main_function));

  // Allocation of Input Buffers
  const int kElementCount0 = 512;
  const int kElementCount1 = 512;
  const int kElementCount2 = 512;
  const int kElementCount3 = 1;
  iree_hal_buffer_t* arg0_buffer = NULL;
  iree_hal_buffer_t* arg1_buffer = NULL;
  iree_hal_buffer_t* arg2_buffer = NULL;
  iree_hal_buffer_t* arg3_buffer = NULL;
  iree_hal_memory_type_t input_memory_type =
      IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), input_memory_type,
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(int) * kElementCount0, &arg0_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), input_memory_type,
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(int) * kElementCount1, &arg1_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), input_memory_type,
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(int) * kElementCount2, &arg2_buffer));
  IREE_RETURN_IF_ERROR(iree_hal_allocator_allocate_buffer(
      iree_hal_device_allocator(device), input_memory_type,
      IREE_HAL_BUFFER_USAGE_ALL, sizeof(int) * kElementCount3, &arg3_buffer));


  // Populate initial values for 4 * 2 = 8.
  const int kInt4 = 4;
  const int kInt2 = 2;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_fill(arg0_buffer, 0, IREE_WHOLE_BUFFER,
                                            &kInt4, sizeof(int)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_fill(arg1_buffer, 0, IREE_WHOLE_BUFFER,
                                            &kInt2, sizeof(int)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_fill(arg2_buffer, 0, IREE_WHOLE_BUFFER,
                                            &kInt4, sizeof(int)));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_fill(arg3_buffer, 0, IREE_WHOLE_BUFFER,
                                            &kInt2, sizeof(int)));


  // Wrap buffers in shaped buffer views.
  iree_hal_dim_t shape0[2] = {1,kElementCount0};
  iree_hal_dim_t shape1[2] = {1,kElementCount1};
  iree_hal_dim_t shape2[2] = {1,kElementCount2};
  iree_hal_dim_t shape3[1] = {kElementCount3};
  iree_hal_buffer_view_t* arg0_buffer_view = NULL;
  iree_hal_buffer_view_t* arg1_buffer_view = NULL;
  iree_hal_buffer_view_t* arg2_buffer_view = NULL;
  iree_hal_buffer_view_t* arg3_buffer_view = NULL;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      arg0_buffer, shape0, IREE_ARRAYSIZE(shape0), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &arg0_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      arg1_buffer, shape1, IREE_ARRAYSIZE(shape1), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &arg1_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      arg2_buffer, shape2, IREE_ARRAYSIZE(shape2), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &arg2_buffer_view));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_view_create(
      arg3_buffer, shape3, IREE_ARRAYSIZE(shape3), IREE_HAL_ELEMENT_TYPE_SINT_32,
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR, &arg3_buffer_view));

  iree_hal_buffer_release(arg0_buffer);
  iree_hal_buffer_release(arg1_buffer);
  iree_hal_buffer_release(arg2_buffer);
  iree_hal_buffer_release(arg3_buffer);

  // Setup call inputs with our buffers.
  iree_vm_list_t* inputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/4, iree_allocator_system(), &inputs),
                       "can't allocate input vm list");

  iree_vm_ref_t arg0_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg0_buffer_view);
  iree_vm_ref_t arg1_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg1_buffer_view);
  iree_vm_ref_t arg2_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg2_buffer_view);
  iree_vm_ref_t arg3_buffer_view_ref =
      iree_hal_buffer_view_move_ref(arg3_buffer_view);
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg0_buffer_view_ref));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg1_buffer_view_ref));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg2_buffer_view_ref));
  IREE_RETURN_IF_ERROR(
      iree_vm_list_push_ref_move(inputs, &arg3_buffer_view_ref));



  // Prepare outputs list to accept the results from the invocation.
  // The output vm list is allocated statically.
  iree_vm_list_t* outputs = NULL;
  IREE_RETURN_IF_ERROR(iree_vm_list_create(
                           /*element_type=*/NULL,
                           /*capacity=*/1, iree_allocator_system(), &outputs),
                       "can't allocate output vm list");

  // Synchronously invoke the function.
  printf("I am already here!\n");
  IREE_RETURN_IF_ERROR(iree_vm_invoke(context, main_function,
                                      /*policy=*/NULL, inputs, outputs,
                                      iree_allocator_system()));

  printf("Done with invoke!\n");
  // Get the result buffers from the invocation.
  iree_hal_buffer_view_t* ret_buffer_view =
      (iree_hal_buffer_view_t*)iree_vm_list_get_ref_deref(
          outputs, 0, iree_hal_buffer_view_get_descriptor());
  if (ret_buffer_view == NULL) {
    return iree_make_status(IREE_STATUS_NOT_FOUND,
                            "can't find return buffer view");
  }
  printf("Returning buffer!\n");
  // Read back the results. The output of the nlpnet model is a 1x10 prediction
  // confidence values for each digit in [0, 9].
  iree_hal_buffer_mapping_t mapped_memory;
  IREE_RETURN_IF_ERROR(iree_hal_buffer_map_range(
      iree_hal_buffer_view_buffer(ret_buffer_view), IREE_HAL_MEMORY_ACCESS_READ,
      0, IREE_WHOLE_BUFFER, &mapped_memory));
  float result_val = FLT_MIN;
  const float* data_ptr = (const float*)mapped_memory.contents.data;
  for (int i = 0; i < mapped_memory.contents.data_length / sizeof(float); ++i) {
    if (data_ptr[i] > result_val) {
      result_val = data_ptr[i];
    }
  }
  iree_hal_buffer_unmap_range(&mapped_memory);
  // Get the highest index from the output.
  fprintf(stdout, "Loss: %f\n", result_val);
  iree_hal_buffer_unmap_range(&mapped_memory);

  iree_vm_list_release(inputs);
  iree_vm_list_release(outputs);
  iree_hal_device_release(device);
  iree_vm_context_release(context);
  iree_vm_instance_release(instance);
  return iree_ok_status();
}

int main(int argc, char** argv) {
  if (argc > 2) {
    fprintf(stderr, "Usage: iree-run-nlpnet-module <image file>\n");
    return -1;
  }
  iree_string_view_t image_path;
  if (argc == 1) {
    image_path = iree_make_cstring_view("mnist_test.png");
  } else {
    image_path = iree_make_cstring_view(argv[1]);
  }
  iree_status_t result = Run(image_path);
  if (!iree_status_is_ok(result)) {
    iree_status_fprint(stderr, result);
    iree_status_ignore(result);
    return -1;
  }
  iree_status_ignore(result);
  return 0;
}
