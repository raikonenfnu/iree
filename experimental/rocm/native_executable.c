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

#include "experimental/rocm/native_executable.h"

#include "experimental/rocm/status_util.h"
#include "iree/base/tracing.h"

// flatcc schemas:
#include "iree/base/internal/flatcc.h"
#include "iree/schemas/rocm_executable_def_reader.h"
#include "iree/schemas/rocm_executable_def_verifier.h"

typedef struct {
  hipFunction_t rocm_function;
  uint32_t block_size_x;
  uint32_t block_size_y;
  uint32_t block_size_z;
} iree_hal_rocm_native_executable_function_t;

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t *context;
  iree_host_size_t entry_count;
  hipModule_t module;
  iree_hal_rocm_native_executable_function_t entry_functions[];
} iree_hal_rocm_native_executable_t;

extern const iree_hal_executable_vtable_t
    iree_hal_rocm_native_executable_vtable;

static iree_hal_rocm_native_executable_t *iree_hal_rocm_native_executable_cast(
    iree_hal_executable_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_native_executable_vtable);
  return (iree_hal_rocm_native_executable_t *)base_value;
}

iree_status_t iree_hal_rocm_native_executable_create(
    iree_hal_rocm_context_wrapper_t *context,
    const iree_hal_executable_spec_t *executable_spec,
    iree_hal_executable_t **out_executable) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(executable_spec);
  IREE_ASSERT_ARGUMENT(out_executable);
  *out_executable = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_native_executable_t *executable = NULL;

  // TODO: Verify the flat buffer.
  iree_ROCMExecutableDef_table_t executable_def =
      iree_ROCMExecutableDef_as_root(executable_spec->executable_data.data);

  // Create the kernel module.
  flatbuffers_string_t hsaco_image =
      iree_ROCMExecutableDef_hsaco_image_get(executable_def);
  flatbuffers_string_vec_t entry_points_vec =
      iree_ROCMExecutableDef_entry_points_get(executable_def);
  iree_ROCMBlockSizeDef_vec_t block_sizes_vec =
      iree_ROCMExecutableDef_block_sizes_get(executable_def);
  iree_host_size_t entry_count = flatbuffers_string_vec_len(entry_points_vec);
  iree_host_size_t total_size =
      sizeof(*executable) +
      entry_count * sizeof(iree_hal_rocm_native_executable_function_t);
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, total_size, (void **)&executable);
  hipModule_t module = NULL;
  ROCM_RETURN_IF_ERROR(context->syms,
                       hipModuleLoadDataEx(&module, hsaco_image, 0, NULL, NULL),
                       "hipModuleLoadDataEx");

  for (iree_host_size_t i = 0; i < entry_count; i++) {
    hipFunction_t function = NULL;
    const char *entry_name = flatbuffers_string_vec_at(entry_points_vec, i);
    ROCM_RETURN_IF_ERROR(context->syms,
                         hipModuleGetFunction(&function, module, entry_name),
                         "hipModuleGetFunction");
    executable->entry_functions[i].rocm_function = function;
    executable->entry_functions[i].block_size_x = block_sizes_vec[i].x;
    executable->entry_functions[i].block_size_y = block_sizes_vec[i].y;
    executable->entry_functions[i].block_size_z = block_sizes_vec[i].z;
  }

  iree_hal_resource_initialize(&iree_hal_rocm_native_executable_vtable,
                               &executable->resource);
  executable->module = module;
  executable->context = context;
  *out_executable = (iree_hal_executable_t *)executable;
  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

hipFunction_t iree_hal_rocm_native_executable_for_entry_point(
    iree_hal_executable_t *base_executable, int32_t entry_point) {
  iree_hal_rocm_native_executable_t *executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  return executable->entry_functions[entry_point].rocm_function;
}

iree_status_t iree_hal_rocm_native_executable_block_size(
    iree_hal_executable_t *base_executable, int32_t entry_point, uint32_t *x,
    uint32_t *y, uint32_t *z) {
  iree_hal_rocm_native_executable_t *executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  *x = executable->entry_functions[entry_point].block_size_x;
  *y = executable->entry_functions[entry_point].block_size_y;
  *z = executable->entry_functions[entry_point].block_size_z;
  return iree_ok_status();
}

static void iree_hal_rocm_native_executable_destroy(
    iree_hal_executable_t *base_executable) {
  iree_hal_rocm_native_executable_t *executable =
      iree_hal_rocm_native_executable_cast(base_executable);
  iree_allocator_t host_allocator = executable->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable);

  IREE_TRACE_ZONE_END(z0);
}

const iree_hal_executable_vtable_t iree_hal_rocm_native_executable_vtable = {
    .destroy = iree_hal_rocm_native_executable_destroy,
};
