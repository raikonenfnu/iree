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

#include "experimental/rocm/nop_executable_cache.h"

#include "experimental/rocm/native_executable.h"
#include "iree/base/tracing.h"

typedef struct {
  iree_hal_resource_t resource;
  iree_hal_rocm_context_wrapper_t *context;
} iree_hal_rocm_nop_executable_cache_t;

extern const iree_hal_executable_cache_vtable_t
    iree_hal_rocm_nop_executable_cache_vtable;

static iree_hal_rocm_nop_executable_cache_t *
iree_hal_rocm_nop_executable_cache_cast(
    iree_hal_executable_cache_t *base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_nop_executable_cache_vtable);
  return (iree_hal_rocm_nop_executable_cache_t *)base_value;
}

iree_status_t iree_hal_rocm_nop_executable_cache_create(
    iree_hal_rocm_context_wrapper_t *context, iree_string_view_t identifier,
    iree_hal_executable_cache_t **out_executable_cache) {
  IREE_ASSERT_ARGUMENT(out_executable_cache);
  *out_executable_cache = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_nop_executable_cache_t *executable_cache = NULL;
  iree_status_t status =
      iree_allocator_malloc(context->host_allocator, sizeof(*executable_cache),
                            (void **)&executable_cache);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_rocm_nop_executable_cache_vtable,
                                 &executable_cache->resource);
    executable_cache->context = context;

    *out_executable_cache = (iree_hal_executable_cache_t *)executable_cache;
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_rocm_nop_executable_cache_destroy(
    iree_hal_executable_cache_t *base_executable_cache) {
  iree_hal_rocm_nop_executable_cache_t *executable_cache =
      iree_hal_rocm_nop_executable_cache_cast(base_executable_cache);
  iree_allocator_t host_allocator = executable_cache->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, executable_cache);

  IREE_TRACE_ZONE_END(z0);
}

static bool iree_hal_rocm_nop_executable_cache_can_prepare_format(
    iree_hal_executable_cache_t *base_executable_cache,
    iree_hal_executable_caching_mode_t caching_mode,
    iree_string_view_t executable_format) {
  return iree_string_view_equal(executable_format,
                                iree_make_cstring_view("PTXE"));
}

static iree_status_t iree_hal_rocm_nop_executable_cache_prepare_executable(
    iree_hal_executable_cache_t *base_executable_cache,
    const iree_hal_executable_spec_t *executable_spec,
    iree_hal_executable_t **out_executable) {
  iree_hal_rocm_nop_executable_cache_t *executable_cache =
      iree_hal_rocm_nop_executable_cache_cast(base_executable_cache);
  return iree_hal_rocm_native_executable_create(
      executable_cache->context, executable_spec, out_executable);
}

const iree_hal_executable_cache_vtable_t
    iree_hal_rocm_nop_executable_cache_vtable = {
        .destroy = iree_hal_rocm_nop_executable_cache_destroy,
        .can_prepare_format =
            iree_hal_rocm_nop_executable_cache_can_prepare_format,
        .prepare_executable =
            iree_hal_rocm_nop_executable_cache_prepare_executable,
};
