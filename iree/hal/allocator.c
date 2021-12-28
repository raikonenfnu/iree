// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator.h"

#include <stddef.h>
#include <stdio.h>

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_format(
    const iree_hal_allocator_statistics_t* statistics,
    iree_string_builder_t* builder) {
#if IREE_STATISTICS_ENABLE

  // This could be prettier/have nice number formatting/etc.

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "  HOST_LOCAL: %12" PRIdsz "B peak / %12" PRIdsz
      "B allocated / %12" PRIdsz "B freed / %12" PRIdsz "B live\n",
      statistics->host_bytes_peak, statistics->host_bytes_allocated,
      statistics->host_bytes_freed,
      (statistics->host_bytes_allocated - statistics->host_bytes_freed)));

  IREE_RETURN_IF_ERROR(iree_string_builder_append_format(
      builder,
      "DEVICE_LOCAL: %12" PRIdsz "B peak / %12" PRIdsz
      "B allocated / %12" PRIdsz "B freed / %12" PRIdsz "B live\n",
      statistics->device_bytes_peak, statistics->device_bytes_allocated,
      statistics->device_bytes_freed,
      (statistics->device_bytes_allocated - statistics->device_bytes_freed)));

#else
  // No-op when disabled.
#endif  // IREE_STATISTICS_ENABLE
  return iree_ok_status();
}

#define _VTABLE_DISPATCH(allocator, method_name) \
  IREE_HAL_VTABLE_DISPATCH(allocator, iree_hal_allocator, method_name)

IREE_HAL_API_RETAIN_RELEASE(allocator);

IREE_API_EXPORT iree_allocator_t
iree_hal_allocator_host_allocator(const iree_hal_allocator_t* allocator) {
  IREE_ASSERT_ARGUMENT(allocator);
  return _VTABLE_DISPATCH(allocator, host_allocator)(allocator);
}

IREE_API_EXPORT void iree_hal_allocator_query_statistics(
    iree_hal_allocator_t* allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  IREE_ASSERT_ARGUMENT(allocator);
  memset(out_statistics, 0, sizeof(*out_statistics));
  IREE_STATISTICS({
    _VTABLE_DISPATCH(allocator, query_statistics)(allocator, out_statistics);
  });
}

IREE_API_EXPORT iree_hal_buffer_compatibility_t
iree_hal_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  IREE_ASSERT_ARGUMENT(allocator);
  return _VTABLE_DISPATCH(allocator, query_buffer_compatibility)(
      allocator, memory_type, allowed_usage, intended_usage, allocation_size);
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_allocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, allocate_buffer)(
      allocator, memory_type, allowed_usage, allocation_size, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_wrap_buffer(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(out_buffer);
  *out_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_status_t status = _VTABLE_DISPATCH(allocator, wrap_buffer)(
      allocator, memory_type, allowed_access, allowed_usage, data,
      data_allocator, out_buffer);
  IREE_TRACE_ZONE_END(z0);
  return status;
}

IREE_API_EXPORT void iree_hal_allocator_deallocate_buffer(
    iree_hal_allocator_t* allocator, iree_hal_buffer_t* buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(buffer);
  IREE_TRACE_ZONE_BEGIN(z0);
  _VTABLE_DISPATCH(allocator, deallocate_buffer)(allocator, buffer);
  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_statistics_fprint(
    FILE* file, iree_hal_allocator_t* allocator) {
#if IREE_STATISTICS_ENABLE
  iree_hal_allocator_statistics_t statistics;
  iree_hal_allocator_query_statistics(allocator, &statistics);

  iree_string_builder_t builder;
  iree_string_builder_initialize(iree_hal_allocator_host_allocator(allocator),
                                 &builder);

  // TODO(benvanik): query identifier for the allocator so we can denote which
  // device is being reported.
  iree_status_t status = iree_string_builder_append_cstring(
      &builder, "[[ iree_hal_allocator_t memory statistics ]]\n");

  if (iree_status_is_ok(status)) {
    status = iree_hal_allocator_statistics_format(&statistics, &builder);
  }

  if (iree_status_is_ok(status)) {
    fprintf(file, "%.*s", (int)iree_string_builder_size(&builder),
            iree_string_builder_buffer(&builder));
  }

  iree_string_builder_deinitialize(&builder);
  return status;
#else
  // No-op.
  return iree_ok_status();
#endif  // IREE_STATISTICS_ENABLE
}
