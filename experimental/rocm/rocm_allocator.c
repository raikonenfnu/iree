// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/rocm/rocm_allocator.h"

#include <stddef.h>

#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/rocm_buffer.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"

typedef struct iree_hal_rocm_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_device_t* base_device;
  iree_hal_rocm_context_wrapper_t* context;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_rocm_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_rocm_allocator_vtable;

static iree_hal_rocm_allocator_t* iree_hal_rocm_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_allocator_vtable);
  return (iree_hal_rocm_allocator_t*)base_value;
}

iree_status_t iree_hal_rocm_allocator_create(
    iree_hal_device_t* base_device, iree_hal_rocm_context_wrapper_t* context,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);
  iree_hal_rocm_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*allocator), (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_rocm_allocator_vtable,
                                 &allocator->resource);
    allocator->context = context;
    allocator->base_device = base_device;
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_rocm_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_rocm_allocator_t* allocator =
      iree_hal_rocm_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_rocm_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_rocm_allocator_t* allocator =
      (iree_hal_rocm_allocator_t*)base_allocator;
  return allocator->context->host_allocator;
}

static iree_status_t iree_hal_rocm_allocator_trim(
    iree_hal_allocator_t* base_allocator) {
  return iree_ok_status();
}

static void iree_hal_rocm_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  IREE_STATISTICS({
    iree_hal_rocm_allocator_t* allocator =
        iree_hal_rocm_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_rocm_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  // TODO(benvanik): check to ensure the allocator can serve the memory type.

  // Disallow usage not permitted by the buffer itself. Since we then use this
  // to determine compatibility below we'll naturally set the right compat flags
  // based on what's both allowed and intended.
  intended_usage &= allowed_usage;

  // All buffers can be allocated on the heap.
  iree_hal_buffer_compatibility_t compatibility =
      IREE_HAL_BUFFER_COMPATIBILITY_ALLOCATABLE;

  // Buffers can only be used on the queue if they are device visible.
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_TRANSFER)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER;
    }
    if (iree_all_bits_set(intended_usage, IREE_HAL_BUFFER_USAGE_DISPATCH)) {
      compatibility |= IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH;
    }
  }

  return compatibility;
}

static void iree_hal_rocm_buffer_free(iree_hal_rocm_context_wrapper_t* context,
                                      iree_hal_memory_type_t memory_type,
                                      hipDeviceptr_t device_ptr,
                                      void* host_ptr) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local.
    ROCM_IGNORE_ERROR(context->syms, hipFree(device_ptr));
  } else {
    // Host local.
    ROCM_IGNORE_ERROR(context->syms, hipHostFree(host_ptr));
  }
}

static iree_status_t iree_hal_rocm_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer) {
  iree_hal_rocm_allocator_t* allocator =
      iree_hal_rocm_allocator_cast(base_allocator);
  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;
  iree_status_t status;
  void* host_ptr = NULL;
  hipDeviceptr_t device_ptr = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      status = ROCM_RESULT_TO_STATUS(
          allocator->context->syms,
          hipMallocManaged(&device_ptr, allocation_size, hipMemAttachGlobal));
      host_ptr = (void*)device_ptr;
    } else {
      // Device only.
      status = ROCM_RESULT_TO_STATUS(allocator->context->syms,
                                     hipMalloc(&device_ptr, allocation_size));
    }
  } else {
    unsigned int flags = hipHostMallocMapped;
    if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      flags |= hipHostMallocWriteCombined;
    }
    status = ROCM_RESULT_TO_STATUS(
        allocator->context->syms,
        hipMemAllocHost(&host_ptr, allocation_size, flags));
    if (iree_status_is_ok(status)) {
      status = ROCM_RESULT_TO_STATUS(
          allocator->context->syms,
          hipHostGetDevicePointer(&device_ptr, host_ptr, /*flags=*/0));
    }
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_rocm_buffer_wrap(
        (iree_hal_allocator_t*)allocator, memory_type,
        IREE_HAL_MEMORY_ACCESS_ALL, allowed_usage, allocation_size,
        /*byte_offset=*/0,
        /*byte_length=*/allocation_size, device_ptr, host_ptr, &buffer);
  }

  // Copy the initial contents into the buffer. This may require staging.
  if (iree_status_is_ok(status) &&
      !iree_const_byte_span_is_empty(initial_data)) {
    status = iree_hal_device_transfer_range(
        allocator->base_device,
        iree_hal_make_host_transfer_buffer_span((void*)initial_data.data,
                                                initial_data.data_length),
        0, iree_hal_make_device_transfer_buffer(buffer), 0,
        initial_data.data_length, IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
  }

  if (iree_status_is_ok(status)) {
    IREE_STATISTICS(iree_hal_allocator_statistics_record_alloc(
        &allocator->statistics, memory_type, allocation_size));
    *out_buffer = buffer;
  } else {
    if (!buffer) {
      iree_hal_rocm_buffer_free(allocator->context, memory_type, device_ptr,
                                host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static iree_status_t iree_hal_rocm_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

static void iree_hal_rocm_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  iree_hal_rocm_allocator_t* allocator =
      iree_hal_rocm_allocator_cast(base_allocator);

  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  iree_hal_rocm_buffer_free(allocator->context, memory_type,
                            iree_hal_rocm_buffer_device_pointer(base_buffer),
                            iree_hal_rocm_buffer_host_pointer(base_buffer));

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, memory_type,
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_rocm_allocator_vtable = {
    .destroy = iree_hal_rocm_allocator_destroy,
    .host_allocator = iree_hal_rocm_allocator_host_allocator,
    .trim = iree_hal_rocm_allocator_trim,
    .query_statistics = iree_hal_rocm_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_rocm_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_rocm_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_rocm_allocator_wrap_buffer,
    .deallocate_buffer = iree_hal_rocm_allocator_deallocate_buffer,
};
