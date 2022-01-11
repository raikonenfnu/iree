// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/cuda_allocator.h"

#include <stddef.h>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/cuda/cuda_buffer.h"
#include "iree/hal/cuda/dynamic_symbols.h"
#include "iree/hal/cuda/status_util.h"

typedef struct iree_hal_cuda_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_device_t* base_device;
  iree_hal_cuda_context_wrapper_t* context;
  CUdevice device;
  CUstream stream;
  bool supports_concurrent_managed_access;

  IREE_STATISTICS(iree_hal_allocator_statistics_t statistics;)
} iree_hal_cuda_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_cuda_allocator_vtable;

static iree_hal_cuda_allocator_t* iree_hal_cuda_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_allocator_vtable);
  return (iree_hal_cuda_allocator_t*)base_value;
}

iree_status_t iree_hal_cuda_allocator_create(
    iree_hal_device_t* base_device, iree_hal_cuda_context_wrapper_t* context,
    CUdevice device, CUstream stream, iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(base_device);
  IREE_ASSERT_ARGUMENT(context);
  IREE_TRACE_ZONE_BEGIN(z0);

  // To support device-local + host-visible memory we need concurrent managed
  // access indicating that the host and devices can concurrently access the
  // device memory. If we don't have this feature then we fall back to forcing
  // all device-local + host-visible memory into host-local + device-visible
  // page-locked memory. The compiler tries to avoid this for high-traffic
  // buffers except for readback staging buffers.
  int supports_concurrent_managed_access = 0;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, CU_RESULT_TO_STATUS(
              context->syms,
              cuDeviceGetAttribute(
                  &supports_concurrent_managed_access,
                  CU_DEVICE_ATTRIBUTE_CONCURRENT_MANAGED_ACCESS, device),
              "cuDeviceGetAttribute"));

  IREE_TRACE_ZONE_APPEND_TEXT(
      z0, supports_concurrent_managed_access
              ? "has CONCURRENT_MANAGED_ACCESS"
              : "no CONCURRENT_MANAGED_ACCESS (expect slow accesses on "
                "device-local + host-visible memory)");

  iree_hal_cuda_allocator_t* allocator = NULL;
  iree_status_t status = iree_allocator_malloc(
      context->host_allocator, sizeof(*allocator), (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_cuda_allocator_vtable,
                                 &allocator->resource);
    allocator->base_device = base_device;
    allocator->context = context;
    allocator->device = device;
    allocator->stream = stream;
    allocator->supports_concurrent_managed_access =
        supports_concurrent_managed_access != 0;
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);
  iree_allocator_t host_allocator = allocator->context->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(host_allocator, allocator);

  IREE_TRACE_ZONE_END(z0);
}

static iree_allocator_t iree_hal_cuda_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_cuda_allocator_t* allocator =
      (iree_hal_cuda_allocator_t*)base_allocator;
  return allocator->context->host_allocator;
}

static iree_status_t iree_hal_cuda_allocator_trim(
    iree_hal_allocator_t* base_allocator) {
  return iree_ok_status();
}

static void iree_hal_cuda_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  IREE_STATISTICS({
    iree_hal_cuda_allocator_t* allocator =
        iree_hal_cuda_allocator_cast(base_allocator);
    memcpy(out_statistics, &allocator->statistics, sizeof(*out_statistics));
  });
}

static iree_hal_buffer_compatibility_t
iree_hal_cuda_allocator_query_buffer_compatibility(
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

static void iree_hal_cuda_buffer_free(iree_hal_cuda_context_wrapper_t* context,
                                      iree_hal_memory_type_t memory_type,
                                      CUdeviceptr device_ptr, void* host_ptr) {
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local.
    CUDA_IGNORE_ERROR(context->syms, cuMemFree(device_ptr));
  } else {
    // Host local.
    CUDA_IGNORE_ERROR(context->syms, cuMemFreeHost(host_ptr));
  }
}

static iree_status_t iree_hal_cuda_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_const_byte_span_t initial_data, iree_hal_buffer_t** out_buffer) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);
  // Guard against the corner case where the requested buffer size is 0. The
  // application is unlikely to do anything when requesting a 0-byte buffer; but
  // it can happen in real world use cases. So we should at least not crash.
  if (allocation_size == 0) allocation_size = 4;

  // If concurrent managed access is not supported then make device-local +
  // host-visible allocations fall back to host-local + device-visible
  // page-locked memory. This will be significantly slower for the device to
  // access but the compiler only uses this type for readback staging buffers
  // and it's better to function than function fast.
  if (!allocator->supports_concurrent_managed_access &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                                         IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    memory_type &= ~(IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL |
                     IREE_HAL_MEMORY_TYPE_HOST_VISIBLE);
    memory_type |=
        IREE_HAL_MEMORY_TYPE_HOST_LOCAL | IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
  }

  iree_status_t status;
  void* host_ptr = NULL;
  CUdeviceptr device_ptr = 0;
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL)) {
    // Device local case.
    if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
      status =
          CU_RESULT_TO_STATUS(allocator->context->syms,
                              cuMemAllocManaged(&device_ptr, allocation_size,
                                                CU_MEM_ATTACH_GLOBAL));
      if (iree_status_is_ok(status)) {
        // Prefetch the buffer on the GPU device.
        status = CU_RESULT_TO_STATUS(
            allocator->context->syms,
            cuMemPrefetchAsync(device_ptr, allocation_size, allocator->device,
                               allocator->stream));
      }
      host_ptr = (void*)device_ptr;
    } else {
      // Device only.
      status = CU_RESULT_TO_STATUS(allocator->context->syms,
                                   cuMemAlloc(&device_ptr, allocation_size));
    }
  } else {
    unsigned int flags = CU_MEMHOSTALLOC_DEVICEMAP;
    if (!iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_CACHED)) {
      flags |= CU_MEMHOSTALLOC_WRITECOMBINED;
    }
    status =
        CU_RESULT_TO_STATUS(allocator->context->syms,
                            cuMemHostAlloc(&host_ptr, allocation_size, flags));
    if (iree_status_is_ok(status)) {
      status = CU_RESULT_TO_STATUS(
          allocator->context->syms,
          cuMemHostGetDevicePointer(&device_ptr, host_ptr, /*flags=*/0));
    }
  }

  iree_hal_buffer_t* buffer = NULL;
  if (iree_status_is_ok(status)) {
    status = iree_hal_cuda_buffer_wrap(
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
      iree_hal_cuda_buffer_free(allocator->context, memory_type, device_ptr,
                                host_ptr);
    } else {
      iree_hal_buffer_release(buffer);
    }
  }
  return status;
}

static iree_status_t iree_hal_cuda_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

static void iree_hal_cuda_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  iree_hal_cuda_allocator_t* allocator =
      iree_hal_cuda_allocator_cast(base_allocator);
  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  iree_hal_cuda_buffer_free(allocator->context, memory_type,
                            iree_hal_cuda_buffer_device_pointer(base_buffer),
                            iree_hal_cuda_buffer_host_pointer(base_buffer));

  IREE_STATISTICS(iree_hal_allocator_statistics_record_free(
      &allocator->statistics, memory_type,
      iree_hal_buffer_allocation_size(base_buffer)));

  iree_hal_buffer_destroy(base_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_cuda_allocator_vtable = {
    .destroy = iree_hal_cuda_allocator_destroy,
    .host_allocator = iree_hal_cuda_allocator_host_allocator,
    .trim = iree_hal_cuda_allocator_trim,
    .query_statistics = iree_hal_cuda_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_cuda_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_cuda_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_cuda_allocator_wrap_buffer,
    .deallocate_buffer = iree_hal_cuda_allocator_deallocate_buffer,
};
