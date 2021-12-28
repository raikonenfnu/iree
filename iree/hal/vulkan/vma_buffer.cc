// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/vma_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/status_util.h"

typedef struct iree_hal_vulkan_vma_buffer_t {
  iree_hal_buffer_t base;

  VmaAllocator vma;
  VkBuffer handle;
  VmaAllocation allocation;
  VmaAllocationInfo allocation_info;
} iree_hal_vulkan_vma_buffer_t;

namespace {
extern const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable;
}  // namespace

static iree_hal_vulkan_vma_buffer_t* iree_hal_vulkan_vma_buffer_cast(
    iree_hal_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_vma_buffer_vtable);
  return (iree_hal_vulkan_vma_buffer_t*)base_value;
}

iree_status_t iree_hal_vulkan_vma_buffer_wrap(
    iree_hal_allocator_t* allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_device_size_t allocation_size,
    iree_device_size_t byte_offset, iree_device_size_t byte_length,
    VmaAllocator vma, VkBuffer handle, VmaAllocation allocation,
    VmaAllocationInfo allocation_info, iree_hal_buffer_t** out_buffer) {
  IREE_ASSERT_ARGUMENT(allocator);
  IREE_ASSERT_ARGUMENT(vma);
  IREE_ASSERT_ARGUMENT(handle);
  IREE_ASSERT_ARGUMENT(allocation);
  IREE_ASSERT_ARGUMENT(out_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_t host_allocator =
      iree_hal_allocator_host_allocator(allocator);
  iree_hal_vulkan_vma_buffer_t* buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(host_allocator, sizeof(*buffer), (void**)&buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_buffer_initialize(
        host_allocator, allocator, &buffer->base, allocation_size, byte_offset,
        byte_length, memory_type, allowed_access, allowed_usage,
        &iree_hal_vulkan_vma_buffer_vtable, &buffer->base);
    buffer->vma = vma;
    buffer->handle = handle;
    buffer->allocation = allocation;
    buffer->allocation_info = allocation_info;

    // TODO(benvanik): set debug name instead and use the
    //     VMA_ALLOCATION_CREATE_USER_DATA_COPY_STRING_BIT flag.
    vmaSetAllocationUserData(buffer->vma, buffer->allocation, buffer);

    // TODO(benvanik): figure out why this is not working - has unbalanced
    // allocs in the tracy UI even though they are definitely balanced here.
    // IREE_TRACE_ALLOC_NAMED("VMA", (void*)buffer->handle, byte_length);

    *out_buffer = &buffer->base;
  } else {
    vmaDestroyBuffer(vma, handle, allocation);
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static void iree_hal_vulkan_vma_buffer_destroy(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  iree_allocator_t host_allocator = base_buffer->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  // IREE_TRACE_FREE_NAMED("VMA", (void*)buffer->handle);

  vmaDestroyBuffer(buffer->vma, buffer->handle, buffer->allocation);
  iree_allocator_free(host_allocator, buffer);

  IREE_TRACE_ZONE_END(z0);
}

VkBuffer iree_hal_vulkan_vma_buffer_handle(iree_hal_buffer_t* base_buffer) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  return buffer->handle;
}

static iree_status_t iree_hal_vulkan_vma_buffer_map_range(
    iree_hal_buffer_t* base_buffer, iree_hal_mapping_mode_t mapping_mode,
    iree_hal_memory_access_t memory_access,
    iree_device_size_t local_byte_offset, iree_device_size_t local_byte_length,
    void** out_data_ptr) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);

  uint8_t* data_ptr = nullptr;
  VK_RETURN_IF_ERROR(
      vmaMapMemory(buffer->vma, buffer->allocation, (void**)&data_ptr),
      "vmaMapMemory");
  *out_data_ptr = data_ptr + local_byte_offset;

  // If we mapped for discard scribble over the bytes. This is not a mandated
  // behavior but it will make debugging issues easier. Alternatively for
  // heap buffers we could reallocate them such that ASAN yells, but that
  // would only work if the entire buffer was discarded.
#ifndef NDEBUG
  if (iree_any_bit_set(memory_access, IREE_HAL_MEMORY_ACCESS_DISCARD)) {
    memset(*out_data_ptr, 0xCD, local_byte_length);
  }
#endif  // !NDEBUG

  return iree_ok_status();
}

static void iree_hal_vulkan_vma_buffer_unmap_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length, void* data_ptr) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  vmaUnmapMemory(buffer->vma, buffer->allocation);
}

static iree_status_t iree_hal_vulkan_vma_buffer_invalidate_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(
      vmaInvalidateAllocation(buffer->vma, buffer->allocation,
                              local_byte_offset, local_byte_length),
      "vmaInvalidateAllocation");
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_vma_buffer_flush_range(
    iree_hal_buffer_t* base_buffer, iree_device_size_t local_byte_offset,
    iree_device_size_t local_byte_length) {
  iree_hal_vulkan_vma_buffer_t* buffer =
      iree_hal_vulkan_vma_buffer_cast(base_buffer);
  VK_RETURN_IF_ERROR(vmaFlushAllocation(buffer->vma, buffer->allocation,
                                        local_byte_offset, local_byte_length),
                     "vmaFlushAllocation");
  return iree_ok_status();
}

namespace {
const iree_hal_buffer_vtable_t iree_hal_vulkan_vma_buffer_vtable = {
    /*.destroy=*/iree_hal_vulkan_vma_buffer_destroy,
    /*.map_range=*/iree_hal_vulkan_vma_buffer_map_range,
    /*.unmap_range=*/iree_hal_vulkan_vma_buffer_unmap_range,
    /*.invalidate_range=*/iree_hal_vulkan_vma_buffer_invalidate_range,
    /*.flush_range=*/iree_hal_vulkan_vma_buffer_flush_range,
};
}  // namespace
