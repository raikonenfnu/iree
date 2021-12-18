// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/allocator_caching.h"

#include "iree/base/tracing.h"
#include "iree/hal/detail.h"
#include "iree/hal/resource.h"

typedef struct iree_hal_buffer_node_t {
  iree_hal_buffer_t* cache_data;
  struct iree_hal_buffer_node_t* next;
} iree_hal_buffer_node_t;

typedef struct iree_hal_caching_allocator_t {
  iree_hal_resource_t resource;
  iree_hal_allocator_t* delegate_allocator;
  iree_string_view_t identifier;
  size_t cache_bucket_size;
  iree_hal_buffer_node_t* cache_list;
} iree_hal_caching_allocator_t;

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable;

iree_hal_caching_allocator_t* iree_hal_caching_allocator_cast(
    iree_hal_allocator_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_caching_allocator_vtable);
  return (iree_hal_caching_allocator_t*)base_value;
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_create_caching(
    iree_string_view_t identifier, iree_hal_allocator_t* delegate_allocator,
    iree_hal_allocator_t** out_allocator) {
  IREE_ASSERT_ARGUMENT(out_allocator);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_caching_allocator_t* allocator = NULL;
  iree_host_size_t total_size =
      iree_sizeof_struct(*allocator) + identifier.size;
  iree_status_t status = iree_allocator_malloc(
      iree_hal_allocator_host_allocator(delegate_allocator), total_size, (void**)&allocator);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_caching_allocator_vtable,
                                  &allocator->resource);

    allocator->delegate_allocator = delegate_allocator;
    allocator->cache_list = NULL;
    allocator->cache_bucket_size = 0;
    iree_string_view_append_to_buffer(
        identifier, &allocator->identifier,
        (char*)allocator + iree_sizeof_struct(*allocator));
    *out_allocator = (iree_hal_allocator_t*)allocator;
  }

  IREE_TRACE_ZONE_END(z0);
  return iree_ok_status();
}

static bool iree_hal_caching_allocator_buffer_available(iree_hal_caching_allocator_t* base_allocator,
    iree_host_size_t requested_size) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  // printf("Checking availability: begin\n");
  iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
  // printf("Checking availability: done reading\n");
  while(cache_list_ptr) {
    if(cache_list_ptr->cache_data->allocation_size >= requested_size) {
      return true;
    }
    cache_list_ptr = cache_list_ptr->next;
  }
  // printf("Checking availability: finish\n");
  return false;
}

static void iree_hal_caching_allocator_remove_buffer_from_cache(iree_hal_caching_allocator_t* allocator,
    int buffer_index_to_remove) {
      // TODO:Replace iree_hal_caching_buffer with iree_hal_buffer_node_t
  iree_hal_buffer_node_t** cache_list = &(allocator->cache_list);
  // printf("removing from cache: begin\n");
  iree_hal_buffer_node_t *cache_list_ptr = *cache_list;
  // printf("removing from cache: finish reading\n");
  // printf("removing from index:%d\n",buffer_index_to_remove);
  if (buffer_index_to_remove == 0) {
    *cache_list = cache_list_ptr->next;
    iree_allocator_free(iree_hal_allocator_host_allocator(allocator->delegate_allocator), cache_list_ptr);
    // printf("new ptr:%p\n",*cache_list);
    // printf("removing from cache: finish everything init\n");
    allocator->cache_bucket_size -= 1;
    // printf("new bucket size after removal:%d\n",allocator->cache_bucket_size);
    return;
  }
  // buffer index to remove 3
  // at 0: cache_list_ptr = 1
  // at 1: cache_list_ptr = 2
  // at 2: cache_list_ptr = 3
  for(int cur_idx = 0; cur_idx < buffer_index_to_remove-1; cur_idx++) {
    cache_list_ptr = cache_list_ptr->next;
  }
  iree_hal_buffer_node_t* next = cache_list_ptr->next->next;
  iree_allocator_free(iree_hal_allocator_host_allocator(allocator->delegate_allocator), cache_list_ptr->next);
  cache_list_ptr->next = next;
  allocator->cache_bucket_size -= 1;
  // printf("new bucket size after removal:%d\n",allocator->cache_bucket_size);
}

static iree_status_t iree_hal_caching_allocator_allocate_from_cache(
    iree_hal_caching_allocator_t* allocator, iree_host_size_t requested_size,
    iree_hal_buffer_t** out_buffer) {
  size_t buffer_index_in_cache = 0;
  // size_t buffer_index_in_cache = allocator->buffer_index;
  // printf("allocating from cache\n");
  iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
  while (cache_list_ptr) {
    if (cache_list_ptr->cache_data->allocation_size >= requested_size) {
      break;
    }
    // printf("nexting\n");
    cache_list_ptr = cache_list_ptr->next;
    buffer_index_in_cache++;
  }
  // printf("ptr to allocate: %p\n",cache_list_ptr->cache_data);
  // printf("ptr to allocate size: %ld\n",cache_list_ptr->cache_data->allocation_size);
  // *out_buffer = NULL;
  // *out_buffer = cache_list_ptr->cache_data;
  iree_hal_subspan_buffer_create(cache_list_ptr->cache_data, /*byte offset*/0,
    requested_size, allocator, iree_hal_allocator_host_allocator(allocator->delegate_allocator), out_buffer);
  // printf("allocating from cache: done\n");
  iree_hal_caching_allocator_remove_buffer_from_cache(allocator, buffer_index_in_cache);
  // printf("goodie!\n");
  return iree_ok_status();
}

IREE_API_EXPORT iree_status_t iree_hal_allocator_add_buffer_to_cache(iree_hal_buffer_t* buffer) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(buffer->device_allocator);
  // printf("Adding to cache: begin - %d\n", allocator->cache_bucket_size);
  iree_hal_buffer_node_t* cache_buffer_node = NULL;
  iree_status_t status =
    iree_allocator_malloc(iree_hal_allocator_host_allocator(allocator->delegate_allocator),
                          iree_sizeof_struct(*cache_buffer_node), (void**)&cache_buffer_node);
  if (!iree_status_is_ok(status)) return status;
  cache_buffer_node->cache_data = buffer;
  cache_buffer_node->next = NULL;
  if (!allocator->cache_list) {
    // printf("inserting at top\n");
    allocator->cache_list = cache_buffer_node;
  } else {
    iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
    // Add buffer to the end of list
    while(cache_list_ptr) {
      // printf("ptr1:%p\n",cache_list_ptr);
      // printf("ptr2:%p\n",cache_list_ptr->next);
      if(cache_list_ptr->next == NULL)
        break;
      else
        cache_list_ptr = cache_list_ptr->next;
    }
    cache_list_ptr->next = cache_buffer_node;
  }
  // printf("Adding to cache: finish %p\n",cache_buffer_node->cache_data);
  // printf("Adding to cache: finish size:%d\n",cache_buffer_node->cache_data->allocation_size);
  allocator->cache_bucket_size += 1;
  // printf("new bucket size after adding:%d\n",allocator->cache_bucket_size);
  return iree_ok_status();
}

static void iree_hal_caching_allocator_destroy(
    iree_hal_allocator_t* base_allocator) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_t* delegate_allocator = allocator->delegate_allocator;
  iree_hal_buffer_node_t* cache_list_ptr = allocator->cache_list;
  // printf("begin destruction\n");
  while (cache_list_ptr) {
    cache_list_ptr->cache_data->device_allocator = delegate_allocator;
    iree_hal_buffer_destroy(cache_list_ptr->cache_data);
    cache_list_ptr = cache_list_ptr->next;
  }
  iree_hal_allocator_destroy(allocator->delegate_allocator);
  // printf("end destruction\n");
}

static iree_allocator_t iree_hal_caching_allocator_host_allocator(
    const iree_hal_allocator_t* base_allocator) {
  iree_hal_caching_allocator_t* allocator = (iree_hal_caching_allocator_t*)base_allocator;
  return iree_hal_allocator_host_allocator(allocator->delegate_allocator);
}

static void iree_hal_caching_allocator_query_statistics(
    iree_hal_allocator_t* base_allocator,
    iree_hal_allocator_statistics_t* out_statistics) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  iree_hal_allocator_query_statistics(allocator->delegate_allocator, out_statistics);
}

static iree_hal_buffer_compatibility_t
iree_hal_caching_allocator_query_buffer_compatibility(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage,
    iree_hal_buffer_usage_t intended_usage,
    iree_device_size_t allocation_size) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  return iree_hal_allocator_query_buffer_compatibility(allocator->delegate_allocator,
    memory_type, allowed_usage, intended_usage, allocation_size);
}

static iree_status_t iree_hal_caching_allocator_allocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_buffer_usage_t allowed_usage, iree_host_size_t allocation_size,
    iree_hal_buffer_t** out_buffer) {
  iree_hal_caching_allocator_t* allocator = iree_hal_caching_allocator_cast(base_allocator);
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    if (iree_hal_caching_allocator_buffer_available(allocator, allocation_size)) {
        return iree_hal_caching_allocator_allocate_from_cache(allocator, allocation_size, out_buffer);
    }
  }
  iree_status_t status;
  status = iree_hal_allocator_allocate_buffer(allocator->delegate_allocator, memory_type,
      allowed_usage, allocation_size, out_buffer);
  (*out_buffer)->device_allocator = (iree_hal_allocator_t*)allocator;
  return status;
}

static iree_status_t iree_hal_caching_allocator_wrap_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_memory_type_t memory_type,
    iree_hal_memory_access_t allowed_access,
    iree_hal_buffer_usage_t allowed_usage, iree_byte_span_t data,
    iree_allocator_t data_allocator, iree_hal_buffer_t** out_buffer) {
  return iree_make_status(IREE_STATUS_UNAVAILABLE,
                          "wrapping of external buffers not supported");
}

static void iree_hal_caching_allocator_deallocate_buffer(
    iree_hal_allocator_t* base_allocator, iree_hal_buffer_t* base_buffer) {
  iree_hal_memory_type_t memory_type = iree_hal_buffer_memory_type(base_buffer);
  iree_hal_caching_allocator_t* allocator =
      iree_hal_caching_allocator_cast(base_allocator);
  if (iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL) &&
      iree_all_bits_set(memory_type, IREE_HAL_MEMORY_TYPE_HOST_VISIBLE)) {
    iree_status_t status = iree_hal_allocator_add_buffer_to_cache(base_buffer);
    // iree_hal_caching_allocator_remove_buffer_from_cache(allocator, (allocator->cache_bucket_size)-1);
    if (iree_status_is_ok(status)) {
      return;
    }
  }
  // printf("dealloc from real buffer!\n");
  iree_hal_allocator_deallocate_buffer(allocator->delegate_allocator, base_buffer);
}

static const iree_hal_allocator_vtable_t iree_hal_caching_allocator_vtable = {
    .destroy = iree_hal_caching_allocator_destroy,
    .host_allocator = iree_hal_caching_allocator_host_allocator,
    .query_statistics = iree_hal_caching_allocator_query_statistics,
    .query_buffer_compatibility =
        iree_hal_caching_allocator_query_buffer_compatibility,
    .allocate_buffer = iree_hal_caching_allocator_allocate_buffer,
    .wrap_buffer = iree_hal_caching_allocator_wrap_buffer,
    .deallocate_buffer = iree_hal_caching_allocator_deallocate_buffer,
};
