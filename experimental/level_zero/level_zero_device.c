// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "experimental/level_zero/level_zero_device.h"

#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "experimental/level_zero/context_wrapper.h"
#include "experimental/level_zero/descriptor_set_layout.h"
#include "experimental/level_zero/direct_command_buffer.h"
#include "experimental/level_zero/dynamic_symbols.h"
#include "experimental/level_zero/event_semaphore.h"
#include "experimental/level_zero/executable_layout.h"
#include "experimental/level_zero/level_zero_allocator.h"
#include "experimental/level_zero/level_zero_event.h"
#include "experimental/level_zero/nop_executable_cache.h"
#include "experimental/level_zero/status_util.h"
#include "iree/base/internal/arena.h"
#include "iree/base/tracing.h"
#include "iree/hal/utils/buffer_transfer.h"

//===----------------------------------------------------------------------===//
// iree_hal_level_zero_device_t
//===----------------------------------------------------------------------===//

typedef struct iree_hal_level_zero_device_t {
  iree_hal_resource_t resource;
  iree_string_view_t identifier;

  // Block pool used for command buffers with a larger block size (as command
  // buffers can contain inlined data uploads).
  iree_arena_block_pool_t block_pool;

  // Optional driver that owns the Level Zero symbols. We retain it for our
  // lifetime to ensure the symbols remains valid.
  iree_hal_driver_t* driver;

  // Level Zero APIs.
  ze_device_handle_t device;
  uint32_t command_queue_ordinal;
  ze_command_queue_handle_t command_queue;

  iree_hal_level_zero_context_wrapper_t context_wrapper;
  iree_hal_allocator_t* device_allocator;

} iree_hal_level_zero_device_t;

static const iree_hal_device_vtable_t iree_hal_level_zero_device_vtable;

static iree_hal_level_zero_device_t* iree_hal_level_zero_device_cast(
    iree_hal_device_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_level_zero_device_vtable);
  return (iree_hal_level_zero_device_t*)base_value;
}

static void iree_hal_level_zero_device_destroy(iree_hal_device_t* base_device) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  iree_allocator_t host_allocator = iree_hal_device_host_allocator(base_device);
  IREE_TRACE_ZONE_BEGIN(z0);

  // There should be no more buffers live that use the allocator.
  iree_hal_allocator_release(device->device_allocator);
  LEVEL_ZERO_IGNORE_ERROR(device->context_wrapper.syms,
                          zeCommandQueueDestroy(device->command_queue));
  // Finally, destroy the device.
  iree_hal_driver_release(device->driver);

  iree_allocator_free(host_allocator, device);

  IREE_TRACE_ZONE_END(z0);
}

static iree_status_t iree_hal_level_zero_device_create_internal(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    ze_device_handle_t level_zero_device, uint32_t command_queue_ordinal,
    ze_command_queue_handle_t command_queue,
    ze_context_handle_t level_zero_context,
    iree_hal_level_zero_dynamic_symbols_t* syms,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_level_zero_device_t* device = NULL;
  iree_host_size_t total_size = sizeof(*device) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&device));
  memset(device, 0, total_size);
  iree_hal_resource_initialize(&iree_hal_level_zero_device_vtable,
                               &device->resource);
  device->driver = driver;
  iree_hal_driver_retain(device->driver);
  uint8_t* buffer_ptr = (uint8_t*)device + sizeof(*device);
  buffer_ptr += iree_string_view_append_to_buffer(
      identifier, &device->identifier, (char*)buffer_ptr);
  device->device = level_zero_device;
  device->command_queue_ordinal = command_queue_ordinal;
  device->command_queue = command_queue;
  device->context_wrapper.level_zero_context = level_zero_context;
  device->context_wrapper.host_allocator = host_allocator;
  device->context_wrapper.syms = syms;
  iree_status_t status = iree_hal_level_zero_allocator_create(
      (iree_hal_device_t*)device, device->device, &device->context_wrapper,
      &device->device_allocator);
  if (iree_status_is_ok(status)) {
    *out_device = (iree_hal_device_t*)device;
  } else {
    iree_hal_device_release((iree_hal_device_t*)device);
  }
  return status;
}

iree_status_t iree_hal_level_zero_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    iree_hal_level_zero_dynamic_symbols_t* syms,
    ze_device_handle_t level_zero_device,
    ze_context_handle_t level_zero_context, iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {
  IREE_TRACE_ZONE_BEGIN(z0);
  // Create a command queue
  uint32_t num_queue_groups = 0;
  iree_status_t status = LEVEL_ZERO_RESULT_TO_STATUS(
      syms,
      zeDeviceGetCommandQueueGroupProperties(level_zero_device,
                                             &num_queue_groups, NULL),
      "zeDeviceGetCommandQueueGroupProperties");
  if (num_queue_groups == 0) {
    return iree_make_status(IREE_STATUS_NOT_FOUND, "No queue groups found");
  }
  ze_command_queue_group_properties_t* queue_properties =
      (ze_command_queue_group_properties_t*)malloc(
          num_queue_groups * sizeof(ze_command_queue_group_properties_t));
  status = LEVEL_ZERO_RESULT_TO_STATUS(
      syms,
      zeDeviceGetCommandQueueGroupProperties(
          level_zero_device, &num_queue_groups, queue_properties),
      "zeDeviceGetCommandQueueGroupProperties");

  ze_command_queue_desc_t command_queue_desc = {};
  for (uint32_t i = 0; i < num_queue_groups; i++) {
    if (queue_properties[i].flags &
        ZE_COMMAND_QUEUE_GROUP_PROPERTY_FLAG_COMPUTE) {
      command_queue_desc.ordinal = i;
    }
  }
  command_queue_desc.index = 0;
  command_queue_desc.mode = ZE_COMMAND_QUEUE_MODE_ASYNCHRONOUS;
  ze_command_queue_handle_t command_queue;
  status = LEVEL_ZERO_RESULT_TO_STATUS(
      syms,
      zeCommandQueueCreate(level_zero_context, level_zero_device,
                           &command_queue_desc, &command_queue),
      "zeCommandQueueCreate");

  if (iree_status_is_ok(status)) {
    status = iree_hal_level_zero_device_create_internal(
        driver, identifier, level_zero_device, command_queue_desc.ordinal,
        command_queue, level_zero_context, syms, host_allocator, out_device);
  }
  if (!iree_status_is_ok(status)) {
    syms->zeCommandQueueDestroy(command_queue);
    syms->zeContextDestroy(level_zero_context);
  }
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static iree_string_view_t iree_hal_level_zero_device_id(
    iree_hal_device_t* base_device) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return device->identifier;
}

static iree_allocator_t iree_hal_level_zero_device_host_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return device->context_wrapper.host_allocator;
}

static iree_hal_allocator_t* iree_hal_level_zero_device_allocator(
    iree_hal_device_t* base_device) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return device->device_allocator;
}

static iree_status_t iree_hal_level_zero_device_query_i64(
    iree_hal_device_t* base_device, iree_string_view_t category,
    iree_string_view_t key, int64_t* out_value) {
  // iree_hal_level_zero_device_t* device =
  // iree_hal_level_zero_device_cast(base_device);
  *out_value = 0;

  if (iree_string_view_equal(category,
                             iree_make_cstring_view("hal.executable.format"))) {
    *out_value =
        iree_string_view_equal(key, iree_make_cstring_view("opencl-spirv-fb"))
            ? 1
            : 0;
    return iree_ok_status();
  }

  return iree_make_status(
      IREE_STATUS_NOT_FOUND,
      "unknown device configuration key value '%.*s :: %.*s'",
      (int)category.size, category.data, (int)key.size, key.data);
}

static iree_status_t iree_hal_level_zero_device_trim(
    iree_hal_device_t* base_device) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  iree_arena_block_pool_trim(&device->block_pool);
  return iree_hal_allocator_trim(device->device_allocator);
}

static iree_status_t iree_hal_level_zero_device_create_command_buffer(
    iree_hal_device_t* base_device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_direct_command_buffer_create(
      base_device, &device->context_wrapper, mode, command_categories,
      queue_affinity, &device->block_pool, device->device,
      device->command_queue_ordinal, out_command_buffer);
}

static iree_status_t iree_hal_level_zero_device_create_descriptor_set(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_t* set_layout,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings,
    iree_hal_descriptor_set_t** out_descriptor_set) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "non-push descriptor sets still need work");
}

static iree_status_t iree_hal_level_zero_device_create_descriptor_set_layout(
    iree_hal_device_t* base_device,
    iree_hal_descriptor_set_layout_usage_type_t usage_type,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_layout_binding_t* bindings,
    iree_hal_descriptor_set_layout_t** out_descriptor_set_layout) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_descriptor_set_layout_create(
      &device->context_wrapper, usage_type, binding_count, bindings,
      out_descriptor_set_layout);
}

static iree_status_t iree_hal_level_zero_device_create_event(
    iree_hal_device_t* base_device, iree_hal_event_t** out_event) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_event_create(&device->context_wrapper, out_event);
}

static iree_status_t iree_hal_level_zero_device_create_executable_cache(
    iree_hal_device_t* base_device, iree_string_view_t identifier,
    iree_loop_t loop, iree_hal_executable_cache_t** out_executable_cache) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_nop_executable_cache_create(
      &device->context_wrapper, identifier, device->device,
      out_executable_cache);
}

static iree_status_t iree_hal_level_zero_device_create_executable_layout(
    iree_hal_device_t* base_device, iree_host_size_t push_constants,
    iree_host_size_t set_layout_count,
    iree_hal_descriptor_set_layout_t** set_layouts,
    iree_hal_executable_layout_t** out_executable_layout) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_executable_layout_create(
      &device->context_wrapper, set_layout_count, set_layouts, push_constants,
      out_executable_layout);
}

static iree_status_t iree_hal_level_zero_device_create_semaphore(
    iree_hal_device_t* base_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  return iree_hal_level_zero_semaphore_create(&device->context_wrapper,
                                              initial_value, out_semaphore);
}

static iree_hal_semaphore_compatibility_t
iree_hal_level_zero_device_query_semaphore_compatibility(
    iree_hal_device_t* base_device, iree_hal_semaphore_t* semaphore) {
  // TODO: implement Level Zero semaphores.
  return IREE_HAL_SEMAPHORE_COMPATIBILITY_HOST_ONLY;
}

static iree_status_t iree_hal_level_zero_device_queue_submit(
    iree_hal_device_t* base_device,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity, iree_host_size_t batch_count,
    const iree_hal_submission_batch_t* batches) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  // TODO(raikonenfnu): Once semaphore is implemented wait for semaphores
  // TODO(thomasraoux): implement semaphores - for now this conservatively
  // synchronizes after every submit.
  for (int i = 0; i < batch_count; i++) {
    for (int j = 0; j < batches[i].command_buffer_count; j++) {
      ze_command_list_handle_t command_list =
          iree_hal_level_zero_direct_command_buffer_exec(
              batches[i].command_buffers[j]);
      LEVEL_ZERO_RETURN_IF_ERROR(device->context_wrapper.syms,
                                 zeCommandListClose(command_list),
                                 "zeCommandListClose");
      LEVEL_ZERO_RETURN_IF_ERROR(
          device->context_wrapper.syms,
          zeCommandQueueExecuteCommandLists(device->command_queue, 1,
                                            &command_list, NULL),
          "zeCommandQueueExecuteCommandLists");
    }
  }

  LEVEL_ZERO_RETURN_IF_ERROR(
      device->context_wrapper.syms,
      zeCommandQueueSynchronize(device->command_queue, IREE_DURATION_INFINITE),
      "zeCommandQueueSynchronize");
  return iree_ok_status();
}

static iree_status_t iree_hal_level_zero_device_wait_semaphores(
    iree_hal_device_t* base_device, iree_hal_wait_mode_t wait_mode,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "semaphore not implemented");
}

static iree_status_t iree_hal_level_zero_device_wait_idle(
    iree_hal_device_t* base_device, iree_timeout_t timeout) {
  iree_hal_level_zero_device_t* device =
      iree_hal_level_zero_device_cast(base_device);
  // Wait until the stream is done.
  iree_time_t timeout_ns = iree_timeout_as_deadline_ns(timeout);
  LEVEL_ZERO_RETURN_IF_ERROR(
      device->context_wrapper.syms,
      zeCommandQueueSynchronize(device->command_queue, timeout_ns),
      "zeCommandQueueSynchronize");
  return iree_ok_status();
}

static const iree_hal_device_vtable_t iree_hal_level_zero_device_vtable = {
    .destroy = iree_hal_level_zero_device_destroy,
    .id = iree_hal_level_zero_device_id,
    .host_allocator = iree_hal_level_zero_device_host_allocator,
    .device_allocator = iree_hal_level_zero_device_allocator,
    .trim = iree_hal_level_zero_device_trim,
    .query_i64 = iree_hal_level_zero_device_query_i64,
    .create_command_buffer = iree_hal_level_zero_device_create_command_buffer,
    .create_descriptor_set = iree_hal_level_zero_device_create_descriptor_set,
    .create_descriptor_set_layout =
        iree_hal_level_zero_device_create_descriptor_set_layout,
    .create_event = iree_hal_level_zero_device_create_event,
    .create_executable_cache =
        iree_hal_level_zero_device_create_executable_cache,
    .create_executable_layout =
        iree_hal_level_zero_device_create_executable_layout,
    .create_semaphore = iree_hal_level_zero_device_create_semaphore,
    .query_semaphore_compatibility =
        iree_hal_level_zero_device_query_semaphore_compatibility,
    .transfer_range = iree_hal_device_submit_transfer_range_and_wait,
    .queue_submit = iree_hal_level_zero_device_queue_submit,
    .wait_semaphores = iree_hal_level_zero_device_wait_semaphores,
    .wait_idle = iree_hal_level_zero_device_wait_idle,
};
