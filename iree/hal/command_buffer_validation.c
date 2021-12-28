// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/command_buffer_validation.h"

#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "iree/base/tracing.h"
#include "iree/hal/allocator.h"
#include "iree/hal/buffer.h"
#include "iree/hal/descriptor_set.h"
#include "iree/hal/detail.h"
#include "iree/hal/device.h"
#include "iree/hal/event.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_layout.h"
#include "iree/hal/resource.h"

#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
#define VALIDATION_STATE(command_buffer) (&(command_buffer)->validation)
#else
#define VALIDATION_STATE(command_buffer) \
  ((iree_hal_command_buffer_validation_state_t*)NULL)
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE

// Returns success iff the queue supports the given command categories.
static iree_status_t iree_hal_command_buffer_validate_categories(
    const iree_hal_command_buffer_t* command_buffer,
    iree_hal_command_category_t required_categories) {
  if (!iree_all_bits_set(command_buffer->allowed_categories,
                         required_categories)) {
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t required_categories_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_command_category_format(required_categories, &temp0);
    iree_string_view_t allowed_categories_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_command_category_format(command_buffer->allowed_categories,
                                         &temp1);
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "operation requires categories %.*s but command buffer only supports "
        "%.*s",
        (int)required_categories_str.size, required_categories_str.data,
        (int)allowed_categories_str.size, allowed_categories_str.data);
  }
  return iree_ok_status();
}

// Returns success iff the buffer is compatible with the device.
static iree_status_t iree_hal_command_buffer_validate_buffer_compatibility(
    const iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer,
    iree_hal_buffer_compatibility_t required_compatibility,
    iree_hal_buffer_usage_t intended_usage) {
  iree_hal_buffer_compatibility_t allowed_compatibility =
      iree_hal_allocator_query_buffer_compatibility(
          iree_hal_device_allocator(VALIDATION_STATE(command_buffer)->device),
          iree_hal_buffer_memory_type(buffer),
          iree_hal_buffer_allowed_usage(buffer), intended_usage,
          iree_hal_buffer_allocation_size(buffer));
  if (!iree_all_bits_set(allowed_compatibility, required_compatibility)) {
    // Buffer cannot be used on the queue for the given usage.
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t allowed_usage_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_buffer_usage_format(iree_hal_buffer_allowed_usage(buffer),
                                     &temp0);
    iree_string_view_t intended_usage_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_buffer_usage_format(intended_usage, &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "requested buffer usage is not supported for the buffer on this queue; "
        "buffer allows %.*s, operation requires %.*s",
        (int)allowed_usage_str.size, allowed_usage_str.data,
        (int)intended_usage_str.size, intended_usage_str.data);
  }
  return iree_ok_status();
}

// Returns success iff the currently bound descriptor sets are valid for the
// given executable entry point.
static iree_status_t iree_hal_command_buffer_validate_dispatch_bindings(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point) {
  // TODO(benvanik): validate buffers referenced have compatible memory types
  // and access rights.
  // TODO(benvanik): validate no aliasing between inputs/outputs.
  return iree_ok_status();
}

void iree_hal_command_buffer_initialize_validation(
    iree_hal_device_t* device, iree_hal_command_buffer_t* command_buffer) {
  VALIDATION_STATE(command_buffer)->device = device;
  VALIDATION_STATE(command_buffer)->is_recording = false;
}

iree_status_t iree_hal_command_buffer_begin_validation(
    iree_hal_command_buffer_t* command_buffer) {
  if (VALIDATION_STATE(command_buffer)->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is already in a recording state");
  }
  VALIDATION_STATE(command_buffer)->is_recording = true;
  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_end_validation(
    iree_hal_command_buffer_t* command_buffer) {
  if (VALIDATION_STATE(command_buffer)->debug_group_depth != 0) {
    return iree_make_status(
        IREE_STATUS_FAILED_PRECONDITION,
        "unbalanced debug group depth (expected 0, is %d)",
        VALIDATION_STATE(command_buffer)->debug_group_depth);
  }
  if (!VALIDATION_STATE(command_buffer)->is_recording) {
    return iree_make_status(IREE_STATUS_FAILED_PRECONDITION,
                            "command buffer is not in a recording state");
  }
  VALIDATION_STATE(command_buffer)->is_recording = false;
  return iree_ok_status();
}

void iree_hal_command_buffer_begin_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location) {
  ++VALIDATION_STATE(command_buffer)->debug_group_depth;
}

void iree_hal_command_buffer_end_debug_group_validation(
    iree_hal_command_buffer_t* command_buffer) {
  --VALIDATION_STATE(command_buffer)->debug_group_depth;
}

iree_status_t iree_hal_command_buffer_execution_barrier_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // NOTE: all command buffer types can perform this so no need to check.

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_signal_event_validation(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_reset_event_validation(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_wait_events_validation(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): additional synchronization validation.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_discard_buffer_validation(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_fill_buffer_validation(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  // Ensure the value length is supported.
  if (pattern_length != 1 && pattern_length != 2 && pattern_length != 4) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "fill value length is not one of the supported "
                            "values (pattern_length=%zu)",
                            pattern_length);
  }

  // Ensure the offset and length have an alignment matching the value length.
  if ((target_offset % pattern_length) != 0 || (length % pattern_length) != 0) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "fill offset and/or length do not match the natural alignment of the "
        "fill value (target_offset=%" PRIdsz ", length=%" PRIdsz
        ", pattern_length=%zu)",
        target_offset, length, pattern_length);
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_update_buffer_validation(
    iree_hal_command_buffer_t* command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(target_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_copy_buffer_validation(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, source_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, target_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_TRANSFER,
      IREE_HAL_BUFFER_USAGE_TRANSFER));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(source_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(source_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(source_buffer, source_offset, length));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(target_buffer),
      IREE_HAL_BUFFER_USAGE_TRANSFER));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(target_buffer),
      IREE_HAL_MEMORY_ACCESS_WRITE));
  IREE_RETURN_IF_ERROR(
      iree_hal_buffer_validate_range(target_buffer, target_offset, length));

  // At least source or destination must be device-visible to enable
  // host->device, device->host, and device->device.
  // TODO(b/117338171): host->host copies.
  if (!iree_any_bit_set(iree_hal_buffer_memory_type(source_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE) &&
      !iree_any_bit_set(iree_hal_buffer_memory_type(target_buffer),
                        IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE)) {
    iree_bitfield_string_temp_t temp0, temp1;
    iree_string_view_t source_memory_type_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_type_format(iree_hal_buffer_memory_type(source_buffer),
                                    &temp0);
    iree_string_view_t target_memory_type_str IREE_ATTRIBUTE_UNUSED =
        iree_hal_memory_type_format(iree_hal_buffer_memory_type(target_buffer),
                                    &temp1);
    return iree_make_status(
        IREE_STATUS_PERMISSION_DENIED,
        "at least one buffer must be device-visible for a copy; "
        "source_buffer=%.*s, target_buffer=%.*s",
        (int)source_memory_type_str.size, source_memory_type_str.data,
        (int)target_memory_type_str.size, target_memory_type_str.data);
  }

  // Check for overlap - just like memcpy we don't handle that.
  if (iree_hal_buffer_test_overlap(source_buffer, source_offset, length,
                                   target_buffer, target_offset, length) !=
      IREE_HAL_BUFFER_OVERLAP_DISJOINT) {
    return iree_make_status(
        IREE_STATUS_INVALID_ARGUMENT,
        "source and target ranges overlap within the same buffer");
  }

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_push_constants_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  if (IREE_UNLIKELY((values_length % 4) != 0)) {
    return iree_make_status(IREE_STATUS_INVALID_ARGUMENT,
                            "invalid alignment %zu, must be 4-byte aligned",
                            values_length);
  }

  // TODO(benvanik): validate offset and value count with layout.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_push_descriptor_set_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate binding_offset.
  // TODO(benvanik): validate bindings.

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_bind_descriptor_set_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));

  // TODO(benvanik): validate set index.
  // TODO(benvanik): validate dynamic offsets (both count and offsets).

  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_dispatch_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, executable, entry_point));
  return iree_ok_status();
}

iree_status_t iree_hal_command_buffer_dispatch_indirect_validation(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_categories(
      command_buffer, IREE_HAL_COMMAND_CATEGORY_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_buffer_compatibility(
      command_buffer, workgroups_buffer,
      IREE_HAL_BUFFER_COMPATIBILITY_QUEUE_DISPATCH,
      IREE_HAL_BUFFER_USAGE_DISPATCH));

  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_memory_type(
      iree_hal_buffer_memory_type(workgroups_buffer),
      IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_access(
      iree_hal_buffer_allowed_access(workgroups_buffer),
      IREE_HAL_MEMORY_ACCESS_READ));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_usage(
      iree_hal_buffer_allowed_usage(workgroups_buffer),
      IREE_HAL_BUFFER_USAGE_DISPATCH));
  IREE_RETURN_IF_ERROR(iree_hal_buffer_validate_range(
      workgroups_buffer, workgroups_offset, sizeof(uint32_t) * 3));

  IREE_RETURN_IF_ERROR(iree_hal_command_buffer_validate_dispatch_bindings(
      command_buffer, executable, entry_point));

  return iree_ok_status();
}
