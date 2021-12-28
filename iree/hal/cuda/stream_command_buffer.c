// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/cuda/stream_command_buffer.h"

#include "iree/base/tracing.h"
#include "iree/hal/cuda/cuda_buffer.h"
#include "iree/hal/cuda/cuda_event.h"
#include "iree/hal/cuda/executable_layout.h"
#include "iree/hal/cuda/native_executable.h"
#include "iree/hal/cuda/status_util.h"

#define IREE_HAL_CUDA_MAX_BINDING_COUNT 64
// Kernel arguments contains binding and push constants.
#define IREE_HAL_CUDA_MAX_KERNEL_ARG 128
// This records the commands on the calling thread without additional threading
// indirection.

typedef struct {
  iree_hal_command_buffer_t base;
  iree_hal_cuda_context_wrapper_t* context;
  CUstream stream;

  int32_t push_constant[IREE_HAL_CUDA_MAX_PUSH_CONSTANT_COUNT];
  // Keep track of the current set of kernel arguments.
  void* current_descriptor[IREE_HAL_CUDA_MAX_KERNEL_ARG];
  CUdeviceptr* device_ptrs[IREE_HAL_CUDA_MAX_KERNEL_ARG];
} iree_hal_cuda_stream_command_buffer_t;

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable;

static iree_hal_cuda_stream_command_buffer_t*
iree_hal_cuda_stream_command_buffer_cast(
    iree_hal_command_buffer_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_cuda_stream_command_buffer_vtable);
  return (iree_hal_cuda_stream_command_buffer_t*)base_value;
}

iree_status_t iree_hal_cuda_stream_command_buffer_create(
    iree_hal_device_t* device, iree_hal_cuda_context_wrapper_t* context,
    iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories, CUstream stream,
    iree_hal_command_buffer_t** out_command_buffer) {
  IREE_ASSERT_ARGUMENT(context);
  IREE_ASSERT_ARGUMENT(out_command_buffer);
  *out_command_buffer = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_cuda_stream_command_buffer_t* command_buffer = NULL;
  iree_status_t status =
      iree_allocator_malloc(context->host_allocator, sizeof(*command_buffer),
                            (void**)&command_buffer);
  if (iree_status_is_ok(status)) {
    iree_hal_command_buffer_initialize(
        device, mode, command_categories, IREE_HAL_QUEUE_AFFINITY_ANY,
        &iree_hal_cuda_stream_command_buffer_vtable, &command_buffer->base);
    command_buffer->context = context;
    command_buffer->stream = stream;
    for (size_t i = 0; i < IREE_HAL_CUDA_MAX_KERNEL_ARG; i++) {
      command_buffer->current_descriptor[i] = &command_buffer->device_ptrs[i];
    }
  }

  *out_command_buffer = &command_buffer->base;
  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_cuda_stream_command_buffer_destroy(
    iree_hal_command_buffer_t* base_command_buffer) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_allocator_free(command_buffer->context->host_allocator, command_buffer);

  IREE_TRACE_ZONE_END(z0);
}

bool iree_hal_cuda_stream_command_buffer_isa(
    iree_hal_command_buffer_t* command_buffer) {
  return iree_hal_command_buffer_dyn_cast(
      command_buffer, &iree_hal_cuda_stream_command_buffer_vtable);
}

static void* iree_hal_cuda_stream_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable) {
  if (vtable == &iree_hal_cuda_stream_command_buffer_vtable) {
    IREE_HAL_ASSERT_TYPE(command_buffer, vtable);
    return command_buffer;
  }
  return NULL;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_begin(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_end(
    iree_hal_command_buffer_t* base_command_buffer) {
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_signal_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_reset_event(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask) {
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_wait_events(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_host_size_t event_count, const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers) {
  // TODO(jinchen62): implement CUDA barrier
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* base_command_buffer, iree_hal_buffer_t* buffer) {
  // nothing to do.
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length, const void* pattern,
    iree_host_size_t pattern_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  size_t num_elements = length / pattern_length;
  switch (pattern_length) {
    case 4: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD32Async(dst, *(const uint32_t*)(pattern), num_elements,
                           command_buffer->stream),
          "cuMemsetD32Async");
      break;
    }
    case 2: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD16Async(dst, *(const uint16_t*)(pattern), num_elements,
                           command_buffer->stream),
          "cuMemsetD16Async");
      break;
    }
    case 1: {
      CUDA_RETURN_IF_ERROR(
          command_buffer->context->syms,
          cuMemsetD8Async(dst, *(const uint8_t*)(pattern), num_elements,
                          command_buffer->stream),
          "cuMemsetD8Async");
      break;
    }
    default:
      return iree_make_status(IREE_STATUS_INTERNAL,
                              "unsupported fill pattern length");
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_update_buffer(
    iree_hal_command_buffer_t* base_command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation of update buffer");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
    iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
    iree_device_size_t length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);

  CUdeviceptr target_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(target_buffer));
  target_offset += iree_hal_buffer_byte_offset(target_buffer);
  CUdeviceptr source_device_buffer = iree_hal_cuda_buffer_device_pointer(
      iree_hal_buffer_allocated_buffer(source_buffer));
  source_offset += iree_hal_buffer_byte_offset(source_buffer);
  CUdeviceptr dst = target_device_buffer + target_offset;
  CUdeviceptr src = source_device_buffer + source_offset;
  CUDA_RETURN_IF_ERROR(command_buffer->context->syms,
                       cuMemcpyAsync(dst, src, length, command_buffer->stream),
                       "cuMemcpyAsync");
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_constants(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  iree_host_size_t constant_base_index = offset / sizeof(int32_t);
  for (iree_host_size_t i = 0; i < values_length / sizeof(int32_t); i++) {
    command_buffer->push_constant[i + constant_base_index] =
        ((uint32_t*)values)[i];
  }
  return iree_ok_status();
}

// Tie together the binding index and its index in |bindings| array.
typedef struct {
  uint32_t index;
  uint32_t binding;
} iree_hal_cuda_binding_mapping_t;

// Helper to sort the binding based on their binding index.
static int compare_binding_index(const void* a, const void* b) {
  const iree_hal_cuda_binding_mapping_t buffer_a =
      *(const iree_hal_cuda_binding_mapping_t*)a;
  const iree_hal_cuda_binding_mapping_t buffer_b =
      *(const iree_hal_cuda_binding_mapping_t*)b;
  return buffer_a.binding < buffer_b.binding ? -1 : 1;
}

static iree_status_t iree_hal_cuda_stream_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  iree_host_size_t base_binding =
      iree_hal_cuda_base_binding_index(executable_layout, set);
  // Convention with the compiler side. We map bindings to kernel argument.
  // We compact the bindings to get a dense set of arguments and keep them order
  // based on the binding index.
  // Sort the binding based on the binding index and map the array index to the
  // argument index.
  iree_hal_cuda_binding_mapping_t binding_used[IREE_HAL_CUDA_MAX_BINDING_COUNT];
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_cuda_binding_mapping_t buffer = {i, bindings[i].binding};
    binding_used[i] = buffer;
  }
  qsort(binding_used, binding_count, sizeof(iree_hal_cuda_binding_mapping_t),
        compare_binding_index);
  assert(binding_count < IREE_HAL_CUDA_MAX_BINDING_COUNT &&
         "binding count larger than the max expected.");
  for (iree_host_size_t i = 0; i < binding_count; i++) {
    iree_hal_descriptor_set_binding_t binding = bindings[binding_used[i].index];
    CUdeviceptr device_ptr =
        iree_hal_cuda_buffer_device_pointer(
            iree_hal_buffer_allocated_buffer(binding.buffer)) +
        iree_hal_buffer_byte_offset(binding.buffer) + binding.offset;
    *((CUdeviceptr*)command_buffer->current_descriptor[i + base_binding]) =
        device_ptr;
  }
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation of bind descriptor set");
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z) {
  iree_hal_cuda_stream_command_buffer_t* command_buffer =
      iree_hal_cuda_stream_command_buffer_cast(base_command_buffer);
  iree_hal_executable_layout_t* layout =
      iree_hal_cuda_executable_get_layout(executable, entry_point);
  iree_host_size_t num_constants =
      iree_hal_cuda_executable_layout_num_constants(layout);
  iree_host_size_t constant_base_index =
      iree_hal_cuda_push_constant_index(layout);
  // Patch the push constants in the kernel arguments.
  for (iree_host_size_t i = 0; i < num_constants; i++) {
    *((uint32_t*)command_buffer->current_descriptor[i + constant_base_index]) =
        command_buffer->push_constant[i];
  }

  int32_t block_size_x, block_size_y, block_size_z;
  IREE_RETURN_IF_ERROR(iree_hal_cuda_native_executable_block_size(
      executable, entry_point, &block_size_x, &block_size_y, &block_size_z));
  CUfunction func =
      iree_hal_cuda_native_executable_for_entry_point(executable, entry_point);
  CUDA_RETURN_IF_ERROR(
      command_buffer->context->syms,
      cuLaunchKernel(func, workgroup_x, workgroup_y, workgroup_z, block_size_x,
                     block_size_y, block_size_z, 0, command_buffer->stream,
                     command_buffer->current_descriptor, NULL),
      "cuLaunchKernel");
  return iree_ok_status();
}

static iree_status_t iree_hal_cuda_stream_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* base_command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer,
    iree_device_size_t workgroups_offset) {
  return iree_make_status(IREE_STATUS_UNIMPLEMENTED,
                          "need cuda implementation of dispatch indirect");
}

static const iree_hal_command_buffer_vtable_t
    iree_hal_cuda_stream_command_buffer_vtable = {
        .destroy = iree_hal_cuda_stream_command_buffer_destroy,
        .dyn_cast = iree_hal_cuda_stream_command_buffer_dyn_cast,
        .begin = iree_hal_cuda_stream_command_buffer_begin,
        .end = iree_hal_cuda_stream_command_buffer_end,
        .execution_barrier =
            iree_hal_cuda_stream_command_buffer_execution_barrier,
        .signal_event = iree_hal_cuda_stream_command_buffer_signal_event,
        .reset_event = iree_hal_cuda_stream_command_buffer_reset_event,
        .wait_events = iree_hal_cuda_stream_command_buffer_wait_events,
        .discard_buffer = iree_hal_cuda_stream_command_buffer_discard_buffer,
        .fill_buffer = iree_hal_cuda_stream_command_buffer_fill_buffer,
        .update_buffer = iree_hal_cuda_stream_command_buffer_update_buffer,
        .copy_buffer = iree_hal_cuda_stream_command_buffer_copy_buffer,
        .push_constants = iree_hal_cuda_stream_command_buffer_push_constants,
        .push_descriptor_set =
            iree_hal_cuda_stream_command_buffer_push_descriptor_set,
        .bind_descriptor_set =
            iree_hal_cuda_stream_command_buffer_bind_descriptor_set,
        .dispatch = iree_hal_cuda_stream_command_buffer_dispatch,
        .dispatch_indirect =
            iree_hal_cuda_stream_command_buffer_dispatch_indirect,
};
