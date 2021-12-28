// Copyright 2020 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_COMMAND_BUFFER_H_
#define IREE_HAL_COMMAND_BUFFER_H_

#include <stdbool.h>
#include <stdint.h>

#include "iree/base/api.h"
#include "iree/hal/buffer.h"
#include "iree/hal/descriptor_set.h"
#include "iree/hal/descriptor_set_layout.h"
#include "iree/hal/event.h"
#include "iree/hal/executable.h"
#include "iree/hal/executable_layout.h"
#include "iree/hal/resource.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

typedef struct iree_hal_device_t iree_hal_device_t;

//===----------------------------------------------------------------------===//
// Types and Enums
//===----------------------------------------------------------------------===//

// A bitfield specifying the mode of operation for a command buffer.
enum iree_hal_command_buffer_mode_bits_t {
  // Command buffer will be submitted once and never used again.
  // This may enable in-place patching of command buffers that reduce overhead
  // when it's known that command buffers will not be reused.
  IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT = 1u << 0,

  // TODO(benvanik): IREE_HAL_COMMAND_BUFFER_MODE_REUSABLE = 1u << 1,
  // TODO(benvanik): IREE_HAL_COMMAND_BUFFER_MODE_PRIMARY = 1u << 2,
  // TODO(benvanik): IREE_HAL_COMMAND_BUFFER_MODE_SECONDARY = 1u << 3,

  // Indicates that the command buffer execution is allowed to execute inline
  // with recording. The exact execution behavior is unspecified by the API and
  // intentionally unknowable and must always assume to happen entirely
  // asynchronously and that it will only have completed after waiting on device
  // idle or the wait semaphores specified in the submission are signaled.
  //
  // Local backends can use this to avoid recording when the calling program can
  // guarantee that it makes no assumptions about execution being deferred until
  // a submission. The command buffer must still be submitted for scheduling and
  // must have no wait semaphores specified. This allows the same program code
  // to execute work both synchronously and asynchronously as remote backends
  // are allowed to ignore this.
  //
  // Remote backends can use this to flush the command buffer more aggressively
  // to begin early execution and overlap with continued recording.
  //
  // Requires IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT and
  // IREE_HAL_COMMAND_BUFFER_MODE_PRIMARY. Compatible with
  // IREE_HAL_COMMAND_BUFFER_MODE_REUSABLE.
  IREE_HAL_COMMAND_BUFFER_MODE_ALLOW_INLINE_EXECUTION = 1u << 4,

  // Disables additional command buffer validation (if present).
  // By default all command buffers will be validated if
  // `IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE=1` - if shimming command buffers
  // or performing replay this validation can be disabled per-command buffer.
  IREE_HAL_COMMAND_BUFFER_MODE_UNVALIDATED = 1u << 5,
};
typedef uint32_t iree_hal_command_buffer_mode_t;

// A bitfield specifying the category of commands in a command queue.
enum iree_hal_command_category_bits_t {
  // Command is considered a transfer operation (memcpy, etc).
  IREE_HAL_COMMAND_CATEGORY_TRANSFER = 1u << 0,
  // Command is considered a dispatch operation (dispatch/execute).
  IREE_HAL_COMMAND_CATEGORY_DISPATCH = 1u << 1,
  // Commands may be of any type.
  // Using this value may prevent optimizations and if possible callers should
  // always specify the strictest set possible (for example, only transfer
  // commands to ensure they get placed on a DMA queue).
  IREE_HAL_COMMAND_CATEGORY_ANY =
      IREE_HAL_COMMAND_CATEGORY_TRANSFER | IREE_HAL_COMMAND_CATEGORY_DISPATCH,
};
typedef uint32_t iree_hal_command_category_t;

// A bitmask indicating affinity for a submission to use a particular set of
// queues.
//
// Upon submission the queue is selected based on the flags set in
// |command_categories| and the |queue_affinity|. As the number of available
// queues can vary the |queue_affinity| is used to hash into the available
// queues for the required categories. For example if 2 queues support transfer
// commands and the affinity is 5 the resulting queue could be index hash(5)=1.
// The affinity can thus be treated as just a way to indicate whether two
// submissions must be placed on to the same queue. Note that the exact hashing
// function is implementation dependent.
typedef uint64_t iree_hal_queue_affinity_t;

// Specifies that any queue may be selected.
#define IREE_HAL_QUEUE_AFFINITY_ANY ((iree_hal_queue_affinity_t)(-1))

// Bitfield specifying which execution stage a barrier should start/end at.
//
// Maps to VkPipelineStageFlagBits.
enum iree_hal_execution_stage_bits_t {
  // Top of the pipeline when commands are initially issued by the device.
  IREE_HAL_EXECUTION_STAGE_COMMAND_ISSUE = 1u << 0,
  // Stage of the pipeline when dispatch parameter data is consumed.
  IREE_HAL_EXECUTION_STAGE_COMMAND_PROCESS = 1u << 1,
  // Stage where dispatch commands execute.
  IREE_HAL_EXECUTION_STAGE_DISPATCH = 1u << 2,
  // Stage where transfer (copy/clear/fill/etc) commands execute.
  IREE_HAL_EXECUTION_STAGE_TRANSFER = 1u << 3,
  // Final stage in the pipeline when commands are retired on the device.
  IREE_HAL_EXECUTION_STAGE_COMMAND_RETIRE = 1u << 4,
  // Pseudo-stage for read/writes by the host. Not executed on device.
  IREE_HAL_EXECUTION_STAGE_HOST = 1u << 5,
};
typedef uint32_t iree_hal_execution_stage_t;

// Bitfield specifying flags controlling an execution dependency.
//
// Maps to VkDependencyFlags.
enum iree_hal_execution_barrier_flag_bits_t {
  IREE_HAL_EXECUTION_BARRIER_FLAG_NONE = 0,
};
typedef uint32_t iree_hal_execution_barrier_flags_t;

// Bitfield specifying which scopes will access memory and how.
//
// Maps to VkAccessFlagBits.
enum iree_hal_access_scope_bits_t {
  // Read access to indirect command data as part of an indirect dispatch.
  IREE_HAL_ACCESS_SCOPE_INDIRECT_COMMAND_READ = 1u << 0,
  // Constant uniform buffer reads by the device.
  IREE_HAL_ACCESS_SCOPE_CONSTANT_READ = 1u << 1,
  // Storage buffer reads by dispatch commands.
  IREE_HAL_ACCESS_SCOPE_DISPATCH_READ = 1u << 2,
  // Storage buffer writes by dispatch commands.
  IREE_HAL_ACCESS_SCOPE_DISPATCH_WRITE = 1u << 3,
  // Source of a transfer operation.
  IREE_HAL_ACCESS_SCOPE_TRANSFER_READ = 1u << 4,
  // Target of a transfer operation.
  IREE_HAL_ACCESS_SCOPE_TRANSFER_WRITE = 1u << 5,
  // Read operation by the host through mapped memory.
  IREE_HAL_ACCESS_SCOPE_HOST_READ = 1u << 6,
  // Write operation by the host through mapped memory.
  IREE_HAL_ACCESS_SCOPE_HOST_WRITE = 1u << 7,
  // External/non-specific read.
  IREE_HAL_ACCESS_SCOPE_MEMORY_READ = 1u << 8,
  // External/non-specific write.
  IREE_HAL_ACCESS_SCOPE_MEMORY_WRITE = 1u << 9,
};
typedef uint32_t iree_hal_access_scope_t;

// Defines a global memory barrier.
// These are cheaper to encode than buffer-specific barriers but may cause
// stalls and bubbles in device pipelines if applied too broadly. Prefer them
// over equivalently large sets of buffer-specific barriers (such as when
// completely changing execution contexts).
//
// Maps to VkMemoryBarrier.
typedef struct iree_hal_memory_barrier_t {
  // All access scopes prior-to the barrier (inclusive).
  iree_hal_access_scope_t source_scope;
  // All access scopes following the barrier (inclusive).
  iree_hal_access_scope_t target_scope;
} iree_hal_memory_barrier_t;

// Defines a memory barrier that applies to a range of a specific buffer.
// Use of these (vs. global memory barriers) provides fine-grained execution
// ordering to device command processors and allows for more aggressive
// reordering.
//
// Maps to VkBufferMemoryBarrier.
typedef struct iree_hal_buffer_barrier_t {
  // All access scopes prior-to the barrier (inclusive).
  iree_hal_access_scope_t source_scope;
  // All access scopes following the barrier (inclusive).
  iree_hal_access_scope_t target_scope;
  // Buffer the barrier is restricted to.
  // The barrier will apply to the entire physical device allocation.
  iree_hal_buffer_t* buffer;
  // Relative offset/length within |buffer| (which may itself be mapped into the
  // device allocation at an offset).
  iree_device_size_t offset;
  iree_device_size_t length;
} iree_hal_buffer_barrier_t;

// An RGBA color.
typedef struct iree_hal_label_color_t {
  uint8_t r;
  uint8_t g;
  uint8_t b;
  uint8_t a;
} iree_hal_label_color_t;

// A source location attached to debug labels.
typedef struct iree_hal_label_location_t {
  iree_string_view_t file;
  int line;
} iree_hal_label_location_t;

// An unspecified color; debugging tools are to choose their own.
static inline iree_hal_label_color_t iree_hal_label_color_unspecified() {
  iree_hal_label_color_t color = {0, 0, 0, 0};
  return color;
}

// Formats a command buffer mode bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t
iree_hal_command_buffer_mode_format(iree_hal_command_buffer_mode_t value,
                                    iree_bitfield_string_temp_t* out_temp);

// Formats a command category bitfield as a string.
// See iree_bitfield_format for usage.
IREE_API_EXPORT iree_string_view_t iree_hal_command_category_format(
    iree_hal_command_category_t value, iree_bitfield_string_temp_t* out_temp);

// Storage for command buffer validation state.
// Designed to be embedded in concrete implementations that want validation.
typedef struct iree_hal_command_buffer_validation_state_t {
  iree_hal_device_t* device;
  bool is_recording;
  int32_t debug_group_depth;
  // TODO(benvanik): current executable layout/descriptor set layout info.
  // TODO(benvanik): valid push constant bit ranges.
} iree_hal_command_buffer_validation_state_t;

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t
//===----------------------------------------------------------------------===//

// Asynchronous command buffer recording interface.
// Commands are recorded by the implementation for later submission to command
// queues.
//
// Buffers and synchronization objects referenced must remain valid and not be
// modified or read while there are commands in-flight. The usual flow is to
// populate input buffers, Dispatch using those buffers, wait on a Semaphore
// until the buffers are guaranteed to no longer be in use, and then reuse or
// release the buffers.
//
// Errors that can be recognized when operations are enqueued will be returned
// immediately, such as invalid argument errors. Errors that can only be
// determined at execution time will be returned on semaphores. Once a failure
// occurs the device queue will enter an error state that invalidates all
// operations on the device queue (as ordering is not strict and any may still
// be in-flight). In this case the user of the device queue should treat all
// in-flight operations as cancelled and fully reset themselves. Other device
// queues that may be waiting on events from the device queue will also enter
// error states. Only once a user has acknowledged and cleared the error state
// with a Reset the queue will become usable, and otherwise all operations will
// return errors.
//
// Command buffers are thread-compatible. Use multiple command buffers if trying
// to record commands from multiple threads. Command buffers must not be mutated
// between when they have are submitted for execution on a queue and when the
// semaphore fires indicating the completion of their execution.
typedef struct iree_hal_command_buffer_t iree_hal_command_buffer_t;

// Creates a command buffer ready to begin recording, possibly reusing an
// existing one from the |device| pool.
//
// |queue_affinity| specifies the device queues the command buffer may be
// submitted to. The queue affinity provided to iree_hal_device_queue_submit
// must match or be a subset of the |queue_affinity|.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_create(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    iree_hal_command_buffer_t** out_command_buffer);

// Retains the given |command_buffer| for the caller.
IREE_API_EXPORT void iree_hal_command_buffer_retain(
    iree_hal_command_buffer_t* command_buffer);

// Releases the given |command_buffer| from the caller.
IREE_API_EXPORT void iree_hal_command_buffer_release(
    iree_hal_command_buffer_t* command_buffer);

IREE_API_EXPORT void* iree_hal_command_buffer_dyn_cast(
    iree_hal_command_buffer_t* command_buffer, const void* vtable);

// Returns a bitmask indicating the behavior of the command buffer.
IREE_API_EXPORT iree_hal_command_buffer_mode_t
iree_hal_command_buffer_mode(const iree_hal_command_buffer_t* command_buffer);

// Returns a bitmask indicating which command categories this command buffer
// can record.
IREE_API_EXPORT iree_hal_command_category_t
iree_hal_command_buffer_allowed_categories(
    const iree_hal_command_buffer_t* command_buffer);

// Resets and begins recording into the command buffer, clearing all
// previously recorded contents.
// The command buffer must not be in-flight.
IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_begin(iree_hal_command_buffer_t* command_buffer);

// Ends recording into the command buffer.
// This must be called prior to submitting the command buffer for execution.
IREE_API_EXPORT iree_status_t
iree_hal_command_buffer_end(iree_hal_command_buffer_t* command_buffer);

// Pushes a new debug group with the given |label|.
// All commands between this and a mandatory matching call to
// iree_hal_command_buffer_end_debug_group will be grouped together with the
// given label. If a source location is available it can be provided via
// |location| to allow mapping back into the source program that issued the
// commands.
//
// An optional RGBA color to show in the debug UI may be provided via
// |label_color|; otherwise iree_hal_label_color_unspecified can be used to let
// the debug tool choose.
IREE_API_EXPORT void iree_hal_command_buffer_begin_debug_group(
    iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
    iree_hal_label_color_t label_color,
    const iree_hal_label_location_t* location);

// Pops a debug group from the stack.
IREE_API_EXPORT void iree_hal_command_buffer_end_debug_group(
    iree_hal_command_buffer_t* command_buffer);

// Defines a memory dependency between commands recorded before and after the
// barrier. One or more memory or buffer barriers can be specified to indicate
// between which stages or buffers the dependencies exist.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_execution_barrier(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_hal_execution_barrier_flags_t flags,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

// Sets an event to the signaled state.
// |source_stage_mask| specifies when the event is signaled.
//
// Events are only valid within a single command buffer. Events can only be
// used on non-transfer queues.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_signal_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask);

// Resets an event to the non-signaled state.
// |source_stage_mask| specifies when the event is unsignaled.
//
// Events are only valid within a single command buffer. Events can only be
// used on non-transfer queues.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_reset_event(
    iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
    iree_hal_execution_stage_t source_stage_mask);

// Waits for one or more events to be signaled and defines a memory dependency
// between the synchronization scope of the signal operations and the commands
// following the wait.
//
// |source_stage_mask| must include ExecutionStage::kHost for Event::Signal to
// be visibile.
//
// Events are only valid within a single command buffer. Events remain
// signaled even after waiting and must be reset to be reused. Events can only
// be used on non-transfer queues.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_wait_events(
    iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count,
    const iree_hal_event_t** events,
    iree_hal_execution_stage_t source_stage_mask,
    iree_hal_execution_stage_t target_stage_mask,
    iree_host_size_t memory_barrier_count,
    const iree_hal_memory_barrier_t* memory_barriers,
    iree_host_size_t buffer_barrier_count,
    const iree_hal_buffer_barrier_t* buffer_barriers);

// Hints to the device queue that the given buffer will not be used again.
// After encoding a discard the buffer contents will be considered undefined.
// This is because the discard may be used to elide write backs to host memory
// or aggressively reuse the allocation for other purposes.
//
// For buffers allocated with IREE_HAL_MEMORY_TYPE_TRANSIENT this may allow
// the device queue to reclaim the memory used by the buffer earlier than
// otherwise possible.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_discard_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer);

// Fills the target buffer with the given repeating value.
// Expects that |pattern_length| is one of 1, 2, or 4 and that the offset and
// length are aligned to the natural alignment of the value.
// The target buffer must be compatible with the devices owned by this
// device queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_fill_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length,
    const void* pattern, iree_host_size_t pattern_length);

// Updates a range of the given target buffer from the source host memory.
// The source host memory is copied immediately into the command buffer and
// occupies command buffer space. It is strongly recommended that large buffer
// updates are performed via iree_hal_command_buffer_copy_buffer where there is
// the possibility of a zero-copy path.
// The |source_buffer| may be releaed by the caller immediately after this
// call returns.
// The |target_buffer| must be compatible with the devices owned by this
// device queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_update_buffer(
    iree_hal_command_buffer_t* command_buffer, const void* source_buffer,
    iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

// Copies a range of one buffer to another.
// Both buffers must be compatible with the devices owned by this device
// queue and be allocated with IREE_HAL_BUFFER_USAGE_TRANSFER. Though the source
// and target buffer may be the same the ranges must not overlap (as with
// memcpy).
//
// This can be used to perform device->host, host->device, and device->device
// copies.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_copy_buffer(
    iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* source_buffer,
    iree_device_size_t source_offset, iree_hal_buffer_t* target_buffer,
    iree_device_size_t target_offset, iree_device_size_t length);

// Pushes an inline set of constants that can be accessed by subsequent
// dispatches using a compatible executable layout.
//
// Push constants are treated as opaque bytes, meaning that they may be
// bit-casted floats, bit-packed booleans, etc. |offset| and |values_length| are
// in bytes.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_push_constants(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
    const void* values, iree_host_size_t values_length);

// Pushes a descriptor set and associates it with |set|.
// This uses an internal ringbuffer inside of the command buffer to avoid the
// need for creating and binding descriptor sets and managing their lifetime.
//
// The descriptor set will remain bound and valid so long as the executable
// layouts used by dispatches are compatible (same descriptor layouts and push
// constant sizes).
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_push_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_host_size_t binding_count,
    const iree_hal_descriptor_set_binding_t* bindings);

// Binds a descriptor set to the given |set| matching that used in the
// executable layout interface.
//
// The descriptor set will remain bound and valid so long as the executable
// layouts used by dispatches are compatible (same descriptor layouts and push
// constant sizes).
//
// If any dynamic descriptor types are defined in the descriptor set layout then
// the dynamic offsets must be provided. These offsets will be added to the base
// offset of the descriptor layout binding.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_bind_descriptor_set(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_layout_t* executable_layout, uint32_t set,
    iree_hal_descriptor_set_t* descriptor_set,
    iree_host_size_t dynamic_offset_count,
    const iree_device_size_t* dynamic_offsets);

// Dispatches an execution request.
// The request may execute overlapped with any other transfer operation or
// dispatch made within the same barrier-defined sequence.
//
// The executable specified must be registered for use with the device driver
// owning this queue. It must not be unregistered until all requests that use
// it have completed.
//
// Fails if the queue does not support dispatch operations (as indicated by
// can_dispatch).
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_dispatch(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z);

// Dispatches an execution request with deferred workgroup counts.
// This is the same as iree_hal_command_buffer_dispatch but the workgroup counts
// are read from the given |workgroups_buffer| at offset |workgroups_offset| as
// 3 uint32_t XYZ values before performing the dispatch. This allows prior
// dispatches within the command sequence to populate the workgroup counts.
//
// The buffer must have been allocated with IREE_HAL_BUFFER_USAGE_DISPATCH and
// be of IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_dispatch_indirect(
    iree_hal_command_buffer_t* command_buffer,
    iree_hal_executable_t* executable, int32_t entry_point,
    iree_hal_buffer_t* workgroups_buffer, iree_device_size_t workgroups_offset);

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t validation wrapper
//===----------------------------------------------------------------------===//

// Wraps |target_command_buffer| with a validation layer that checks the
// parameters to each call in an attempt to return errors where usage may result
// in failed or incorrect execution. This layer adds many additional checks to
// each call but must be used when dealing with untrusted incoming commands.
//
// The validation is strictly input argument and permission-based and not a full
// verification of the correctness of any barriers or memory dependencies. A
// command buffer recording that has passed validation does not indicate that it
// is guaranteed to make forward progress or properly observe memory visibility
// or availability rules. Instead, validation ensures that no command references
// memory outside of the allowed ranges or accesses memory in violation of the
// allowed usage or access rights.
IREE_API_EXPORT iree_status_t iree_hal_command_buffer_wrap_validation(
    iree_hal_device_t* device, iree_hal_command_buffer_t* target_command_buffer,
    iree_hal_command_buffer_t** out_command_buffer);

//===----------------------------------------------------------------------===//
// iree_hal_command_buffer_t implementation details
//===----------------------------------------------------------------------===//

typedef struct iree_hal_command_buffer_vtable_t {
  // << HAL C porting in progress >>
  IREE_API_UNSTABLE

  void(IREE_API_PTR* destroy)(iree_hal_command_buffer_t* command_buffer);

  void*(IREE_API_PTR* dyn_cast)(iree_hal_command_buffer_t* command_buffer,
                                const void* vtable);

  iree_status_t(IREE_API_PTR* begin)(iree_hal_command_buffer_t* command_buffer);
  iree_status_t(IREE_API_PTR* end)(iree_hal_command_buffer_t* command_buffer);

  void(IREE_API_PTR* begin_debug_group)(
      iree_hal_command_buffer_t* command_buffer, iree_string_view_t label,
      iree_hal_label_color_t label_color,
      const iree_hal_label_location_t* location);
  void(IREE_API_PTR* end_debug_group)(
      iree_hal_command_buffer_t* command_buffer);

  iree_status_t(IREE_API_PTR* execution_barrier)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_execution_stage_t source_stage_mask,
      iree_hal_execution_stage_t target_stage_mask,
      iree_hal_execution_barrier_flags_t flags,
      iree_host_size_t memory_barrier_count,
      const iree_hal_memory_barrier_t* memory_barriers,
      iree_host_size_t buffer_barrier_count,
      const iree_hal_buffer_barrier_t* buffer_barriers);

  iree_status_t(IREE_API_PTR* signal_event)(
      iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
      iree_hal_execution_stage_t source_stage_mask);

  iree_status_t(IREE_API_PTR* reset_event)(
      iree_hal_command_buffer_t* command_buffer, iree_hal_event_t* event,
      iree_hal_execution_stage_t source_stage_mask);

  iree_status_t(IREE_API_PTR* wait_events)(
      iree_hal_command_buffer_t* command_buffer, iree_host_size_t event_count,
      const iree_hal_event_t** events,
      iree_hal_execution_stage_t source_stage_mask,
      iree_hal_execution_stage_t target_stage_mask,
      iree_host_size_t memory_barrier_count,
      const iree_hal_memory_barrier_t* memory_barriers,
      iree_host_size_t buffer_barrier_count,
      const iree_hal_buffer_barrier_t* buffer_barriers);

  iree_status_t(IREE_API_PTR* discard_buffer)(
      iree_hal_command_buffer_t* command_buffer, iree_hal_buffer_t* buffer);

  iree_status_t(IREE_API_PTR* fill_buffer)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length, const void* pattern,
      iree_host_size_t pattern_length);

  iree_status_t(IREE_API_PTR* update_buffer)(
      iree_hal_command_buffer_t* command_buffer, const void* source_buffer,
      iree_host_size_t source_offset, iree_hal_buffer_t* target_buffer,
      iree_device_size_t target_offset, iree_device_size_t length);

  iree_status_t(IREE_API_PTR* copy_buffer)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_buffer_t* source_buffer, iree_device_size_t source_offset,
      iree_hal_buffer_t* target_buffer, iree_device_size_t target_offset,
      iree_device_size_t length);

  iree_status_t(IREE_API_PTR* push_constants)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_executable_layout_t* executable_layout, iree_host_size_t offset,
      const void* values, iree_host_size_t values_length);

  iree_status_t(IREE_API_PTR* push_descriptor_set)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_executable_layout_t* executable_layout, uint32_t set,
      iree_host_size_t binding_count,
      const iree_hal_descriptor_set_binding_t* bindings);

  iree_status_t(IREE_API_PTR* bind_descriptor_set)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_executable_layout_t* executable_layout, uint32_t set,
      iree_hal_descriptor_set_t* descriptor_set,
      iree_host_size_t dynamic_offset_count,
      const iree_device_size_t* dynamic_offsets);

  iree_status_t(IREE_API_PTR* dispatch)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_executable_t* executable, int32_t entry_point,
      uint32_t workgroup_x, uint32_t workgroup_y, uint32_t workgroup_z);

  iree_status_t(IREE_API_PTR* dispatch_indirect)(
      iree_hal_command_buffer_t* command_buffer,
      iree_hal_executable_t* executable, int32_t entry_point,
      iree_hal_buffer_t* workgroups_buffer,
      iree_device_size_t workgroups_offset);
} iree_hal_command_buffer_vtable_t;

struct iree_hal_command_buffer_t {
  iree_hal_resource_t resource;
  iree_hal_command_buffer_mode_t mode;
  iree_hal_command_category_t allowed_categories;
  iree_hal_queue_affinity_t queue_affinity;

#if IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
  iree_hal_command_buffer_validation_state_t validation;
#endif  // IREE_HAL_COMMAND_BUFFER_VALIDATION_ENABLE
};

IREE_API_EXPORT void iree_hal_command_buffer_initialize(
    iree_hal_device_t* device, iree_hal_command_buffer_mode_t mode,
    iree_hal_command_category_t command_categories,
    iree_hal_queue_affinity_t queue_affinity,
    const iree_hal_command_buffer_vtable_t* vtable,
    iree_hal_command_buffer_t* command_buffer);

IREE_API_EXPORT void iree_hal_command_buffer_destroy(
    iree_hal_command_buffer_t* command_buffer);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_COMMAND_BUFFER_H_
