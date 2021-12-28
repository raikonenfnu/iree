// Copyright 2019 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/hal/vulkan/native_semaphore.h"

#include <cstddef>

#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/vulkan/dynamic_symbol_tables.h"
#include "iree/hal/vulkan/dynamic_symbols.h"
#include "iree/hal/vulkan/status_util.h"
#include "iree/hal/vulkan/util/ref_ptr.h"

// The maximum valid payload value of an iree_hal_semaphore_t.
// Payload values larger than this indicate that the semaphore has failed.
//
// This originates from Vulkan having a lower-bound of INT_MAX for
// maxTimelineSemaphoreValueDifference and many Android devices only supporting
// that lower-bound. At ~100 signals per second it'll take 1.5+ years to
// saturate. We may increase this value at some point but so long as there are
// some devices in the wild that may have this limitation we can ensure better
// consistency across the backends by observing this.
//
// The major mitigation here is that in proper usage of IREE there are no
// semaphores that are implicitly referenced by multiple VMs (each creates their
// own internally) and in a multitenant system each session should have its own
// semaphores - so even if the process lives for years it's highly unlikely any
// particular session does. Whatever, 640K is enough for anyone.
//
// See:
//   https://vulkan.gpuinfo.org/displayextensionproperty.php?name=maxTimelineSemaphoreValueDifference
#define IREE_HAL_VULKAN_SEMAPHORE_MAX_VALUE (2147483647ull - 1)

using namespace iree::hal::vulkan;

typedef struct iree_hal_vulkan_native_semaphore_t {
  iree_hal_resource_t resource;
  VkDeviceHandle* logical_device;
  VkSemaphore handle;
  iree_atomic_intptr_t failure_status;
} iree_hal_vulkan_native_semaphore_t;

namespace {
extern const iree_hal_semaphore_vtable_t
    iree_hal_vulkan_native_semaphore_vtable;
}  // namespace

static iree_hal_vulkan_native_semaphore_t*
iree_hal_vulkan_native_semaphore_cast(iree_hal_semaphore_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_vulkan_native_semaphore_vtable);
  return (iree_hal_vulkan_native_semaphore_t*)base_value;
}

iree_status_t iree_hal_vulkan_native_semaphore_create(
    iree::hal::vulkan::VkDeviceHandle* logical_device, uint64_t initial_value,
    iree_hal_semaphore_t** out_semaphore) {
  IREE_ASSERT_ARGUMENT(logical_device);
  IREE_ASSERT_ARGUMENT(out_semaphore);
  *out_semaphore = NULL;
  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphoreTypeCreateInfo timeline_create_info;
  timeline_create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
  timeline_create_info.pNext = NULL;
  timeline_create_info.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
  timeline_create_info.initialValue = initial_value;

  VkSemaphoreCreateInfo create_info;
  create_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
  create_info.pNext = &timeline_create_info;
  create_info.flags = 0;
  VkSemaphore handle = VK_NULL_HANDLE;
  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, VK_RESULT_TO_STATUS(logical_device->syms()->vkCreateSemaphore(
                                  *logical_device, &create_info,
                                  logical_device->allocator(), &handle),
                              "vkCreateSemaphore"));

  iree_hal_vulkan_native_semaphore_t* semaphore = NULL;
  iree_status_t status = iree_allocator_malloc(
      logical_device->host_allocator(), sizeof(*semaphore), (void**)&semaphore);
  if (iree_status_is_ok(status)) {
    iree_hal_resource_initialize(&iree_hal_vulkan_native_semaphore_vtable,
                                 &semaphore->resource);
    semaphore->logical_device = logical_device;
    semaphore->handle = handle;
    iree_atomic_store_intptr(&semaphore->failure_status, 0,
                             iree_memory_order_release);
    *out_semaphore = (iree_hal_semaphore_t*)semaphore;
  } else {
    logical_device->syms()->vkDestroySemaphore(*logical_device, handle,
                                               logical_device->allocator());
  }

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static void iree_hal_vulkan_native_semaphore_destroy(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_allocator_t host_allocator = semaphore->logical_device->host_allocator();
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_free((iree_status_t)iree_atomic_load_intptr(
      &semaphore->failure_status, iree_memory_order_acquire));
  semaphore->logical_device->syms()->vkDestroySemaphore(
      *semaphore->logical_device, semaphore->handle,
      semaphore->logical_device->allocator());
  iree_allocator_free(host_allocator, semaphore);

  IREE_TRACE_ZONE_END(z0);
}

VkSemaphore iree_hal_vulkan_native_semaphore_handle(
    iree_hal_semaphore_t* base_semaphore) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  return semaphore->handle;
}

static iree_status_t iree_hal_vulkan_native_semaphore_query(
    iree_hal_semaphore_t* base_semaphore, uint64_t* out_value) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  *out_value = 0;

  uint64_t value = 0;
  IREE_RETURN_IF_ERROR(VK_RESULT_TO_STATUS(
      semaphore->logical_device->syms()->vkGetSemaphoreCounterValue(
          *semaphore->logical_device, semaphore->handle, &value),
      "vkGetSemaphoreCounterValue"));

  if (value > IREE_HAL_VULKAN_SEMAPHORE_MAX_VALUE) {
    iree_status_t failure_status = (iree_status_t)iree_atomic_load_intptr(
        &semaphore->failure_status, iree_memory_order_acquire);
    if (iree_status_is_ok(failure_status)) {
      return iree_make_status(IREE_STATUS_RESOURCE_EXHAUSTED,
                              "overflowed timeline semaphore max value");
    }
    return iree_status_clone(failure_status);
  }

  *out_value = value;
  return iree_ok_status();
}

static iree_status_t iree_hal_vulkan_native_semaphore_signal(
    iree_hal_semaphore_t* base_semaphore, uint64_t new_value) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = new_value;
  return VK_RESULT_TO_STATUS(
      semaphore->logical_device->syms()->vkSignalSemaphore(
          *semaphore->logical_device, &signal_info),
      "vkSignalSemaphore");
}

static void iree_hal_vulkan_native_semaphore_fail(
    iree_hal_semaphore_t* base_semaphore, iree_status_t status) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);

  // Try to set our local status - we only preserve the first failure so only
  // do this if we are going from a valid semaphore to a failed one.
  iree_status_t old_status = iree_ok_status();
  if (!iree_atomic_compare_exchange_strong_intptr(
          &semaphore->failure_status, (intptr_t*)&old_status, (intptr_t)status,
          iree_memory_order_seq_cst, iree_memory_order_seq_cst)) {
    // Previous status was not OK; drop our new status.
    IREE_IGNORE_ERROR(status);
    return;
  }

  VkSemaphoreSignalInfo signal_info;
  signal_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO;
  signal_info.pNext = NULL;
  signal_info.semaphore = semaphore->handle;
  signal_info.value = IREE_HAL_VULKAN_SEMAPHORE_MAX_VALUE + 1;
  // NOTE: we don't care about the result in case of failures as we are
  // failing and the caller will likely be tearing everything down anyway.
  semaphore->logical_device->syms()->vkSignalSemaphore(
      *semaphore->logical_device, &signal_info);
}

iree_status_t iree_hal_vulkan_native_semaphore_multi_wait(
    iree::hal::vulkan::VkDeviceHandle* logical_device,
    const iree_hal_semaphore_list_t* semaphore_list, iree_timeout_t timeout,
    VkSemaphoreWaitFlags wait_flags) {
  if (semaphore_list->count == 0) return iree_ok_status();

  iree_time_t deadline_ns = iree_timeout_as_deadline_ns(timeout);

  uint64_t timeout_ns;
  if (deadline_ns == IREE_TIME_INFINITE_FUTURE) {
    timeout_ns = UINT64_MAX;
  } else if (deadline_ns == IREE_TIME_INFINITE_PAST) {
    timeout_ns = 0;
  } else {
    iree_time_t now_ns = iree_time_now();
    if (deadline_ns < now_ns) {
      return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
    }
    timeout_ns = (uint64_t)(deadline_ns - now_ns);
  }

  IREE_TRACE_ZONE_BEGIN(z0);

  VkSemaphore* semaphore_handles =
      (VkSemaphore*)iree_alloca(semaphore_list->count * sizeof(VkSemaphore));
  for (iree_host_size_t i = 0; i < semaphore_list->count; ++i) {
    semaphore_handles[i] =
        iree_hal_vulkan_native_semaphore_handle(semaphore_list->semaphores[i]);
  }

  VkSemaphoreWaitInfo wait_info;
  wait_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO;
  wait_info.pNext = nullptr;
  wait_info.flags = wait_flags;
  wait_info.semaphoreCount = semaphore_list->count;
  wait_info.pSemaphores = semaphore_handles;
  wait_info.pValues = semaphore_list->payload_values;
  static_assert(
      sizeof(wait_info.pValues[0]) == sizeof(semaphore_list->payload_values[0]),
      "payload value type must match vulkan expected size");

  // NOTE: this may fail with a timeout (VK_TIMEOUT) or in the case of a
  // device loss event may return either VK_SUCCESS *or* VK_ERROR_DEVICE_LOST.
  // We may want to explicitly query for device loss after a successful wait
  // to ensure we consistently return errors.
  VkResult result = logical_device->syms()->vkWaitSemaphores(
      *logical_device, &wait_info, timeout_ns);

  IREE_TRACE_ZONE_END(z0);

  if (result == VK_SUCCESS) {
    return iree_ok_status();
  } else if (result == VK_ERROR_DEVICE_LOST) {
    // Nothing we do now matters.
    return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
  } else if (result == VK_TIMEOUT) {
    return iree_status_from_code(IREE_STATUS_DEADLINE_EXCEEDED);
  }
  return VK_RESULT_TO_STATUS(result, "vkWaitSemaphores");
}

static iree_status_t iree_hal_vulkan_native_semaphore_wait(
    iree_hal_semaphore_t* base_semaphore, uint64_t value,
    iree_timeout_t timeout) {
  iree_hal_vulkan_native_semaphore_t* semaphore =
      iree_hal_vulkan_native_semaphore_cast(base_semaphore);
  iree_hal_semaphore_list_t semaphore_list = {
      /*.count=*/1,
      /*.semaphores=*/&base_semaphore,
      /*.payload_values=*/&value,
  };
  return iree_hal_vulkan_native_semaphore_multi_wait(
      semaphore->logical_device, &semaphore_list, timeout, 0);
}

namespace {
const iree_hal_semaphore_vtable_t iree_hal_vulkan_native_semaphore_vtable = {
    /*.destroy=*/iree_hal_vulkan_native_semaphore_destroy,
    /*.query=*/iree_hal_vulkan_native_semaphore_query,
    /*.signal=*/iree_hal_vulkan_native_semaphore_signal,
    /*.fail=*/iree_hal_vulkan_native_semaphore_fail,
    /*.wait=*/iree_hal_vulkan_native_semaphore_wait,
};
}  // namespace
