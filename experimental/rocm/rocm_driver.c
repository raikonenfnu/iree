// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <stdint.h>
#include <string.h>

#include "experimental/rocm/api.h"
#include "experimental/rocm/dynamic_symbols.h"
#include "experimental/rocm/rocm_device.h"
#include "experimental/rocm/status_util.h"
#include "iree/base/api.h"
#include "iree/base/tracing.h"
#include "iree/hal/api.h"

typedef struct iree_hal_rocm_driver_t {
  iree_hal_resource_t resource;
  iree_allocator_t host_allocator;
  // Identifier used for the driver in the IREE driver registry.
  // We allow overriding so that multiple ROCM versions can be exposed in the
  // same process.
  iree_string_view_t identifier;
  int default_device_index;
  // ROCM symbols.
  iree_hal_rocm_dynamic_symbols_t syms;
} iree_hal_rocm_driver_t;

// Pick a fixed lenght size for device names.
#define IREE_MAX_ROCM_DEVICE_NAME_LENGTH 100

static const iree_hal_driver_vtable_t iree_hal_rocm_driver_vtable;

static iree_hal_rocm_driver_t* iree_hal_rocm_driver_cast(
    iree_hal_driver_t* base_value) {
  IREE_HAL_ASSERT_TYPE(base_value, &iree_hal_rocm_driver_vtable);
  return (iree_hal_rocm_driver_t*)base_value;
}

IREE_API_EXPORT void iree_hal_rocm_driver_options_initialize(
    iree_hal_rocm_driver_options_t* out_options) {
  memset(out_options, 0, sizeof(*out_options));
  out_options->default_device_index = 0;
}

static iree_status_t iree_hal_rocm_driver_create_internal(
    iree_string_view_t identifier,
    const iree_hal_rocm_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  iree_hal_rocm_driver_t* driver = NULL;
  iree_host_size_t total_size = sizeof(*driver) + identifier.size;
  IREE_RETURN_IF_ERROR(
      iree_allocator_malloc(host_allocator, total_size, (void**)&driver));
  iree_hal_resource_initialize(&iree_hal_rocm_driver_vtable, &driver->resource);
  driver->host_allocator = host_allocator;
  iree_string_view_append_to_buffer(
      identifier, &driver->identifier,
      (char*)driver + total_size - identifier.size);
  driver->default_device_index = options->default_device_index;
  iree_status_t status =
      iree_hal_rocm_dynamic_symbols_initialize(host_allocator, &driver->syms);
  if (iree_status_is_ok(status)) {
    *out_driver = (iree_hal_driver_t*)driver;
  } else {
    iree_hal_driver_release((iree_hal_driver_t*)driver);
  }
  return status;
}

static void iree_hal_rocm_driver_destroy(iree_hal_driver_t* base_driver) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  iree_allocator_t host_allocator = driver->host_allocator;
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_hal_rocm_dynamic_symbols_deinitialize(&driver->syms);
  iree_allocator_free(host_allocator, driver);

  IREE_TRACE_ZONE_END(z0);
}

IREE_API_EXPORT iree_status_t iree_hal_rocm_driver_create(
    iree_string_view_t identifier,
    const iree_hal_rocm_driver_options_t* options,
    iree_allocator_t host_allocator, iree_hal_driver_t** out_driver) {
  IREE_ASSERT_ARGUMENT(options);
  IREE_ASSERT_ARGUMENT(out_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  iree_status_t status = iree_hal_rocm_driver_create_internal(
      identifier, options, host_allocator, out_driver);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

// Populates device information from the given ROCM physical device handle.
// |out_device_info| must point to valid memory and additional data will be
// appended to |buffer_ptr| and the new pointer is returned.
static uint8_t* iree_hal_rocm_populate_device_info(
    hipDevice_t device, iree_hal_rocm_dynamic_symbols_t* syms,
    uint8_t* buffer_ptr, iree_hal_device_info_t* out_device_info) {
  char device_name[IREE_MAX_ROCM_DEVICE_NAME_LENGTH];
  ROCM_IGNORE_ERROR(syms,
                    hipDeviceGetName(device_name, sizeof(device_name), device));
  memset(out_device_info, 0, sizeof(*out_device_info));
  out_device_info->device_id = (iree_hal_device_id_t)device;

  iree_string_view_t device_name_string =
      iree_make_string_view(device_name, strlen(device_name));
  buffer_ptr += iree_string_view_append_to_buffer(
      device_name_string, &out_device_info->name, (char*)buffer_ptr);
  return buffer_ptr;
}

static iree_status_t iree_hal_rocm_driver_query_available_devices(
    iree_hal_driver_t* base_driver, iree_allocator_t host_allocator,
    iree_hal_device_info_t** out_device_infos,
    iree_host_size_t* out_device_info_count) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  // Query the number of available ROCM devices.
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(&driver->syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");

  // Allocate the return infos and populate with the devices.
  iree_hal_device_info_t* device_infos = NULL;
  iree_host_size_t total_size = device_count * sizeof(iree_hal_device_info_t);
  for (iree_host_size_t i = 0; i < device_count; ++i) {
    total_size += IREE_MAX_ROCM_DEVICE_NAME_LENGTH * sizeof(char);
  }
  iree_status_t status =
      iree_allocator_malloc(host_allocator, total_size, (void**)&device_infos);
  if (iree_status_is_ok(status)) {
    uint8_t* buffer_ptr =
        (uint8_t*)device_infos + device_count * sizeof(iree_hal_device_info_t);
    for (iree_host_size_t i = 0; i < device_count; ++i) {
      hipDevice_t device;
      iree_status_t status = ROCM_RESULT_TO_STATUS(
          &driver->syms, hipDeviceGet(&device, i), "hipDeviceGet");
      if (!iree_status_is_ok(status)) break;
      buffer_ptr = iree_hal_rocm_populate_device_info(
          device, &driver->syms, buffer_ptr, &device_infos[i]);
    }
  }
  if (iree_status_is_ok(status)) {
    *out_device_info_count = device_count;
    *out_device_infos = device_infos;
  } else {
    iree_allocator_free(host_allocator, device_infos);
  }
  return status;
}

static iree_status_t iree_hal_rocm_driver_select_default_device(
    iree_hal_rocm_dynamic_symbols_t* syms, int default_device_index,
    iree_allocator_t host_allocator, hipDevice_t* out_device) {
  int device_count = 0;
  ROCM_RETURN_IF_ERROR(syms, hipGetDeviceCount(&device_count),
                       "hipGetDeviceCount");
  iree_status_t status = iree_ok_status();
  if (device_count == 0 || default_device_index >= device_count) {
    status = iree_make_status(IREE_STATUS_NOT_FOUND,
                              "default device %d not found (of %d enumerated)",
                              default_device_index, device_count);
  } else {
    hipDevice_t device;
    ROCM_RETURN_IF_ERROR(syms, hipDeviceGet(&device, default_device_index),
                         "hipDeviceGet");
    *out_device = device;
  }
  return status;
}

static iree_status_t iree_hal_rocm_driver_create_device(
    iree_hal_driver_t* base_driver, iree_hal_device_id_t device_id,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device) {
  iree_hal_rocm_driver_t* driver = iree_hal_rocm_driver_cast(base_driver);
  IREE_TRACE_ZONE_BEGIN(z0);

  IREE_RETURN_AND_END_ZONE_IF_ERROR(
      z0, ROCM_RESULT_TO_STATUS(&driver->syms, hipInit(0), "hipInit"));
  // Use either the specified device (enumerated earlier) or whatever default
  // one was specified when the driver was created.
  hipDevice_t device = (hipDevice_t)device_id;
  if (device == 0) {
    IREE_RETURN_AND_END_ZONE_IF_ERROR(
        z0, iree_hal_rocm_driver_select_default_device(
                &driver->syms, driver->default_device_index, host_allocator,
                &device));
  }

  iree_string_view_t device_name = iree_make_cstring_view("rocm");

  // Attempt to create the device.
  iree_status_t status =
      iree_hal_rocm_device_create(base_driver, device_name, &driver->syms,
                                  device, host_allocator, out_device);

  IREE_TRACE_ZONE_END(z0);
  return status;
}

static const iree_hal_driver_vtable_t iree_hal_rocm_driver_vtable = {
    .destroy = iree_hal_rocm_driver_destroy,
    .query_available_devices = iree_hal_rocm_driver_query_available_devices,
    .create_device = iree_hal_rocm_driver_create_device,
};
