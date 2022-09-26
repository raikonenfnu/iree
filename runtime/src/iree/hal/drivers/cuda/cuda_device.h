// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_
#define IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_

#include "iree/base/api.h"
#include "iree/hal/api.h"
#include "iree/hal/drivers/cuda/api.h"
#include "iree/hal/drivers/cuda/dynamic_symbols.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a device that owns and manages its own CUcontext.
iree_status_t iree_hal_cuda_device_create(
    iree_hal_driver_t* driver, iree_string_view_t identifier,
    const iree_hal_cuda_device_params_t* params,
    iree_hal_cuda_dynamic_symbols_t* syms, CUdevice device,
    iree_allocator_t host_allocator, iree_hal_device_t** out_device);

iree_status_t iree_cuda_set_current_thread(iree_hal_device_t* device);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_DRIVERS_CUDA_CUDA_DEVICE_H_
