// Copyright 2021 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef IREE_HAL_LEVEL_ZERO_EVENT_H_
#define IREE_HAL_LEVEL_ZERO_EVENT_H_

#include "experimental/level_zero/context_wrapper.h"
#include "experimental/level_zero/level_zero_headers.h"
#include "iree/base/api.h"
#include "iree/hal/api.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// Creates a dummy event object. Object will be represented by level_zero
// command list nodes so nothing is created at creation time. When an event is
// signaled in the command buffer we will add the appropriate edges to enforce
// the right synchronization.
iree_status_t iree_hal_level_zero_event_create(
    iree_hal_level_zero_context_wrapper_t* context_wrapper,
    iree_hal_event_t** out_event);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_LEVEL_ZERO_EVENT_H_
