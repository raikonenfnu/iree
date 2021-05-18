// Copyright 2021 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_
#define IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_

#include "experimental/rocm/rocm_headers.h"
#include "iree/base/api.h"
#include "iree/base/internal/dynamic_library.h"

#ifdef __cplusplus
extern "C" {
#endif  // __cplusplus

// DynamicSymbols allow loading dynamically a subset of ROCM driver API. It
// loads all the function declared in `dynamic_symbol_tables.def` and fail if
// any of the symbol is not available. The functions signatures are matching
// the declarations in `hipruntime.h`.
typedef struct {
  iree_dynamic_library_t *loader_library;

#define RC_PFN_DECL(rocmSymbolName, ...) \
  hipError_t (*rocmSymbolName)(__VA_ARGS__);
#define RC_PFN_STR_DECL(rocmSymbolName, ...) \
  const char *(*rocmSymbolName)(__VA_ARGS__);
#include "experimental/rocm/dynamic_symbol_tables.h"
#undef RC_PFN_DECL
#undef RC_PFN_STR_DECL
} iree_hal_rocm_dynamic_symbols_t;

// Initializes |out_syms| in-place with dynamically loaded ROCM symbols.
// iree_hal_rocm_dynamic_symbols_deinitialize must be used to release the
// library resources.
iree_status_t iree_hal_rocm_dynamic_symbols_initialize(
    iree_allocator_t allocator, iree_hal_rocm_dynamic_symbols_t *out_syms);

// Deinitializes |syms| by unloading the backing library. All function pointers
// will be invalidated. They _may_ still work if there are other reasons the
// library remains loaded so be careful.
void iree_hal_rocm_dynamic_symbols_deinitialize(
    iree_hal_rocm_dynamic_symbols_t *syms);

#ifdef __cplusplus
}  // extern "C"
#endif  // __cplusplus

#endif  // IREE_HAL_ROCM_DYNAMIC_SYMBOLS_H_
