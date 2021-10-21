# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT} OR NOT ${IREE_HAL_DRIVER_DYLIB})
  return()
endif()

iree_cc_binary(
  NAME
    iree-run-nlplearn-module-rocm
  SRCS
    "device_rocm.c"
    "iree-run-nlpnet-train.c"
  DEPS
    ::nlpnet_bytecode_module_c
    iree::tools::utils::image_util
    experimental::rocm::registration
    iree::base
    iree::hal
    iree::modules::hal
    iree::vm
    iree::vm::bytecode_module
)

# Build the bytecode from the nlpnet.mlir in iree/samples/models.
iree_bytecode_module(
  NAME
    nlpnet_bytecode_module
  SRC
    "nlp.mlir"
  C_IDENTIFIER
    "nlpnet_bytecode_module_c"
  FLAGS
    "-iree-mlir-to-vm-bytecode-module"
    "-iree-hal-target-backends=rocm"
    "-iree-rocm-target-chip=gfx908"
    "-iree-rocm-link-bc=true"
    "-iree-rocm-bc-dir=/home/stanley/nod/amdgcn/bitcode"
    "-iree-llvm-debug-symbols=false"
  PUBLIC
)
