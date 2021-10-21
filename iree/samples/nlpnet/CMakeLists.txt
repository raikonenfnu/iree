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

if(NOT "${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT}" OR
   NOT "${IREE_HAL_DRIVER_DYLIB_SYNC}" OR
   NOT "${BACKEND}")
  return()
endif()


# iree_cc_test(
#   NAME
#     iree-run-nlp-module
#   SRCS
#     "iree-run-nlpnet-module.c"
#   DEPS
#     ::nlpnet_bytecode_module_c
#     iree::compiler::Dialect::Nod::IR::module
#     iree::base
#     iree::base::logging
#     iree::hal
#     iree::hal::dylib::registration::sync
#     iree::modules::hal
#     iree::vm
#     iree::vm::bytecode_module
#     iree::vm::cc
# )

# iree_cc_test(
#   NAME
#     iree-run-nlp-nod-module
#   SRCS
#     "iree-run-nlpnet-module.c"
#   DEPS
#     ::nlpnet_nod_bytecode_module_c
#     iree::compiler::Dialect::Nod::IR::module
#     iree::base
#     iree::base::logging
#     iree::hal
#     iree::hal::dylib::registration::sync
#     iree::modules::hal
#     iree::vm
#     iree::vm::bytecode_module
#     iree::vm::cc
# )
# target_compile_definitions(iree_samples_nlpnet_iree-run-nlp-nod-module PRIVATE USE_NOD=1)

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
  "-iree-hal-target-backends=dylib-llvm-aot"
  "-iree-llvm-debug-symbols=false"
  PUBLIC
)

# Build the bytecode from the nlpnet.mlir in iree/samples/models.
iree_bytecode_module(
  NAME
    nlpnet_nod_bytecode_module
  SRC
    "nlp_nod.mlir"
  C_IDENTIFIER
    "nlpnet_nod_bytecode_module_c"
  FLAGS
  "-iree-mlir-to-vm-bytecode-module"
  "-iree-hal-target-backends=dylib-llvm-aot"
  "-iree-llvm-debug-symbols=false"
  PUBLIC
)

