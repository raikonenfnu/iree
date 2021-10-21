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

if(NOT ${IREE_TARGET_BACKEND_DYLIB-LLVM-AOT})
  return()
endif()

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

