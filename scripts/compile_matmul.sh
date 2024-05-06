#! /usr/bin/env bash

# Make sure to symlink your `tools` build directory here.

set -xeuo pipefail

readonly INPUT="$1"
shift

./tools/iree-compile "$INPUT" \
  --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx1100 \
  --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-preprocessing-pad-to-intrinsics{intrinsic-multiple=4}))" \
  --iree-codegen-llvmgpu-use-vector-distribution \
  --iree-llvmgpu-enable-prefetch \
  --iree-stream-resource-max-allocation-size=4294967296 \
  --mlir-disable-threading --verify=true \
  -o "$(basename "$INPUT" .mlir).vmfb" "$@"
