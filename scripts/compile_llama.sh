#! /usr/bin/env bash

# Make sure to symlink your `tools` build directory here.

set -xeuo pipefail

./tools/iree-compile --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-preprocessing-pad-to-intrinsics))" \
  --iree-codegen-llvmgpu-use-vector-distribution --iree-input-type=auto \
  --iree-hal-target-backends=rocm --iree-rocm-target-chip=gfx1100 \
  --iree-stream-resource-max-allocation-size=4294967296 \
  --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 \
  --iree-vm-target-truncate-unsupported-floats --iree-codegen-llvmgpu-enable-transform-dialect-jit=false \
  --iree-preprocessing-transform-spec-filename=spec.mlir "$@"
