#! /usr/bin/env bash

set -eou pipefail

readonly BENCHMARK="$(realpath $1)"
shift

readonly BUILD_DIR="${HOME}/iree/build/relass"
readonly IREE_COMPILE="${BUILD_DIR}/tools/iree-compile"
readonly IREE_BENCHMARK="${BUILD_DIR}/tools/iree-benchmark-module"

readonly ITERS=1024

set -x

if ! cat /sys/class/drm/card0/device/power_dpm_force_performance_level | grep -q profile_peak ; then
  echo profile_peak | sudo tee /sys/class/drm/card0/device/power_dpm_force_performance_level
fi

"${IREE_COMPILE}" "${BENCHMARK}" \
  --iree-hal-target-backends=vulkan-spirv \
  --iree-vulkan-target-triple=rdna3-7900-linux \
  --iree-hal-benchmark-dispatch-repeat-count="${ITERS}" \
  --iree-hal-dump-executable-files-to=dumps \
  -o benchmark.vmfb

spirv-cross -V dumps/*.spv || spirv-dis dumps/*.spv

"${IREE_BENCHMARK}" --module=benchmark.vmfb --device=vulkan --function=main \
  --batch_size="${ITERS}"


