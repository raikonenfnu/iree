#! /usr/bin/env bash

# Make sure to symlink your `tools` build directory here.

set -xeuo pipefail

readonly INPUT="$1"
shift

./tools/iree-benchmark-module --module="$INPUT" --device=hip \
  --function=main --input=2511x64x128xf16 --device_allocator=caching \
  --benchmark_repetitions=5
