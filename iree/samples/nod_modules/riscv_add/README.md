# IREE "Relu Static Library for RISCV Backend" sample

## Environment Setup
```sh
export RISCV_TOOLCHAIN_ROOT=/home/stanley/riscv/toolchain/clang/linux/RISCV
```
## Background

IREE's `static_library_loader` allows applications to inject a set of static
libraries that can be resolved at runtime by name. This can be particularly
useful on "bare metal" or embedded systems running IREE that lack operating
systems or the ability to load shared libraries in binaries.

When static library output is enabled, `iree-translate` produces a separate
static library to compile into the target program. At runtime bytecode module
instructs the VM which static libraries to load exported functions from the
model.

## Instructions
_Note: run the following commands from IREE's github repo root._

1. Configure CMake for building the static library then demo. You'll need to set
the flags building samples, the compiler, and the `dylib-llvm-aot`
driver/backend. See
[here](https://google.github.io/iree/building-from-source/getting-started/)
for general instructions on building using CMake):

  ```shell
  cmake -B ../iree-build/
    -DIREE_BUILD_SAMPLES=ON \
    -DIREE_TARGET_BACKENDS_TO_BUILD=DYLIB-LLVM-AOT \
    -DIREE_HAL_DRIVERS_TO_BUILD=DYLIB \
    -DIREE_BUILD_COMPILER=ON \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo .
  ```

2. Build the `riscv_add_exec` CMake target to create the static demo. This
target has several dependencies that will translate `riscv_add.mlir` into a
static library (`riscv_add.h` & `riscv_add.c`) as well as a bytecode module
(`riscv_add.vmfb`) which are finally built into the demo binary:

  ```shell
  cmake --build ../iree-build/ --target iree_samples_static_library_demo
  ```

3. Run the sample binary:

  ```shell
  export RISCV_TOOLCHAIN_ROOT=${HOME}/riscv/toolchain/clang/linux/RISCV
  $QEMU_BIN -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 -L $RISCV_TOOLCHAIN_ROOT/sysroot ./riscv_add_exec
  # Output: static_library_run passed
  ```

### Changing compilation options

The steps above build both the compiler for the host (machine doing the
compiling) and the demo for the target using same options as the host machine.
If you wish to target a different deployment other than the host, you'll need to
compile the library and demo with different options.

For example, see
[documentation](https://google.github.io/iree/building-from-source/android/)
on cross compiling on Android.

Note: separating the target from the host will require modifying dependencies in
the demos `CMakeLists.txt`. See included comments for more info.
