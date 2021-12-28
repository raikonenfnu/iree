# CMake Options and Variables

## Frequently-used CMake Variables

#### `CMAKE_BUILD_TYPE`:STRING

Sets the build type. Possible values are `Release`, `Debug`,
`RelWithDebInfo` and `MinSizeRel`. If unset, build type is set to `Release`.

#### `CMAKE_<LANG>_COMPILER`:STRING

This is the command that will be used as the `<LANG>` compiler, which are `C`
and `CXX` in IREE. These variables are set to compile IREE with `clang` or
rather `clang++`. Once set, these variables can not be changed.

## IREE-specific CMake Options and Variables

This gives a brief explanation of IREE specific CMake options and variables.

#### `IREE_ENABLE_RUNTIME_TRACING`:BOOL

Enables instrumented runtime tracing. Defaults to `OFF`.

#### `IREE_ENABLE_COMPILER_TRACING`:BOOL

Enables instrumented compiler tracing. This requires that
`IREE_ENABLE_RUNTIME_TRACING` also be set. Defaults to `OFF`.

#### `IREE_ENABLE_EMITC`:BOOL

Enables the build of the out-of-tree MLIR dialect EmitC. Defaults to `OFF`.

#### `IREE_BUILD_COMPILER`:BOOL

Builds the IREE compiler. Defaults to `ON`.

#### `IREE_BUILD_TESTS`:BOOL

Builds IREE unit tests. Defaults to `ON`.

#### `IREE_BUILD_DOCS`:BOOL

Builds IREE documentation. Defaults to `OFF`.

#### `IREE_BUILD_SAMPLES`:BOOL

Builds IREE sample projects. Defaults to `ON`.

#### `IREE_BUILD_PYTHON_BINDINGS`:BOOL

Builds the IREE python bindings. Defaults to `OFF`.

#### `IREE_BUILD_BINDINGS_TFLITE`:BOOL

Builds the IREE TFLite C API compatibility shim. Defaults to `ON`.

#### `IREE_BUILD_BINDINGS_TFLITE_JAVA`:BOOL

Builds the IREE TFLite Java bindings with the C API compatibility shim. Defaults to `ON`.

#### `IREE_BUILD_EXPERIMENTAL_REMOTING`:BOOL

Builds experimental remoting component. Defaults to `OFF`.

#### `IREE_HAL_DRIVER_DEFAULTS`:BOOL

Default setting for each `IREE_HAL_DRIVER_*` option.

#### `IREE_HAL_DRIVER_*`:BOOL

Individual options enabling the build for each runtime HAL driver.

#### `IREE_TARGET_BACKEND_DEFAULTS`:BOOL

Default setting for each `IREE_TARGET_BACKEND_*` option.

#### `IREE_TARGET_BACKEND_*`:BOOL

Individual options enabling the build for each compiler target backend.

#### `IREE_DEV_MODE`:BOOL

Configure settings to optimize for IREE development (as opposed to CI or
release). Defaults to `OFF`. For example, this will downgrade some compiler
diagnostics from errors to warnings.

#### `IREE_ENABLE_LLD`:BOOL

Use lld when linking. Defaults to `OFF`. This option is equivalent to
`-DIREE_USE_LINKER=lld`. The option `IREE_ENABLE_LLD` and `IREE_USE_LINKER` can
not be set at the same time.

#### `IREE_ENABLE_ASAN`:BOOL

Enable [address sanitizer](https://clang.llvm.org/docs/AddressSanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_ENABLE_MSAN`:BOOL

Enable [memory sanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_ENABLE_TSAN`:BOOL

Enable [thread sanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) if
the current build type is Debug and the compiler supports it.

#### `IREE_MLIR_DEP_MODE`:STRING

Defines the MLIR dependency mode. Case-sensitive. Can be `BUNDLED`, `DISABLED`
or `INSTALLED`. Defaults to `BUNDLED`. If set to `INSTALLED`, the variable
`MLIR_DIR` needs to be passed and that LLVM needs to be compiled with
`LLVM_ENABLE_RTTI` set to `ON`.

#### `IREE_BUILD_TENSORFLOW_COMPILER`:BOOL

Enables building of the TensorFlow to IREE compiler under
`integrations/tensorflow`, including some native binaries and Python packages.
Note that TensorFlow's build system is bazel and this will require having
previously built (or installed) the iree-import-tf at the path specified by
`IREE_TF_TOOLS_ROOT`.

#### `IREE_BUILD_TFLITE_COMPILER`:BOOL

Enables building of the TFLite to IREE compiler under `integrations/tensorflow`,
including some native binaries and Python packages. Note that TensorFlow's build
system is bazel and this will require having previously built (or installed) the
iree-import-tf at the path specified by `IREE_TF_TOOLS_ROOT`.

#### `IREE_BUILD_XLA_COMPILER`:BOOL

Enables building of the XLA to IREE compiler under `integrations/tensorflow`,
including some native binaries and Python packages. Note that TensorFlow's build
system is bazel and this will require having previously built (or installed) the
iree-import-tf at the path specified by `IREE_TF_TOOLS_ROOT`.

#### `IREE_TF_TOOLS_ROOT`:STRING

Path to prebuilt TensorFlow integration binaries to be used by the Python
bindings. Defaults to
"${CMAKE_SOURCE_DIR}/integrations/tensorflow/bazel-bin/iree_tf_compiler", which
is where they would be placed by a `bazel build` invocation.

## MLIR-specific CMake Options and Variables

#### `MLIR_DIR`:STRING

Specifies the path where to look for the installed MLIR/LLVM packages. Required
if `IREE_MLIR_DEP_MODE` is set to `INSTALLED`.

## Cross-compilation

When cross compiling (using a toolchain file like
[`android.toolchain.cmake`](https://android.googlesource.com/platform/ndk/+/master/build/cmake/android.toolchain.cmake)),
first build and install IREE's tools for your host configuration, then use the
`IREE_HOST_BINARY_ROOT` CMake option to point the cross compiled build at the
host tools.
