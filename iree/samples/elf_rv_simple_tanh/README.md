# Torch to RISC-V

## Initial/First Time Setups
#### Coming Soon!

## Environment Setup
```sh
export PYTHONPATH=$PYTHONPATH:/nodclouddata/stanley/npcomp-build/python_packages/npcomp_core
export PYTHONPATH=$PYTHONPATH:/nodclouddata/stanley/npcomp-build/python_packages/npcomp_torch
export RISCV_TOOLCHAIN_ROOT=/home/stanley/riscv/toolchain/clang/linux/RISCV
```
## Torch to MLIR RISCV
```sh
python tanh_e2e_codegen.py

/nodclouddata/stanley/iree-riscv/iree/tools/iree-translate -iree-mlir-to-vm-bytecode-module \
-iree-hal-target-backends=dylib-llvm-aot -iree-llvm-target-triple=riscv64-pc-linux-elf \
-iree-llvm-target-cpu=generic-rv64 -iree-llvm-target-abi=lp64d \
-iree-llvm-target-cpu-features=+m,+a,+f,+d,+experimental-v -riscv-v-vector-bits-min=128 \
-riscv-v-fixed-length-vector-lmul-max=8 -iree-input-type=mhlo -iree-llvm-link-embedded=true -iree-llvm-keep-linker-artifacts=true \
--iree-llvm-loop-vectorization --iree-llvm-slp-vectorization \
/tmp/tanh.mlir -o /tmp/tanh.vmfb 2> /tmp/tanh-ir.txt

mv /tmp/forward_dispatch_*.so sharedobject/simple_tanh_riscv_64.so
```

## Building example
```sh
cmake -G Ninja -B /nodclouddata/stanley/iree-exec-riscv/ \
  -DCMAKE_TOOLCHAIN_FILE="./build_tools/cmake/riscv.toolchain.cmake" \
  -DIREE_HOST_BINARY_ROOT=$(realpath /nodclouddata/stanley/iree-riscv/install) \
  -DRISCV_CPU=rv64 \
  -DIREE_BUILD_COMPILER=OFF \
  -DRISCV_TOOLCHAIN_ROOT=${RISCV_TOOLCHAIN_ROOT} \
  .

cmake --build /nodclouddata/stanley/iree-exec-riscv/
```

## Running Example
```sh
$QEMU_BIN -cpu rv64,x-v=true,x-k=true,vlen=256,elen=64,vext_spec=v1.0 -L $RISCV_TOOLCHAIN_ROOT/sysroot ./elf_module_test_binary_tanh
```
