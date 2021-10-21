# Nod Hardware Custom Dialect Plug-In

### MKL:
**Build direction**
```bash
#ls in $MKL_DIR should contain lib/intel64 where libmkl lives 
export MKL_DIR=/path/to/mkl
cmake -G Ninja -B /path/to/iree-build -DCMAKE_INSTALL_PREFIX=/path/to/iree-build/install -DCMAKE_BUILD_TYPE=RelWithDebInfo -DIREE_VM_EXECUTION_TRACING_ENABLE=1 -DMKL_DIR=$MKL_DIR /path/to/iree
```
**Setting Environment:**
```bash
export LD_LIBRARY_PATH="$MKL_DIR/lib/intel64:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```
**Example Usage with Benchmark-Module:**
```bash
export LD_LIBRARY_PATH="$MKL_DIR/lib/intel64:$LD_LIBRARY_PATH"
/path/to/iree-build-host/install/bin/iree-benchmark-module --driver=dylib --function_input=1x512xi32 --function_input=1x512xi32 --function_input=1x512xi32 --entry_function=predict --module_file=nlpnet_nod_bytecode_module.vmfb
```
