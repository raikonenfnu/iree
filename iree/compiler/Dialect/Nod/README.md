# Nod Hardware Custom Dialect Plug-In

### MKL:
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
