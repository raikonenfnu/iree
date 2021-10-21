# MiniLM with Nod Module

### Prerequisite:
0. We assume we already have iree-opt and iree-translate with nod passes(--iree-linalg-to-nod) built
1. If not comment out the CMakeLists.txt first, build iree, and then proceed with next step.
2. Follow instruction to setup ModelCompiler [here](https://github.com/google/iree-samples/tree/main/ModelCompiler)
3. `cd iree-samples/ModelCompiler/nlp_models` 
4. `python huggingface_MiniLM_gen.py`
5. mv model.mlir > /path/to/iree/iree/samples/nlpnet/nlp.mlir
6. /path/to/iree-opt model.mlir --iree-linalg-to-nod --allow-unregistered-dialect > /path/to/iree/iree/samples/nlpnet/nlp\_nod.mlir

### Running with MKL:
**Setting Environment:**
```bash
#ls in $MKL_DIR should contain lib/intel64 where libmkl lives 
export MKL_DIR=/path/to/mkl
export LD_LIBRARY_PATH="$MKL_DIR/lib/intel64:$LD_LIBRARY_PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
```
**Running with Benchmark-Module:**
```bash
/path/to/iree-build/install/bin/iree-benchmark-module --driver=dylib --function_input=1x512xi32 --function_input=1x512xi32 --function_input=1x512xi32 --entry_function=predict --module_file=/path/to/iree-build/iree/samples/nlpnet/nlpnet_nod_bytecode_module.vmfb
```

**Benchmarking:**
```bash
perf record /path/to/iree-build/install/bin/iree-benchmark-module --driver=dylib --function_input=1x512xi32 --function_input=1x512xi32 --function_input=1x512xi32 --entry_function=predict --module_file=/path/to/iree-build/iree/samples/nlpnet/nlpnet_nod_bytecode_module.vmfb
perf report
```

**Tracing**
```bash
gdb
file /path/to/iree-build/install/bin/iree-benchmark-module
set env LD_LIBRARY_PATH /path/to/mkl/latest/lib/intel64
break <function_to_trace>
run --driver=dylib --function_input=1x512xi32 --function_input=1x512xi32 --function_input=1x512xi32 --entry_function=predict --module_file=/path/to/iree-build/iree/samples/nlpnet/nlpnet_nod_bytecode_module.vmfb
bt # as desired
c # as desired
```
