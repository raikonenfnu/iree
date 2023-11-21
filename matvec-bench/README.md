# MatVec Utilities

[[TOC]]

## Files

1. `run.sh` -- a script to run the matvec microbenchmark on Vulkan.
2. `vmt.mlir` -- input mlir with a matvec operations (actually vector-times-matrix-transposed).
3. `target-spirv.mlir` -- spirv demonstrating the target output for vmt.
4. `out.log` -- output showing the intended transformations in the vector-reduction-to-gpu pass.
5. `all.log` -- output of the whole compilation pipeline.

## Useful Commands

1. Run the microbenchmark: `run.sh vmt.mlir`
2. Compile vmt and dump relevant into:

   ```shell
   tools/iree-compile vmt.mlir --iree-hal-target-backends=vulkan-spirv --iree-vulkan-target-triple=rdna3-7900-linux --iree-hal-benchmark-dispatch-repeat-count=1 --iree-hal-dump-executable-files-to=dumps -o benchmark.vmfb --debug-only=iree-codegen-vector-reduction-to-gpu &> out.log
   ```
