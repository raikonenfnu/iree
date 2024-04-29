# Llama2 70B ROCm W7900 Artifacts

## Downloads

* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf.mlir
* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf_f16_int4.safetensors

## Run instruction

```sh
iree-build/install/bin/iree-compile Llama_2_70b_chat_hf.mlir --iree-preprocessing-pass-pipeline="builtin.module(util.func(iree-preprocessing-pad-to-intrinsics))"  --iree-codegen-llvmgpu-use-vector-distribution --iree-input-type=auto --iree-vm-bytecode-module-output-format=flatbuffer-binary --iree-hal-target-backends=rocm --iree-vulkan-target-triple=rdna3-unknown-linux --iree-stream-resource-max-allocation-size=4294967296 --mlir-print-op-on-diagnostic=false --iree-input-type=torch --mlir-print-op-on-diagnostic=false --iree-llvmcpu-target-cpu-features=host --iree-llvmcpu-target-triple=x86_64-linux-gnu --iree-stream-resource-index-bits=64 --iree-vm-target-index-bits=64 --iree-rocm-target-chip=gfx1100 --iree-vm-bytecode-module-strip-source-map=true --iree-opt-strip-assertions=true --iree-vm-target-truncate-unsupported-floats --iree-codegen-llvmgpu-enable-transform-dialect-jit=false --iree-preprocessing-transform-spec-filename=/home/stanley/nod/macroHipKernel/embedded_example_transform_spec.mlir -o llama_70b_rocm_vecdist.vmfb

python test_2511_128.py --vmfb_path=llama_70b_rocm_vecdist.vmfb --external_weight_path=Llama_2_70b_chat_hf_f16_int4.safetensors --device=rocm
```
