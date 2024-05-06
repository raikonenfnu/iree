# Llama2 70B ROCm W7900 Artifacts

## Downloads

* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf.mlir
* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf_f16_int4.safetensors

## Venv setup instruction

```shell
ln -s /path/to/iree-build/tools .
ln -s /path/to/iree-build/.env .

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip uninstall iree-runtime

source .env
export PYTHONPATH
```

## E2E run instruction

### Model Setup Instructions:

```shell
# Download IR and weights.
wget https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf.mlir
wget https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf_f16_int4.safetensors

# Compile to VMFB/Runnables.
./compile_llama.sh Llama_2_70b_chat_hf.mlir -o llama.vmfb
```

### Running b_ai benchmark
```shell
python run_e2e.py --vmfb_path=llama.vmfb --external_weight_path=Llama_2_70b_chat_hf_f16_int4.safetensors --device=rocm --benchmar b_ai
```

### Running MLC benchmark
```shell
python run_e2e.py --vmfb_path=llama.vmfb --external_weight_path=Llama_2_70b_chat_hf_f16_int4.safetensors --device=rocm --benchmar b_ai
```

## Matmul run instruction

```shell
./compile_matmul.sh dynamic_matmul_1.mlir

tools/iree-benchmark-module --module=dynamic_matmul_1.vmfb --device=rocm --function=main --input=2511x64x128xf16 \
  --device_allocator=caching --benchmark_repetitions=5
```
