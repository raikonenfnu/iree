# Llama2 70B ROCm W7900 Artifacts

## Downloads

* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf.mlir
* https://sharkpublic.blob.core.windows.net/sharkpublic/ian/Llama_2_70b_chat_hf_f16_int4.safetensors

## Venv setup instruction

```sh
pip install -r requirements.txt
pip uninstall iree-runtime
sorce /path/to/iree-build/.env && export PYTHONPATH
```

## E2E run instruction

```sh
./compile_llama.sh Llama_2_70b_chat_hf.mlir -o llama.vmfb

python run_e2e.py --vmfb_path=llama.vmfb --external_weight_path=Llama_2_70b_chat_hf_f16_int4.safetensors --device=rocm
```
