export CUDA_VISIBLE_DEVICES=5
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
python -m vllm.entrypoints.openai.api_server \
    --host 0.0.0.0 --port 8073 \
    --model \
    --dtype=half \
    --api-key  \
    --enable-lora \
    --gpu_memory_utilization 0.9 \
    --max_model_len 10000 \
    --disable-frontend-multiprocessing \
    --disable-custom-all-reduce \
