#!/bin/bash
TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"
HF_TOKEN=$1
DRAFT_MODEL="google/gemma-4-31B-it-assistant"

echo "-> Stopping existing vLLM containers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command='sudo docker rm -f vllm-gemma4 || true'

echo "-> Booting container..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker run -d --name vllm-gemma4 --privileged --network host --shm-size 16g \
  -v /dev/shm:/dev/shm -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN='${HF_TOKEN}' -e HF_HOME='/root/.cache/huggingface' \
  -e USE_BATCHED_RPA_KERNEL=1 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
  vllm/vllm-tpu:nightly-20260611-1043491-248e33c \
  sleep infinity
"

echo "-> Upgrading transformers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 pip install git+https://github.com/huggingface/transformers.git
"

echo "-> Starting API server..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec -d vllm-gemma4 bash -c \"python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 --seed 42 --tensor-parallel-size 4 \
  --max-model-len 8192 --max-num-batched-tokens 4096 \
  --block-size 256 \
  --download-dir /root/.cache/huggingface \
  --no-enable-prefix-caching \
  --additional_config '{\"quantization\": { \"qwix\": { \"rules\": [{ \"module_path\": \"^model\\\\..*\", \"weight_qtype\": \"float8_e4m3fn\", \"act_qtype\": \"float8_e4m3fn\"}]}}}' \
  --model google/gemma-4-31B-it \
  --speculative-config '{\\\"model\\\": \\\"${DRAFT_MODEL}\\\", \\\"num_speculative_tokens\\\": 5}' \
  --kv-cache-dtype fp8 \
  --trust-remote-code \
  --async-scheduling --gpu-memory-utilization 0.90 \
  --disable_chunked_mm_input --enable-auto-tool-choice --tool-call-parser gemma4 > /vllm.log 2>&1\"
"

echo "-> Waiting 60 seconds for it to compile/crash..."
sleep 60

echo "-> Reading logs..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 cat /vllm.log
"
