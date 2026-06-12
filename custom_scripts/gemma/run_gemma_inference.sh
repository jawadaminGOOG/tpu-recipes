#!/bin/bash

# Configuration Variables
TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"

if [ -z "$1" ]; then
  echo "Usage: ./run_gemma_inference.sh <HF_TOKEN>"
  exit 1
fi
HF_TOKEN=$1

echo "-> Stopping any existing containers..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command='sudo docker rm -f vllm-gemma4 || true'

echo "-> Starting Gemma 4 inference server on TPU VM..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker run -d \
  --name vllm-gemma4 \
  --privileged \
  --network host \
  --shm-size 16g \
  -v /dev/shm:/dev/shm \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -e HF_TOKEN='${HF_TOKEN}' \
  -e HF_HOME='/root/.cache/huggingface' \
  -e USE_BATCHED_RPA_KERNEL=1 \
  -e MOE_REQUANTIZE_WEIGHT_DTYPE=float8_e4m3fn \
  vllm/vllm-tpu:nightly-20260514-4690ef3-bf0d2dc \
  python3 -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 --port 8000 --seed 42 \
  --tensor-parallel-size 4 \
  --max-model-len 8192 \
  --max-num-batched-tokens 4096 \
  --block-size 256 \
  --download-dir /root/.cache/huggingface \
  --no-enable-prefix-caching \
  --additional_config '{\"quantization\": { \"qwix\": { \"rules\": [{ \"module_path\": \".*\", \"weight_qtype\": \"float8_e4m3fn\", \"act_qtype\": \"float8_e4m3fn\"}]}}}' \
  --model google/gemma-4-31B-it \
  --kv-cache-dtype fp8 \
  --async-scheduling \
  --gpu-memory-utilization 0.90 \
  --disable_chunked_mm_input \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4
"
