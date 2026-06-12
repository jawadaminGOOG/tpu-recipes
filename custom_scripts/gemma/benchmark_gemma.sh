#!/bin/bash
# Runs the standard vLLM serving benchmark against the Gemma API

TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"

echo "-> Downloading benchmark dataset into the container..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="sudo docker exec vllm-gemma4 wget -qO ShareGPT.json https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"

echo "-> Running serving benchmark (100 prompts at 10 requests/sec)..."
gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="
sudo docker exec vllm-gemma4 vllm bench serve \
  --backend openai \
  --endpoint /v1/completions \
  --model google/gemma-4-31B-it \
  --dataset-name sharegpt \
  --dataset-path ShareGPT.json \
  --num-prompts 100 \
  --request-rate 10
"
