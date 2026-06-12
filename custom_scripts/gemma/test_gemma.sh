#!/bin/bash
# Tests the Gemma inference server running on the TPU VM

TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"

echo "-> Testing Gemma inference server on ${TPU_NAME}..."

gcloud compute tpus tpu-vm ssh ${TPU_NAME} --zone=${ZONE} --command="curl -s http://localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    \"model\": \"google/gemma-4-31B-it\",
    \"messages\": [
      {
        \"role\": \"user\",
        \"content\": \"Explain how TPU architectures speed up matrix multiplication in 2 short sentences.\"
      }
    ],
    \"max_tokens\": 100
  }'"
