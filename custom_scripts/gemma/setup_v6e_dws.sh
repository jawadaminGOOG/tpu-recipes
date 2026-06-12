#!/bin/bash
set -e

PROJECT_ID="northam-ce-mlai-tpu"
QR_NAME="gemma-v6e-qr"
TPU_NAME="gemma-v6e-vm"
ZONE="us-east5-b"
ACCELERATOR_TYPE="v6e-4"
VERSION="v2-alpha-tpuv6e"

echo "=== Gemma v6e Standard DWS Setup ==="
gcloud config set project "${PROJECT_ID}"

echo "-> Submitting Queued Resource request for ${TPU_NAME} in ${ZONE}..."

# Standard DWS Queued Resource modeled after the successful ones in your project.
# No spot, no flex-start, runs indefinitely once it gets capacity!
gcloud alpha compute tpus queued-resources create "${QR_NAME}" \
    --node-id="${TPU_NAME}" \
    --zone="${ZONE}" \
    --accelerator-type="${ACCELERATOR_TYPE}" \
    --runtime-version="${VERSION}" \
    --valid-until-duration="7d" \
    --quiet

echo "✅ Standard Queued Resource submitted successfully!"
echo "Status: gcloud compute tpus queued-resources describe ${QR_NAME} --zone=${ZONE}"
