#!/bin/bash

# Stop script execution on error
set -e

# Configuration Variables
TPU_NAME="recipe-vm"
PROJECT_ID="northam-ce-mlai-tpu"
ZONE="us-east5-b"
ACCELERATOR_TYPE="v6e-1"
VERSION="v2-alpha-tpuv6e"

echo "=== TPU VM Setup Script ==="

# 1. Set the default project
echo "-> Configuring default GCP project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# 2. Check if the TPU VM already exists to ensure idempotency
echo "-> Checking if TPU VM '${TPU_NAME}' already exists in zone '${ZONE}'..."
if gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" > /dev/null 2>&1; then
    echo "✅ TPU VM '${TPU_NAME}' already exists. Skipping creation."
else
    echo "-> Creating TPU VM '${TPU_NAME}'..."
    # We do not quote variables here to avoid breaking the gcloud command structure if they are empty, 
    # though they are guaranteed not to be in this script.
    gcloud compute tpus tpu-vm create "${TPU_NAME}" \
        --zone="${ZONE}" \
        --accelerator-type="${ACCELERATOR_TYPE}" \
        --version="${VERSION}"
    echo "✅ TPU VM created successfully."
fi

echo "=== Setup Complete ==="
