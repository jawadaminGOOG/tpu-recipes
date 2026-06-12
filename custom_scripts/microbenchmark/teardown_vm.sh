#!/bin/bash

# Stop script execution on error
set -e

# Configuration Variables
TPU_NAME="recipe-vm"
PROJECT_ID="northam-ce-mlai-tpu"
ZONE="us-east5-b"

echo "=== TPU VM Teardown Script ==="

# 1. Set the default project
echo "-> Configuring default GCP project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

# 2. Check if the TPU VM exists and delete it if it does
echo "-> Checking if TPU VM '${TPU_NAME}' exists in zone '${ZONE}'..."
if gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" > /dev/null 2>&1; then
    echo "-> Deleting TPU VM '${TPU_NAME}'..."
    # The --quiet flag automatically confirms the prompt to delete
    gcloud compute tpus tpu-vm delete "${TPU_NAME}" \
        --zone="${ZONE}" \
        --quiet
    echo "✅ TPU VM deleted successfully."
else
    echo "✅ TPU VM '${TPU_NAME}' does not exist. Nothing to delete."
fi

echo "=== Teardown Complete ==="
