#!/bin/bash
set -e

PROJECT_ID="northam-ce-mlai-tpu"
QR_NAME="gemma-qr"
TPU_NAME="gemma-vm"
VERSION="v2-alpha-tpuv5"
ZONES=("us-central1-a" "europe-west4-b" "europe-west1-b" "europe-west1-c" "europe-west1-d" "us-east1-d" "us-east5-b" "us-east5-c" "us-south1-a" "us-south1-b" "us-south1-c")

echo "=== Gemma DWS Flex Start Setup ==="
gcloud config set project "${PROJECT_ID}"

for ZONE in "${ZONES[@]}"; do
    echo "------------------------------------------------------"
    echo "-> Attempting to submit Queued Resource for ${TPU_NAME} in ${ZONE}..."

    if gcloud alpha compute tpus queued-resources create "${QR_NAME}" \
        --node-id="${TPU_NAME}" \
        --zone="${ZONE}" \
        --type="v5p" \
        --topology="2x2x1" \
        --runtime-version="${VERSION}" \
        --provisioning-model="flex-start" \
        --valid-until-duration="7d" \
        --max-run-duration="24h" \
        --quiet; then
        
        echo "✅ Queued Resource submitted successfully in ${ZONE}!"
        exit 0
    else
        echo "❌ Failed to submit flex-start request in ${ZONE}. Trying next..."
    fi
done

echo "❌ Exhausted all zones. Flex-start might not be supported for v5p in any of them."
exit 1

