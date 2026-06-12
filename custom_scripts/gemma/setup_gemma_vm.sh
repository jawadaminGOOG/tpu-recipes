#!/bin/bash

# Configuration Variables
TPU_NAME="gemma-vm"
PROJECT_ID="northam-ce-mlai-tpu"
VERSION="v2-alpha-tpuv5"
# Prioritizing US zones known to have v5p
ZONES=("us-central1-a" "europe-west4-b" "europe-west1-b" "europe-west1-c" "europe-west1-d" "us-east1-d" "us-east5-b" "us-east5-c" "us-south1-a" "us-south1-b" "us-south1-c")

echo "=== Gemma TPU VM Setup Script (v5p, 8 chips) with Zone Fallback ==="

echo "-> Configuring default GCP project to ${PROJECT_ID}..."
gcloud config set project "${PROJECT_ID}"

for ZONE in "${ZONES[@]}"; do
    echo "------------------------------------------------------"
    echo "-> Checking if TPU VM '${TPU_NAME}' already exists in zone '${ZONE}'..."
    if gcloud compute tpus tpu-vm describe "${TPU_NAME}" --zone="${ZONE}" > /dev/null 2>&1; then
        echo "✅ TPU VM '${TPU_NAME}' already exists in ${ZONE}. Skipping creation."
        exit 0
    fi

    echo "-> Attempting to create Gemma TPU VM '${TPU_NAME}' (8 chips) in ${ZONE}..."
    # We are using --spot here because your large quota for v5p is preemptible
    if gcloud alpha compute tpus tpu-vm create "${TPU_NAME}" \
        --zone="${ZONE}" \
        --type="v5p" \
        --topology="2x2x1" \
        --version="${VERSION}" \
        --spot \
        --quiet; then
        
        echo "✅ TPU VM created successfully in ${ZONE}!"
        exit 0
    else
        echo "❌ Failed to create in ${ZONE} (likely out of capacity). Trying next zone..."
    fi
done

echo "❌ Exhausted all listed zones. Could not provision the TPU VM due to capacity limits."
exit 1
