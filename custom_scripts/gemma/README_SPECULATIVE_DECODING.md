# Speculative Decoding on TPU v6e for Gemma-4 31B

This directory contains the consolidated scripts and patches required to run **Speculative Decoding** with `google/gemma-4-31B-it` and the MTP draft assistant `google/gemma-4-31B-it-assistant` on a Trillium TPU v6e-4 VM using vLLM.

## Overview of Scripts

* **`run_speculative_server.sh <HF_TOKEN>`**
  This is the unified startup script that:
  1. Pulls the `vllm/vllm-tpu:nightly-20260611` container and starts it in `sleep infinity` mode.
  2. Hot-patches `transformers` from the Hugging Face main branch.
  3. Patches `processing_gemma4.py` to bypass a dummy validation error.
  4. Patches `gemma4_mtp.py` to support `Qwix` FP8 quantization syntax properly.
  5. Patches `tpu_runner.py` to preserve original `input_ids` and avoid crashes when feeding multimodal inputs to the speculative rejection sampler.
  6. Copies and runs `patch_qwix.py` and `configs.py`.
  7. Boots the vLLM API server in the background with `FP8` quantization on both models, `--tensor-parallel-size 4`, and `--num-speculative-tokens 4`.

* **`gate_speculative_performance.sh`**
  This script executes the standard `vllm bench serve` command against the running server using the `ShareGPT` dataset to verify performance.

## Prerequisites
* A running TPU v6e-4 instance (default assumed name: `gemma-v6e-vm` in zone `us-east5-b`).
* A valid Hugging Face access token with permission to download the Gemma-4-31B models.

## Usage

1. **Start the API Server**
   Run the setup script with your Hugging Face token:
   ```bash
   ./run_speculative_server.sh hf_your_token_here
   ```
   The script will stream logs to standard output while it waits for the `http://localhost:8000/v1/models` endpoint to become ready. *(Note: JAX compilation can take ~5-10 minutes on the first request)*.

2. **Run Text Benchmarks**
   Once the server is running, execute the text benchmark:
   ```bash
   ./gate_speculative_performance.sh
   ```
   > **Important:** The very first inference request triggers JAX compile-cache loading from disk, which heavily skews the benchmark duration and TTFT metrics. Run the benchmark a *second* time to collect accurate, warm-cache performance numbers.

3. **Run Multimodal Benchmarks**
   For multimodal benchmarks (random images + text), run:
   ```bash
   gcloud compute tpus tpu-vm ssh gemma-v6e-vm --zone=us-east5-b --command="
     sudo docker exec vllm-gemma4 vllm bench serve \
       --backend openai --endpoint /v1/completions \
       --model google/gemma-4-31B-it \
       --dataset-name random --random-input-len 1024 \
       --random-output-len 128 --random-mm-base-items-per_request 1 \
       --random-limit-mm-per-prompt '{\"image\": 1}' \
       --num-prompts 100 --request-rate 10
   "
   ```

## Expected Performance 
*(Metrics recorded with FP8 quantization and `--num-speculative-tokens 4` on TPU v6e-4)*

* **Active Token Throughput:** ~590 tok/s
* **Median Time Per Output Token (TPOT):** ~50.6 ms (20% faster than non-speculative BF16 baseline)
* **Median Inter-Token Latency (ITL):** ~115.5 ms
* **MTP Acceptance Rate:** ~63.9% (Average 3.56 tokens per draft step)
