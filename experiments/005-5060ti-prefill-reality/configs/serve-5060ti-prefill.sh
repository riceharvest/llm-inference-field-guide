#!/usr/bin/env bash
# Reproducible vLLM serving command for the 5060 Ti prefill benchmark.
# Uses generic placeholders so the recipe stays machine-agnostic.

set -euo pipefail

MODEL="${MODEL_PATH:-Qwen3.6-27B-NVFP4}"
PORT="${PORT:-18000}"
HOST="${HOST:-127.0.0.1}"
TP="${TP:-2}"
MAX_LEN="${MAX_LEN:-120000}"
GPU_MEM="${GPU_MEM:-0.90}"
MAX_SEQS="${MAX_SEQS:-4}"

python -m vllm.entrypoints.openai.api_server \
  --model "${MODEL}" \
  --port "${PORT}" \
  --host "${HOST}" \
  --tensor-parallel-size "${TP}" \
  --max-model-len "${MAX_LEN}" \
  --gpu-memory-utilization "${GPU_MEM}" \
  --enable-chunked-prefill \
  --max-num-seqs "${MAX_SEQS}"
