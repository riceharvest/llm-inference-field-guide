#!/bin/bash
# Production MTP launcher for Qwen3.6-35B-A3B on a 12GB card.
# This is the winning config from experiment 008.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

exec llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" --port "${PORT}" \
  -ngl 32 -ncmoe 28 -t 14 -tb 16 \
  -b 2048 -ub 512 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 131072 -np 1 \
  --prio 2 --prio-batch 2 \
  --jinja --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --kv-unified --cache-idle-slots \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --no-warmup
