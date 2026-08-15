#!/bin/bash
# DFlash launcher for Qwen3.6-35B-A3B. REPRODUCTION ONLY.
# This config measured 42-52% SLOWER than built-in MTP on RTX 3060.
# Kept so you can verify the loss yourself. Do not use in production.
#
# Notes:
# - The DFlash draft runtime needs ~3 GB (cross-attention buffers) even
#   though the draft GGUF is ~279 MB. Hence -ngl 28, not 32.
# - --spec-branch-budget 0 = flat drafting. Setting it to 3 with
#   --spec-draft-top-k 4 (DDTree) aborts the server on the first request
#   at long context: output_reserve OOM inside draft_tree.
set -euo pipefail

MODEL_PATH="${MODEL_PATH:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
DRAFT_PATH="${DRAFT_PATH:-qwen36-35b-a3b-dflash-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
HOST="${HOST:-127.0.0.1}"

exec llama-server \
  -m "${MODEL_PATH}" \
  --host "${HOST}" --port "${PORT}" \
  -ngl 28 -ncmoe 28 -t 14 -tb 16 \
  -b 2048 -ub 512 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 32768 -np 1 \
  --jinja --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --spec-type dflash \
  --spec-draft-model "${DRAFT_PATH}" \
  --spec-draft-n-max 8 \
  --spec-dflash-cross-ctx 512 \
  --spec-branch-budget 0 \
  --spec-draft-temp auto \
  --no-warmup --no-mmap
