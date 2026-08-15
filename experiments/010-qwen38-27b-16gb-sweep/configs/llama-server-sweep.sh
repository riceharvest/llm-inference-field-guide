#!/bin/bash
# Launch Qwen3.8-27B for ONE sweep cell. Called by scripts/sweep.sh.
# One process per (quant, KV type, context). Binds loopback only.
#
# Env:
#   MODEL_PATH  absolute path to the GGUF
#   PORT        server port
#   CTX         context size in tokens
#   CTK, CTV    KV dtype override (q8_0/q4_0) — ONLY for standard K-quants.
#               Leave EMPTY for Unsloth UD quants: they embed their own V-cache
#               format and REJECT -ctk/-ctv (cf. experiment 009).
#   LLAMA_SERVER  path to the llama-server binary (default: llama-server)

set -eu
PORT="${PORT:-1235}"
CTX="${CTX:-16384}"
SRV="${LLAMA_SERVER:-llama-server}"

if [ -z "${MODEL_PATH:-}" ]; then
  echo "ERROR: MODEL_PATH not set" >&2; exit 1
fi

# Build KV args only when explicitly provided (standard quants).
KV_ARGS=()
if [ -n "${CTK:-}" ] && [ -n "${CTV:-}" ]; then
  KV_ARGS=(-ctk "$CTK" -ctv "$CTV")
fi

exec "$SRV" \
  -m "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" \
  -c "$CTX" -ngl 999 -fa on \
  --jinja -np 1 --no-warmup "${KV_ARGS[@]}"
