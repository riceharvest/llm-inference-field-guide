#!/bin/bash
# Known-working launch for Gemma 4 12B QAT on a 12 GB card (RTX 3060).
# Upstream llama.cpp build. The UD quant sets its own KV format, so do NOT
# pass -ctk/-ctv. Flash attention is required for the UD V-cache quant.
#
# Usage: MODEL_PATH=/path/to/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf ./llama-server-qat.sh

set -euo pipefail

MODEL_PATH="${MODEL_PATH:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
DRAFT_PATH="${DRAFT_PATH:-gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf}"
PORT="${PORT:-1235}"

# Identity guard: refuse to launch a non-QAT file. The -qat- token in the
# filename is the only reliable local signal that the quant was trained in.
if [[ "${MODEL_PATH}" != *-qat-* ]]; then
  echo "ERROR: ${MODEL_PATH} is not a QAT artifact (-qat- missing in filename)." >&2
  echo "Gemma 4 at 4-bit without QAT fails Hermes agent tasks (0/13 observed)." >&2
  exit 1
fi

exec llama-server \
  -m "${MODEL_PATH}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c 196608 -ngl 99 -fa on \
  --jinja -np 1 --no-warmup \
  --model-draft "${DRAFT_PATH}"
