#!/bin/bash
# Convert Ling-3.0-tiny from HuggingFace to GGUF with proper 128K context metadata.
# The HF-converted GGUFs cap context at 32K — this script rebuilds with 128K metadata.

set -euo pipefail

HF_MODEL="${HF_MODEL:-inclusionAI/Ling-3.0-tiny-int4}"
OUTPUT="${OUTPUT:-./Ling-3.0-tiny-Q4_K_M-128K.gguf}"
CONVERT_SCRIPT="${CONVERT_SCRIPT:-./convert_hf_to_gguf.py}"

echo "=== Downloading model from HuggingFace ==="
huggingface-cli download "${HF_MODEL}" --local-dir /tmp/ling-hf --local-dir-use-symlinks false

echo "=== Converting to GGUF (Q4_K_M, 128K context) ==="
python3 "${CONVERT_SCRIPT}" \
  /tmp/ling-hf \
  --outfile "${OUTPUT}" \
  --outtype q4_k_m \
  --context-length 131072 \
  --model-name Ling-3.0-tiny

echo "=== Done: ${OUTPUT} ==="
ls -lh "${OUTPUT}"
