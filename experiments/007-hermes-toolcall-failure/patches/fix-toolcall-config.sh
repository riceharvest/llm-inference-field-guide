#!/bin/bash
# Hotfix 3: corrected model config for Hermes tool-calling
# This patch replaces the turbo4 KV quant that corrupts long generations.
set -euo pipefail

MODEL_PATH="${1:?Usage: patch-007-fix-toolcall-config.sh <model_path>}"

echo "=== Hermes tool-calling config patch ==="
echo ""
echo "Replace your current model config with:"
echo ""
echo "Model: ${MODEL_PATH}"
echo ""
echo "Required settings:"
echo "  - KV cache: q8_0 (or bf16), NOT turbo4"
echo "  - Context: 131072"
echo "  - Parallel: 1"
echo "  - Reasoning: off for tool calls, or budget >= 1024"
echo ""
echo "Example llama-server command:"
echo ""
echo "  llama-server \\"
echo "    -m ${MODEL_PATH} \\"
echo "    -c 131072 \\"
echo "    -ngl 99 \\"
echo "    -ctk q8_0 -ctv q8_0 \\"
echo "    --reasoning off \\"
echo "    --parallel 1 \\"
echo "    --temp 0.6 --top-p 0.95 --top-k 20"
echo ""
echo "Why:"
echo "  - turbo4 KV quant corrupts long generations; error accumulates with sequence length"
echo "  - short direct API calls survive, long agent runs decay into fragments"
echo "  - reasoning with a small budget produces thinking-only output and empty assistant messages"
