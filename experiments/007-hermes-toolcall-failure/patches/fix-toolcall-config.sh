#!/bin/bash
# Hotfix 3: corrected model config for Hermes tool-calling
# This patch fixes KV config and reasoning settings.
set -euo pipefail

MODEL_PATH="${1:?Usage: patch-007-fix-toolcall-config.sh <model_path>}"

echo "=== Hermes tool-calling config patch ==="
echo ""
echo "Replace your current model config with:"
echo ""
echo "Model: ${MODEL_PATH}"
echo ""
echo "Required settings:"
echo "  - KV cache: bf16"
echo "  - Context: 32768"
echo "  - Parallel: 1"
echo "  - Reasoning: off for tool calls, or budget >= 1024"
echo ""
echo "Example llama-server command:"
echo ""
echo "  llama-server \\"
echo "    -m ${MODEL_PATH} \\"
echo "    -c 32768 \\"
echo "    -ngl 999 \\"
echo "    -ctk bf16 -ctv bf16 \\"
echo "    --parallel 1 \\"
echo "    --temp 0.1 --top-p 0.9"
echo ""
echo "Why:"
echo "  - q8_0 KV + turbo4 caused KV corruption under long generations"
echo "  - parallel > 1 serializes turns and corrupts KV state"
echo "  - reasoning-only output produces empty assistant messages"
