#!/bin/bash
# Hotfix 2: corrected sglang launch flags for production
# This patch replaces the buggy sglang launch with the working config.
set -euo pipefail

PORT="${1:-8000}"
MODEL="${2:-Qwen/Qwen3.6-27B}"

echo "=== sglang production launch ==="
echo ""
echo "Replace your current sglang launch with:"
echo ""
echo "  python -m sglang.launch_server \\"
echo "    --model ${MODEL} \\"
echo "    --port ${PORT} \\"
echo "    --tensor-parallel-size 1 \\"
echo "    --context-length 32768 \\"
echo "    --mem-fraction-static 0.90 \\"
echo "    --cuda-graph-max-bs 32 \\"
echo "    --max-running-requests 64 \\"
echo "    --max-prefill-tokens 8192 \\"
echo "    --schedule-policy fcfs"
echo ""
echo "Key changes:"
echo "  - removed: --chat-template"
echo "  - removed: --enable-thinking"
echo "  - mem-fraction-static: 0.80 -> 0.90"
echo "  - cuda-graph-max-bs: 64 -> 32"
