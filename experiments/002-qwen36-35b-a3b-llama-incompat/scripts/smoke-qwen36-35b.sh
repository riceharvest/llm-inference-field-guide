#!/bin/bash
# Smoke test for Qwen3.6-35B-A3B GGUF checksum verification
# Usage: ./smoke-qwen36-35b.sh <gguf_path>

set -euo pipefail

GGUF_PATH="${1:-}"
if [ -z "${GGUF_PATH}" ]; then
  echo "Usage: $0 <path/to/model.gguf>"
  exit 2
fi

if [ ! -f "${GGUF_PATH}" ]; then
  echo "FAIL: file not found: ${GGUF_PATH}"
  exit 1
fi

echo "=== Smoke test: GGUF integrity ==="
echo ""

SIZE=$(stat -c%s "${GGUF_PATH}" 2>/dev/null || stat -f%z "${GGUF_PATH}" 2>/dev/null || echo 0)
echo "File: ${GGUF_PATH}"
echo "Size: ${SIZE} bytes"

if [ "${SIZE}" -lt 1000000000 ]; then
  echo "FAIL: file looks too small for a 23 GB quant"
  exit 1
fi

echo "PASS: file size looks plausible"
echo ""
echo "Next step: compare against the publisher's SHA256."
echo "See scripts/verify-checksum.sh"
