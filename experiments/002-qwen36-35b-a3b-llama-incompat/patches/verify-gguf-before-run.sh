#!/bin/bash
# Hotfix 1: verify GGUF before serving
# This patch adds a checksum gate before running llama-server.
set -euo pipefail

MODEL_PATH="${1:?Usage: patch-002-verify-gguf.sh <model_path> [expected_sha256]}"
EXPECTED_SHA256="${2:-}"

if [ -n "${EXPECTED_SHA256}" ]; then
  echo "Verifying ${MODEL_PATH} against ${EXPECTED_SHA256}..."
  ACTUAL_SHA256=$(sha256sum "${MODEL_PATH}" | awk '{print $1}')
  if [ "${ACTUAL_SHA256}" != "${EXPECTED_SHA256}" ]; then
    echo "FAIL: SHA256 mismatch"
    echo "Expected: ${EXPECTED_SHA256}"
    echo "Actual:   ${ACTUAL_SHA256}"
    exit 1
  fi
  echo "PASS: checksum matches"
else
  echo "WARN: no expected SHA256 provided, skipping verification"
fi

echo "You can now safely run:"
echo "  llama-server -m ${MODEL_PATH} ..."
