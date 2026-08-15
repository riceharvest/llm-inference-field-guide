#!/bin/bash
# Verify worker GPU matches your filters
# Usage: ./verify-worker.sh <expected_gpu> <expected_count>

EXPECTED_GPU="${1:?Expected GPU name (e.g., "RTX 5060 Ti")}"
EXPECTED_COUNT="${2:?Expected GPU count}"

echo "=== Checking running workers ==="
vastai show instances --raw | jq -r --arg gpu "$EXPECTED_GPU" --arg count "$EXPECTED_COUNT" '
  .[] | 
  select(.actual_status == "running") |
  {id, gpu_name, num_gpus, actual_status} |
  . as $w |
  if .gpu_name == $gpu and (.num_gpus | tostring) == $count then
    "✓ Worker \(.id): \(.gpu_name) x\(.num_gpus) — OK"
  else
    "✗ Worker \(.id): \(.gpu_name // "unknown") x\(.num_gpus // "unknown") — MISMATCH (expected: \($gpu) x\($count))"
  end
'
