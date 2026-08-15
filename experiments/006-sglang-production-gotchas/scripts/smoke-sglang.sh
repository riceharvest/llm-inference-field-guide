#!/bin/bash
# Smoke test for sglang flag sanity
# Usage: ./smoke-sglang.sh <api_url> <model>

set -euo pipefail

API_URL="${1:-http://127.0.0.1:8000}"
MODEL="${2:-test-model}"
BASE_URL="${API_URL}/v1"

echo "=== Smoke test: sglang serving ==="
echo ""

# 1. Health
echo "[1/4] Health..."
if ! curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "FAIL: sglang not responding"
  exit 1
fi
echo "PASS: sglang is up"

# 2. Model listing
echo "[2/4] Model listing..."
MODELS=$(curl -sf --max-time 5 "${BASE_URL}/models" || true)
if ! echo "${MODELS}" | grep -qi "${MODEL}"; then
  echo "FAIL: ${MODEL} not listed"
  echo "Got: ${MODELS}"
  exit 1
fi
echo "PASS: ${MODEL} is listed"

# 3. Non-streaming completion
echo "[3/4] Non-streaming completion..."
RESPONSE=$(curl -sf --max-time 30 "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [{"role":"user","content":"Reply with OK"}],
    "max_tokens": 8,
    "temperature": 0.0,
    "stream": false
  }' || true)
if ! echo "${RESPONSE}" | grep -q '"choices"'; then
  echo "FAIL: no completion choices"
  echo "Got: ${RESPONSE}"
  exit 1
fi
echo "PASS: completion returned"

# 4. Empty response check
echo "[4/4] Empty response check..."
CONTENT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || true)
if [ -z "${CONTENT}" ]; then
  echo "FAIL: empty response content"
  exit 1
fi
echo "PASS: response has content: ${CONTENT}"

echo ""
echo "SMOKE PASS: sglang is serving"
