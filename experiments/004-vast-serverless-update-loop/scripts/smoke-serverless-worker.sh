#!/bin/bash
# Smoke test for serverless worker sanity
# Usage: ./smoke-serverless-worker.sh <api_url>

set -euo pipefail

API_URL="${1:-}"
if [ -z "${API_URL}" ]; then
  echo "Usage: $0 <serverless_api_url>"
  exit 2
fi

echo "=== Smoke test: serverless inference endpoint ==="
echo ""

# 1. Health/model listing
echo "[1/3] Endpoint reachable..."
if ! curl -sf --max-time 10 "${API_URL}/health" >/dev/null; then
  echo "FAIL: endpoint not reachable"
  exit 1
fi
echo "PASS: endpoint is reachable"

# 2. Minimal completion
echo "[2/3] Minimal completion..."
RESPONSE=$(curl -sf --max-time 30 "${API_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role":"user","content":"Reply with OK"}],
    "max_tokens": 8,
    "temperature": 0.0
  }' || true)
if ! echo "${RESPONSE}" | grep -q '"choices"'; then
  echo "FAIL: no completion choices"
  echo "Got: ${RESPONSE}"
  exit 1
fi
echo "PASS: completion returned"

# 3. Empty/error response check
echo "[3/3] Response sanity..."
EMPTY=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print('yes' if not d.get('choices',[{}])[0].get('message',{}).get('content') else 'no')" 2>/dev/null || echo yes)
if [ "${EMPTY}" = "yes" ]; then
  echo "FAIL: empty response from endpoint"
  exit 1
fi
echo "PASS: response has content"

echo ""
echo "SMOKE PASS: endpoint is healthy"
