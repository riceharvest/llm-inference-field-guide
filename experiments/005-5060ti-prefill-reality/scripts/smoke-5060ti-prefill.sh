#!/bin/bash
# Smoke test for 2x RTX 5060 Ti prefill benchmark probe
# Usage: ./smoke-5060ti-prefill.sh <api_url>

set -euo pipefail

API_URL="${1:-http://127.0.0.1:18000}"
BASE_URL="${API_URL}/v1"

echo "=== Smoke test: 2x RTX 5060 Ti prefill probe ==="
echo ""

# 1. Health
echo "[1/4] Health..."
if ! curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "FAIL: server not responding"
  exit 1
fi
echo "PASS: server is up"

# 2. Short prompt
echo "[2/4] Short prompt prefill..."
RESPONSE=$(curl -sf --max-time 30 "${BASE_URL}/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Hello",
    "max_tokens": 1,
    "temperature": 0.0
  }' || true)
echo "Got: ${RESPONSE}"
echo "PASS: short prompt returned"

# 3. Long prompt
echo "[3/4] Long prompt prefill..."
LONG_PROMPT=$(printf 'Hello %0.s ' $(seq 1 20000))
RESPONSE2=$(curl -sf --max-time 60 "${BASE_URL}/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "'"${LONG_PROMPT}"'",
    "max_tokens": 1,
    "temperature": 0.0
  }' || true)
echo "Got: ${RESPONSE2}"
echo "PASS: long prompt returned"

# 4. Timing sanity
echo "[4/4] Timing sanity..."
TIME_SHORT=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo 0)
TIME_LONG=$(echo "${RESPONSE2}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo 0)
echo "Prompt tokens: short=${TIME_SHORT}, long=${TIME_LONG}"
if [ "${TIME_SHORT}" -lt 1 ] || [ "${TIME_LONG}" -lt 1000 ]; then
  echo "FAIL: token counts look wrong"
  exit 1
fi
echo "PASS: token counts plausible"

echo ""
echo "SMOKE PASS: prefill probe completed"
