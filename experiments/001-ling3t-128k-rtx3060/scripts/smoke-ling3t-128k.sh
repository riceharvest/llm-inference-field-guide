#!/bin/bash
# Smoke test for Ling-3.0-tiny 128K on RTX 3060
# Usage: SMOKE_MODEL=ling-3.0-tiny ./smoke-ling3t-128k.sh [PORT] [HOST]

set -euo pipefail

MODEL="${SMOKE_MODEL:-ling-3.0-tiny}"
PORT="${1:-${PORT:-12445}}"
HOST="${2:-${HOST:-127.0.0.1}}"
BASE_URL="http://${HOST}:${PORT}/v1"

echo "=== Smoke test: ${MODEL} on ${HOST}:${PORT} ==="
echo ""

# 1. Health check
echo "[1/4] Health check..."
if ! curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "FAIL: server not responding on ${BASE_URL}/health"
  exit 1
fi
echo "PASS: server is up"

# 2. Model list
echo "[2/4] Model list..."
MODELS=$(curl -sf --max-time 5 "${BASE_URL}/models" || true)
if ! echo "${MODELS}" | grep -qi "${MODEL}"; then
  echo "FAIL: ${MODEL} not in model list"
  echo "Got: ${MODELS}"
  exit 1
fi
echo "PASS: ${MODEL} is listed"

# 3. Short completion
echo "[3/4] Short completion..."
RESPONSE=$(curl -sf --max-time 30 "${BASE_URL}/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "prompt": "Hello, world!",
    "max_tokens": 16,
    "temperature": 0.0
  }' || true)
if ! echo "${RESPONSE}" | grep -q '"choices"'; then
  echo "FAIL: no completion choices in response"
  echo "Got: ${RESPONSE}"
  exit 1
fi
echo "PASS: completion returned"

# 4. Token count sanity
echo "[4/4] Token count sanity..."
PROMPT_TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('prompt_tokens',0))" 2>/dev/null || echo 0)
COMPLETION_TOKENS=$(echo "${RESPONSE}" | python3 -c "import sys,json; print(json.load(sys.stdin).get('usage',{}).get('completion_tokens',0))" 2>/dev/null || echo 0)
if [ "${PROMPT_TOKENS}" -lt 1 ] || [ "${COMPLETION_TOKENS}" -lt 1 ]; then
  echo "FAIL: token counts look wrong: prompt=${PROMPT_TOKENS}, completion=${COMPLETION_TOKENS}"
  exit 1
fi
echo "PASS: prompt_tokens=${PROMPT_TOKENS}, completion_tokens=${COMPLETION_TOKENS}"

echo ""
echo "SMOKE PASS: ${MODEL} is serving on ${HOST}:${PORT}"
