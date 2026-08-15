#!/bin/bash
# Smoke test for SSE proxy streaming
# Usage: ./smoke-sse-proxy.sh <proxy_port> <upstream_port>

set -euo pipefail

PROXY_PORT="${1:-8081}"
UPSTREAM_PORT="${2:-8080}"
PROXY_URL="http://127.0.0.1:${PROXY_PORT}"
UPSTREAM_URL="http://127.0.0.1:${UPSTREAM_PORT}"

echo "=== Smoke test: SSE proxy on port ${PROXY_PORT} -> upstream ${UPSTREAM_PORT} ==="
echo ""

# 1. Proxy health
echo "[1/4] Proxy health..."
if ! curl -sf --max-time 5 "${PROXY_URL}/health" >/dev/null; then
  echo "FAIL: proxy not responding"
  exit 1
fi
echo "PASS: proxy is up"

# 2. Upstream health
echo "[2/4] Upstream health..."
if ! curl -sf --max-time 5 "${UPSTREAM_URL}/health" >/dev/null; then
  echo "FAIL: upstream not responding"
  exit 1
fi
echo "PASS: upstream is up"

# 3. Streaming behavior
echo "[3/4] Streaming behavior..."
STREAM_OUT=$(mktemp)
trap 'rm -f "${STREAM_OUT}"' EXIT

curl -N --max-time 15 "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "test",
    "messages": [{"role":"user","content":"Say hi in 2 words."}],
    "max_tokens": 16,
    "stream": true
  }' > "${STREAM_OUT}" 2>&1 || true

if ! grep -q 'data:' "${STREAM_OUT}"; then
  echo "FAIL: no SSE data lines received"
  echo "Got:"
  cat "${STREAM_OUT}"
  exit 1
fi
echo "PASS: proxy returned SSE chunks"

# 4. Buffer size check
echo "[4/4] Buffer size check..."
CHUNK_SIZE=$(grep -o 'data:.*' "${STREAM_OUT}" | head -1 | wc -c || true)
if [ "${CHUNK_SIZE}" -gt 4096 ]; then
  echo "WARN: first chunk is large (${CHUNK_SIZE} bytes). If streaming feels laggy, reduce proxy read buffer."
else
  echo "PASS: first chunk size=${CHUNK_SIZE} bytes"
fi

echo ""
echo "SMOKE PASS: proxy is streaming on port ${PROXY_PORT}"
