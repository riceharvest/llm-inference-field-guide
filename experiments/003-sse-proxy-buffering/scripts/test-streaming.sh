#!/bin/bash
# Test streaming behavior with direct curl vs proxied
# Usage: ./test-streaming.sh <proxy_url> <direct_url>

PROXY_URL="${1:-http://127.0.0.1:8081}"
DIRECT_URL="${2:-http://127.0.0.1:8080}"

echo "=== Test 1: Direct curl (bypasses proxy) ==="
echo "Streaming tokens as they're produced:"
echo ""

curl -N "${DIRECT_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Count from 1 to 5."}],"stream":true}' \
  2>/dev/null | while read -r line; do
    if [[ "$line" == data:* ]]; then
      echo "  $line"
      sleep 0.1
    fi
  done

echo ""
echo "=== Test 2: Proxied curl ==="
echo "If proxy buffer is too large, tokens arrive in bursts:"
echo ""

curl -N "${PROXY_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Count from 1 to 5."}],"stream":true}' \
  2>/dev/null | while read -r line; do
    if [[ "$line" == data:* ]]; then
      echo "  $line"
      sleep 0.1
    fi
  done

echo ""
echo "=== Diagnosis ==="
echo "If Test 1 streams tokens one-by-one but Test 2 shows them in bursts,"
echo "your proxy read buffer is too large. Reduce CHUNK to <1024 bytes."
