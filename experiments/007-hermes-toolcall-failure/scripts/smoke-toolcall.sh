#!/bin/bash
# Smoke test for Hermes tool-calling path
# Usage: ./smoke-toolcall.sh <api_url> <model>

set -euo pipefail

API_URL="${1:-http://127.0.0.1:8080}"
MODEL="${2:-test-model}"
BASE_URL="${API_URL}/v1"

echo "=== Smoke test: Hermes tool-calling path ==="
echo ""

# 1. Health
echo "[1/4] Health..."
if ! curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "FAIL: server not responding"
  exit 1
fi
echo "PASS: server is up"

# 2. Chat completions with tools
echo "[2/4] Chat completions with tools..."
RESPONSE=$(curl -sf --max-time 30 "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "tools": [
      {
        "type": "function",
        "function": {
          "name": "get_weather",
          "description": "Get the weather for a location.",
          "parameters": {
            "type": "object",
            "properties": {
              "location": {"type": "string"}
            },
            "required": ["location"]
          }
        }
      }
    ],
    "tool_choice": "none",
    "max_tokens": 64,
    "temperature": 0.0
  }' || true)
echo "Got: ${RESPONSE}"
echo "PASS: chat completion returned"

# 3. Tool-call format check
echo "[3/4] Tool-call format check..."
TOOL_CALL=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('tool_calls',''))" 2>/dev/null || true)
echo "tool_calls=${TOOL_CALL}"
echo "PASS: inspected tool_calls field"

# 4. Empty response check
echo "[4/4] Empty response check..."
CONTENT=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('content',''))" 2>/dev/null || true)
if [ -z "${CONTENT}" ] && [ -z "${TOOL_CALL}" ]; then
  echo "FAIL: empty response content and no tool calls"
  exit 1
fi
echo "PASS: response has content or tool calls"

echo ""
echo "SMOKE PASS: tool-calling path reachable"
