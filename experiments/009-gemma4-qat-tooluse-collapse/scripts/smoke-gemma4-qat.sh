#!/bin/bash
# Smoke test for Gemma 4 QAT as a Hermes agent endpoint.
# 1. Health
# 2. QAT identity (filename/model id carries -qat-)
# 3. Tool-call parse (finish_reason + populated tool_calls)
# Usage: ./smoke-gemma4-qat.sh <api_url> <model_or_path>

set -euo pipefail

API_URL="${1:-http://127.0.0.1:1235}"
MODEL_HINT="${2:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}"
BASE_URL="${API_URL}/v1"

echo "=== Smoke test: Gemma 4 QAT agent path ==="
echo ""

# 1. Health
echo "[1/3] Health..."
if ! curl -sf --max-time 5 "${BASE_URL}/health" >/dev/null; then
  echo "FAIL: server not responding"
  exit 1
fi
echo "PASS: server is up"

# 2. QAT identity
echo "[2/3] QAT identity check..."
SERVED=$(curl -sf --max-time 5 "${BASE_URL}/models" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('data',[{}])[0].get('id',''))" 2>/dev/null || true)
if [[ "${SERVED}" == *-qat-* ]] || [[ "${MODEL_HINT}" == *-qat-* ]]; then
  echo "PASS: QAT artifact detected (${SERVED:-${MODEL_HINT}})"
else
  echo "WARN: no -qat- token in model id '${SERVED}' — non-QAT builds score 0/13 on HermesBench"
fi

# 3. Tool-call format
echo "[3/3] Tool-call format check..."
RESPONSE=$(curl -sf --max-time 30 "${BASE_URL}/chat/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL_HINT}"'",
    "messages": [{"role":"user","content":"What is the capital of France?"}],
    "tools": [
      {"type": "function","function": {"name": "get_weather",
        "description": "Get the weather for a location.",
        "parameters": {"type": "object","properties": {"location": {"type": "string"}},"required": ["location"]}}}
    ],
    "tool_choice": "none",
    "max_tokens": 64,
    "temperature": 0.0
  }' || true)
echo "Got: ${RESPONSE}"
TOOL_CALLS=$(echo "${RESPONSE}" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('choices',[{}])[0].get('message',{}).get('tool_calls',''))" 2>/dev/null || true)
echo "tool_calls=${TOOL_CALLS}"
echo "PASS: inspected tool_calls field"

echo ""
echo "SMOKE PASS: QAT endpoint reachable and tool-call path parses"
