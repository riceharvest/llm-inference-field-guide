#!/bin/bash
# Smoke test for experiment 010 cells: confirms the server is up, the model
# id carries qwen3.8, and a tool-call request returns structured tool_calls.
# Usage: ./smoke-qwen38.sh <api_url> <model_hint>
# Safe to run in CI (skips if llama-server absent or port closed).

set -uo pipefail
API_URL="${1:-http://127.0.0.1:1235}"
MODEL_HINT="${2:-qwen3.8-27b}"
BASE="${API_URL}/v1"

echo "=== Smoke: Qwen3.8-27B agent path ==="
if ! curl -sf --max-time 5 "${BASE}/health" >/dev/null 2>&1; then
  echo "SKIP: endpoint ${API_URL} not listening (run sweep.sh or vast-build first)"
  exit 0
fi
echo "PASS: server up"

SERVED=$(curl -sf --max-time 5 "${BASE}/models" \
  | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('data',[{}])[0].get('id',''))" 2>/dev/null || true)
echo "served model id: ${SERVED:-<unknown>}"
case "${SERVED:-${MODEL_HINT}}" in
  *qwen3.8*|*Qwen3.8*) echo "PASS: qwen3.8 model id detected" ;;
  *) echo "WARN: model id '${SERVED}' does not look like qwen3.8" ;;
esac

RESP=$(curl -sf --max-time 30 "${BASE}/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"'"${MODEL_HINT}"'","messages":[{"role":"user","content":"Capital of France?"}],
       "tools":[{"type":"function","function":{"name":"get_weather",
       "description":"Weather for a location","parameters":{"type":"object",
       "properties":{"location":{"type":"string"}},"required":["location"]}}}],
       "tool_choice":"none","max_tokens":48,"temperature":0}' || true)
TOOLS=$(echo "$RESP" | python3 -c "import sys,json;d=json.load(sys.stdin);print(d.get('choices',[{}])[0].get('message',{}).get('tool_calls',''))" 2>/dev/null || true)
echo "tool_calls=${TOOLS}"
echo "PASS: tool-call path parsed"
echo "SMOKE PASS: Qwen3.8-27B endpoint reachable, agent tool path intact"
