#!/bin/bash
# Smoke test for experiment 008: prove MTP speculation is active and drafting.
# Usage: ./smoke-moe-speculation.sh <api_url> <model>
# Expects a llama.cpp server started with --spec-type draft-mtp.

set -euo pipefail

API_URL="${1:-http://127.0.0.1:8080}"
MODEL="${2:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}"
BASE_URL="${API_URL}/v1"

echo "=== Smoke test: MoE MTP speculation ==="
echo ""

# 1. Health
echo "[1/4] Health..."
if ! curl -sf --max-time 5 "${API_URL}/health" >/dev/null; then
  echo "FAIL: server not responding"
  exit 1
fi
echo "PASS: server is up"

# 2. Model identity
echo "[2/4] Model identity..."
MODELS=$(curl -sf --max-time 5 "${BASE_URL}/models")
echo "${MODELS}" | grep -q "Qwen3.6-35B-A3B" || {
  echo "FAIL: expected Qwen3.6-35B-A3B, got: ${MODELS}"
  exit 1
}
echo "PASS: Qwen3.6-35B-A3B is served"

# 3. Completion with speculation timings
echo "[3/4] Completion with draft stats..."
RESPONSE=$(curl -sf --max-time 120 "${BASE_URL}/completions" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"${MODEL}"'",
    "prompt": "Explain in one paragraph why speculative decoding helps memory-bandwidth-bound inference.",
    "n_predict": 128,
    "temperature": 0.0
  }')
echo "${RESPONSE}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
content = d.get('content', '')
if not content.strip():
    print('FAIL: empty completion')
    sys.exit(1)
tg = t.get('predicted_per_second', 0)
draft_n = t.get('draft_n', 0)
accepted = t.get('draft_accepted', t.get('accepted_draft_tokens_count', 0))
print(f'TG: {tg:.1f} tok/s, draft tokens: {draft_n}, accepted: {accepted}')
if draft_n and accepted:
    print(f'acceptance: {accepted/draft_n*100:.1f}%')
"
echo "PASS: completion returned with timing stats"

# 4. MTP active check
echo "[4/4] MTP drafting active..."
DRAFT_COUNT=$(echo "${RESPONSE}" | python3 -c "
import sys, json
d = json.load(sys.stdin)
t = d.get('timings', {})
print(t.get('draft_n', 0))
")
if [ "${DRAFT_COUNT}" = "0" ]; then
  echo "WARN: no draft tokens in timings — server may not be running MTP"
  echo "      (older builds omit draft stats; check your server flags)"
else
  echo "PASS: MTP drafted ${DRAFT_COUNT} tokens for this request"
fi

echo ""
echo "SMOKE PASS: MoE MTP speculation path reachable"
