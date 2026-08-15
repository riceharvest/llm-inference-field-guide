# 007 — Model Works Standalone, Fails Through the Agent

> Qwen3.6-27B returned clean responses in direct API calls, but failed through the agent layer. The model emitted thinking-only output, then empty assistant messages, then stopped. Root cause: a combination of reasoning-only output handling and a KV config that corrupted long generations.

**Hardware:** Tesla V100 32GB
**Model:** Qwen3.6-27B-Heretic-v2-Q4_K_M
**Engine:** llama.cpp turboquant fork
**Date:** 2026-08-10

---

## Quick Start

If your model works standalone but fails through the agent layer:

```bash
# Start with this config (known-working):
llama-server \
  -m Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf \
  --port 8080 --host 127.0.0.1 \
  -c 131072 -ngl 99 --jinja \
  --reasoning off \
  -ctk q8_0 -ctv q8_0 \
  --parallel 1 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

**Key flags:**
- `--reasoning off` — prevents thinking-only output that the agent can't handle
- `-ctk q8_0 -ctv q8_0` — KV quant, avoiding the config that corrupts long generations
- `--parallel 1` — single slot, avoiding cache issues

## Exact flags

```bash
llama-server \
  -m "${MODEL_PATH:-Qwen3.6-27B-uncensored-heretic-v2-Q4_K_M.gguf}" \
  --port "${PORT:-8080}" --host "${HOST:-127.0.0.1}" \
  -c 131072 -ngl 99 --jinja \
  --reasoning off \
  -ctk q8_0 -ctv q8_0 \
  --parallel 1 \
  --temp 0.6 --top-p 0.95 --top-k 20
```

---

## The symptom

Direct API calls returned correct responses. Through the agent layer, tool calls failed before the first tool execution.

**Failure log:**
```
API call 1: 48 output tokens, thinking-only
Agent prefilling continuation 1
API call 2: 6 output tokens, thinking-only
Agent prefilling continuation 2
Agent repaired an empty assistant message
API call 3: 1 output token, then stop
tool_turns=0
```

**UI showed:** `"fying the two calls: 32-32assistant "` followed by `"[response interrupted]"`

## The diagnosis path

### 1. GGUF integrity — ruled out

SHA-256 matched the publisher's LFS hash. The file loaded correctly and produced normal text in direct API calls.

### 2. Chat template — ruled out

The embedded model template was active. A custom community JSON template was tested but made tool discovery worse, so it was removed.

### 3. Proxy buffering — already fixed

The SSE proxy was using 128-byte reads. This was not the cause.

### 4. Reasoning-only output — partial cause

The model emitted thinking-only output that the agent couldn't parse into tool calls. The agent tried to repair an empty assistant message, then the model stopped.

### 5. KV config — the real culprit

The server was running with a KV config that corrupted long generations. This caused the model to produce malformed output that the agent couldn't handle.

## The fix

```bash
# Before (broken):
-ctk turbo4 -ctv turbo4
--reasoning on --reasoning-budget 256
--parallel 3

# After (working):
-ctk q8_0 -ctv q8_0
--reasoning off
--parallel 1
```

**Measured after fix:**

| Metric | Before fix | After fix |
|---|---|---|
| Tool calls | 0 (failed) | working |
| Output | empty/garbage | clean |
| Finish reason | `length` (truncated) | `stop` |

## Troubleshooting

### Model works standalone but fails through the agent

**Symptom:** Direct API calls return correct responses. Through the agent, tool calls fail before the first execution.

**Cause:** Combination of reasoning-only output and KV corruption on long generations.

**Fix:** Try `--reasoning off` first. If that doesn't help, check your KV config.

### Thinking-only output with no tool calls

**Symptom:** The model emits thinking tokens but never produces a tool call.

**Cause:** The reasoning budget is too small, or the agent can't parse the thinking output into structured tool calls.

**Fix:** Either increase `--reasoning-budget` significantly or disable reasoning entirely with `--reasoning off`.

### Empty assistant messages

**Symptom:** Agent logs "repaired an empty assistant message" then the model stops.

**Cause:** The model produced output that the agent couldn't parse into a valid assistant message. This is often caused by KV corruption on long generations.

**Fix:** Switch from the problematic KV config to q8_0 or bf16. Verify with a known-good config.

---

## What transfers

1. **A model that works standalone can still fail through the agent layer.** The agent adds parsing requirements that expose edge cases in model output.

2. **Reasoning-only output breaks tool-calling flows.** If the model only emits thinking tokens, the agent can't extract tool calls. Either disable reasoning or give it a large enough budget.

3. **Some KV configs corrupt long generations.** Use q8_0 or bf16 instead for production workloads.

4. **Test through the full pipeline, not just direct API calls.** Direct calls can succeed while the agent layer fails.

---

## Files

| Path | Purpose |
|---|---|
| [`scripts/smoke-toolcall.sh`](scripts/smoke-toolcall.sh) | ~30s agent tool-call smoke test |
| [`scripts/test-hermes-toolcall.sh`](scripts/test-hermes-toolcall.sh) | Reproduce the agent tool-calling failure |
| [`patches/fix-toolcall-config.sh`](patches/fix-toolcall-config.sh) | Hotfix patch for toolcall config |
| [`results/2026-08-10-hermes-toolcall-fix.md`](results/2026-08-10-hermes-toolcall-fix.md) | Full diagnosis path + test results |
| [`results/2026-08-09-toolcall-kv-fix.json`](results/2026-08-09-toolcall-kv-fix.json) | raw benchmark matrix |
