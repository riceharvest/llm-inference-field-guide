# 007 — KV Cache Corruption That Looked Like an Agent Failure

> Qwen3.6-27B returned clean responses in direct API calls, but produced garbage through the agent layer. The agent was not the bug. The turbo4 KV cache quant corrupted long generations. Short replies survived. Long agent outputs decayed into thinking-only output, empty messages, and zero tool calls.

**Hardware:** Tesla V100 32GB
**Model:** Qwen3.6-27B-Heretic-v2-Q4_K_M
**Engine:** llama.cpp turboquant fork
**Date:** 2026-08-10

---

## Quick Start

If short API tests pass but long agent runs produce garbage or empty messages:

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
- `-ctk q8_0 -ctv q8_0` — replaces turbo4 KV quant, which corrupts long generations
- `--reasoning off` — prevents thinking-only output that the agent cannot parse into tool calls
- `--parallel 1` — single slot, keeps the cache state simple while you debug

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

Note the fragment. `"fying the two calls"` is the middle of a word. A chat template bug does not produce mid-word fragments. Corrupted cache state does.

## The diagnosis path

### 1. GGUF integrity — ruled out

SHA-256 matched the publisher's LFS hash. The file loaded correctly and produced normal text in direct API calls.

### 2. Chat template — ruled out

The embedded model template was active. A custom community JSON template was tested but made tool discovery worse, so it was removed.

### 3. Proxy buffering — already fixed

The SSE proxy was using 128-byte reads. This was not the cause.

### 4. Reasoning-only output — contributing cause

The server ran with `--reasoning on --reasoning-budget 256`. The model spent its small budget on thinking tokens. The visible assistant message came out empty. The agent tried to repair the empty message, then the model stopped. This explains the thinking-only calls. It does not explain the mid-word garbage.

### 5. Turbo4 KV quant — the root cause

The server ran with `-ctk turbo4 -ctv turbo4`. Turbo4 is an aggressive KV cache quant. Quantization error accumulates in the cache as the sequence grows. Short generations read a short, low-error cache and stay clean. Long agent generations attend over thousands of tokens of degraded cache. The output decays mid-stream: coherent at the start, fragments at the end.

This is why direct API calls passed. They were short. The agent's multi-turn continuations were long. The agent layer was the trigger, not the cause.

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

### Output starts clean, then decays into fragments

**Symptom:** Long generations begin coherently, then break into mid-word fragments or empty messages.

**Cause:** Aggressive KV cache quant. Error accumulates in the cache as the sequence grows.

**Fix:** Switch to `-ctk q8_0 -ctv q8_0` or bf16. Retest with a long generation, not a short one.

### Thinking-only output with no tool calls

**Symptom:** The model emits thinking tokens but never produces a tool call.

**Cause:** The reasoning budget is too small, or the agent cannot parse the thinking output into structured tool calls.

**Fix:** Either increase `--reasoning-budget` significantly or disable reasoning entirely with `--reasoning off`.

### Model works standalone but fails through the agent

**Symptom:** Direct API calls return correct responses. Through the agent, output is empty or garbage.

**Cause:** Usually not the agent. Agent prompts and continuations are longer than manual tests. Length-dependent failures, like KV quant corruption, only appear on the long path.

**Fix:** Reproduce with a long direct generation first. If that also decays, check the KV config before you touch the agent.

---

## What transfers

1. **Aggressive KV quants can pass short tests and fail long generations.** Quantization error accumulates with sequence length. A 20-token smoke test proves nothing about a 2,000-token agent run.

2. **Length-dependent failures look like agent or parser bugs.** The long path breaks first. Whatever sits on the long path gets the blame. Check cache and sampling config before you debug the caller.

3. **Mid-word garbage fragments point at corrupted state, not templates.** A template bug produces wrong structure. A corrupted cache produces broken tokens like `"fying the two calls"`.

4. **Reasoning with a small budget produces empty visible messages.** The model spends the budget on thinking tokens. The agent sees an empty assistant message and cannot recover a tool call from it.

5. **Test at production length.** Direct calls can succeed while the agent layer fails, because the agent layer generates longer.

---

## Files

| Path | Purpose |
|---|---|
| [`scripts/smoke-toolcall.sh`](scripts/smoke-toolcall.sh) | ~30s agent tool-call smoke test |
| [`patches/fix-toolcall-config.sh`](patches/fix-toolcall-config.sh) | Hotfix patch for toolcall config |
| [`results/2026-08-09-toolcall-kv-fix.json`](results/2026-08-09-toolcall-kv-fix.json) | raw benchmark matrix |
