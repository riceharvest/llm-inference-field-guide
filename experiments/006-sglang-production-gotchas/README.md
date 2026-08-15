# 006 — sglang Production Gotchas

> Running a hybrid MoE model on sglang? Default flags are not production flags. Here's what each one actually does — and what to change.

**Hardware:** RTX 3060 12GB (Ampere, cc 8.6)
**Model:** Ling-3.0-tiny (7.9B total / 1.3B active, `bailingmoe3` arch)
**Engine:** SGLang
**Date:** 2026-08-14 to 2026-08-15

---

## Quick Start

```bash
# The WRONG way (default flags that hurt production):
sglang serve \
  --model-path inclusionAI/Ling-3.0-tiny-int4 \
  --max-running-requests 1 \        # ← SERIALIZES all turns
  --enable-hierarchical-cache \      # ← unnecessary CPU↔GPU overhead
  --hicache-size 8 \
  --mem-fraction-static 0.80 \       # ← starves KV cache
  --cuda-graph-backend-prefill disabled  # ← slows prefill

# The RIGHT way (production flags for Ampere):
sglang serve \
  --model-path inclusionAI/Ling-3.0-tiny-int4 \
  --max-running-requests 4 \
  --mem-fraction-static 0.90 \
  --cuda-graph-backend-prefill enabled \
  --moe-runner-backend auto \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser glm
```

**Also set this env var** (selects Marlin MoE kernel instead of Triton):
```bash
SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=True
```

## Exact flags

```bash
# Production flags for Ampere / RTX 3060
sglang serve \
  --model-path ${MODEL_PATH:-inclusionAI/Ling-3.0-tiny-int4} \
  --max-running-requests 4 \
  --mem-fraction-static 0.90 \
  --cuda-graph-backend-prefill enabled \
  --moe-runner-backend auto \
  --kv-cache-dtype bfloat16 \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser glm
```

---

## The symptom

Model on sglang was getting ~17.5 tok/s. The same model on llama.cpp gets ~126–133 tok/s. A 7× gap.

## The diagnosis (6 stacked issues)

### 1. `--max-running-requests 1` — serialized all turns

**Symptom:** Only one request processes at a time. Multi-turn agent workflows stall waiting for each turn to complete.

**Cause:** This flag limits the server to a single concurrent request. sglang's scheduler can't batch or overlap requests.

**Fix:** Increase to `--max-running-requests 4` (or higher for your concurrency needs).

### 2. `--enable-hierarchical-cache --hicache-size 8` — CPU↔GPU overhead

**Symptom:** Added latency with no benefit for single-GPU setups.

**Cause:** Hierarchical cache offloads KV to CPU RAM. For a single 12GB GPU with a small model, this just adds unnecessary memory copies.

**Fix:** Remove these flags unless you're running multi-GPU with CPU offloading.

### 3. `--mem-fraction-static 0.80` — starved KV cache

**Symptom:** OOM on long contexts, or the model can't use the full context length.

**Cause:** This reserves 80% of VRAM for model weights, leaving only 20% for KV cache. For small models at INT4, you're wasting most of your VRAM.

**Fix:** Increase to `--mem-fraction-static 0.90` (or higher). Let the KV cache use more VRAM.

### 4. `--cuda-graph-backend-prefill disabled` — slowed prefill

**Symptom:** Prefill (prompt processing) is slower than expected.

**Cause:** CUDA graphs accelerate prefill by reducing kernel launch overhead. Disabling them forces eager execution.

**Fix:** Enable with `--cuda-graph-backend-prefill enabled`.

### 5. `enable_thinking: false` — empty responses

**Symptom:** When `chat_template_kwargs: {"enable_thinking": false}` is sent, the model returns empty choices.

**Cause:** sglang's reasoning parser doesn't handle this kwarg the same way llama.cpp does.

**Fix:** Either don't send `enable_thinking: false`, or handle empty responses client-side by retrying without the kwarg.

### 6. The Ampere wall — KDA Triton kernel

**Symptom:** Even with all flags fixed, sglang on RTX 3060 caps at ~32 tok/s. llama.cpp gets ~126–133 tok/s.

**Cause:** Hybrid MoE models with 3:1 linear-attention to full-attention ratios run a custom KDA kernel. On Ampere (cc 8.6), sglang can only run the slow Triton reference KDA kernel — all faster paths are gated to SM90+ (Hopper) or SM100 (Blackwell). llama.cpp has native CUDA KDA kernels that run on Ampere.

**Fix:** Use llama.cpp on Ampere for hybrid architectures. SGLang's KDA support for consumer GPUs is a known limitation.

## Measured Results

**Date:** 2026-08-14 to 2026-08-15

| Config | Decode tok/s |
|---|---|
| Default flags | ~17.5 |
| Fixed flags (no hicache, mem-fraction 0.90, Marlin MoE) | ~31.7 |
| llama.cpp (same GPU) | ~126–133 |

## Troubleshooting

### Empty response with `enable_thinking: false`

**Symptom:** sglang returns empty choices when `enable_thinking: false` is sent.

**Fix:** Remove the kwarg from the request, or handle empty responses client-side.

### ~17 tok/s on a "fast" model

**Symptom:** Stuck at ~17 tok/s.

**Cause:** Missing `SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=True` — you're on the Triton MoE fallback.

**Fix:** Set the env var to select the Marlin MoE kernel. This alone can take you from ~17 to ~32 tok/s.

### ~32 tok/s ceiling on RTX 3060

**Symptom:** Can't get above ~32 tok/s even with all flags fixed.

**Cause:** The KDA linear-attention kernel is gated to newer architectures on Ampere.

**Fix:** Use llama.cpp — it has native CUDA KDA kernels that run on Ampere at ~126–133 tok/s.

---

## What transfers

1. **Default sglang flags are not production flags.** `--max-running-requests 1` serializes everything. `--mem-fraction-static 0.80` starves KV. `--enable-hierarchical-cache` adds overhead without benefit on single-GPU.

2. **`SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK=True` is required on Ampere.** Without it, you're on the Triton MoE fallback — half the speed.

3. **The Ampere KDA wall is structural.** sglang's fast KDA kernels are gated to SM90+/SM100. On consumer Ampere, use llama.cpp for hybrid MoE architectures.

4. **`enable_thinking: false` can break sglang responses.** Test with and without this kwarg if you get empty responses.

---

## Files

| Path | Purpose |
|---|---|
| [`scripts/smoke-sglang.sh`](scripts/smoke-sglang.sh) | ~30s sglang smoke test |
| [`scripts/bench-sglang.sh`](scripts/bench-sglang.sh) | Benchmark sglang with different flag combinations |
| [`patches/sglang-production-flags.sh`](patches/sglang-production-flags.sh) | Hotfix patch for production flags |
| [`results/2026-08-15-sglang-production-fix.md`](results/2026-08-15-sglang-production-fix.md) | Full flag-by-flag comparison |
| [`results/2026-08-09-sglang-flags-audit.json`](results/2026-08-09-sglang-flags-audit.json) | raw benchmark matrix |
