# 009 — The 4-Bit Quant That Fired Tools but Finished Nothing

> Two Gemma 4 GGUFs at 4-bit. Same parameter class. Both chat fine in direct use. Through Hermes Agent, the non-QAT build fires tool calls and completes zero tasks. The QAT build completes 9 of 13. The difference is not size. It is quantization-aware training.

**Hardware:** RTX 3060 12GB + Ryzen 5950X (62 GB RAM)
**Model:** Gemma 4 12B it QAT UD-Q4_K_XL (6.7 GB) vs Gemma 4 E4B Q4_K_M (non-QAT, 5.0 GB)
**Engine:** llama.cpp upstream (build-mtp), Hermes Agent + HermesBench Core v2
**Date:** 2026-07-14

---

## Quick Start

For Gemma 4 as a Hermes agent on a 12 GB card, download the QAT artifact. Look for `-qat-` in the filename. Then use these flags.

```bash
llama-server \
  -m "${MODEL_PATH:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}" \
  --host 127.0.0.1 --port "${PORT:-1235}" \
  -c 196608 -ngl 99 -fa on \
  --jinja -np 1 --no-warmup
```

**Key flags:**
- `-fa on` — required. The Unsloth Dynamic (UD) quant uses V-cache quantization that needs flash attention. Without it the load fails with `V cache quantization requires flash_attn`.
- **No `-ctk` / `-ctv`** — the UD file sets its own KV cache format in metadata. Explicit flags clash with it and produce the same `flash_attn` error.
- **Use the upstream build** — the turboquant fork can hang at `fitting params to device memory` when it loads UD quants. The standard build loads them reliably.

Optional MTP draft head (upstream builds only, ~2x decode):

```bash
  --model-draft "${DRAFT_PATH:-gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf}"
```

## Exact flags

```bash
llama-server \
  -m "${MODEL_PATH:-gemma-4-12B-it-qat-UD-Q4_K_XL.gguf}" \
  --host 127.0.0.1 --port "${PORT:-1235}" \
  -c 196608 -ngl 99 -fa on \
  --jinja -np 1 --no-warmup \
  --model-draft "${DRAFT_PATH:-gemma-4-12B-it-qat-assistant-MTP-Q8_0.gguf}"
```

Hermes provider wiring:

```yaml
providers:
  gemma4-local:
    api: http://127.0.0.1:1235/v1
    default_model: gemma-4-12b-it-qat
    api_key: dummy
    context_length: 196608
```

Then: `hermes chat --provider gemma4-local -m gemma-4-12b-it-qat`

---

## The symptom

A Gemma 4 E4B Q4_K_M GGUF loaded fine. Direct chat looked normal. Through Hermes Agent, every task failed.

**HermesBench Core v2 (13 agent tasks):**

| Metric | Non-QAT E4B | QAT 12B |
|---|---:|---:|
| Tasks completed | **0/13** | **9/13** |
| Tool-use pass | 12/13 | 13/13 |
| Tool classes used | 13/14 | 14/14 |
| Aggregate tok/s | 185.9 | 134.3 |
| Wall time | 11m 39s | 15m 40s |

Read the second row carefully. The non-QAT model *emitted* tool calls. It parsed the schema. It called the right tools. It still completed zero tasks. The quant did not remove the grammar. It removed the reasoning quality needed to finish multi-step work.

## The diagnosis path

### 1. Server tool parser — ruled out

Raw API objects showed `finish_reason: tool_calls` with populated `message.tool_calls` and valid JSON arguments. The parser worked.

### 2. Chat template — ruled out

The embedded template was active. Tool-call structure was correct in both models.

### 3. Sampling and agent config — ruled out

Both runs used the same Hermes agent, the same suite, the same flags, sequential on the same GPU with VRAM verified back at baseline between runs.

### 4. Quantization provenance — the root cause

The failing file was a standard post-training 4-bit quant. Gemma 4 loses a large amount of capability at 4-bit unless the quant was trained in. Google's QAT checkpoints keep the 4-bit error inside what the model was trained to tolerate. The Unsloth `-qat-` GGUFs are built from those checkpoints.

The MoE variant (E4B) suffers most. Naive 4-bit quantization disturbs expert routing, and agentic multi-step tasks die first. Even the QAT E4B only reached 2/13. The dense 12B QAT reached 9/13 at the same VRAM budget.

## The fix

Download the QAT artifact instead:

```bash
# Wrong (0/13): standard 4-bit quant
gemma-4-E4B-it-Q4_K_M.gguf

# Right (9/13): QAT artifact from the Unsloth QAT collection
hf download unsloth/gemma-4-12B-it-qat-GGUF \
  gemma-4-12B-it-qat-UD-Q4_K_XL.gguf --local-dir models/gemma4-qat
```

Verify the artifact before benchmarking anything:

```bash
# 1. Filename must contain -qat-
# 2. SHA256 must match the publisher's LFS oid
sha256sum gemma-4-12B-it-qat-UD-Q4_K_XL.gguf
# cc9ff072e0a8203429ed854e6662c17a6c2bc1e5dca5b475dd4736caaacbc165  (our copy, 6,716,355,328 bytes)
```

## Measured Results

HermesBench Core v2, RTX 3060 12GB, `--jobs 1`, sequential servers, 2026-07-14:

| Model | Quant | QAT? | Score | Tok/s | Wall |
|---|---|---|---:|---:|---:|
| Gemma 4 12B | UD-Q4_K_XL | Yes | **9/13** | 134.3 | 15m 40s |
| Gemma 4 E2B | q4_0 | Yes | 6/13 | 337.3 | 5m 49s |
| Gemma 4 E4B | UD-Q4_K_XL | Yes | 2/13 | 220.3 | 13m 03s |
| Gemma 4 E4B | Q4_K_M | **No** | **0/13** | 185.9 | 11m 39s |

The 12B QAT score reproduced on the older hermes-core suite three days earlier: 9/13 (0.692), 1008 s wall, 108,481 input / 8,244 output tokens, server resident at 9,472 MiB VRAM.

MTP draft head sweep (upstream llama.cpp, 131K context, decode tok/s):

| Config | Decode tok/s |
|---|---:|
| No draft (baseline) | 37.6 |
| MTP Q8_0, n=1 | 56.9 |
| MTP Q8_0, n=2 | 76.3 |
| MTP Q8_0, n=3 | 79.5 |
| MTP Q8_0, n=8 | 81.7 |

Decode roughly doubles with the Q8_0 assistant-MTP head. Prefill stays at 360–397 tok/s.

## Troubleshooting

### Load fails with `V cache quantization requires flash_attn`

**Cause:** UD quant without `-fa on`, or explicit `-ctk`/`-ctv` flags clashing with the UD metadata.

**Fix:** Add `-fa on`. Remove `-ctk` and `-ctv`. Let the file choose its KV format.

### Server hangs at `fitting params to device memory`

**Cause:** The turboquant fork can deadlock when it loads UD quants, especially with `-np > 1` on tight VRAM.

**Fix:** Use the upstream llama.cpp build. If you must keep turboquant, add `-fit off` (not `--no-fit`, that flag does not exist), or omit `-ngl` and let the fitter choose the layer count.

### MTP draft head fails to load or is ignored

**Cause:** Head/build mismatch. The Unsloth Q8_0 assistant-MTP head works with upstream llama.cpp only. The ik_llama Q4 MTP head is incompatible with both upstream and turboquant builds.

**Fix:** Pair the Q8_0 head with the upstream build. Check the base GGUF actually has MTP layers before you pass speculation flags: `grep -a mtp model.gguf`.

### OOM when you add the MTP head at 262K context

**Cause:** The draft model's CUDA buffer (~428 MB) plus the 444 MB head pushes past 12 GB after the base model and KV cache are loaded.

**Fix:** Cap context at 196K with MTP on a 12 GB card. 262K works only without the draft head.

### Model completes tasks in chat but scores zero as an agent

**Cause:** Non-QAT 4-bit quant. Short chat survives the quant error. Multi-step agent tasks do not.

**Fix:** Switch to the `-qat-` artifact. Then re-run the full agent suite, not a chat probe.

---

## What transfers

1. **QAT is a capability feature, not a quality nicety.** At the same bit-width, QAT was the difference between 9/13 and 0/13. Check training provenance before you benchmark flags.

2. **Tool-call emission is not task completion.** The failing model fired valid tool calls in 12/13 tasks and finished none. Score agent benchmarks on completed tasks, not on whether `tool_calls` was populated.

3. **MoE quants degrade first.** The E4B MoE lost more at 4-bit than the dense 12B, even with QAT (2/13 vs 9/13 at the same VRAM). Routing quality is quant-sensitive.

4. **Smaller + QAT beats larger + non-QAT.** The 3.2 GB QAT E2B (6/13) beat the 5.0 GB non-QAT E4B (0/13). Provenance beat size.

5. **Match the draft head to the build.** MTP heads are not portable across llama.cpp forks. Verify the pairing before you measure a speculative speedup.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/llama-server-qat.sh`](configs/llama-server-qat.sh) | Known-working server launch (UD + MTP) |
| [`configs/hermes-provider.yaml`](configs/hermes-provider.yaml) | Hermes provider snippet |
| [`scripts/smoke-gemma4-qat.sh`](scripts/smoke-gemma4-qat.sh) | ~30s QAT identity + tool-call smoke test |
| [`results/2026-07-14-gemma4-qat-matrix.json`](results/2026-07-14-gemma4-qat-matrix.json) | Raw benchmark matrix |
