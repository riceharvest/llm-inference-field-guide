# 008 — Speculation Shootout on a 3B-Active MoE: MTP Wins, DFlash Loses, DDTree Crashes

> Every llama.cpp speculative decoding path was tested on Qwen3.6-35B-A3B (3B active, 8-of-256 routed experts) on an RTX 3060. Built-in MTP is the only one that helps (+8%). External DFlash draft heads lose 42–52%. DDTree tree speculation aborts on the first request — the tree verify pass OOMs. The limit is physics, not configuration: the verify pass pays for the union of expert weights.

**Hardware:** RTX 3060 12GB + Ryzen 5950X (62 GB RAM)
**Model:** Qwen3.6-35B-A3B-UD-Q3_K_M.gguf (16 GB, MoE, 3B active)
**Engine:** llama.cpp (MTP build) and BeeLlama fork (turboquant + DFlash support)
**Date:** 2026-07-03 → 2026-07-04

---

## Quick Start

For Qwen3.6-35B-A3B on a 12 GB card, use built-in MTP. Do not use DFlash or tree speculation.

```bash
llama-server \
  -m "${MODEL_PATH:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}" \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  -ngl 32 -ncmoe 28 -t 14 -tb 16 \
  -b 2048 -ub 512 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 131072 -np 1 \
  --jinja --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --kv-unified --cache-idle-slots \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --no-warmup
```

**Key flags:**
- `--spec-type draft-mtp --spec-draft-n-max 3` — the model's own MTP head drafts 3 tokens. Trained jointly with the base model, so it understands the expert routing.
- `-ncmoe 28` — keeps routed MoE experts of the first 28 layers in CPU RAM. The dense/attention path stays on GPU. This is the VRAM/speed sweet spot; see the sweep below.
- `-ctk q4_0 -ctv q4_0` — q4 KV makes 131K context fit. Note: this model's GQA 8:1 ratio makes turbo4 K auto-upgrade to q8_0, so plain q4_0 is the predictable choice.

## Exact flags

**Working MTP config (production):**

```bash
llama-server \
  -m "${MODEL_PATH:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}" \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  -ngl 32 -ncmoe 28 -t 14 -tb 16 \
  -b 2048 -ub 512 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 131072 -np 1 \
  --prio 2 --prio-batch 2 \
  --jinja --chat-template-kwargs '{"preserve_thinking":true}' \
  --temp 0.6 --top-p 0.95 --top-k 20 --min-p 0.0 \
  --presence-penalty 0.0 --repeat-penalty 1.0 \
  --kv-unified --cache-idle-slots \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --no-warmup
```

**Losing DFlash config (for reproduction only):**

```bash
llama-server \
  -m "${MODEL_PATH:-Qwen3.6-35B-A3B-UD-Q3_K_M.gguf}" \
  --host 127.0.0.1 --port "${PORT:-8080}" \
  -ngl 28 -ncmoe 28 -t 14 -tb 16 \
  -b 2048 -ub 512 -fa on \
  -ctk q4_0 -ctv q4_0 \
  -c 32768 -np 1 \
  --spec-type dflash \
  --spec-draft-model "${DRAFT_PATH:-qwen36-35b-a3b-dflash-Q4_K_M.gguf}" \
  --spec-draft-n-max 8 \
  --spec-dflash-cross-ctx 512 \
  --spec-branch-budget 0 \
  --no-warmup --no-mmap
```

Note the lower `-ngl 28` and lower context. The DFlash draft head needs ~2.9 GB of runtime state (cross-attention buffers) even though the draft GGUF file is only 279 MB. The MTP config at `-ngl 32` does not leave room for it.

**Crashing DDTree config (for reproduction only):** same as DFlash but with `--spec-branch-budget 3 --spec-draft-top-k 4` at `-c 131072`. The server aborts on the first request.

---

## The symptom

Decode on this card is bandwidth-bound (~336 GB/s effective). Speculative decoding should help: draft cheap tokens, verify them in one pass. Three speculation paths were tested:

1. **DFlash with an external draft head — slower than no speculation at all.** The server loaded, drafted, and verified, but decode dropped to 16.9 tok/s against MTP's 29.3 tok/s.
2. **A higher-precision DFlash head — even worse.** The Q8_0 head (491 MB) scored lower acceptance than the Q4_K_M head (279 MB): 44% vs 53%. More draft precision did not help.
3. **DDTree tree speculation — server crash on first request.** The process aborted inside the tree-verify path:

```
ggml_abort
llama_context::output_reserve(int)
llama_context::decode(llama_batch const&)
common_speculative_impl_dflash::draft_tree(...)
common_speculative_draft_tree(...)
server_context_impl::update_slots()
```

`output_reserve` fails when the verify batch does not fit. A tree of drafts (branch budget 3, top-k 4) multiplies the tokens the target must verify in one pass. At 131K context on 12 GB, that allocation does not exist.

## The diagnosis path

### 1. Draft model file size — ruled out as the win condition

The DFlash head is only 279 MB. It loads easily. Small draft, fast draft — and still a net loss. Draft cost was never the problem.

### 2. Draft head precision — ruled out

Q8_0 (491 MB) scored *worse* than Q4_K_M (279 MB): 13.9 vs 16.9 tok/s, 44% vs 53% acceptance. If the draft head were the bottleneck, precision would help. It did not, because the misalignment is on the other side: the draft head was trained on BF16 hidden states, and the Q3_K_M target's hidden states are shifted by quantization. Drafter-verifier alignment breaks no matter how precise the drafter is.

### 3. Tree verification — the crash

Flat DFlash (`--spec-branch-budget 0`) runs. Tree DFlash (`--spec-branch-budget 3 --spec-draft-top-k 4`) aborts in `output_reserve` during `draft_tree`. The verify pass for a tree pays for every branch at once. On a card where the base config already uses ~9.5 GB, the tree verify batch has nowhere to go.

### 4. Root cause — MoE expert saturation

Qwen3.6-35B-A3B activates 3B parameters per token: 8 routed experts out of 256, plus shared dense weights. Per MoESD (arXiv 2505.19645), the expert-saturation threshold for this architecture is T_thres ≈ 94. Any realistic draft depth K is far below 94.

Each drafted token routes to its own experts. The verify pass must load the *union* of expert weights for all K drafted tokens. On a bandwidth-bound GPU, that union costs more than the speculation saves. **100% draft acceptance cannot rescue this** — the external RTX 3090 benchmark below measured a −39% loss with a verified 100% acceptance rate.

MTP survives because it is not an external drafter. It is a head trained jointly with the base model. It operates on the model's own representation space, so quantization shift and routing behavior are already priced in. Three-token drafts at 60% acceptance with a cheap verify cycle beat eight-token DFlash drafts at 53% with an expensive one.

### 5. Expert offload sweep (`-ncmoe`) — the blunt lever wins

While tuning, a manual tensor-override sweep tested whether forcing specific layers' routed experts to CPU beats the blunt `-ncmoe` knob:

| Config | VRAM | Decode (tiny prompt) | MTP acceptance | Status |
|---|---|---|---|---|
| `-ncmoe 28` (baseline) | 10,090 MiB | 34.62 tok/s | 0.643 | OK |
| `-ncmoe 27` + blk.27 experts forced CPU | 10,422 MiB | 34.90 tok/s | 0.691 | No useful gain |
| `-ncmoe 26` + blk.26/27 experts forced CPU | — | — | — | CUDA OOM on first request (`cudaGraphLaunch`) |

Fine-grained `-ot` overrides bought nothing. Keep `-ncmoe 28`.

## The fix

Use MTP. Do not use DFlash draft heads or tree speculation on this model + hardware.

```bash
# Before (DFlash, losing):
--spec-type dflash --spec-draft-model <dflash-head.gguf> --spec-draft-n-max 8 -ngl 28

# Before (DDTree, crashing):
--spec-type dflash --spec-branch-budget 3 --spec-draft-top-k 4

# After (MTP, working):
--spec-type draft-mtp --spec-draft-n-max 3 -ngl 32
```

## Measured Results

Head-to-head decode, same model, same card (July 2026):

| Speculation | TG tok/s | Acceptance | vs MTP |
|---|---|---|---|
| None (baseline) | ~27.1 | — | −8% |
| **MTP n=3 (built-in)** | **29.3** | 60.1% | reference |
| DFlash Q4_K_M head, n=8 | 16.9 | 53% (31/58) | −42% |
| DFlash Q8_0 head, n=8 | 13.9 | 44% (25/57) | −52% |
| DDTree (branch 3, top-k 4) | crash | — | `output_reserve` abort |

Production MTP config at 131K context, 15K prompt → 1K output:

| Metric | Value |
|---|---|
| Prompt processing | 404 tok/s |
| Token generation | 26.7 tok/s |
| MTP acceptance | 60.1% |
| VRAM | ~9,477 MiB |

External corroboration (thc1006, RTX 3090, UD-Q4_K_XL, 19-config llama.cpp matrix): baseline 139.9 tok/s; best speculative config 85.6 tok/s **with 100% draft acceptance** (−39%); best DFlash 77.0 tok/s (−44.6%). No llama.cpp speculation mode beat baseline on that card either. The same source measured MTP on vLLM at **+27.5%** — but only on dual-GPU TP=2, not single-card.

## Troubleshooting

### Server aborts on the first request with tree speculation

**Symptom:** `ggml_abort` in `llama_context::output_reserve`, called from `draft_tree`.

**Cause:** The tree verify batch does not fit in VRAM. Branch budget multiplies the tokens verified in one pass.

**Fix:** Set `--spec-branch-budget 0` (flat drafting), lower `-c`, or lower `-ngl`. Or accept that tree speculation is not viable at long context on 12 GB and use MTP.

### DFlash server estimates gigabytes for a tiny draft head

**Symptom:** Log shows `[spec] estimated memory usage of draft model is 2865.44 MiB` for a 279 MB file.

**Cause:** DFlash cross-attention buffers and draft KV state dwarf the weight file.

**Fix:** Budget ~3 GB for the draft runtime, not the file size. Drop `-ngl` or context to make room.

### `failed to fit params to free device memory: n_gpu_layers already set by user`

**Symptom:** Warning at startup when `-ngl` is set explicitly.

**Cause:** The `--fit` auto-placer refuses to override your manual layer count.

**Fix:** Benign. Nothing to do.

### Should I just use a smaller quant instead of speculation?

A smaller quant is the one lever that beat MTP on raw speed here: IQ3_XXS fits more of the model on GPU and reached 53.3 tok/s all-GPU in a separate test. It failed a real-repo agent eval on quality, so Q3_K_M + MTP remained the production default. Treat quant reduction and speculation as separate decisions: benchmark speed with the quant, then validate quality on a real task before adopting it.

## What transfers

1. **Acceptance rate is not speedup.** A 100%-accepted draft can still lose by 39%. The verify pass pays for the union of expert weights. Measure wall-clock decode, never acceptance alone.

2. **Small-active MoE has a speculation ceiling.** When active params per token are tiny (3B of 35B) and routing is sparse (8 of 256), each drafted token pulls fresh expert slices through the memory hierarchy. Draft K ≪ T_thres ≈ 94 means no amortization. This is architecture physics, not a tuning problem.

3. **Jointly-trained MTP beats external draft heads on quantized targets.** External heads are trained on BF16 hidden states. Quantization shifts those states and breaks drafter-verifier alignment. Higher draft precision makes it *worse* (Q8_0 < Q4_K_M here), because the problem is the quantized target, not the draft.

4. **Tree speculation multiplies verify cost.** Every branch is verified in one batch. On memory-bound cards at long context, the tree verify batch is the first allocation to OOM. Flat drafting or MTP first; trees only with headroom.

5. **The winning speculation depends on the engine and topology.** llama.cpp single-GPU: MTP +8%. vLLM dual-GPU TP=2: MTP +27.5%. llama.cpp draft-spec and DFlash, any card tested: net loss. Port the conclusion, not just the flag.

6. **The blunt offload knob beat fine-grained overrides.** `-ncmoe 28` outperformed manual per-layer expert pinning, and one step lower OOMed. Sweep the simple lever to its edge before writing regex tensor maps.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/llama-server-mtp.sh`](configs/llama-server-mtp.sh) | Production MTP launcher (the fix) |
| [`configs/llama-server-dflash.sh`](configs/llama-server-dflash.sh) | DFlash launcher (reproduction of the loss) |
| [`scripts/smoke-moe-speculation.sh`](scripts/smoke-moe-speculation.sh) | ~30s check that MTP speculation is active and drafting |
| [`results/2026-07-04-moe-speculation-matrix.json`](results/2026-07-04-moe-speculation-matrix.json) | Raw benchmark matrix for all speculation modes |
| [`results/2026-07-04-moe-speculation-matrix.md`](results/2026-07-04-moe-speculation-matrix.md) | Tracked summary of the same matrix |
