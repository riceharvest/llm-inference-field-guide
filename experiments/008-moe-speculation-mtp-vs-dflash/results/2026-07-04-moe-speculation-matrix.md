# 008 Results — MoE Speculation Matrix (2026-07-04)

**Hardware:** RTX 3060 12GB + Ryzen 5950X (62 GB RAM)
**Model:** Qwen3.6-35B-A3B-UD-Q3_K_M.gguf (MoE, 3B active, 8-of-256 routed experts, GQA 8:1)
**Engines:** llama.cpp MTP build; BeeLlama fork (turboquant + DFlash)

## Head-to-head decode

| Speculation | TG tok/s | Acceptance | vs MTP | Status |
|---|---|---|---|---|
| None | ~27.1 | — | −8% | baseline |
| **MTP n=3 (built-in)** | **29.3** | 60.1% | reference | **winner** |
| DFlash Q4_K_M head, n=8 | 16.9 | 53% (31/58) | −42% | slower than baseline |
| DFlash Q8_0 head, n=8 | 13.9 | 44% (25/57) | −52% | slower than baseline |
| DDTree (branch 3, top-k 4) | — | — | — | crash on first request |

DFlash notes: 279 MB draft file but ~2,865 MiB estimated draft runtime (cross-attention buffers). Q8_0 head scored worse than Q4_K_M — the misalignment lives in the quantized target's hidden states, not the draft head.

DDTree crash backtrace:

```
ggml_abort
llama_context::output_reserve(int)
llama_context::decode(llama_batch const&)
common_speculative_impl_dflash::draft_tree(...)
server_context_impl::update_slots()
```

## Production MTP config at 131K context

Benchmark shape: 15K prompt → 1K output.

| Metric | Value |
|---|---|
| Prompt processing | 404 tok/s |
| Token generation | 26.7 tok/s |
| MTP acceptance | 60.1% |
| VRAM | ~9,477 MiB |

Flags: `-ngl 32 -ncmoe 28 -t 14 -tb 16 -b 2048 -ub 512 -fa on -ctk q4_0 -ctv q4_0 -c 131072 -np 1 --kv-unified --cache-idle-slots --spec-type draft-mtp --spec-draft-n-max 3`

## `-ncmoe` expert-offload sweep

| Config | VRAM | Decode (tiny prompt) | MTP acceptance | Status |
|---|---|---|---|---|
| `-ncmoe 28` (baseline) | 10,090 MiB | 34.62 tok/s | 0.643 | optimal |
| `-ncmoe 27` + blk.27 experts forced CPU (`-ot`) | 10,422 MiB | 34.90 tok/s | 0.691 | no useful gain |
| `-ncmoe 26` + blk.26/27 experts forced CPU (`-ot`) | — | — | — | CUDA OOM (`cudaGraphLaunch`) |

## External corroboration

thc1006 RTX 3090 matrix (UD-Q4_K_XL, llama.cpp, 19 configs):

| Config | tok/s | vs baseline |
|---|---|---|
| Baseline (no spec) | 139.9 | reference |
| Best speculative (`--draft-min 48 --draft-max 64`, 100% acceptance) | 85.6 | −39% |
| Best DFlash (`--draft-max 8`) | 77.0 | −44.6% |
| MTP on vLLM, dual 3090 TP=2 | — | **+27.5%** |

## Root cause

MoE expert saturation (arXiv 2505.19645). A3B activates 3B parameters per token with 8-of-256 routed experts. Expert-saturation threshold T_thres ≈ 94, far above any practical draft depth K. The verify pass pays for the union of expert weights across all drafted tokens. On a bandwidth-bound card, 100% acceptance cannot rescue speculation.

MTP survives because it is trained jointly with the base model and shares its representation space. Three-token drafts at 60% acceptance with a cheap verify cycle beat eight-token external drafts at 53% with an expensive one.
