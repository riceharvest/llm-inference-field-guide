# Choosing a model for one DGX Spark

**Last checked: 2026-07-13.** This guide ranks models that fit a **single 128 GB DGX Spark**, using public LocalMaxxing runs for speed/configuration and Hugging Face model metadata/discussions for community signals.

## Short answer

1. **Best default: Qwen3.6-27B NVFP4 + MTP.** Prefer the dense 27B when consistent per-token capacity matters more than the 35B-A3B MoE's headline throughput. A Spark vLLM run reports 32.8 tok/s; see the [dedicated guide](dgx-spark-qwen36-27b.md).
2. **Fast alternative: Qwen3.6-35B-A3B NVFP4.** Use it when throughput is the priority; its 42.98 tok/s autoregressive baseline and 102.05 tok/s DFlash result are not evidence that it is the better model for your workload.
3. **Best capability-per-box experiment: MiniMax-M2.7 IQ4.** A 229B/10B-active-class model fits as a 101 GB GGUF and reached 30.88 tok/s at 108K with n-gram speculation.
4. **DeepSeek-V4-Flash fits only as an extreme experiment on one Spark.** A hybrid 2-bit checkpoint uses ~110 GiB resident and runs below 2 tok/s. Practical 36–46 tok/s configurations use two Sparks.
5. **MiniMax-M3 is not the same size as M2.7 and does not properly fit.** M3 is 428B/23B-active versus M2.7's 229B/~10B-active. Its smallest 1-bit GGUF is 128 GB and requires at least 133 GB total memory; useful 4-bit/NVFP4 versions need multiple Sparks.

Do not choose from parameter count alone. On Spark, a sparse MoE can be both larger and faster than a dense 27B because only a small expert subset is active per token.

## Evidence table

These are **community measurements, not controlled cross-model benchmarks**. `c=1` means single-stream. Model quality is not inferred from tok/s.

| Model / artifact | One-Spark fit | LocalMaxxing single-stream evidence | Context | Community signal | Verdict |
|---|---:|---:|---:|---|---|
| [Qwen3.6-27B](https://huggingface.co/Qwen/Qwen3.6-27B) NVFP4 + MTP | Yes | **32.8 tok/s** reported with vLLM NVFP4 | Run-specific | Dense model; preferred here for capability consistency rather than maximum tok/s | **Default** |
| [Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) + NVFP4/DFlash | Yes | **102.05 tok/s** median; 42.98 tok/s autoregressive baseline | 262K | Strong adoption, but sparse A3B activation is a quality/capability tradeoff—not a free speedup | **Throughput alternative** |
| [Qwen3.5-35B-A3B GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF) Q4_K_XL | Yes | **60.3 tok/s** | 262K | Positive for agentic/context use, but mixed on quant/sampler reliability; 858 likes / 172K downloads | **Easy llama.cpp path** |
| [MiniMax-M2.7](https://huggingface.co/MiniMaxAI/MiniMax-M2.7) via [GGUF](https://huggingface.co/unsloth/MiniMax-M2.7) UD-IQ4_XS | Tight: 101 GB weights | **30.88 tok/s** median over 38 warm trials | 108K tested | Positive: HF post calls interactive coding “excellent,” focused, and better at following prompts than tested GLM/Qwen alternatives; 1,229 likes / 1.06M downloads on base model | **Capability experiment** |
| [Step-3.7-Flash GGUF](https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF) UD-IQ4_NL | Yes | **23.0 tok/s** | 2K measured; 262K configured elsewhere | Mixed/negative operationally: practical-tips thread reports infinite reasoning loops from a llama.cpp parser bug and a separate >100K hallucination report | **Experimental** |
| [Qwen3.5-122B-A10B](https://huggingface.co/Qwen/Qwen3.5-122B-A10B) INT4 AutoRound | Yes, tight | **28.1 tok/s**; incomplete public recipe | 262K claimed | Healthy adoption (583 likes / 628K downloads), but users are already requesting a Qwen3.6 successor | **Only if 122B quality wins your eval** |
| [Nemotron-3-Super-120B-A12B NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) | Context-dependent | No sufficiently reproducible one-Spark row selected | 200K–240K problematic in report | Negative Spark-specific signal: open OOM/system-stall report; separate report says 16 tok/s | **Skip initially** |
| [DeepSeek-V4-Flash hybrid 2-bit](https://huggingface.co/bleysg/DeepSeek-V4-Flash-IQ2XXS-Q2K-FP8-120GB-target) | Barely: ~110 GiB resident | **<2 tok/s** on one Spark | 16K default in bring-up | Coherent output demonstrated, but custom/community vLLM patches, long autotune, crashes and KV-accounting bugs were reported | **Proof-of-life only** |
| [MiniMax-M3 GGUF](https://huggingface.co/unsloth/MiniMax-M3-GGUF) UD-IQ1_M | No stable fit | No selected one-Spark run | KV/context excluded from 128 GB file | Experimental, text-only, no MiniMax Sparse Attention; Unsloth states ≥133 GB required | **Does not fit one 128 GB Spark** |

HF likes/downloads are popularity signals, not quality scores. Discussion sentiment is anecdotal and is reported as such.

## Recommended configurations

### 1. Qwen3.6-27B NVFP4 + MTP — preferred default

Use the dense 27B first. The dedicated guide covers Spark-specific NVFP4, MTP, engine choices, and the measured 32.8 tok/s vLLM result:

**[DGX Spark: Qwen3.6-27B NVFP4 + MTP](dgx-spark-qwen36-27b.md)**

This is a preference for the model's dense per-token capacity, not a speed claim. Benchmark both on the same quality set if deciding between 27B and 35B-A3B.

### 2. Qwen3.6-35B-A3B NVFP4 + DFlash — throughput alternative

The strongest reproducible LocalMaxxing submission reports a 42.98 tok/s autoregressive baseline and 102.05 tok/s median with DFlash. It used vLLM `0.19.2rc1` plus an off-by-one patch; without that patch, acceptance reportedly collapsed from about 80% to 3%. Do not copy only the flags and omit the matching runtime patch/revision.

```bash
vllm serve /models/Qwen3.6-35B-A3B-NVFP4 \
  --served-model-name qwen36-35b-a3b \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 262144 \
  --max-num-batched-tokens 4096 \
  --max-num-seqs 1 \
  --load-format fastsafetensors \
  --attention-backend flash_attn \
  --enable-prefix-caching \
  --speculative-config \
    '{"method":"dflash","num_speculative_tokens":15,"model":"/models/Qwen3.6-35B-A3B-DFlash"}' \
  --trust-remote-code
```

LocalMaxxing source run: https://localmaxxing.com/en/models/Qwen/Qwen3.6-35B-A3B?run=cmon8glp0000el204m6i8dge6

Before adopting it, run the same workload with speculation disabled. DFlash acceptance is prompt-dependent and the submission reports high variance.

### 3. Qwen3.5-35B-A3B Q4_K_XL — simplest fast GGUF

```bash
llama-server \
  --model Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf \
  --ctx-size 262144 \
  --parallel 1 \
  --flash-attn on \
  --cache-type-k q8_0 \
  --cache-type-v q8_0
```

LocalMaxxing reports 60.3 tok/s for a sustained 2,048-token completion. HF discussion is **mixed rather than uniformly positive**: one user preferred standard Q4_K_M and conservative sampling for factual/creative reliability, while another found K_XL better at summaries, context retention, coding, and agentic work.

Relevant discussion: https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF/discussions/43

### 4. MiniMax-M2.7 UD-IQ4_XS — largest useful one-box option

The reported GGUF is 101 GB. Leave headroom for KV cache, CUDA workspaces, and the OS; do not allocate the full 128 GB to weights plus an enormous context.

```bash
llama-server \
  --jinja --flash-attn on --no-warmup \
  --threads 20 --ctx-size 108000 \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  --parallel 1 --n-gpu-layers 99 \
  --model /models/MiniMax-M2.7-UD-IQ4_XS-00001-of-00004.gguf \
  --cache-reuse 256 --ctx-checkpoints 0 -cram 100 \
  --spec-type ngram-simple \
  --draft-max 16 --draft-min 1 --draft-p-min 0.5 \
  --spec-ngram-size-n 4
```

The submission pins `nvcr.io/nvidia/cuda:13.0.0-devel-ubuntu24.04` and reports silent cuBLAS corruption with CUDA 13.2/13.3 on GB10. Verify that claim on your current stack before standardizing it; do not assume newer is safe merely because it is newer.

LocalMaxxing source run: https://localmaxxing.com/en/models/MiniMaxAI/MiniMax-M2.7?run=cmon7v1f70007l204wtsw8ogj

Positive coding post: https://huggingface.co/MiniMaxAI/MiniMax-M2.7/discussions/42

### 5. DeepSeek-V4-Flash — fits, but not usefully yet

Two distinct answers matter:

- **Physical fit: yes.** The community hybrid checkpoint uses IQ2_XXS/Q2_K experts, FP8 dense/attention tensors, and BF16 embeddings/norms: ~85 GiB on disk and ~110 GiB resident on one Spark.
- **Practical fit: no, currently.** The published one-Spark vLLM bring-up reports **below 2 tok/s**, >10-minute first autotuning, only ~18K KV tokens in one log, and follow-up crashes/patch fixes. A separate custom C++ MXFP4 runtime proves correctness but has no OpenAI server and no competitive absolute TPS.

Use two Sparks for the current practical path: public LocalMaxxing rows report 36.04–45.7 tok/s with TP=2 at 200K–500K context.

Sources: [single-Spark hybrid vLLM report](https://forums.developer.nvidia.com/t/deepseekv4-flash-hybrid-quant-1x-dgx-spark-antirezs-optimized-128-gb-mlx-recipe-ported-to-vllm-for-gb10/369584), [single-Spark custom-runtime proof](https://forums.developer.nvidia.com/t/deepseek-v4-flash-mxfp4-proof-of-life-on-a-single-gb10-gx10/369131).

### 6. MiniMax-M3 — much larger than M2.7

| | MiniMax-M2.7 | MiniMax-M3 |
|---|---:|---:|
| Total parameters | ~229B | ~428B |
| Active parameters | ~10B | ~23B |
| Practical one-Spark artifact | 101 GB UD-IQ4_XS | None |
| Smallest cited GGUF | — | 128 GB UD-IQ1_M; needs ≥133 GB total |
| 4-bit artifact | ~101–122 GB depending quant | 208 GB UD-IQ4_XS |

M3's NVIDIA NVFP4 repository occupies roughly 250 GB and its example uses TP=8; it is not a one-Spark checkpoint. The 128 GB 1-bit GGUF is also an invalid fit for a 128 GB unified-memory machine because the file alone consumes the budget before KV cache, CUDA workspace, and the OS. Current GGUF support is experimental, text-only, and falls back to dense attention because MiniMax Sparse Attention is unsupported.

Sources: [official NVIDIA NVFP4 card](https://huggingface.co/nvidia/MiniMax-M3-NVFP4), [Unsloth M3 sizing guide](https://unsloth.ai/docs/models/minimax-m3).

### 7. Step-3.7-Flash — only after a quality canary

A community workaround uses:

```bash
llama-server --no-mmap --no-warmup \
  -hf stepfun-ai/Step-3.7-Flash-GGUF:iq4_xs \
  --ctx-size 262144 --parallel 1 \
  --temp 1.0 --top-p 0.95 \
  --reasoning-budget 16384 \
  --spec-type ngram-simple
```

This is not a clean recommendation: the same HF thread records an infinite-reasoning issue attributed to llama.cpp parsing and a report of hallucinations beyond 100K context. Run a long-context retrieval and tool-use canary before production.

Discussion: https://huggingface.co/stepfun-ai/Step-3.7-Flash-GGUF/discussions/6

## Models that need two Sparks

Do not mix these with the one-box shortlist:

- DeepSeek-V4-Flash: the **useful-performance** path currently needs two Sparks; the extreme one-Spark hybrid is documented above.
- MiniMax-M3: 4-bit/NVFP4-class versions need at least two to four Sparks depending checkpoint/runtime; the 128 GB 1-bit GGUF still needs ≥133 GB total memory.
- MiniMax-M2.7 AWQ: the 43.4 tok/s / 196K recipe is **two** Sparks with TP=2 over RoCE.
- Larger GLM-5.x quants may load only at severe quantization or across multiple boxes; loading is not the same as being a good interactive choice.

## Selection protocol

1. Start with Qwen3.6-27B NVFP4 + MTP.
2. Compare Qwen3.6-35B-A3B only if its higher throughput is worth the sparse-capacity tradeoff; add DFlash only after measuring its autoregressive baseline.
3. Run your real quality set: coding patch, tool calling, long-context retrieval, and instruction adherence.
4. If quality is insufficient, try MiniMax-M2.7 IQ4; do not substitute M3 on one Spark because it does not leave runtime headroom.
5. Reject any model that merely loads but leaves insufficient KV/workspace headroom or fails the quality canary.

## Data boundaries

- LocalMaxxing is community-submitted. Rows differ in runtime, context, quantization, output length, speculation, and measurement method.
- Some submissions reference private/local artifacts or omit exact commands. Those are discovery leads, not reproducible evidence.
- Speeds above are copied only from relevant single-Spark, single-stream rows; dual-Spark and high-concurrency aggregate numbers are labeled separately.
- HF discussions are anecdotal. Positive posts establish that positive community experience exists; they do not establish consensus.

## Sources

- DGX Spark LocalMaxxing hardware page: https://localmaxxing.com/en/hardware/UNIFIED%3Agb10%20dgx%20spark%20gb10?name=GB10+DGX+Spark+GB10
- LocalMaxxing leaderboard API: `https://localmaxxing.com/api/leaderboard`
- Hugging Face model metadata/discussion API: `https://huggingface.co/api/models/{owner}/{model}`
- Individual model and discussion links are embedded above.
