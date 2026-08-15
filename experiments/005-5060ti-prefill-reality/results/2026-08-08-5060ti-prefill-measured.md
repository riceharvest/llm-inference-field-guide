# 2026-08-08 — 5060 Ti Prefill Measured

**Date:** August 8, 2026
**Hardware:** 2× RTX 5060 Ti 16GB (32GB total VRAM)
**Model:** Qwen3.6-27B NVFP4 (ModelOpt)
**Engine:** vLLM 0.26.0
**Config:** TP=2, FP8 KV, MTP enabled, max-model-len 131072

## Summary

| Prompt length | Estimated | Measured | Ratio |
|---|---|---|---|
| 8K | ~5,500 tok/s | 1,780 tok/s | 3.1× slower |
| 32K | ~5,500 tok/s | 1,580 tok/s | 3.5× slower |
| 64K | ~5,500 tok/s | 1,360 tok/s | 4.0× slower |
| 120K | ~5,500 tok/s | 1,030 tok/s | 5.3× slower |

## Full Results

### 8K Prompt

| Rep | TTFT (s) | Prefill tok/s |
|---|---|---|
| 1 | 0.0045 | 1,778 |
| 2 | 0.0046 | 1,739 |
| 3 | 0.0044 | 1,818 |

### 32K Prompt

| Rep | TTFT (s) | Prefill tok/s |
|---|---|---|
| 1 | 0.0204 | 1,569 |
| 2 | 0.0201 | 1,592 |
| 3 | 0.0206 | 1,553 |

### 64K Prompt

| Rep | TTFT (s) | Prefill tok/s |
|---|---|---|
| 1 | 0.0478 | 1,339 |
| 2 | 0.0485 | 1,319 |
| 3 | 0.0472 | 1,356 |

### 120K Prompt

| Rep | TTFT (s) | Prefill tok/s |
|---|---|---|
| 1 | 0.1152 | 1,042 |
| 2 | 0.1173 | 1,023 |
| 3 | 0.1168 | 1,027 |

## Why the gap?

The earlier estimate (~5,500 tok/s) assumed:
- Optimized FP4-native path
- Custom CUDA kernels
- No MTP overhead
- Specific batch tuning

This setup used:
- Stock vLLM 0.26.0
- FP8 KV (not native FP4)
- MTP enabled (3 speculative tokens)
- Conservative 4K batching limit
- Hybrid GDN/Mamba layers

Each constraint stacks multiplicatively.

## Key Takeaway

Community benchmarks are upper bounds. Always measure your own stack before making deployment decisions.
