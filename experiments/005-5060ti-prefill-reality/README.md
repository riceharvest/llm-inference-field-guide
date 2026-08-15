# 005 — The 5060 Ti Prefill Reality Check

> Community estimates predicted ~5.5K prefill tok/s for 2× RTX 5060 Ti. Your actual measured numbers: **1.78K tok/s at 8K prompt, 1.03K at 120K.** A 3–5× gap. The estimates assumed optimizations you're not running.

**Hardware:** 2× RTX 5060 Ti 16GB (32GB total VRAM)
**Model:** Qwen3.6-27B NVFP4
**Engine:** vLLM 0.26.0
**Date:** 2026-08-08

---

## Quick Start

Before trusting community benchmarks for your hardware, measure your own stack:

```bash
# Start the benchmark server
bash configs/serve-5060ti-prefill.sh

# Then run a prefill sweep at multiple prompt lengths
for ctx in 8192 32768 65536 120000; do
  python3 -c "
import requests, time
r = requests.post('http://127.0.0.1:18000/v1/completions', json={
  'model': 'Qwen3.6-27B-NVFP4',
  'prompt': 'x' * $ctx,
  'max_tokens': 1,
  'stream': False
}, timeout=300)
d = r.json()
ttft = d.get('response_metadata', {}).get('ttft', 0)
print(f'ctx=$ctx ttft={ttft:.3f}s prefill_tok_s={$ctx/max(ttft,0.001):.0f}')
"
done
```

**Full sweep script:** [`scripts/prefill-sweep.py`](scripts/prefill-sweep.py)
**Serving config:** [`configs/serve-5060ti-prefill.sh`](configs/serve-5060ti-prefill.sh)

## Exact flags

```bash
# Example vLLM serving command
python -m vllm.entrypoints.openai.api_server \
  --model Qwen3.6-27B-NVFP4 \
  --port 18000 \
  --tensor-parallel-size 2 \
  --max-model-len 120000 \
  --gpu-memory-utilization 0.90 \
  --enable-chunked-prefill \
  --max-num-seqs 4
```

---

## The symptom

Community estimates predicted ~5.5K prefill tok/s for this hardware. Actual measured numbers were dramatically lower.

## The wrong suspicion

We suspected:
- The GPUs were defective or thermally throttling
- The model was the wrong quant
- TP configuration was wrong

None of these were the cause.

## The real cause

The earlier estimate assumed an **optimized FP4-native path**. This setup used:
- Stock vLLM 0.26.0
- ModelOpt NVFP4 weights
- FP8 KV cache
- Hybrid GDN/Mamba layers
- MTP enabled
- Conservative 4K batching limit

Each of those "reasonable" constraints stacked multiplicatively. The gap wasn't a bug — it was the difference between a clean FP4-native benchmark and a production-featured serving stack.

## Measured Results

**Date:** 2026-08-08
**Full data:** [`results/2026-08-08-5060ti-prefill-measured.md`](results/2026-08-08-5060ti-prefill-measured.md)

| Prompt length | Estimated | Measured | Ratio |
|---|---|---|---|
| 8K | ~5,500 tok/s | 1,780 tok/s | 3.1× slower |
| 32K | ~5,500 tok/s | 1,580 tok/s | 3.5× slower |
| 64K | ~5,500 tok/s | 1,360 tok/s | 4.0× slower |
| 120K | ~5,500 tok/s | 1,030 tok/s | 5.3× slower |

## Troubleshooting

### Benchmark numbers don't match community reports

**Symptom:** Your measured throughput is 3–5× lower than published benchmarks.

**Cause:** Community benchmarks assume specific optimizations you may not be running.

**Fix:** Always measure your own stack before making decisions. Community numbers are upper bounds, not guarantees.

### Prefill speed drops with context length

**Symptom:** Prefill tok/s at 120K is much lower than at 8K.

**Cause:** Prefill is memory-bandwidth bound. Longer contexts = more KV to write = lower effective throughput.

**Fix:** This is expected behavior. If you need consistent prefill speed at long context, you need more memory bandwidth.

---

## What transfers

1. **Benchmark estimates assume optimizations you may not be running.** Expect 30-50% of optimistic estimates until you've tuned your stack.

2. **Each "reasonable" constraint stacks multiplicatively.** Stock vLLM + FP8 KV + MTP + 4K batching can reduce throughput by 3-5× vs an optimized FP4-native path.

3. **Measure your own hardware.** Don't make deployment decisions based on community benchmarks alone.

4. **Prefill is memory-bandwidth bound.** It naturally drops with context length. This is not a bug.

---

## Files

| Path | Purpose |
|---|---|
| [`scripts/smoke-5060ti-prefill.sh`](scripts/smoke-5060ti-prefill.sh) | ~30s prefill smoke test |
| [`scripts/prefill-sweep.py`](scripts/prefill-sweep.py) | Automated prefill benchmark script |
| [`results/2026-08-08-5060ti-prefill-measured.json`](results/2026-08-08-5060ti-prefill-measured.json) | raw benchmark matrix |
| [`results/2026-08-08-5060ti-prefill-measured.md`](results/2026-08-08-5060ti-prefill-measured.md) | Full sweep data (4 context lengths × 3 repeats) |
