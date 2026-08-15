# 001 — Ling-3.0-tiny 128K on RTX 3060

> Run Ling-3.0-tiny at 128K context fully on an RTX 3060 12GB. Includes llama-server config, systemd unit, memory budget breakdown, and benchmark numbers.

**Hardware:** RTX 3060 12GB
**Model:** `Ling-3.0-tiny-Q4_K_M.gguf` (7.9B total / 1.3B active, `bailingmoe3` arch)
**Engine:** aetherbird/llama.cpp `bailingmoe3-support` fork (commit `db4480bc8`, build 10411)
**Endpoint:** local inference provider wired into Hermes Agent as a local model provider

---

## Quick Start

1. **Get the fork**
   ```bash
   git clone https://github.com/aetherbird-llama-cpp /tmp/aetherbird-llama-cpp
   cd /tmp/aetherbird-llama-cpp
   git checkout bailingmoe3-support
   ```

2. **Build**
   ```bash
   cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/gcc-14
   cmake --build build -j$(nproc)
   ```

3. **Get the GGUF**
   The HF-converted GGUFs cap at 32K — you need one rebuilt with 128K metadata.
   See [`configs/convert-128k.sh`](configs/convert-128k.sh).

4. **Run**
   ```bash
   ./build/bin/llama-server \
     -m /path/to/Ling-3.0-tiny-Q4_K_M-128K.gguf \
     --host 0.0.0.0 --port 12445 \
     -c 131072 -n 16384 -b 2048 -ub 512 \
     -ngl 999 --device CUDA0 -fa auto \
     -ctk bf16 -ctv bf16 \
     --load-mode mmap+mlock \
     --parallel 2 --poll 50 --poll-batch 1 \
     --prio 2 --prio-batch 2 -t 14 -tb 16 \
     --temp 1.0 --top-p 0.95 --top-k 20
   ```

**Full systemd unit:** [`configs/ling-llama.service`](configs/ling-llama.service) — auto-starts on boot, restarts on failure, 20G memory cap.

---

## The symptom

Embedded GGUF metadata caps context at 32K despite `-c 131072`. Server loads, health returns OK, but loses coherence past ~30K tokens. First long prompt OOMs because `--parallel 4` pre-allocates 4 × 128K × BF16 KV = ~5.1 GB.

## The fix

1. Regenerate the GGUF with 128K context metadata. See [`configs/convert-128k.sh`](configs/convert-128k.sh).
2. Reduce `--parallel` from 4 to 2. KV cache is pre-allocated for all slots at full context.
3. If you need more concurrency, use Q4 KV (`-ctk q4_0 -ctv q4_0`) to halve KV VRAM.

**VRAM budget (12GB card, ~10.9GB usable after desktop):**

| Component | VRAM |
|---|---|
| Desktop | ~799 MiB |
| Model weights (Q4_K_M) | ~910 MiB |
| KV cache × 2 slots @ 128K BF16 | ~2,560 MiB |
| Batch buffers (ubatch 512) | ~384 MiB |
| CUDA overhead + graphs | ~450 MiB |
| **Total** | **~5,093 MiB** |
| **Free** | **~5,807 MiB** |

> **Note:** `--parallel 4` OOMs. KV cache is pre-allocated for all slots at full context: 4 × 128K × BF16 = ~5.1 GB just for KV. With 2 slots it's ~2.5 GB — stable.

---

## Measured Results

**Date:** 2026-08-15
**Full data:** [`results/2026-08-15-ling3t-128k-bench.md`](results/2026-08-15-ling3t-128k-bench.md)

| KV quant | Model VRAM | Decode speed | Quality hit |
|---|---|---|---|
| BF16 | ~910 MiB | ~126–133 tok/s | baseline |
| Q8_0 | ~720 MiB | ~126–133 tok/s | none measurable |
| Q4_0 | ~530 MiB | ~126–133 tok/s | none measurable |

---

## Troubleshooting

### GGUF behaves as 32K context despite `-c 131072`

**Symptom:** Model loads, health returns OK, but loses coherence past ~30K tokens.

**Cause:** The GGUF's embedded `context_length` metadata is 32768. llama.cpp respects the embedded cap — CLI `-c` doesn't override it.

**Fix:** Regenerate the GGUF from `inclusionAI/Ling-3.0-tiny-int4` with 128K context metadata. See [`configs/convert-128k.sh`](configs/convert-128k.sh).

### OOM on first long prompt

**Symptom:** Server health is OK, but a 100K prompt triggers OOM.

**Cause:** `--parallel 4` pre-allocates 4 × 128K × BF16 KV = ~5.1 GB. With model + overhead, this exceeds the 12GB card.

**Fix:** Reduce to `--parallel 2`. If you need more concurrency, use Q4 KV (`-ctk q4_0 -ctv q4_0`) to halve KV VRAM.

### `fused_topk_deepseek does not support compute capability 86`

**Symptom:** Trying to run Ling-3.0-tiny on SGLang crashes immediately.

**Cause:** SGLang's KDA kernels are gated to SM90+ (Hopper) or SM100 (Blackwell). The RTX 3060 is SM8.6 (Ampere).

**Fix:** Use the aetherbird llama.cpp fork — it has native CUDA KDA kernels that run on Ampere. SGLang is not viable for this model on this hardware.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/ling-llama.service`](configs/ling-llama.service) | systemd user service (auto-start, restart, 20G cap) |
| [`configs/convert-128k.sh`](configs/convert-128k.sh) | Regenerate GGUF with 128K context metadata |
| [`results/2026-08-15-ling3t-128k-bench.md`](results/2026-08-15-ling3t-128k-bench.md) | Full benchmark data (15 prompts × 3 KV quants) |
| [`scripts/bench.py`](scripts/bench.py) | Benchmark harness — spins up server, sends prompts, captures VRAM + output |
