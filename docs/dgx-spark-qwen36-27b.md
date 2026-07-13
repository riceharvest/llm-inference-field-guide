# DGX Spark case: Qwen3.6-27B NVFP4 + MTP

## First conclusion

DGX Spark is not a miniature H100. Its GB10 exposes strong low-precision compute and 128 GB unified memory, but the official hardware guide specifies **273 GB/s memory bandwidth**. Dense batch-1 decoding repeatedly streams a large fraction of a 27B model, so bandwidth and kernel efficiency remain hard constraints.

If the effective NVFP4 model footprint touched per token is 14–16 GB, the **derived ideal bandwidth ceiling** is only:

```text
273 / 14 = 19.5 tok/s
273 / 16 = 17.1 tok/s
```

Real speed must be lower because attention/KV, recurrent hybrid layers, dequant/scales, output projection, launch overhead, and imperfect bandwidth utilization also cost time. Therefore 10–16 tok/s may be plausible; 2–5 tok/s indicates an additional bottleneck. Replace these assumptions with the actual file/tensor footprint and measured bandwidth.

## Triage order

### 1. Establish exact facts

Capture:

```bash
./tools/collect-system.sh > spark-system.txt
llama-server --version
llama-server --help > llama-help.txt
```

Record exact model repository, revision, file, runtime commit, full command, prompt/output tokens, active context, no-MTP tok/s, MTP tok/s, drafted/accepted counts, and peak memory.

### 2. Confirm the native path

- The binary must target the Spark/GB10 architecture rather than a generic CUDA fallback.
- Startup logs must identify CUDA and the expected quantization path.
- Compare current upstream/runtime release against the checkpoint publisher's tested build.
- A GGUF labeled NVFP4 is not proof every layer uses an optimized NVFP4 kernel.

### 3. Run target-only

Disable MTP and measure 512/256 at low context, then at production context. If target-only is already slow, MTP is not the root cause.

### 4. Sweep context

Test 2K, 8K, 32K, and production active context with fixed 256-token output.

- Flat slow decode: weight traffic, unsupported kernels, clocks, or placement.
- Strong context-dependent slowdown: attention/KV or hybrid-state implementation.

### 5. Sweep MTP depth

Test target-only and `n_max` 1, 2, 3, 4 (only syntax supported by the exact build). Record drafted, accepted, and wall time. Stop increasing depth when verification/draft cost exceeds accepted-token savings.

Published/runtime examples can show gains around 20%, but that is workload- and build-specific—not a guaranteed multiplier.

### 6. Check unified-memory residency and migration

128 GB capacity does not mean all pages are equally cheap. Look for memory migration/fault activity and ensure no competing workload forces paging or migration. Keep the model resident and benchmark after warm-up.

### 7. Check power and clocks

Run a long decode and log clocks, power, temperature, and utilization. Compare first and last 256 tokens. A compact system may sustain lower clocks than a short benchmark suggests.

### 8. Compare runtimes without changing the model semantics

For the safetensors NVFP4 checkpoint, compare a current vLLM path with MTP support against llama.cpp/GGUF only if the exact checkpoint format and outputs are equivalent. Runtime comparison isolates kernel maturity; changing quant at the same time does not.

## Fast interpretation table

| Observation | Likely cause | Next test |
|---|---|---|
| 10–16 tok/s, memory traffic high | near Spark bandwidth reality | smaller bytes/token or MTP |
| 2–5 tok/s, CPU high/GPU idle | fallback/host work/sync | logs, placement, native build |
| 2–5 tok/s, GPU compute high | inefficient/unsupported hybrid kernels | runtime/build A/B |
| Fast at 2K, slow at 32K+ | attention/KV | context sweep, FA/KV A/B |
| MTP acceptance high but no speedup | draft/verification overhead | reduce n_max, inspect wall time |
| Performance falls during run | power/thermal or migration | clocks/power/page activity |
| More offload makes it slower | memory pressure/graph fallback | leave workspace headroom |

## Ask the friend for this bundle

```text
- exact DGX OS/driver/runtime versions
- exact model URL + revision + filename
- complete server command and startup log
- low-context no-MTP benchmark
- same benchmark with MTP and acceptance counters
- production-context benchmark
- system collector output
```

Without that bundle, “NVFP4 is slow” is a symptom, not a diagnosis.

## Sources

- NVIDIA, DGX Spark Hardware Overview: https://docs.nvidia.com/dgx/dgx-spark/hardware.html
- vLLM Qwen3.6-27B recipe: https://recipes.vllm.ai/Qwen/Qwen3.6-27B
- Example Spark-targeted NVFP4 MTP GGUF/model notes: https://huggingface.co/nilayparikh/Qwen3.6-27B-Text-NVFP4-MTP-GGUF
- NVIDIA developer discussion: https://forums.developer.nvidia.com/t/mtp-llama-cpp-a-look-at-qwen3-6-27b/370298

Hardware facts are source-backed; ceiling arithmetic is derived and explicitly assumption-dependent.
