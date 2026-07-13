# LLM Inference Field Guide

A measurement-first guide to making local LLM inference faster—and identifying **why** it is slow before changing flags.

Built from real llama.cpp/ik_llama.cpp work on dense, MoE, hybrid-attention/SSM, long-context, quantized, and speculative-decoding workloads.

## Start here

1. Read [the bottleneck decision tree](docs/bottleneck-decision-tree.md).
2. Capture a reproducible baseline using [the benchmark protocol](docs/benchmark-protocol.md).
3. Apply only the section matching the measured bottleneck.
4. Change one variable at a time; use paired repeats.

## Install the Hermes skills

Install one or all directly from this repository:

```bash
hermes skills install https://raw.githubusercontent.com/riceharvest/llm-inference-field-guide/main/skills/inference-bottleneck-diagnostics/SKILL.md
hermes skills install https://raw.githubusercontent.com/riceharvest/llm-inference-field-guide/main/skills/llama-cpp-serving-operator/SKILL.md
hermes skills install https://raw.githubusercontent.com/riceharvest/llm-inference-field-guide/main/skills/hermes-local-llm-integration/SKILL.md
```

Then start a fresh session and load the relevant skill with `/skill <name>` or preload it with `hermes -s <name>`. Inspect before installation with `hermes skills inspect <URL>`.

## Included Hermes skills

- **inference-bottleneck-diagnostics** — measurement-first root-cause isolation.
- **llama-cpp-serving-operator** — GGUF server bring-up, fitting, MTP, and production gates.
- **hermes-local-llm-integration** — structured tool calls, context, real Hermes workflow tests, and fallback design.

## Guides

- [Bottleneck decision tree](docs/bottleneck-decision-tree.md)
- [Optimization levers by bottleneck](docs/optimization-levers.md)
- [Benchmark protocol](docs/benchmark-protocol.md)
- [DGX Spark: Qwen3.6-27B NVFP4 + MTP](docs/dgx-spark-qwen36-27b.md)
- [`llama.cpp` command cookbook](docs/llama-cpp-cookbook.md)
- [`collect-system.sh`](tools/collect-system.sh): captures hardware, processes, clocks, build, and model metadata without secrets

## Core model

Decode is usually bounded by one of these:

| Bottleneck | Signature | First lever |
|---|---|---|
| Weight bandwidth | GPU busy; tok/s tracks weight bytes | smaller/faster quant, full-device placement |
| Compute/kernel | compute busy; quant/kernel sensitive | native kernels, correct backend/build |
| CPU or host memory | CPU saturated or GPU waiting | full offload, NUMA/threads, smaller model |
| Interconnect/synchronization | low utilization, split placement | eliminate per-token transfers |
| Attention/context | decode falls with context | reduce active context, Flash Attention, KV format |
| Launch/shape inefficiency | low GPU use at batch 1 | MTP/speculation, better kernels/runtime |
| Memory pressure | OOM, graph fallback, regressions near capacity | leave headroom, reduce KV/slots/context |
| Scheduler/concurrency | aggregate and per-user throughput diverge | benchmark concurrency separately |

A faster-looking precision name does not prove faster inference. NVFP4 can reduce weight traffic and accelerate supported matrix kernels, but it cannot fix unsupported kernels, recurrent SSM operations, long-context attention, host/device splits, or scheduler overhead.

## Evidence labels

- **Measured:** command, hardware, build, workload, repeats, and raw output are available.
- **Derived:** arithmetic from disclosed specs; assumptions shown.
- **Hypothesis:** mechanism is plausible but not yet isolated by an A/B test.

Contributions are welcome; see [CONTRIBUTING.md](CONTRIBUTING.md). MIT licensed.
