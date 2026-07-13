---
name: llama-cpp-serving-operator
description: Use when deploying, tuning, benchmarking, or troubleshooting a llama.cpp-compatible GGUF server, including GPU/CPU placement, KV cache, long context, MTP/speculative decoding, and OpenAI-compatible API operation.
version: 1.0.0
author: riceharvest contributors
license: MIT
metadata:
  hermes:
    tags: [llama.cpp, gguf, serving, cuda, speculative-decoding]
    related_skills: [inference-bottleneck-diagnostics, hermes-local-llm-integration]
---

# llama.cpp Serving Operator

## Overview

Operate llama.cpp from exact binary capabilities and measured behavior. Fork and release flag syntax changes; `llama-server --help` and startup logs are authoritative for the installed binary.

## When to Use

- Launching a GGUF through `llama-server`.
- Choosing GPU offload, KV types, context, batch/ubatch, slots, or Flash Attention.
- Enabling MTP, draft-model, or n-gram speculation.
- Investigating OOM, slow load, malformed tool calls, or poor throughput.

## Bring-up Sequence

1. **Discover live state.** Check listeners, accelerator processes/memory, RAM/swap, model metadata, binary version, and `--help`. Completion criterion: no unknown server owns the target port and idle memory is known.
2. **Compatibility gate.** Start conservatively with one slot, modest context, no speculation, and safe memory headroom. Verify `/v1/models` and one completion. Completion criterion: response content and timings are valid.
3. **Record placement.** Retain the startup log showing offloaded layers/tensors, model/KV/compute buffers, graph splits, and backend. Completion criterion: the actual placement—not requested placement—is known.
4. **Fit context empirically.** Calculate weights + KV/state + workspace, then binary-search context. Back off from the load/OOM boundary and run a real request. Completion criterion: stable maximum and headroom are reported.
5. **Tune one variable at a time.** Suggested order: quant, placement, KV type, slots, batch/ubatch, Flash Attention, speculation. Use paired repeats. Completion criterion: every accepted flag has a measured wall-time benefit or necessary capacity benefit.
6. **Production gate.** Test realistic prompt/output, structured tool call if required, concurrency target, and sustained run. Completion criterion: API semantics, quality, latency, memory, and restart behavior all pass.

## Baseline Pattern

```bash
llama-server -m /path/model.gguf   --host 127.0.0.1 --port 8080   -c 8192 -np 1
```

Confirm the local spellings for GPU offload, Flash Attention, KV types, Jinja/chat templates, and speculative flags before adding them.

## Measurement Probe

```bash
curl -s http://127.0.0.1:8080/v1/completions   -H 'content-type: application/json'   -d '{"prompt":"Explain memory-bound decode.","n_predict":256,"temperature":0}'   > result.json
```

Preserve the raw JSON and parse server timing fields. Discard cold graph-capture results from warm medians, but report cold latency separately.

## Placement Rules

- Full accelerator residency is preferred when it leaves workspace headroom.
- More offload is not automatically faster near capacity.
- For hybrid host/device runs, tensor-class placement may beat whole-layer placement when supported; derive tensor sizes and verify actual placement.
- MoE expert placement and dense attention placement are separate decisions.
- CPU thread count must be swept; oversubscription can hurt.

## KV and Context Rules

Calculate KV from architecture metadata; do not infer it from total parameters. Include hybrid attention/SSM state and runtime workspace. KV quantization is a capacity/traffic lever that requires a quality test. `-c` allocation behavior differs across builds, so verify with live memory and requests.

## Speculation Rules

Establish target-only performance first. Verify architecture/state compatibility. Sweep depth and record drafted, accepted, target passes, memory, and wall time. High acceptance without wall-time improvement fails. Add speculation only after the base server is stable.

## Tool-Calling Gate

If serving Hermes or another agent, send a real request containing tools and require structured `tool_calls`, valid JSON arguments, correct finish reason, visible final content, and preserved reasoning behavior. A plain chat response is insufficient.

## Common Pitfalls

1. Copying flags between upstream and forks.
2. Trusting requested GPU layers instead of startup logs.
3. Benchmarking only `llama-bench` when server-only MTP behavior matters.
4. Mixing cold graph capture with warm decode.
5. Using maximum context/slots by default and starving workspace.
6. Setting token caps that truncate reasoning before visible content.
7. Assuming GGUF chat templates guarantee OpenAI tool-call conversion.
8. Ignoring swap/page pressure during large model loads.

## Verification Checklist

- [ ] Exact commit/version and build backend recorded
- [ ] Startup log retained
- [ ] `/v1/models` and completion pass
- [ ] Stable context and memory headroom measured
- [ ] Production-shaped benchmark repeated
- [ ] Speculation compared against target-only
- [ ] Tool call tested when agent use is intended
- [ ] Restart and port ownership verified
