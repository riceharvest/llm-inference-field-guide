---
name: inference-bottleneck-diagnostics
description: Use when local LLM inference is slow, unstable, unexpectedly memory-hungry, or regresses after a runtime/config change. Diagnose the limiting phase and mechanism with controlled measurements before recommending flags.
version: 1.0.0
author: riceharvest contributors
license: MIT
metadata:
  hermes:
    tags: [llm, inference, profiling, bottlenecks, benchmarking]
    related_skills: [llama-cpp-serving-operator, hermes-local-llm-integration]
---

# Inference Bottleneck Diagnostics

## Overview

Find the limiting mechanism before tuning. Separate load, TTFT, prefill, decode, and end-to-end wall time. Label claims as measured, derived, or hypothesis.

## When to Use

- Decode or prefill is slower than expected.
- Lower precision did not improve speed.
- More accelerator offload made no difference or caused a regression.
- MTP/speculative decoding has low or negative speedup.
- Long context, concurrency, memory pressure, or thermal behavior may be involved.

Do not use a cross-model comparison to diagnose a same-model regression.

## Procedure

1. **Snapshot prerequisites.** Capture accelerator/CPU/memory, competing processes, clocks, driver, runtime commit/version, build flags, exact model revision/file, and full argv. Completion criterion: another operator could reconstruct the run.
2. **Phase the baseline.** Run one cold request, one warm-up, then at least three warm repeats. Record TTFT, prompt tok/s, decode tok/s, wall time, peak memory, context, output validity, and speculative counters. Completion criterion: raw per-run values are retained.
3. **Check residency.** Determine which tensors/KV/state live on accelerator versus host and whether paging/migration occurs. Completion criterion: startup logs and live process/memory evidence agree.
4. **Derive ceilings.** For dense batch-1 decode, calculate `sustained_bandwidth / bytes_touched_per_token`. For MoE, include dense attention/shared weights plus routed experts—not total weights blindly. Completion criterion: assumptions and units are shown.
5. **Run isolation sweeps.** Change exactly one dimension: context, placement, quant, speculation depth, runtime/build, then concurrency. Alternate A/B/A/B/A/B for candidates. Completion criterion: each result answers one causal question.
6. **Inspect utilization over time.** Distinguish compute saturation, memory traffic, host stalls, launch gaps, migration, and throttling. Completion criterion: the diagnosis explains both metrics and sweep behavior.
7. **Recommend only matching levers.** Preserve required context, quality, concurrency, and semantics. Label reductions as tradeoffs, not free optimizations.

## Interpretation

| Signature | Likely bottleneck | Next isolation test |
|---|---|---|
| Decode near bandwidth ceiling | weight/KV bandwidth | same-model quant A/B |
| GPU idle while CPU active | host work or synchronization | placement/offload sweep |
| GPU compute high, bandwidth low | compute/kernel | runtime/build A/B |
| Speed falls with active context | attention/KV/hybrid state | fixed-output context sweep |
| Regression near memory capacity | workspace/graph fallback | leave 10–20% headroom |
| High MTP acceptance, no wall win | draft/verification overhead | lower depth; target-only A/B |
| Run slows over time | thermal/power/migration | long telemetry trace |
| Aggregate rises but per-user collapses | scheduler/concurrency | 1/2/4/8 client sweep |

## Reporting Contract

Report exact hardware, versions, model revision/file, command, prompt/output/context, every sample and median, peak memory, output validity, speculative acceptance, diagnosis confidence, rejected hypotheses, and the next falsifying test.

## Common Pitfalls

1. GPU utilization alone does not identify the bottleneck.
2. Checkpoint size is not always bytes touched per token.
3. A faster quant name does not prove a faster kernel.
4. Acceptance rate is not speculative speedup.
5. Short synthetic prompts do not represent agent workloads.
6. A model loading successfully does not prove safe memory headroom.
7. Parallel slots may round-robin rather than batch attention.

## Verification Checklist

- [ ] Every phase measured separately
- [ ] Exact build/model/command recorded
- [ ] Baseline and candidate paired repeatedly
- [ ] Output quality/format validated
- [ ] Required capability held fixed
- [ ] Root cause tied to evidence
- [ ] Recommendation includes a rollback condition
