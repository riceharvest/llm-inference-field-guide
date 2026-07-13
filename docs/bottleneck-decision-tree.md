# Bottleneck decision tree

## 0. Separate the phases

Never report one blended “speed.” Record:

- model load time;
- TTFT;
- prompt/prefill tok/s;
- decode tok/s;
- end-to-end wall time;
- speculative drafted/accepted tokens;
- peak memory and context length.

## 1. Establish a clean baseline

Use one request, fixed prompt/output token counts, fixed sampling, fixed context, no other GPU jobs, and at least one discarded warm-up. Record the exact binary commit and command.

## 2. Is the model fully resident on the accelerator?

- **No:** host memory bandwidth, CPU kernels, and synchronization can dominate. Measure CPU utilization and device/host placement. Move critical tensors or the whole model to the accelerator if memory allows.
- **Yes:** continue.

## 3. Does decode approximate the bandwidth ceiling?

For dense batch-1 decode:

```text
ideal_tok_s = sustained_memory_bandwidth_GB_s / bytes_read_per_token_GB
bandwidth_efficiency = measured_tok_s / ideal_tok_s
```

Use **actual bytes touched per token**, not merely the checkpoint filename or parameter count. Dense models generally touch nearly all weights each token; MoE models touch attention/shared weights plus routed experts.

- High efficiency and GPU memory traffic near saturation → bandwidth-bound.
- Low efficiency → inspect compute, kernels, synchronization, context, and launch shape.

## 4. Sweep context length

Hold output length fixed and compare 2K, 8K, 32K, and the production context.

- Decode degrades strongly with context → attention/KV work is material.
- Prefill changes but decode is flat → prompt batching/kernel behavior.
- Both are flat and slow → weights, compute, or placement.

Do not call lowering context an “optimization” when maximum context is a requirement; label it a capability tradeoff.

## 5. Sweep placement

For llama.cpp, compare conservative, midpoint, and maximum GPU offload while keeping all other flags fixed.

- More offload improves speed → host bandwidth/interconnect bottleneck.
- Flat speed across placement → accelerator kernel/compute or another serial section.
- Speed regresses near full memory → memory pressure, graph splits, or reduced workspace.

MoE: keep attention placement fixed and sweep expert placement independently where the runtime permits.

## 6. Disable speculation, then re-enable it

Measure target-only first. Then record:

```text
acceptance = accepted_draft_tokens / drafted_tokens
speedup = target_only_wall_time / speculative_wall_time
```

High acceptance does not guarantee a win: draft cost, verification shape, context copying, and memory pressure can erase it. Sweep draft depth; the maximum is rarely automatically optimal.

## 7. Inspect utilization over time

Averages hide alternating CPU/GPU stalls. Sample at 100–500 ms when possible:

- GPU compute high + bandwidth low → compute/kernel bound.
- Bandwidth high + compute moderate → weight/KV bandwidth bound.
- GPU repeatedly idle while CPU active → host/synchronization bound.
- Both low → launch overhead, scheduler wait, unsupported path, throttling, or I/O.

## 8. Check clocks and thermals

Sustained decode must be long enough to reveal power or thermal throttling. Compare beginning/end tok/s and log clocks, power, and temperature.

## Minimal isolation matrix

| Test | Fixed | Changed | Answers |
|---|---|---|---|
| Context sweep | model/quant/output | context | attention/KV cost? |
| Placement sweep | context/build | offload | CPU/PCIe bottleneck? |
| Quant sweep | architecture/context | quant | bandwidth vs kernel quality? |
| MTP sweep | target/context | n_max | acceptance/cost optimum? |
| Runtime sweep | model/workload | runtime/build | kernel implementation? |
| Concurrency sweep | total work | parallel users | latency vs aggregate throughput? |
