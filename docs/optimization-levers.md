# Optimization levers by bottleneck

## Weight-bandwidth bound

- Reduce bytes read per token with a quant supported efficiently by the hardware/runtime.
- Keep weights on the fastest memory tier.
- Avoid assuming lower bits are faster: dequantization and kernel quality can reverse the result.
- Dense batch-1 decode is the classic case. Approximate ceiling: sustained bandwidth / resident weight bytes.

## Compute or kernel bound

- Verify the build targets the actual GPU architecture and enables the intended backend.
- Compare a current stable build with a known-good commit; new model architectures often land before optimized kernels.
- Test native formats (NVFP4/FP8/INT4) only on hardware and kernels that execute them natively.
- Profile recurrent SSM/DeltaNet, MoE routing, norm, sampling, and output projection separately when possible.

## Host memory / interconnect bound

- Full accelerator residency is usually the largest win.
- If impossible, prioritize attention, embeddings/output, and other serial/critical tensors on device; measure rather than assuming layer-level placement is optimal.
- Pin CPU threads, respect NUMA locality, and avoid oversubscription.
- Eliminate per-token host/device copies.

## Long-context attention / KV bound

- Use Flash Attention when supported and correct.
- Quantize KV only after a quality check.
- Reduce active context through application-level compaction or sliding-window mechanisms when semantics permit.
- Distinguish allocated maximum context from tokens actually attended; runtimes differ.
- Keep concurrency/slots explicit because each live sequence consumes KV and workspace.

## Launch/shape inefficiency

- Batch-1 decode creates narrow operations. MTP, speculative decoding, and runtime-specific fused kernels can increase useful work per target pass.
- Larger prompt batches improve prefill utilization but do not necessarily improve single-user decode.
- Parallel server slots may round-robin rather than batch attention; measure aggregate and per-request throughput.

## Memory pressure

- Leave workspace and speculative-decoder headroom; “loads successfully” is not a stable operating point.
- Back off from the OOM boundary and repeat the benchmark.
- Reduce context, KV precision, parallel slots, or offloaded weights—one at a time.
- Watch for graph splits/fallbacks and non-monotonic speed near capacity.

## MTP/speculative decoding

- Benchmark target-only first.
- Record draft count, accepted count, acceptance distribution, and wall-clock—not acceptance alone.
- Sweep `n_max` and sampling settings under the production workload.
- Cold graph capture and warm requests must be separated.
- Check that target and draft state management supports hybrid attention/SSM architectures.

## MoE-specific

Total parameters determine storage traffic; active parameters approximate compute, but attention and shared experts remain dense. Diagnose separately:

- routed expert placement;
- router/kernel efficiency;
- CPU expert bandwidth;
- device synchronization;
- active-expert imbalance;
- context-dependent attention.

## What not to do

- Change quant, context, offload, batch, and runtime together.
- Do not use a 32-token prompt to represent an agent with 20K context.
- report the best single run;
- treat lower context or quality as a free speedup;
- infer a bottleneck from GPU utilization alone;
- compare different models before establishing a same-model baseline.
