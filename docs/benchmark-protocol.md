# Reproducible benchmark protocol

## Required record

```yaml
hardware:
  accelerator: exact model and count
  memory: capacity, type, advertised and measured bandwidth if available
  cpu: model, cores, NUMA
software:
  os: version
  driver: version
  runtime: name, version/commit
  build_flags: exact
model:
  repository: owner/name
  revision: commit
  file: exact filename
  format: GGUF/safetensors/etc
  quantization: exact
workload:
  prompt_tokens: exact
  output_tokens: exact
  active_context_tokens: exact
  max_context: exact
  concurrency: exact
  sampling: exact
command: exact argv
results:
  warmups: count
  repeats: count
  prefill_tok_s: each run + median
  decode_tok_s: each run + median
  ttft_ms: each run + median
  wall_s: each run + median
  peak_memory: value
  mtp: drafted, accepted, n_max
```

## Procedure

1. Reboot or document competing processes; capture idle memory.
2. Start the server and retain the complete startup log.
3. Run one cold request and label it cold; do not mix with warm results.
4. Run at least one warm-up.
5. Alternate baseline and candidate: A/B/A/B/A/B.
6. Use medians and show every sample. Investigate outliers; do not silently delete them.
7. Verify output validity. A fast malformed or truncated response fails.
8. Repeat using a production-representative context and output length.

## Recommended workloads

- **Micro:** 512 prompt / 256 output—kernel sanity only.
- **Interactive:** 2K / 512.
- **Agent:** 20K / 2K, including realistic tool/schema text.
- **Long context:** production active context / fixed 512 output.
- **Concurrency:** 1, 2, 4, 8 clients; report per-user latency and aggregate tok/s.

## Acceptance rule

Adopt a reversible speed flag only after paired repeats show a stable wall-clock improvement with equal output quality and safe memory headroom. Treat <3% as noise unless it improves stability; 3–8% needs more repeats; >=8% is a candidate, not proof, until the realistic workload passes.
