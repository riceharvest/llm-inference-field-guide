# llama.cpp command cookbook

Flag names change. Confirm all syntax with the exact binary's `--help`; do not copy speculative-decoding flags across forks blindly.

## Baseline server

```bash
llama-server -m MODEL.gguf -c 8192 -np 1 --host 127.0.0.1 --port 8080
```

Add GPU offload and Flash Attention only after confirming the local syntax.

## API timing probe

```bash
curl -s http://127.0.0.1:8080/v1/completions   -H 'content-type: application/json'   -d '{"prompt":"Explain why decode is memory-bound.","n_predict":256,"temperature":0}'   > result.json
python3 - <<'PY'
import json
x=json.load(open('result.json'))
print(json.dumps(x.get('timings', {}), indent=2))
PY
```

## A/B rules

- fixed prompt bytes and output cap;
- temperature 0 or fixed seed/sampling;
- one warm-up;
- A/B/A/B/A/B;
- preserve raw JSON and startup logs;
- validate response content;
- report median plus every run.

## Placement sweep

Use three points: conservative, midpoint, maximum. Keep context, KV, batch, and speculation fixed. If maximum is slower, inspect memory headroom and graph splits before concluding placement is irrelevant.

## MTP sweep

Start target-only. Then test supported depths one at a time. Parse the runtime's timing counters for drafted and accepted tokens. Acceptance without lower wall time is not a win.

## Threading

More CPU threads can hurt due to contention and synchronization. Sweep physical-core counts around the runtime's default; keep affinity/NUMA stable and report it.

## Tensor-level placement

Some forks support regex tensor overrides. This can outperform whole-layer offload for hybrid CPU/GPU runs, but tensor names and semantics are fork-specific. Dump tensor metadata, estimate bytes by tensor class, and verify placement from startup logs.
