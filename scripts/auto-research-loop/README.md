# Auto-Research Loop — Ling-3.0-tiny 128K (RTX 3060)

This folder applies the `auto-research-loop` skill methodology to the
Ling-3.0-tiny 128K server described in
[`../../experiments/001-ling3t-128k-rtx3060/`](../../experiments/001-ling3t-128k-rtx3060/).

## What it optimizes

Ling-3.0-tiny is a 7.9B-total / 1.3B-active `bailingmoe3` MoE. On an RTX 3060
12 GB, decode throughput is **bandwidth-bound** and lands at ~150–155 tok/s
regardless of most flags. The real optimization surface is the **VRAM budget**:

- how many `--parallel` slots fit under 12 GB,
- what KV precision (`bf16` / `q8_0` / `q4_0`) buys you headroom,
- how `-ub` / `-b` change prefill throughput and TTFT.

So the primary metric is **decode tok/s (geomean, temp=0)** and the constraint
is **peak VRAM GB**. A higher parallel count doesn't raise decode but raises
concurrency headroom and (with lower KV precision) shifts the Pareto point.

## Layout

```
scripts/auto-research-loop/
├── harness.py          # spawns a candidate llama-server on :12499, measures decode/prefill/peak VRAM
├── sweep_ling3t.py     # domain beam-grid driver (4 families A–E, never prematurely stops)
├── finalize.py         # prints ranked table + writes results/summary.md
├── configs/
│   └── baseline.yaml   # mirrors the live :12445 service, bound to :12499
├── templates/prompts.json
├── logs/               # raw harness output per candidate (gitignored)
├── archive/            # failed/retired configs + logs
└── results/            # best_*.json (winners), pareto_frontier.json, beam_log.md, summary.md
```

> The skill's generic `sweep.py` is NOT used here — it stops on the first
> goal-met config, which on a bandwidth-bound model would halt at run 1.
> `sweep_ling3t.py` runs the full curated grid so the Pareto frontier and
> parameter sensitivity are meaningful.

## Run it

```bash
# 1. Free the card: the live ling-llama service on :12445 holds ~7.7 GB.
systemctl --user stop ling-llama

# 2. Run the beam sweep (spawns ~18 candidate servers sequentially).
python3.11 sweep_ling3t.py

# 3. Rank + summarize.
python3.11 finalize.py

# 4. Graphs (needs matplotlib/numpy; repo venv is broken, use system 3.11).
#    plot_results.py ships with the auto-research-loop skill:
python3.11 ~/.hermes/skills/mlops/auto-research-loop/scripts/plot_results.py \
  --domain-dir scripts/auto-research-loop

# 5. Restore production service.
systemctl --user start ling-llama
```

## How measurement works

`harness.py` sends streaming `/v1/completions` with
`stream_options.include_usage=true`. Each final SSE chunk carries
`timings.predicted_per_second` (decode) and `timings.prompt_per_second`
(prefill) — measured server-side, so there is no client clock jitter. Peak
VRAM is sampled every 100 ms on a background thread during generation.

Port discipline: candidate servers always bind **:12499**, never :12445, so the
production service is only affected when you explicitly stop it.
