# LLM Inference Field Guide

> Real-world LLM inference experiments — problems, fixes, and lessons.
>
> Every entry is a real problem we hit, how we diagnosed it, and what actually fixed it. No theory. No marketing. Just measurement and honest tradeoffs.

## How to read this

Each experiment is a standalone case study. Read the one that matches your problem. If none match, use the [bottleneck decision tree](docs/bottleneck-decision-tree.md) to find the right lever.

---

## Experiments

| # | Experiment | Problem | Lesson |
|---|-----------|---------|--------|
| 001 | [Ling-3.0-tiny 128K on RTX 3060](experiments/001-ling3t-128k-rtx3060/) | GGUF embedded metadata capped context at 32K; then OOM on first long prompt | Embedded context_length caps CLI flags; KV cache × parallel slots is the real memory cost |
| 002 | [The Corrupt GGUF That Looked Like an Architecture Bug](experiments/002-qwen36-35b-a3b-llama-incompat/) | 23 GB GGUF produced garbage output | Verify SHA256 before debugging behavior — a corrupt download mimics any bug |
| 003 | [Reasoning Came in Big Blocks](experiments/003-sse-proxy-buffering/) | Friend's V100 box: streaming felt laggy, GPU was fine | The proxy's 64 KiB read buffer hid real-time generation behind a fill-then-release pattern |
| 004 | [The Rolling Update Loop](experiments/004-vast-serverless-update-loop/) | Vast endpoint stuck "updating" for hours | Config drift on serverless platforms causes infinite recreate loops — delete and recreate, don't patch |
| 005 | [The 5060 Ti Prefill Reality Check](experiments/005-5060ti-prefill-reality/) | 3–5× slower than community estimates | Optimistic estimates assume optimizations you're not running; measure your own stack |
| 006 | [sglang Production Gotchas](experiments/006-sglang-production-gotchas/) | Default sglang flags hurt production | Default flags are not production flags — here's what each one does |

---

## The bottleneck decision tree

If you don't see your problem above, start here:

1. Read [the bottleneck decision tree](docs/bottleneck-decision-tree.md).
2. Capture a reproducible baseline using [the benchmark protocol](docs/benchmark-protocol.md).
3. Apply only the section matching the measured bottleneck.
4. Change one variable at a time; use paired repeats.

## Evidence labels

- **Measured:** command, hardware, build, workload, repeats, and raw output are available.
- **Derived:** arithmetic from disclosed specs; assumptions shown.
- **Hypothesis:** mechanism is plausible but not yet isolated by an A/B test.

## Contributing

Found a problem that's not here? Open an issue with your hardware, the symptom, and what you've tried. If we hit it too, we'll add it as an experiment.

MIT licensed.
