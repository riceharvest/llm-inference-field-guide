# LLM Inference Field Guide

> Real problems, exact fixes, measured results for local LLM inference.

This is a collection of standalone experiments. Each entry is one model + one hardware stack + one problem we actually hit. Read the experiment that matches your setup, copy the flags, and run it.

## Experiments

| # | Experiment | Problem | Lesson |
|---|-----------|---------|--------|
| 001 | [Ling-3.0-tiny 128K on RTX 3060](experiments/001-ling3t-128k-rtx3060/) | GGUF metadata caps context at 32K; OOM on long prompts | Embedded `context_length` overrides CLI flags; `--parallel` pre-allocates KV for all slots |
| 002 | [The Corrupt GGUF That Looked Like an Architecture Bug](experiments/002-qwen36-35b-a3b-llama-incompat/) | 23 GB GGUF produced garbage output | Verify SHA256 before debugging behavior — a corrupt download mimics any bug |
| 003 | [Streaming Lag Was the Proxy Buffer](experiments/003-sse-proxy-buffering/) | External inference felt slow, GPU was healthy | Proxy read buffer hides real-time generation behind fill-then-release |
| 004 | [Serverless Worker Drift](experiments/004-vast-serverless-update-loop/) | Endpoint stuck updating after template changes | Serverless config updates are destructive, not rolling — delete and recreate |
| 005 | [The 5060 Ti Prefill Reality Check](experiments/005-5060ti-prefill-reality/) | 3–5× slower than community estimates | Optimistic estimates assume optimizations you're not running; measure your own stack |
| 006 | [sglang Production Gotchas](experiments/006-sglang-production-gotchas/) | Default sglang flags hurt production | Default flags are not production flags — here's what each one does |
| 007 | [Model Works Standalone, Fails Through the Agent](experiments/007-hermes-toolcall-failure/) | Clean API responses, empty agent responses | Reasoning-only output plus KV config interaction breaks tool-calling flows |

## How to use this

1. Find the experiment matching your model + hardware.
2. Use the Quick Start commands.
3. Run the smoke test for that experiment.
4. Use the dated results file as your benchmark target.
5. If it breaks, check Troubleshooting first.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

MIT licensed.
