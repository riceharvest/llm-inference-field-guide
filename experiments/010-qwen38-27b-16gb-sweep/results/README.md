# Measured results for experiment 010 (pending the Vast RTX 5060 Ti 16 GB run).

Subject cell — the quant dario specified:

- `Qwen3.8-27B-UD-Q2_K_XL` at its max feasible context (single cell; UD embeds
  its own KV/state format, so no q4/q8 switch).

Outputs the sweep writes:

- `results/Qwen3.8-27B-UD-Q2_K_XL-native-<ctx>/perf.json`  — real-prompt TTFT, prefill tok/s, decode tok/s (MODE=perf, 3 repeats)
- `results/Qwen3.8-27B-UD-Q2_K_XL-native-<ctx>/vram.csv`  — peak VRAM during the run (recorded, not estimated)
- `results/Qwen3.8-27B-UD-Q2_K_XL-native-<ctx>/<run>.json` — hermesbench hermes-core score (MODE=agent)
- `results/system-5060ti.txt` — `../../tools/collect-system.sh` output
- `results/2026-08-<dd>-qwen38-udq2xl.json` — consolidated dated result (hand-compiled from the single cell)

The q4-vs-q8 KV question is intentionally out of scope for this experiment:
UD-Q2_K_XL fixes its KV format and rejects `-ctk/-ctv`. That comparison belongs
on a standard K-quant (which does not fit 16 GiB with usable context anyway).
