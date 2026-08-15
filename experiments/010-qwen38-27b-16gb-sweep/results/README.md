# Placeholder for measured sweep data.

This directory will hold the real results once the Vast RTX 5060 Ti 16 GB run
completes. Naming:

- `results/<quant>-<kv>-<ctx>/perf.json`   — real-prompt decode/prefill (MODE=perf)
- `results/<quant>-<kv>-<ctx>/<run>.json`  — hermesbench hermes-core score (MODE=agent)
- `results/<quant>-<kv>-<ctx>/vram.csv`    — peak VRAM per cell
- `results/system-5060ti.txt`              — `../../tools/collect-system.sh` output
- `2026-08-<dd>-qwen38-16gb-sweep.json`    — consolidated dated matrix (hand-compiled from cells)

Expected cells (fit 16 GiB): UD-IQ2_XXS, UD-IQ2_M, UD-Q2_K_XL, UD-IQ3_XXS
(native KV; 16K/32K, plus 64K for the three smallest), and Q3_K_S, Q3_K_M
(q8_0 and q4_0 KV; 16K/32K). The q4-vs-q8 KV comparison lives on Q3_K_S/M.
