#!/usr/bin/env python3
"""Memory-budget + sweep-plan calculator for Qwen3.8-27B on a 16 GiB GPU.

Qwen3.8-27B is a HYBRID model: 48 Gated DeltaNet layers + 16 Gated Attention
layers. Only the 16 attention layers need a conventional KV cache; the 48 GDN
layers carry a recurrent state that llama.cpp stores in the same allocation
and that -ctk/-ctv quantize. The GDN state dominates the per-token cost, which
is exactly why 262K context is even thinkable at 2-bit on 16 GiB.

KEY FACT (drives the experiment design):
  - Unsloth UD quants embed their OWN KV/state format and REJECT -ctk/-ctv.
    So a UD quant has ONE fixed state cost; it cannot be switched q8<->q4.
  - Standard K-quants (Q3_K_*) DO honor -ctk/-ctv, so q8_0 vs q4_0 is a real,
    decidable axis for them.
  - Because of (a)+(b), on 16 GiB you canNOT hold context length equal while
    comparing q4 vs q8 KV: the smaller-KV mode simply buys more context. The
    honest measurement is agent quality at each mode's MAX context, plus a note
    on where the dtype is even decidable.

Usage:
    python3 budget.py                 # human feasibility matrix
    python3 budget.py --json          # full matrix + feasible cells (CI)
    python3 budget.py --plan          # TSV: "<file stem> <kv> <ctx>" per feasible cell
    python3 budget.py --overhead 1.0e9   # tune CUDA/workspace overhead assumption
"""
import argparse
import json
import sys

VRAM_BYTES = 16 * 1_000_000_000
OVERHEAD_BYTES = 1.5 * 1_000_000_000   # CUDA ctx + cuBLAS + graph + buffers (tunable)

N_LAYERS = 64
ATT_LAYERS = 16
GDN_LAYERS = 48
N_EMBD = 5120
N_KV_HEAD = 4
HEAD_DIM = 256

# Candidate quants that fit 16 GiB (sizes from unsloth/Qwen3.8-27B-GGUF card).
QUANTS = {
    "UD-IQ2_XXS": {"bytes": 9.01e9, "ud": True},
    "UD-IQ2_M":   {"bytes": 10.3e9, "ud": True},
    "UD-Q2_K_XL": {"bytes": 10.7e9, "ud": True},
    "UD-IQ3_XXS": {"bytes": 11.9e9, "ud": True},
    "Q3_K_S":     {"bytes": 12.6e9, "ud": False},
    "UD-Q3_K_XL": {"bytes": 13.4e9, "ud": True},
    "Q3_K_M":     {"bytes": 13.8e9, "ud": False},
    # Q4+ (>=15.7e9) leaves <0.5 GiB for KV -> excluded from the live sweep.
}


def kv_bytes_per_token(kv_type):
    att = 2 * N_KV_HEAD * HEAD_DIM * ATT_LAYERS
    gdn = GDN_LAYERS * N_EMBD
    scale = {"f16": 2.0, "q8_0": 1.0, "q4_0": 0.5}[kv_type]
    return int((att + gdn) * scale)


def max_ctx(weights, kv_type):
    free = VRAM_BYTES - weights - OVERHEAD_BYTES
    per = kv_bytes_per_token(kv_type)
    return max(0, free // per)


def build_matrix():
    rows = []
    for name, info in QUANTS.items():
        if info["ud"]:
            # UD: fixed native state. Estimate cost at q8_0 (V-cache is ~8-bit
            # class); the file chooses the exact dtype and ignores -ctk/-ctv.
            mc = max_ctx(info["bytes"], "q8_0")
            rows.append({
                "quant": name, "ud": True, "kv_type": "native",
                "weights_gb": round(info["bytes"] / 1e9, 2),
                "max_ctx": mc, "fits_16k": mc >= 16384,
                "fits_32k": mc >= 32768, "fits_64k": mc >= 65536,
            })
        else:
            for kv in ("q8_0", "q4_0"):
                mc = max_ctx(info["bytes"], kv)
                rows.append({
                    "quant": name, "ud": False, "kv_type": kv,
                    "weights_gb": round(info["bytes"] / 1e9, 2),
                    "max_ctx": mc, "fits_16k": mc >= 16384,
                    "fits_32k": mc >= 32768, "fits_64k": mc >= 65536,
                })
    return rows


def feasible_cells():
    """Every (quant, KV) mode at ITS OWN max feasible context (rounded down to
    1K, capped at 32K). Running each cell at its own max context means the
    q4-vs-q8 comparison lives on Q3_K_S (q4_0 ~12K vs q8_0 ~6K) — same quant,
    so the KV-dtype effect is isolated rather than confounded by context."""
    cells = []
    for name, info in QUANTS.items():
        kv_modes = ("native",) if info["ud"] else ("q8_0", "q4_0")
        for kv in kv_modes:
            mc = max_ctx(info["bytes"], "q8_0" if info["ud"] else kv)
            if mc < 4096:
                continue  # not worth running below 4K context
            ctx = min(32768, (mc // 1024) * 1024)
            cells.append((f"Qwen3.8-27B-{name}", kv, ctx))
    return cells


def main():
    global OVERHEAD_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--plan", action="store_true", help="emit TSV feasible cells")
    ap.add_argument("--overhead", type=float, default=OVERHEAD_BYTES,
                    help="CUDA/workspace overhead in bytes (default 1.5e9)")
    args = ap.parse_args()

    OVERHEAD_BYTES = args.overhead

    if args.plan:
        for stem, kv, ctx in feasible_cells():
            print(f"{stem}\t{kv}\t{ctx}")
        return

    if args.json:
        print(json.dumps({
            "vram_bytes": VRAM_BYTES,
            "overhead_bytes": OVERHEAD_BYTES,
            "kv_bytes": {k: kv_bytes_per_token(k) for k in ("f16", "q8_0", "q4_0")},
            "matrix": build_matrix(),
            "feasible_cells": [{"file": s, "kv": k, "ctx": c} for s, k, c in feasible_cells()],
        }))
        return

    print(f"VRAM budget : {VRAM_BYTES/1e9:.1f} GB  |  overhead {OVERHEAD_BYTES/1e9:.2f} GB")
    print("Hybrid state per token (KiB): "
          f"f16={kv_bytes_per_token('f16')/1024:.0f}  "
          f"q8_0={kv_bytes_per_token('q8_0')/1024:.0f}  "
          f"q4_0={kv_bytes_per_token('q4_0')/1024:.0f}")
    print("  (GDN recurrent state dominates: 48 layers x 5120 embd. The hybrid")
    print("   design is what makes 262K context even thinkable on 16 GiB.)\n")
    print(f"{'quant':<12}{'GB':>6}{'KVtype':>8}{'maxCtx':>9}{'16K':>5}{'32K':>5}{'64K':>5}")
    print("-" * 50)
    for r in build_matrix():
        print(f"{r['quant']:<12}{r['weights_gb']:>5.1f}{r['kv_type']:>8}"
              f"{r['max_ctx']:>9}{'Y' if r['fits_16k'] else '-':>5}"
              f"{'Y' if r['fits_32k'] else '-':>5}{'Y' if r['fits_64k'] else '-':>5}")
    print("\nFeasible cells (max_ctx >= target):")
    for stem, kv, ctx in feasible_cells():
        print(f"  {stem}  kv={kv:<6} ctx={ctx}")


if __name__ == "__main__":
    main()
