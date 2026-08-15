#!/usr/bin/env python3
"""Memory-budget + cell calculator for Qwen3.8-27B-UD-Q2_K_XL on a 16 GiB GPU.

Qwen3.8-27B is a HYBRID model: 48 Gated DeltaNet layers + 16 Gated Attention
layers. Only the 16 attention layers need a conventional KV cache; the 48 GDN
layers carry a recurrent state that llama.cpp stores in the same allocation
and that -ctk/-ctv quantize. The GDN state dominates the per-token cost, which
is exactly why 262K context is even thinkable at 2-bit on 16 GiB.

SUBJECT OF THIS EXPERIMENT: unsloth/Qwen3.8-27B-UD-Q2_K_XL (the quant dario
specified). It is a single cell: one quant, its own native KV/state format, at
the max context that fits 16 GiB. There is no quant ladder — the other quants
are not the subject and, at 4-bit+, do not fit a 16 GiB card with usable
context (Q4+ is >=15.7 GB and leaves <0.5 GiB for state).

KEY FACT about UD quants:
  - Unsloth UD quants embed their OWN KV/state format and REJECT -ctk/-ctv.
    So UD-Q2_K_XL has ONE fixed state cost; it cannot be switched q8<->q4.
    The launcher omits -ctk/-ctv for it (see configs/llama-server-sweep.sh).

Usage:
    python3 budget.py                 # human feasibility + max context
    python3 budget.py --json          # full data (CI)
    python3 budget.py --plan          # TSV: "<file stem> <kv> <ctx>" (one cell)
    python3 budget.py --overhead 1.0e9   # tune CUDA/workspace overhead assumption
"""
import argparse
import json

VRAM_BYTES = 16 * 1_000_000_000
OVERHEAD_BYTES = 1.5 * 1_000_000_000   # CUDA ctx + cuBLAS + graph + buffers (tunable)

N_LAYERS = 64
ATT_LAYERS = 16
GDN_LAYERS = 48
N_EMBD = 5120
N_KV_HEAD = 4
HEAD_DIM = 256

# The subject quant. Size verified against the HF Xet bridge: 10,676,423,744 B.
SUBJECT = {
    "stem": "Qwen3.8-27B-UD-Q2_K_XL",
    "bytes": 10_676_423_744,
    "ud": True,
}

# For reference only (why there is no ladder): 4-bit+ does not fit 16 GiB with
# usable context. Q4_K_M / Q5 / Q8 / BF16 are >=15.7 GB.
EXCLUDED = {
    "UD-IQ2_XXS": 9.01e9, "UD-IQ2_M": 10.3e9, "UD-Q2_K_XL": 10.7e9,
    "UD-IQ3_XXS": 11.9e9, "Q3_K_S": 12.6e9, "UD-Q3_K_XL": 13.4e9,
    "Q3_K_M": 13.8e9, "IQ4_XS": 15.7e9, "Q4_K_M": 16.9e9,
    "Q8_0": 29e9, "BF16": 54.7e9,
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


def subject_cell():
    """UD-Q2_K_XL at its own max context (rounded down to 1K, capped 32K)."""
    # UD embeds its native state; estimate its per-token cost at q8_0.
    mc = max_ctx(SUBJECT["bytes"], "q8_0")
    ctx = min(32768, (mc // 1024) * 1024)
    return SUBJECT["stem"], "native", ctx, mc


def main():
    global OVERHEAD_BYTES
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON")
    ap.add_argument("--plan", action="store_true", help="emit TSV cell")
    ap.add_argument("--overhead", type=float, default=OVERHEAD_BYTES,
                    help="CUDA/workspace overhead in bytes (default 1.5e9)")
    args = ap.parse_args()

    OVERHEAD_BYTES = args.overhead
    stem, kv, ctx, mc = subject_cell()

    if args.plan:
        print(f"{stem}\t{kv}\t{ctx}")
        return

    if args.json:
        print(json.dumps({
            "vram_bytes": VRAM_BYTES,
            "overhead_bytes": OVERHEAD_BYTES,
            "subject": SUBJECT["stem"],
            "kv_bytes": {k: kv_bytes_per_token(k) for k in ("f16", "q8_0", "q4_0")},
            "max_ctx_estimate": mc,
            "launch_ctx": ctx,
            "excluded_quants_too_big": {k: round(v / 1e9, 2) for k, v in EXCLUDED.items()},
        }))
        return

    print(f"VRAM budget : {VRAM_BYTES/1e9:.1f} GB  |  overhead {OVERHEAD_BYTES/1e9:.2f} GB")
    print("Hybrid state per token (KiB): "
          f"f16={kv_bytes_per_token('f16')/1024:.0f}  "
          f"q8_0={kv_bytes_per_token('q8_0')/1024:.0f}  "
          f"q4_0={kv_bytes_per_token('q4_0')/1024:.0f}")
    print("  (GDN recurrent state dominates: 48 layers x 5120 embd. The hybrid")
    print("   design is what makes large context thinkable on 16 GiB.)\n")
    w = SUBJECT["bytes"]
    print(f"Subject quant : {SUBJECT['stem']}")
    print(f"  weights     : {w/1e9:.2f} GB  (verified vs HF Xet bridge)")
    print(f"  free for KV : {(VRAM_BYTES - w - OVERHEAD_BYTES)/1e9:.2f} GB")
    print(f"  est max ctx : {mc} tokens (q8_0-state estimate)")
    print(f"  launch ctx  : {ctx} tokens")
    print("\nWhy no quant ladder: 4-bit+ quants do not fit 16 GiB with usable")
    print("context. Sizes (GB): " + ", ".join(
        f"{k}={v/1e9:.1f}" for k, v in sorted(EXCLUDED.items(), key=lambda x: x[1])))


if __name__ == "__main__":
    main()
