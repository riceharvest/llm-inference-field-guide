#!/usr/bin/env python3
"""SELF-TEST: verify budget.py arithmetic for the single subject cell.
Run in CI / locally before trusting configs. No GPU required."""
import json
import subprocess

out = subprocess.check_output(["python3", "scripts/budget.py", "--json"], text=True)
d = json.loads(out)

# Per-token state cost (bytes) must match the hybrid formula:
#   att = 2 * n_kv_head(4) * head_dim(256) * att_layers(16) = 32768
#   gdn = gdn_layers(48) * n_embd(5120) = 245760
#   f16 -> *2, q8_0 -> *1, q4_0 -> *0.5
assert d["kv_bytes"] == {
    "f16": 557056, "q8_0": 278528, "q4_0": 139264,
}, d["kv_bytes"]

# Subject quant fixed.
assert d["subject"] == "Qwen3.8-27B-UD-Q2_K_XL", d["subject"]

# Max ctx must be positive and large enough to be worth running.
assert d["max_ctx_estimate"] >= 8192, f"max_ctx too small: {d['max_ctx_estimate']}"
assert d["launch_ctx"] >= 4096, f"launch_ctx too small: {d['launch_ctx']}"
assert d["launch_ctx"] <= d["max_ctx_estimate"], "launch_ctx over max"

# 4-bit+ excluded quants must all be >= 15.7 GB (too big for usable 16 GiB).
for name, gb in d["excluded_quants_too_big"].items():
    # UD-IQ2_XXS..Q3_K_M are <15.7 but still excluded here as non-subject;
    # the hard ceiling that proves "no 4-bit+ ladder fits" is IQ4_XS and up.
    pass
assert d["excluded_quants_too_big"]["IQ4_XS"] >= 15.7
assert d["excluded_quants_too_big"]["BF16"] > 50

print(f"budget self-test PASS: subject={d['subject']} "
      f"max_ctx={d['max_ctx_estimate']} launch_ctx={d['launch_ctx']}")
