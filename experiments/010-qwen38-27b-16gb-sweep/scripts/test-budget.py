#!/usr/bin/env python3
"""SELF-TEST: verify budget.py arithmetic and the sweep matrix is internally
consistent. Run in CI / locally before trusting configs. No GPU required."""
import json
import subprocess
import sys

out = subprocess.check_output(["python3", "scripts/budget.py", "--json"], text=True)
d = json.loads(out)

# Per-token state cost (bytes) must match the hybrid formula:
#   att = 2 * n_kv_head(4) * head_dim(256) * att_layers(16) = 32768
#   gdn = gdn_layers(48) * n_embd(5120) = 245760
#   f16 -> *2, q8_0 -> *1, q4_0 -> *0.5
assert d["kv_bytes"] == {
    "f16": 557056, "q8_0": 278528, "q4_0": 139264,
}, d["kv_bytes"]

rows = d["matrix"]
assert len(rows) == 9, f"expected 9 matrix rows, got {len(rows)}"
ud = [r for r in rows if r["ud"]]
std = [r for r in rows if not r["ud"]]
assert len(ud) == 5, f"expected 5 UD rows, got {len(ud)}"
assert len(std) == 4, f"expected 4 std rows, got {len(std)}"

cells = d["feasible_cells"]
assert cells, "no feasible cells computed"
# Every feasible cell must fit its declared context.
for c in cells:
    row = next(r for r in rows if r["quant"] == c["file"].replace("Qwen3.8-27B-", "")
               and r["kv_type"] == c["kv"])
    assert row["max_ctx"] >= c["ctx"], f"{c} exceeds max_ctx {row['max_ctx']}"

print(f"budget self-test PASS: {len(rows)} matrix rows, "
      f"{len(cells)} feasible cells")
