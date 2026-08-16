#!/usr/bin/env python3
"""
Ling-3.0-tiny 128K — finalize auto-research artifacts.

Reads results/best_*.json produced by sweep_ling3t.py, prints a ranked table,
and writes results/summary.md. Run AFTER the sweep completes.

Usage:
  python3 finalize.py
"""
import json
from pathlib import Path

DOMAIN = Path(__file__).resolve().parent
RESULTS = DOMAIN / "results"


def load():
    rows = []
    for p in sorted(RESULTS.glob("best_*.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    return rows


def main():
    rows = load()
    if not rows:
        print("No results/best_*.json found. Run sweep_ling3t.py first.")
        return
    rows.sort(key=lambda r: (-(r.get("primary_metric") or 0),
                             r.get("constraint_metric") or 999))

    print("\n=== RANKED (decode desc, vram asc) ===")
    print(f"{'#':>2} {'config':<22} {'dec':>7} {'vram':>6} {'pref':>7} {'ttft':>6} {'par':>4} {'kv':>5} {'ub':>5} {'b':>6}")
    for i, r in enumerate(rows, 1):
        c = r.get("configUsed", {})
        m = r.get("metrics", {})
        print(f"{i:>2} {r.get('config_name','?'):<22} "
              f"{r.get('primary_metric',0):>7.1f} {r.get('constraint_metric',0):>6.2f} "
              f"{m.get('tokSPrefill',0):>7.0f} {m.get('ttftMs',0):>6.1f} "
              f"{c.get('parallel','?'):>4} {c.get('kv_quant','?'):>5} "
              f"{c.get('ubatch','?'):>5} {c.get('batch','?'):>6}")

    # best-by-family notes
    fams = {}
    for r in rows:
        f = r["config_name"].split("_")[0]
        fams.setdefault(f, []).append(r)
    print("\n=== Family winners (best decode per family) ===")
    for f, rs in sorted(fams.items()):
        best = max(rs, key=lambda x: x.get("primary_metric") or 0)
        print(f"  {f}: {best['config_name']} decode={best['primary_metric']:.1f} "
              f"vram={best['constraint_metric']:.2f} prefill={best['metrics']['tokSPrefill']:.0f}")

    # write summary.md
    lines = ["# Ling-3.0-tiny 128K — Auto-Research Summary\n",
             f"Generated from {len(rows)} promoted configs.\n",
             "## Ranked\n",
             "| # | config | decode tok/s | peak VRAM GB | prefill tok/s | ttft ms | parallel | kv | ubatch | batch |",
             "|---|---------|------------:|-------------:|-------------:|--------:|---------:|----|-------:|------:|"]
    for i, r in enumerate(rows, 1):
        c = r.get("configUsed", {})
        m = r.get("metrics", {})
        lines.append(
            f"| {i} | {r.get('config_name')} | {r.get('primary_metric',0):.1f} | "
            f"{r.get('constraint_metric',0):.2f} | {m.get('tokSPrefill',0):.0f} | "
            f"{m.get('ttftMs',0):.1f} | {c.get('parallel')} | {c.get('kv_quant')} | "
            f"{c.get('ubatch')} | {c.get('batch')} |")
    lines.append("\n## Family winners\n")
    for f, rs in sorted(fams.items()):
        best = max(rs, key=lambda x: x.get("primary_metric") or 0)
        lines.append(f"- **{f}**: `{best['config_name']}` — decode {best['primary_metric']:.1f} tok/s, "
                     f"{best['constraint_metric']:.2f} GB VRAM, prefill {best['metrics']['tokSPrefill']:.0f} tok/s")
    (RESULTS / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"\nWrote {RESULTS / 'summary.md'}")


if __name__ == "__main__":
    main()
