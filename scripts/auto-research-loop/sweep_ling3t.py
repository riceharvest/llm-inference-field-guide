#!/usr/bin/env python3
"""
Ling-3.0-tiny 128K auto-research sweep driver (domain-specific).

Why not the generic sweep.py? On a bandwidth-bound 7.9B/1.3B-active MoE,
decode tok/s is a near-fixed ceiling (~155 tok/s on an RTX 3060). The real
optimization surface is the VRAM budget: how many parallel slots, what KV
precision, and what prefill batching fit inside 12 GB, and how those trade
against prefill throughput and TTFT. The generic driver stops on the first
goal-met config, which would halt exploration at run 1.

This driver instead runs a curated BEAM GRID (4 families x variants) and
NEVER prematurely stops -- it fills beam_log.md + results/best_*.json so the
Pareto frontier (decode vs VRAM) and parameter sensitivity are meaningful.

It only writes configs/harness runs; the live production service (ling-llama
on 12445) must already be stopped by the operator to free the 11.5 GB headroom.

Each candidate is written to configs/<name>.yaml, measured by harness.py, and
promoted/archived with a beam-log entry. The full result JSON is saved to
results/best_<name>.json (winner) or archive/.
"""

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import yaml

DOMAIN = Path(__file__).resolve().parent
HARNESS = DOMAIN / "harness.py"
RESULTS = DOMAIN / "results"
ARCHIVE = DOMAIN / "archive"
CONFIGS = DOMAIN / "configs"
LOGS = DOMAIN / "logs"


def load_base() -> Dict[str, Any]:
    return yaml.safe_load((CONFIGS / "baseline.yaml").read_text())


def derive(base: Dict[str, Any], name: str, **overrides) -> Dict[str, Any]:
    cfg = dict(base)
    for k, v in overrides.items():
        cfg[k] = v
    cfg["name"] = name
    return cfg


# ---------------------------------------------------------------------------
# Beam grid: 4 active families (beam-search discipline, AGENTS.md)
# ---------------------------------------------------------------------------
def build_grid(base: Dict[str, Any]) -> List[Dict[str, Any]]:
    grid = []

    # Family A: PARALLEL SCALING (decode stays ~155; measure VRAM + prefill + TTFT)
    for p in (1, 2, 3, 4, 6, 8):
        grid.append(derive(base, f"A_par{p}", parallel=p))

    # Family B: KV PRECISION DROP (free VRAM for more parallel / longer ctx)
    for kv in ("bf16", "q8_0", "q4_0"):
        grid.append(derive(base, f"B_kv{kv}", kv_quant=kv, parallel=4))

    # Family C: PREFILL BATCH BUDGET (ubatch/batch affect prefill + TTFT)
    for ub, b in ((256, 512), (512, 1024), (1024, 2048), (2048, 4096)):
        grid.append(derive(base, f"C_ub{ub}_b{b}", ubatch=ub, batch=b, parallel=4))

    # Family D: THREADING (CPU-side prompt prefill / contention)
    for t, tb in ((8, 8), (14, 16), (16, 16), (20, 20)):
        grid.append(derive(base, f"D_t{t}_tb{tb}", threads=t, threads_batch=tb, parallel=4))

    # Family E (structural): combine the best-looking knobs (filled after run,
    # but seed an aggressive-low-VRAM config to probe max parallel under q4_0).
    grid.append(derive(base, "E_kvq4_par8", kv_quant="q4_0", parallel=8))

    return grid


def run_one(cfg: Dict[str, Any]) -> Dict[str, Any]:
    name = cfg["name"]
    cfg_path = CONFIGS / f"{name}.yaml"
    yaml.safe_dump(cfg, open(cfg_path, "w"), sort_keys=False)
    out_path = LOGS / f"run_{name}.json"
    cmd = [sys.executable, str(HARNESS), "--config", str(cfg_path),
           "--output", str(out_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=900)
    for line in proc.stdout.splitlines():
        if line.startswith("Saved to "):
            out_file = line[9:].strip()
            break
    else:
        out_file = str(out_path)
    if Path(out_file).exists():
        return json.loads(Path(out_file).read_text())
    return {"success": False, "error": (proc.stderr or "no output")[-800:],
            "configUsed": cfg}


def log_entry(name: str, status: str, result: Dict[str, Any]) -> None:
    RESULTS.mkdir(exist_ok=True)
    line = (f"- `{name}` {status} | "
            f"decode={result.get('primary_metric', '?')} "
            f"vram={result.get('constraint_metric', '?')} "
            f"prefill={result.get('metrics', {}).get('tokSPrefill', '?')} "
            f"ttft={result.get('metrics', {}).get('ttftMs', '?')} "
            f"@ {datetime.utcnow().isoformat()}Z\n")
    with open(RESULTS / "beam_log.md", "a") as f:
        f.write(line)


def promote(result: Dict[str, Any]) -> None:
    cfg = result.get("configUsed", {})
    name = cfg.get("name", "unknown")
    json.dump(result, open(RESULTS / f"best_{name}.json", "w"), indent=2)
    yaml.safe_dump(cfg, open(CONFIGS / f"{name}.yaml", "w"), sort_keys=False)
    log_entry(name, "promoted", result)


def archive(result: Dict[str, Any], reason: str) -> None:
    cfg = result.get("configUsed", {})
    name = cfg.get("name", "unknown")
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    cf = CONFIGS / f"{name}.yaml"
    if cf.exists():
        shutil.move(str(cf), str(ARCHIVE / f"{name}_{ts}.yaml"))
    json.dump(result, open(ARCHIVE / f"{name}_{ts}.json", "w"), indent=2)
    log_entry(name, f"archived: {reason}", result)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-runs", type=int, default=999)
    ap.add_argument("--only", default=None, help="comma list of family prefixes A,B,C,D,E")
    args = ap.parse_args()

    base = load_base()
    grid = build_grid(base)
    if args.only:
        fams = {f.strip() for f in args.only.split(",")}
        grid = [c for c in grid if c["name"].split("_")[0] in fams]

    grid = grid[:args.max_runs]
    print(f"Grid: {len(grid)} candidates", flush=True)
    print(f"  VRAM headroom check: live service MUST be stopped. "
          f"Current free should be > 10GB.", flush=True)

    # init beam log
    (RESULTS / "beam_log.md").write_text(
        f"# Ling-3.0-tiny 128K auto-research beam log\n"
        f"Started {datetime.utcnow().isoformat()}Z | "
        f"primary=decode tok/s (geomean), constraint=peak VRAM GB\n\n")

    for i, cfg in enumerate(grid, 1):
        name = cfg["name"]
        print(f"\n=== [{i}/{len(grid)}] {name} ===", flush=True)
        res = run_one(cfg)
        if res.get("success"):
            print(f"  decode={res['primary_metric']} vram={res['constraint_metric']} "
                  f"prefill={res['metrics']['tokSPrefill']} ttft={res['metrics']['ttftMs']}",
                  flush=True)
            promote(res)
        else:
            print(f"  FAILED: {str(res.get('error',''))[:160]}", flush=True)
            archive(res, str(res.get("error", "unknown"))[:120])

    # final ranking + pareto
    best = sorted(RESULTS.glob("best_*.json"))
    rows = []
    for p in best:
        try:
            rows.append(json.loads(p.read_text()))
        except Exception:
            pass
    rows.sort(key=lambda r: (-(r.get("primary_metric") or 0),
                             r.get("constraint_metric") or 999))
    with open(RESULTS / "beam_log.md", "a") as f:
        f.write(f"\n## Final ranking ({len(rows)} promoted)\n\n")
        for rank, r in enumerate(rows, 1):
            n = r.get("config_name")
            f.write(f"{rank}. `{n}` decode={r.get('primary_metric')} "
                    f"vram={r.get('constraint_metric')} "
                    f"prefill={r.get('metrics',{}).get('tokSPrefill')}\n")

    # pareto frontier (decode vs vram)
    pareto = []
    for r in rows:
        d = r.get("primary_metric") or 0
        v = r.get("constraint_metric") or 999
        dominated = any((o.get("primary_metric") or 0) >= d and
                        (o.get("constraint_metric") or 999) <= v and
                        ((o.get("primary_metric") or 0) > d or
                         (o.get("constraint_metric") or 999) < v)
                        for o in rows if o is not r)
        if not dominated:
            pareto.append({"name": r.get("config_name"),
                           "primary_metric": d, "constraint_metric": v,
                           "prefill": r.get("metrics", {}).get("tokSPrefill"),
                           "ttft": r.get("metrics", {}).get("ttftMs")})
    json.dump(pareto, open(RESULTS / "pareto_frontier.json", "w"), indent=2)

    print(f"\nDone. {len(rows)} promoted, pareto={len(pareto)}. "
          f"Reports in {RESULTS}", flush=True)


if __name__ == "__main__":
    main()
