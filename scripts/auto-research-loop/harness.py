#!/usr/bin/env python3
"""
Auto-Research Loop — Ling-3.0-tiny (llama.cpp) DOMAIN HARNESS
=============================================================

Measures decode tok/s (primary) and prefill tok/s on a LOCAL llama.cpp
server for a candidate config, without disturbing the live production
service running on port 12445.

How it works
------------
1. Spawns a fresh `llama-server` on a dedicated port (default 12499) using
   the candidate config's flags. The production `ling-llama` service on
   12445 is never touched by default.
2. Waits for /health, then runs `warmup_runs` throwaway generations.
3. Benchmarks `len(prompts)` fixed prompts, streaming with
   stream_options.include_usage so each final SSE chunk carries
   `timings.predicted_per_second` (decode) and `timings.prompt_per_second`
   (prefill) -- measured server-side, no client clock juggling.
4. Samples peak GPU VRAM (nvidia-smi) on a background thread during the run.
5. Tears the candidate server down cleanly and returns the STANDARDIZED
   result dict (see skill Result Schema).

Run:
  python3 harness.py --config configs/baseline.yaml --output logs/run.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml

# --------------------------------------------------------------------------
# Fixed prompts (LLM-inference-field-guide / MiaAI-Lab style variety)
# --------------------------------------------------------------------------
DEFAULT_PROMPTS = [
    ("Write a concise technical summary of how Flash Attention reduces the "
     "quadratic memory cost of self-attention in transformer decoders."),
    ("Explain the trade-off between KV-cache precision (fp16 vs q8_0) and "
     "maximum context length on a 12 GB consumer GPU."),
    ("List five failure modes when serving a Mixture-of-Experts model with "
     "speculative decoding enabled."),
    ("Describe the memory budget of a 7.9B-parameter MoE model loaded into "
     "VRAM at Q4_K_M with a 128K context window."),
    ("Compare mmap+mlock against full GPU offload for model weight loading, "
     "focusing on cold-start latency and page-cache pressure."),
    ("A server reports 131 tok/s decode but 40 ms TTFT at 8K prompt. Diagnose "
     "the likely bottleneck and propose two concrete fixes."),
]


# --------------------------------------------------------------------------
# System metadata helpers (Linux)
# --------------------------------------------------------------------------
def _nvidia(args: List[str]) -> Optional[str]:
    try:
        return subprocess.check_output(
            ["nvidia-smi"] + args, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return None


def gpu_name() -> str:
    return (_nvidia(["--query-gpu=name", "--format=csv,noheader"]) or "Unknown").split("\n")[0]


def gpu_vram_gb() -> int:
    try:
        return int(int((_nvidia(["--query-gpu=memory.total",
                                "--format=csv,noheader,nounits"]) or "0").split("\n")[0]) / 1024)
    except Exception:
        return 0


def gpu_power_watts() -> List[float]:
    out = _nvidia(["--query-gpu=power.draw", "--format=csv,noheader,nounits"])
    if not out:
        return []
    return [float(x.strip()) for x in out.split("\n") if x.strip()]


def system_metadata() -> Dict[str, Any]:
    cpu = "Unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu = line.split(":")[1].strip()
                    break
    except Exception:
        pass
    ram = 0
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    ram = int(line.split()[1]) // (1024 * 1024)
                    break
    except Exception:
        pass
    os_name = "Linux"
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if line.startswith("PRETTY_NAME="):
                    os_name = line.split("=", 1)[1].strip().strip('"')
                    break
    except Exception:
        pass
    return {
        "hwClass": "DISCRETE_GPU",
        "gpuName": gpu_name(),
        "gpuCount": 1,
        "vramGb": gpu_vram_gb(),
        "cpu": cpu,
        "ramGb": ram,
        "os": os_name,
        "powerWatts": round(sum(gpu_power_watts()), 1),
    }


# --------------------------------------------------------------------------
# VRAM sampler thread
# --------------------------------------------------------------------------
class VramSampler(threading.Thread):
    def __init__(self, stop_event: threading.Event, interval: float = 0.10):
        super().__init__(daemon=True)
        self._stop_ev = stop_event
        self._interval = interval
        self.peak_mib = 0
        self.samples: List[int] = []

    def run(self) -> None:
        while not self._stop_ev.is_set():
            used = _nvidia(["--query-gpu=memory.used",
                           "--format=csv,noheader,nounits"])
            if used:
                try:
                    mib = int(used.split("\n")[0])
                    self.peak_mib = max(self.peak_mib, mib)
                    self.samples.append(mib)
                except ValueError:
                    pass
            time.sleep(self._interval)


# --------------------------------------------------------------------------
# Server lifecycle
# --------------------------------------------------------------------------
def build_args(cfg: Dict[str, Any]) -> List[str]:
    m = os.path.expanduser(cfg["model_path"])
    kv = cfg.get("kv_quant", "bf16")
    return [
        os.path.expanduser(cfg["server_binary"]),
        "-m", m,
        "--host", cfg.get("host", "127.0.0.1"),
        "--port", str(cfg.get("port", 12499)),
        "-c", str(cfg.get("context", 131072)),
        "-n", str(cfg.get("n", 16384)),
        "-b", str(cfg.get("batch", 1024)),
        "-ub", str(cfg.get("ubatch", 512)),
        "-ngl", str(cfg.get("ngl", 999)),
        "--device", cfg.get("device", "CUDA0"),
        "-fa", str(cfg.get("flash_attn", "auto")),
        "-ctk", kv, "-ctv", kv,
        "--load-mode", cfg.get("load_mode", "mmap+mlock"),
        "--parallel", str(cfg.get("parallel", 4)),
        "--poll", str(cfg.get("poll", 50)),
        "--poll-batch", str(cfg.get("poll_batch", 1)),
        "--prio", str(cfg.get("prio", 2)),
        "--prio-batch", str(cfg.get("prio_batch", 2)),
        "-t", str(cfg.get("threads", 14)),
        "-tb", str(cfg.get("threads_batch", 16)),
        "--temp", str(cfg.get("temp", 0.0)),
        "--top-p", str(cfg.get("top_p", 0.95)),
        "--top-k", str(cfg.get("top_k", 20)),
    ]


def wait_health(base: str, timeout: float, proc: Optional[subprocess.Popen] = None) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        # Fast-fail: if the server process already exited (e.g. cudaMalloc OOM),
        # don't burn the full timeout polling a dead process.
        if proc is not None and proc.poll() is not None:
            return False
        try:
            r = requests.get(base + "/health", timeout=2)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def stream_complete(base: str, model: str, prompt: str, max_tokens: int,
                    timeout: float) -> Optional[Dict[str, Any]]:
    """Send a streaming completion; return the final timings dict or None."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    try:
        resp = requests.post(base + "/v1/completions", json=payload,
                             stream=True, timeout=timeout)
    except Exception:
        return None
    if resp.status_code != 200:
        return None
    last = None
    for raw in resp.iter_lines(decode_unicode=True):
        if not raw or not raw.startswith("data:"):
            continue
        data = raw[len("data:"):].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except Exception:
            continue
        t = chunk.get("timings")
        if t:
            last = t
    return last


# --------------------------------------------------------------------------
# Main run
# --------------------------------------------------------------------------
def run_config(config: Dict[str, Any]) -> Dict[str, Any]:
    name = config.get("name", "unknown")
    base = f"http://{config.get('host', '127.0.0.1')}:{config.get('port', 12499)}"
    model = config.get("hf_id", "ling-3.0-tiny")
    max_tokens = int(config.get("max_tokens", 1024))
    warmup = int(config.get("warmup_runs", 2))
    startup_timeout = float(config.get("startup_timeout", 120))
    prompts_file = config.get("prompts_file")

    # load prompts
    prompts = DEFAULT_PROMPTS
    if prompts_file and Path(prompts_file).exists():
        try:
            data = json.loads(Path(prompts_file).read_text())
            if isinstance(data, list) and data:
                prompts = [(p.get("prompt", p) if isinstance(p, dict) else p)
                           for p in data]
        except Exception:
            pass

    args = build_args(config)
    proc = None
    started = time.time()
    try:
        log_path = Path("logs") / f"{name}.server.log"
        log_path.parent.mkdir(exist_ok=True)
        proc = subprocess.Popen(args, stdout=open(log_path, "w"),
                                stderr=subprocess.STDOUT)
        if not wait_health(base, startup_timeout, proc):
            return _fail(name, config, "server did not become healthy",
                         exit_code=proc.poll(), proc=proc)

        # warmup
        for i in range(warmup):
            stream_complete(base, model, prompts[i % len(prompts)][1]
                            if isinstance(prompts[i % len(prompts)], tuple)
                            else prompts[i % len(prompts)], max_tokens // 4 or 16,
                            startup_timeout)

        # benchmark
        stop = threading.Event()
        sampler = VramSampler(stop)
        sampler.start()
        per_prompt = []
        decode_rates, prefill_rates, prompt_tok, comp_tok = [], [], [], []
        for p in prompts:
            text = p[1] if isinstance(p, tuple) else p
            t0 = time.time()
            timings = stream_complete(base, model, text, max_tokens, 600)
            if not timings:
                stop.set()
                return _fail(name, config, "no timings returned (generation failed)",
                             exit_code=proc.poll(), proc=proc)
            dec = timings.get("predicted_per_second") or timings.get("completion_per_second")
            pre = timings.get("prompt_per_second")
            pn = timings.get("prompt_n") or timings.get("prompt_tokens")
            prn = timings.get("predicted_n") or timings.get("completion_tokens")
            if dec:
                decode_rates.append(dec)
            if pre:
                prefill_rates.append(pre)
            if pn:
                prompt_tok.append(pn)
            if prn:
                comp_tok.append(prn)
            per_prompt.append({
                "ttft_ms": round((timings.get("prompt_ms") or 0), 2),
                "tok_s_out": round(dec, 2) if dec else None,
                "tok_s_prefill": round(pre, 2) if pre else None,
                "prompt_tokens": int(pn) if pn else None,
                "completion_tokens": int(prn) if prn else None,
            })
        stop.set()
        sampler.join(timeout=2)

        # aggregate
        dec_geomean = _geomean(decode_rates)
        pre_geomean = _geomean(prefill_rates)
        peak_vram_gb = round(sampler.peak_mib / 1024, 2)
        model_vram_gb = None  # unknown without idle baseline subtraction
        kv_per_slot_gb = None
        power = gpu_power_watts()

        result = {
            "config_name": name,
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "success": True,
            "primary_metric": round(dec_geomean, 3),
            "constraint_metric": peak_vram_gb,
            "metrics": {
                "tokSOut": round(dec_geomean, 3),
                "tokSPrefill": round(pre_geomean, 3),
                "ttftMs": round(sum(p["ttft_ms"] for p in per_prompt) / len(per_prompt), 2),
                "peakVramGb": peak_vram_gb,
                "modelVramGb": model_vram_gb,
                "kvVramPerSlotGb": kv_per_slot_gb,
                "gpuPowerWatts": [round(x, 1) for x in power],
                "totalPowerWatts": round(sum(power), 1),
                "decodeMin": round(min(decode_rates), 2) if decode_rates else None,
                "decodeMax": round(max(decode_rates), 2) if decode_rates else None,
            },
            "hardware": system_metadata(),
            "engine": {
                "engineName": "llama.cpp",
                "engineVersion": _binary_version(config.get("server_binary", "")),
                "backend": "cuda",
                "engineFlags": {
                    "commandSnippet": " ".join(args),
                    "tensorParallel": 1,
                    "kvCacheDtype": config.get("kv_quant", "bf16"),
                    "flashAttn": str(config.get("flash_attn", "auto")),
                    "concurrency": int(config.get("parallel", 4)),
                    "temperature": 0.0,
                },
            },
            "model": {
                "hfId": config.get("hf_id", "ling-3.0-tiny"),
                "modelRevision": config.get("model_revision", "main"),
                "quantization": config.get("quantization", "Q4_K_M"),
            },
            "benchmark": {
                "promptTokens": int(sum(prompt_tok)) if prompt_tok else 0,
                "outputTokens": int(sum(comp_tok)) if comp_tok else 0,
                "contextLength": int(config.get("context", 131072)),
                "batchSize": int(config.get("parallel", 4)),
                "prefillTokens": 0,
            },
            "perPrompt": per_prompt,
            "configUsed": dict(config),
            "notes": (f"Auto-research run: {name}. Dec geomean over {len(decode_rates)} "
                      f"prompts, temp=0, stream timings. Peak VRAM {peak_vram_gb} GB."),
        }
        return result
    except subprocess.TimeoutExpired:
        return _fail(name, config, "generation timed out", proc=proc)
    except Exception as e:  # noqa
        import traceback
        return _fail(name, config, f"{type(e).__name__}: {e}",
                     traceback=traceback.format_exc(), proc=proc)
    finally:
        if proc is not None:
            proc.terminate()
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
            # be sure the port is free
            try:
                proc.wait(timeout=5)
            except Exception:
                pass


def _fail(name, config, reason, exit_code=None, proc=None, traceback=None):
    # attempt to surface OOM from log
    detail = reason
    if proc is not None and proc.poll() is not None and proc.poll() != 0:
        log_path = Path("logs") / f"{name}.server.log"
        if log_path.exists():
            tail = log_path.read_text()[-1500:]
            if any(k in tail.lower() for k in ("out of memory", "oom", "vram", "cuda error", "failed to allocate")):
                detail = "likely OOM/crash: " + reason
    return {
        "config_name": name,
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "success": False,
        "error": detail,
        "exit_code": exit_code,
        "traceback": traceback,
        "configUsed": dict(config),
    }


def _geomean(xs: List[float]) -> float:
    xs = [x for x in xs if x and x > 0]
    if not xs:
        return 0.0
    logsum = sum(__import__("math").log(x) for x in xs)
    return __import__("math").exp(logsum / len(xs))


def _binary_version(binary: str) -> str:
    try:
        out = subprocess.check_output([binary, "--version"],
                                      stderr=subprocess.STDOUT, timeout=15).decode()
        for line in out.splitlines():
            if "version" in line.lower() or "bailingmoe" in line.lower():
                return line.strip()[:80]
        return out.strip().splitlines()[0][:80] if out else "unknown"
    except Exception:
        return "unknown"


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Ling-3.0-tiny auto-research harness")
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", help="Output JSON (default logs/<ts>_<name>.json)")
    ap.add_argument("--extra", action="append", default=[],
                    help="Extra key=value overrides merged into config")
    args = ap.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    for kv in args.extra:
        k, _, v = kv.partition("=")
        config[k] = v

    result = run_config(config)

    if not args.output:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        args.output = str(log_dir / f"{ts}_{config.get('name', 'run')}.json")

    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
