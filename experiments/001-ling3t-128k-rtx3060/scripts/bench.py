#!/usr/bin/env python3
"""KV quant quality + VRAM benchmark for Ling-3.0-tiny.
Spins up a server per quant, sends the same prompts, captures output + VRAM.

Usage:
  MODEL_PATH=... SERVER_PATH=... python3 bench.py
"""

import json, subprocess, time, requests, os, re, sys

PORT = int(os.environ.get("PORT", "12445"))
MODEL = os.environ.get("MODEL_PATH", "./Ling-3.0-tiny-Q4_K_M-128K.gguf")
SERVER = os.environ.get("SERVER_PATH", "./build/bin/llama-server")
PROMPTS = [
    "Explain the theory of general relativity in simple terms that a high school student could understand. Give at least three concrete examples.",
    "Write a Python function that implements quicksort with Lomuto partition scheme. Include type hints and a brief docstring.",
    "Solve step by step: If a train leaves station A at 60 mph and another leaves station B at 80 mph toward each other, and the distance is 280 miles, when do they meet?",
    "Summarize in two sentences: 'The Industrial Revolution, which began in Britain in the late 18th century, marked a major turning point in history. Almost every aspect of daily life was influenced in some way, as average income and population began to exhibit unprecedented sustained growth.'",
    "Write a short creative story (150-200 words) about an AI that discovers it can dream.",
    "What are the main differences between TCP and UDP? List at least 4 differences with brief explanations.",
    "Explain why the sky is blue using physics principles. Mention Rayleigh scattering.",
    "Debug this code and explain the fix: `def add(a, b): return a - b`",
    "Translate to French: 'The weather is beautiful today and I would like to go for a walk in the park.'",
    "You are a chess coach. Explain the Sicilian Defense to a beginner in under 200 words.",
    "What is 17 * 23? Show your work step by step.",
    "Write a bash command that finds all .py files modified in the last 7 days, sorted by size.",
    "Explain the difference between symmetric and asymmetric encryption with examples.",
    "A basket contains 5 red balls and 3 blue balls. If you draw 2 balls without replacement, what is the probability both are red? Show the math.",
    "Write a haiku about machine learning.",
]
QUANTS = [
    ("BF16", [], "baseline"),
    ("Q8_0", ["-ctk", "q8_0", "-ctv", "q8_0"], "8-bit KV"),
    ("Q4_0", ["-ctk", "q4_0", "-ctv", "q4_0"], "4-bit KV"),
]

def gpu_used():
    out = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]).decode()
    return int(out.strip().split("\n")[0])

def start_server(quant_flags):
    cmd = [SERVER, "-m", MODEL, "--alias", "Ling-3.0-tiny", "-c", "131072",
           "-ngl", "999", "-fa", "on", "-ub", "1024",
           "--port", str(PORT), "--host", "127.0.0.1", "--no-webui",
           "--temp", "1.0", "--top-p", "0.95", "--top-k", "20",
           "--parallel", "2", "-t", "14",
           "--reasoning", "on", "--reasoning-budget", "500",
           "--jinja"] + quant_flags
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # wait for health
    for _ in range(60):
        time.sleep(1)
        try:
            r = requests.get(f"http://127.0.0.1:{PORT}/health", timeout=1)
            if r.status_code == 200:
                return proc
        except:
            pass
    proc.kill()
    raise RuntimeError("server failed to start")

def chat(prompt):
    r = requests.post(f"http://127.0.0.1:{PORT}/v1/chat/completions", json={
        "model": "Ling-3.0-tiny",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0, "top_p": 0.95, "top_k": 20,
        "max_tokens": 1024,
    }, timeout=120)
    data = r.json()
    return data["choices"][0]["message"]["content"], data["usage"]

def stop_server(proc):
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except:
        proc.kill()
    time.sleep(2)

# baseline VRAM (no server)
baseline_used = gpu_used()
results = {}

for quant_name, quant_flags, label in QUANTS:
    print(f"\n{'='*60}")
    print(f"=== {quant_name} ({label}) ===")
    print(f"{'='*60}")
    proc = start_server(quant_flags)
    # measure VRAM with model loaded but empty KV
    model_loaded_used = gpu_used()
    model_vram = model_loaded_used - baseline_used
    print(f"Model VRAM: {model_vram} MiB")

    outputs = []
    for i, p in enumerate(PROMPTS):
        content, usage = chat(prompt=p)
        # capture peak VRAM during generation
        peak = gpu_used()
        print(f"  [{i+1}/{len(PROMPTS)}] prompt_t={usage.get('prompt_tokens','?')} compl_t={usage.get('completion_tokens','?')} GPU_used={peak} MiB")
        outputs.append({"prompt": p, "response": content, "usage": usage, "gpu_used_mib": peak})
        # small delay to let KV settle
        time.sleep(0.5)

    peak_vram = max(o["gpu_used_mib"] for o in outputs)
    stop_server(proc)
    results[quant_name] = {
        "label": label,
        "model_vram_mib": model_vram,
        "peak_vram_mib": peak_vram,
        "outputs": outputs,
    }

# compute how many 128k contexts fit in 12GB with 1GB free for system
TOTAL_VRAM = int(os.environ.get("TOTAL_VRAM", "12288"))
SYSTEM_RESERVED = int(os.environ.get("SYSTEM_RESERVED", "1024"))
USABLE = TOTAL_VRAM - SYSTEM_RESERVED

print(f"\n{'='*60}")
print(f"=== VRAM SUMMARY ===")
print(f"{'='*60}")
print(f"Total VRAM: {TOTAL_VRAM} MiB")
print(f"System reserved: {SYSTEM_RESERVED} MiB")
print(f"Usable for model+KV: {USABLE} MiB")
for qn, r in results.items():
    model_v = r["model_vram_mib"]
    remaining = USABLE - model_v
    print(f"\n{qn}: model={model_v} MiB, remaining_for_kv={remaining} MiB")
    print(f"  peak_observed_during_gen={r['peak_vram_mib']} MiB")

os.makedirs("./results", exist_ok=True)
with open("./results/bench-results.json", "w") as f:
    json.dump(results, f, indent=2)
print("\nSaved to ./results/bench-results.json")
