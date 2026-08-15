#!/usr/bin/env python3
"""Agent-shaped decode/prefill probe for experiment 010 (stdlib-only, no deps).

Sends a REALISTIC Hermes-agent payload (sized system prompt + nested tool
schemas + a concrete coding task) to a llama-server /v1/chat/completions
endpoint, streaming, and measures:
  - prompt_tokens / completion_tokens   (from usage)
  - TTFT                                 (time to first token)
  - prefill tok/s = prompt_tokens / TTFT
  - decode  tok/s = completion_tokens / (E2E - TTFT)
  - E2E wall time

This is NOT a 256-token synthetic completion. The prompt alone is ~700 tokens
and the model must emit a structured tool call, mirroring agent traffic so the
numbers transfer to real Hermes use. Uses only the standard library.

Usage:
  python3 perf-probe.py --url http://127.0.0.1:1235 --model qwen3.8-27b \
    --output results/CELL/perf.json --repeats 3
"""
import argparse
import json
import time
import sys
import urllib.request
import urllib.error

SYSTEM_PROMPT = (
    "You are a careful coding agent working inside a Linux repository. "
    "You have access to file-reading, shell-execution, and patch-application "
    "tools. You must inspect the repository before proposing changes, prefer "
    "minimal well-scoped edits, and never fabricate file paths or command "
    "output. When the task is ambiguous, ask a clarifying question instead of "
    "guessing. After acting, verify the change with a real build or test "
    "command and report observed output. Keep reasoning short and the final "
    "answer grounded in tool evidence. You operate on a Ryzen 5950X workstation "
    "with an RTX 3060 GPU and 64 GiB RAM; the working tree is a Rust crate that "
    "implements a local LLM inference server. The user expects production-grade "
    "code, not a prototype. Remember prior turns; reuse established conventions "
    "from the surrounding files rather than inventing new ones."
)

TOOLS = [
    {"type": "function", "function": {"name": "read_file",
        "description": "Read a text file with line numbers and pagination.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "Absolute or relative path."},
            "offset": {"type": "integer", "description": "1-indexed start line."},
            "limit": {"type": "integer", "description": "Max lines to return."}},
            "required": ["path"]}}},
    {"type": "function", "function": {"name": "run_shell",
        "description": "Execute a shell command and return stdout/stderr + exit code.",
        "parameters": {"type": "object", "properties": {
            "command": {"type": "string", "description": "Command line to execute."},
            "timeout": {"type": "integer", "description": "Seconds before kill."},
            "workdir": {"type": "string", "description": "Working directory."}},
            "required": ["command"]}}},
    {"type": "function", "function": {"name": "apply_patch",
        "description": "Apply a unified or V4A patch to the working tree.",
        "parameters": {"type": "object", "properties": {
            "path": {"type": "string", "description": "File to patch."},
            "patch": {"type": "string", "description": "Patch content."},
            "mode": {"type": "string", "enum": ["replace", "patch"]}},
            "required": ["path", "patch", "mode"]}}},
]


def one_request(url, model, user_prompt):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        "tools": TOOLS,
        "tool_choice": "auto",
        "max_tokens": 512,
        "temperature": 0.0,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions", data=data,
        headers={"content-type": "application/json"})

    t0 = time.perf_counter()
    ttft = None
    completion = ""
    usage = None
    with urllib.request.urlopen(req, timeout=180) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line or not line.startswith("data:"):
                continue
            chunk = line[5:].strip()
            if chunk == "[DONE]":
                break
            try:
                obj = json.loads(chunk)
            except json.JSONDecodeError:
                continue
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                if ttft is None:
                    ttft = time.perf_counter() - t0
                completion += delta["content"]
            if obj.get("usage"):
                usage = obj["usage"]
    e2e = time.perf_counter() - t0
    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "e2e_s": round(e2e, 3),
        "prompt_tokens": usage.get("prompt_tokens") if usage else None,
        "completion_tokens": usage.get("completion_tokens") if usage else None,
        "completion_chars": len(completion),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--output", required=True)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--user-prompt", default=(
        "Inspect the crate's main server module and add a GET /healthz endpoint "
        "that returns 200 OK. Show the exact patch you would apply."))
    args = ap.parse_args()

    results = []
    one_request(args.url, args.model, args.user_prompt)  # warm-up
    for _ in range(args.repeats):
        r = one_request(args.url, args.model, args.user_prompt)
        pt, ct = r["prompt_tokens"] or 0, r["completion_tokens"] or 0
        ttft, e2e = r["ttft_s"] or 0, r["e2e_s"] or 0
        r["prefill_tok_s"] = round(pt / ttft, 1) if ttft > 0 else None
        r["decode_tok_s"] = round(ct / (e2e - ttft), 1) if (e2e - ttft) > 0 else None
        results.append(r)

    out = {"url": args.url, "model": args.model,
           "repeats": args.repeats, "runs": results}
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    ds = [r["decode_tok_s"] for r in results if r["decode_tok_s"]]
    ps = [r["prefill_tok_s"] for r in results if r["prefill_tok_s"]]
    if ds:
        print(f"decode  median: {sorted(ds)[len(ds)//2]} tok/s")
    if ps:
        print(f"prefill median: {sorted(ps)[len(ps)//2]} tok/s")


if __name__ == "__main__":
    main()
