# 010 — Qwen3.8-27B-UD-Q2_K_XL on 16 GB VRAM for Hermes Agents

> A 27 B dense hybrid (48 Gated DeltaNet + 16 Gated Attention layers) does not fit a 16 GB card at 4-bit with usable context. The quant dario specified — **`unsloth/Qwen3.8-27B-UD-Q2_K_XL`** (10.7 GB weights) — is the smallest viable option: it leaves ~3.5 GB for the hybrid recurrent/KV state, enough for a large agent context. This experiment measures that one quant: real-prompt **decode/prefill tok/s**, achievable **context length**, and **HermesBench tool-use completion** on a 16 GB card, so anyone with that card can launch the same config.

**Hardware:** RTX 5060 Ti 16 GB (Blackwell, sm_120) on Vast
**Model:** unsloth/Qwen3.8-27B-UD-Q2_K_XL (2-bit Unsloth Dynamic)
**Engine:** llama.cpp (CUDA + Flash Attention, sm_120), HermesBench `hermes-core`
**Date:** 2026-08 (pending Vast run)

---

## Quick Start

For a 16 GB card and Hermes agent use, the subject config is `UD-Q2_K_XL` at the max context that fits. To reproduce the measurement:

```bash
# 1. On the Vast box: build, download + verify the one quant, arch-probe
bash configs/vast-build.sh
bash scripts/download-verify.sh          # UD-Q2_K_XL only, SHA256-verified

# 2. Measure real-prompt decode/prefill + record peak VRAM
MODE=perf bash scripts/sweep.sh

# 3. Wire into Hermes + run the agent eval
hermes config set providers.qwen38-local.api http://127.0.0.1:1235/v1
hermes config set providers.qwen38-local.context_length 32768
MODE=agent bash scripts/sweep.sh         # hermesbench hermes-core
```

The budget calculator shows the fit before you download anything:

```bash
python3 scripts/budget.py
# -> weights 10.68 GB, ~3.5 GB free for state
# -> est max ctx ~13312 (q8_0-state estimate), launch ctx 12288
```

---

## Exact flags

Per-cell server launch (`configs/llama-server-sweep.sh`):

```bash
llama-server \
  -m "${MODEL_PATH}" \
  --host 127.0.0.1 --port "${PORT}" \
  -c "${CTX}" -ngl 999 -fa on \
  --jinja -np 1 --no-warmup
  # NOTE: no -ctk/-ctv — UD-Q2_K_XL embeds its own KV/state format and
  # rejects those flags (experiment 009 lesson). -fa on is required for
  # the UD V-cache quantization.
```

Hermes provider wiring (`configs/hermes-provider.yaml`):

```yaml
providers:
  qwen38-local:
    api: http://127.0.0.1:1235/v1
    default_model: qwen3.8-27b
    api_key: dummy
    context_length: 32768
```

---

## The symptom / the question

A 27 B model at 4-bit is ≥15.7 GB. On a 16 GB card that leaves <0.5 GB for the hybrid state — no usable context. Sub-4-bit is mandatory. The specific open question: **does the 2-bit `UD-Q2_K_XL` survive agent use** — i.e. does it emit and *complete* HermesBench tasks, or does it collapse into the experiment-009 failure mode (valid tool calls, zero task completion) that hit non-QAT quants?

---

## The fix / the method

No code "fix" — this is a measurement experiment. The method enforces the field guide's own rules:

- **Verify before you debug** (exp 002): `download-verify.sh` uses `hf_hub_download`, which computes the GGUF's SHA256 and refuses to return it unless it matches the repo LFS/Xet oid. It then arch-probes the file to confirm llama.cpp can load the `qwen35` hybrid before you trust it.
- **UD rejects `-ctk/-ctv`** (exp 009): the launcher omits KV overrides for UD quants. UD-Q2_K_XL runs at its embedded native state format.
- **Real prompts, not synthetic** (exp 005): `perf-probe.py` sends a ~700-token agent system prompt + nested tool schemas and parses `TTFT`, `prefill tok/s`, and `decode tok/s` from streaming usage — mirroring actual Hermes traffic.
- **Score on completion, not tool emission** (exp 009): agent quality comes from HermesBench `hermes-core` task completion, not whether `tool_calls` is populated.
- **Measure at the quant's own max context**: `budget.py --plan` emits the single cell (UD-Q2_K_XL at its max feasible context). The sweep runs exactly that one cell.

---

## Measured Results

_Pending the Vast 16 GB run._ The single cell `Qwen3.8-27B-UD-Q2_K_XL/native/<ctx>` produces:

| Artifact | Meaning |
|---|---|
| `perf.json` | real-prompt TTFT, prefill tok/s, decode tok/s (3 repeats) |
| `vram.csv` | peak VRAM for the run (recorded, not estimated) |
| hermesbench run JSON + score | Hermes-core task completion /13 |

The planned cell (default 1.5 GB overhead assumption):

```
Qwen3.8-27B-UD-Q2_K_XL  native  12288
```

The dated consolidated result will be committed as `results/2026-08-<dd>-qwen38-udq2xl.json`.

---

## Troubleshooting

### `V cache quantization requires flash_attn`
**Cause:** UD quant without `-fa on`, or explicit `-ctk/-ctv` clashing with the UD metadata.
**Fix:** `-fa on` and no `-ctk/-ctv` (handled automatically by the launcher for UD files).

### `unknown model architecture 'qwen35'` / load fails
**Cause:** llama.cpp build predates qwen35.cpp (the Qwen3.8 arch string).
**Fix:** bump `LLAMA_COMMIT` and rebuild (`configs/vast-build.sh`). The arch-probe in `download-verify.sh` catches this before you download.

### CUDA `no kernel image available` / sm_120 mismatch
**Cause:** built for the wrong compute capability (5060 Ti is Blackwell sm_120).
**Fix:** ensure `-DCMAKE_CUDA_ARCHITECTURES="120"` and a CUDA 12.4+ toolkit.

### OOM at context start
**Cause:** `--ctx` exceeds what fits for the 10.7 GB weights + ~3.5 GB state budget.
**Fix:** the budget calculator shows the max ctx. Drop the context, don't change the quant (UD-Q2_K_XL is fixed).

### Agent scores valid tool calls but completes 0/13
**Cause:** too-aggressive 2-bit quant.
**Fix:** record the result — this IS the experiment's hypothesis test. If it fails, the conclusion is "2-bit is below the agent floor for 27 B and you need a bigger card or Q3+."

---

## What transfers

1. **Sub-4-bit is mandatory for 27 B on 16 GB.** Q4+ leaves no context. Q2_K_XL is the smallest Unsloth Dynamic option and the only one with a large agent context.
2. **The hybrid (GDN) state dominates KV cost** — 48 recurrent layers at full embedding width. This is what makes even ~12K context thinkable at 2-bit on 16 GB; a pure-attention 27 B would not fit.
3. **UD quants own their KV format** — never pass `-ctk/-ctv` to them; `-fa on` is the only required cache flag.
4. **Agent viability ≠ tool emission.** Score completion (HermesBench), not `tool_calls` presence — 2-bit is where that gap bites.

---

## Files

| Path | Purpose |
|---|---|
| [`configs/llama-server-sweep.sh`](configs/llama-server-sweep.sh) | Single-cell launcher; no KV flags for UD |
| [`configs/vast-build.sh`](configs/vast-build.sh) | llama.cpp build for sm_120 + Flash Attention |
| [`configs/hermes-provider.yaml`](configs/hermes-provider.yaml) | Hermes provider snippet |
| [`scripts/budget.py`](scripts/budget.py) | VRAM/context feasibility for UD-Q2_K_XL |
| [`scripts/download-verify.sh`](scripts/download-verify.sh) | Download + SHA256 verify (via hf_hub) + arch-probe |
| [`scripts/sweep.sh`](scripts/sweep.sh) | Perf + agent orchestrator (MODE=perf\|agent) |
| [`scripts/perf-probe.py`](scripts/perf-probe.py) | Real-prompt decode/prefill measurement |
| [`scripts/smoke-qwen38.sh`](scripts/smoke-qwen38.sh) | Endpoint + tool-call smoke test |
| [`scripts/test-budget.py`](scripts/test-budget.py) | Self-test for budget arithmetic (CI) |
| [`results/LAUNCH.md`](results/LAUNCH.md) | Step-by-step Vast run checklist |
