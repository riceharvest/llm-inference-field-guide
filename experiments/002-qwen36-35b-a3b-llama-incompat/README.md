# 002 — The Corrupt GGUF That Looked Like an Architecture Bug

> Downloaded a 23 GB GGUF. It produced garbage output. I spent hours trying different quants, builds, and flags. Nothing worked. The real problem: the file's SHA256 didn't match the publisher's LFS oid. A re-download fixed everything.

**Hardware:** RTX 3060 12GB
**Model:** Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf (23.4 GB)
**Engine:** llama.cpp (multiple builds tested: upstream, turboquant)
**Key verifications:** GGUF SHA-256 checked against HF LFS oid; live llama.cpp build `284ffc7` contains PR #26252’s Qwen3 specialized parser; custom community template tested and removed.

---

## Quick Start

Before debugging any model behavior, do this:

```bash
# 1. Get the expected SHA256 from the publisher's repo
#    (usually in the model card or release notes)
echo "8d344a4336d8ea7da0cbfc12792d1471e568be7abe8930c52260698bfd01d731  Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf" > expected.sha256

# 2. Verify your download
sha256sum -c expected.sha256
# → OK = file is intact
# → MISMATCH = re-download. Don't debug further until this passes.

# 3. If using aria2 and it dies mid-download, switch to curl:
curl -L -C - -o model.gguf --retry 5 --retry-delay 5 \
  -A "go-http-client/1.1" -sS \
  "https://huggingface.co/<repo>/resolve/main/<filename>.gguf"
```

**Full verifier:** [`scripts/verify-checksum.sh`](scripts/verify-checksum.sh)

## Exact flags

Tested configurations during diagnosis:

```bash
# Failed runs with corrupt GGUF
llama-server -m model.gguf --jinja --temp 0.0 -c 32768 --parallel 1
llama-server -m model.gguf --jinja --temp 0.7 -c 32768 --parallel 2
llama-server -m model.gguf --no-jinja --temp 0.0 -c 32768 --parallel 1
```

Final working run after re-download:

```bash
llama-server \
  -m "${MODEL_PATH:-./Qwen3.6-35B-A3B-Q4_K_P-verified.gguf}" \
  --jinja --temp 0.7 -c 32768 \
  --parallel 2 -t 14 -tb 16 \
  -ngl 999 --device CUDA0
```

---

## The fix

1. Verify the GGUF SHA256 against the publisher's LFS oid before changing any flags.
2. If the hash mismatches, re-download with curl or `huggingface-cli` instead of trusting aria2 resume.
3. Once integrity passes, use the serving flags in `## Exact flags`.

**Why this order matters:** The header/tokenizer can load cleanly while the weight region is corrupt. That failure mode looks exactly like an architecture or quant bug.

---

## The symptom

Downloaded the Q4_K_P quant (23,424,536,704 bytes). It loaded fine — health endpoint returned OK, tokenizer worked, architecture was recognized. But every response was garbage:

- Raw completion of "What is the capital of Japan?" → "Tokyo Tokyo Japan … ENTI-[garbage]"
- Repeated output: `a a a a…` or `def is_pal(s)\n return s == s` looping infinitely
- Same garbage across every config: `--jinja` on/off, thinking on/off, temp 0.0→0.7, 80→2048 token budgets

## The wrong diagnosis path (4 attempts before checking the file)

**Attempt 1:** Suspected the quant. UD/Dynamic-2.0 quants are known to produce silent garbage in mainline llama.cpp. But this was a plain Q4_K_M-style quant — should work.

**Attempt 2:** Suspected the build. Tested both upstream and turboquant. Same garbage on both.

**Attempt 3:** Suspected the architecture. The `qwen3_5_moe` hybrid is only officially supported by SGLang/vLLM. Maybe llama.cpp's partial support was degenerating.

**Attempt 4:** Suspected the repo's conversion. Checked HF discussions — other users reported the model working clean. Nobody reported full token-soup on this quant.

All 4 theories were wrong. The file itself was corrupt.

## The actual fix

```bash
sha256sum Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_P.gguf
# → ee07cfda… (WRONG)

# Expected (from HF LFS oid):
# → 8d344a4336d8ea7da0cbfc12792d1471e568be7abe8930c52260698bfd01d731
```

**Root cause:** The aria2 download died at ~52% with 403s from the Xet CDN. The resume/error aftermath left a file with the **correct byte count but corrupted bytes in the weight region**. The header and tokenizer (early in the file) parsed clean, so I blamed the conversion. The weights (later in the file) were garbage.

**Fix:** Re-downloaded with curl via the resolve CDN (not Xet). SHA256 matched the LFS oid verbatim. The model worked perfectly.

**Honcho data:** The same Qwen3.6-27B lineage later used build `284ffc7` with PR #26252’s specialized parser, embedded model template active, and a 128-byte SSE proxy — all confirmed working once integrity was verified. That later stack is not the cause of 002; it just shows how many downstream knobs get blamed before the file is checked.

## Measured Results

**Date:** 2026-08-06
**Full data:** [`results/2026-08-06-qwen36-35b-a3b-sha256-fix.md`](results/2026-08-06-qwen36-35b-a3b-sha256-fix.md)

| Metric | Corrupt file | Clean file |
|---|---|---|
| SHA256 | `ee07cfda…` (wrong) | `8d344a43…` (matches LFS) |
| Output | token soup | clean English |
| Finish reason | `length` (never stops) | `stop` |
| Decode speed | N/A (garbage) | ~31 tok/s |
| VRAM | ~12 GB | ~12 GB |

## Troubleshooting

### Model loads fine but produces garbage

**Symptom:** Health endpoint returns OK, tokenizer works, architecture recognized. But every response is repetitive nonsense or random characters.

**Cause:** Corrupt download. The file has the correct byte count but corrupted bytes in the weight region.

**Fix:** Always verify SHA256 against the publisher's LFS oid *before* debugging behavior. Don't trust file size alone.

### aria2 dies mid-download with 403s

**Symptom:** aria2c reports completion but the file is corrupt. Xet CDN returns 403 on some byte ranges.

**Cause:** HuggingFace's Xet CDN has intermittent 403s. aria2's resume logic can produce a byte-count-identical file with corrupted regions.

**Fix:** Use curl with `-C -` (resume) and `--retry 5`. Or use the `huggingface-cli` tool which handles Xet correctly.

### How to tell header vs weight corruption apart

**Symptom:** Tokenizer works, but generation is garbage.

**Cause:** GGUF files store the header and tokenizer at the beginning of the file. Weights are stored later. A partial download that completes the header but corrupts the weight region will pass initial checks but fail at generation.

**Fix:** SHA256 check catches this. Don't waste time diagnosing "architecture incompatibility" until you've verified the hash.

---

## What transfers

1. **Always verify SHA256 before debugging behavior.** A corrupt download can perfectly mimic an architecture bug, a quant issue, or a build problem. Check the hash first.

2. **File size matching ≠ file integrity.** A 23 GB file can have the right byte count and still have corrupted regions. Only a cryptographic hash proves integrity.

3. **Xet CDN 403s are real.** If using aria2 and it dies mid-download, switch to curl or `huggingface-cli`. Don't trust the resume.

4. **Header-clean + weight-corrupt is a specific failure mode.** If the tokenizer works but generation is garbage, suspect the weight region — not the architecture.

---

## Files

| Path | Purpose |
|---|---|
| [`results/2026-08-06-qwen36-35b-a3b-sha256-fix.md`](results/2026-08-06-qwen36-35b-a3b-sha256-fix.md) | Full diagnosis path + SHA256 verification |
| [`scripts/verify-checksum.sh`](scripts/verify-checksum.sh) | Verify GGUF SHA256 against HF LFS oid |
