#!/bin/bash
# Download + verify the SUBJECT quant for experiment 010:
#   unsloth/Qwen3.8-27B-UD-Q2_K_XL  (the quant dario specified)
#
# VERIFY step is automatic: huggingface_hub's hf_hub_download computes the
# file's SHA256 and refuses to return it unless it matches the repo's LFS oid
# (the "verify before you debug" rule from experiment 002, enforced by the lib).
# Xet-backed repos are handled by huggingface_hub transparently.
#
# Usage:
#   MODEL_DIR=/path/to/ggufs bash scripts/download-verify.sh
# Emits /path/to/ggufs/manifest.json: { "Qwen3.8-27B-UD-Q2_K_XL": "/abs/path" }

set -eu
MODEL="unsloth/Qwen3.8-27B-GGUF"
SUBJECT="Qwen3.8-27B-UD-Q2_K_XL.gguf"
MODEL_DIR="${MODEL_DIR:-./results/models}"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

python3 - <<PY
import json, sys, importlib.util
if importlib.util.find_spec("huggingface_hub") is None:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import hf_hub_download

f = "$SUBJECT"
# hf_hub_download verifies SHA256 == repo LFS oid before returning.
path = hf_hub_download(repo_id="$MODEL", filename=f, repo_type="model")
manifest = {f.replace(".gguf", ""): path}
with open("manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print(f"verified: {f} -> {path}")
print("manifest written: manifest.json")
PY

# Arch-probe: confirm llama.cpp can LOAD the qwen35 hybrid before we trust it.
if command -v llama-cli >/dev/null 2>&1; then
  echo "arch-probe: loading $SUBJECT (expects no 'unknown architecture qwen35')"
  timeout 180 llama-cli -m "$SUBJECT" -p "hi" -n 1 --no-warmup >/dev/null 2>&1 \
    && echo "arch-probe OK: qwen35 loadable" \
    || echo "arch-probe WARN: load failed — bump LLAMA_COMMIT (see configs/vast-build.sh)"
else
  echo "arch-probe skipped: llama-cli not on PATH (run after configs/vast-build.sh)"
fi
