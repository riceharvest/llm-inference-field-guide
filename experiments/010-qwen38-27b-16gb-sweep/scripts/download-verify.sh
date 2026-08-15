#!/bin/bash
# Download + verify the Qwen3.8-27B GGUFs for the 16 GB sweep, then arch-probe
# the smallest one to confirm llama.cpp can actually load the qwen35 hybrid
# before pulling the rest (saves bandwidth if the build is too old).
#
# VERIFY step is automatic: huggingface_hub's hf_hub_download computes the
# file's SHA256 and refuses to return it unless it matches the repo's LFS oid
# (the "verify before you debug" rule from experiment 002, enforced by the lib).
#
# Usage:
#   MODEL_DIR=/path/to/ggufs bash scripts/download-verify.sh
# Emits /path/to/ggufs/manifest.json: { "Qwen3.8-27B-UD-IQ2_XXS": "/abs/path", ... }

set -eu
MODEL="unsloth/Qwen3.8-27B-GGUF"
MODEL_DIR="${MODEL_DIR:-./results/models}"
mkdir -p "$MODEL_DIR"
cd "$MODEL_DIR"

# Only these fit a 16 GB card with usable context (sizes from the HF card).
FILES=(
  Qwen3.8-27B-UD-IQ2_XXS.gguf
  Qwen3.8-27B-UD-IQ2_M.gguf
  Qwen3.8-27B-UD-Q2_K_XL.gguf
  Qwen3.8-27B-UD-IQ3_XXS.gguf
  Qwen3.8-27B-Q3_K_S.gguf
  Qwen3.8-27B-UD-Q3_K_XL.gguf
  Qwen3.8-27B-Q3_K_M.gguf
)

python3 - <<PY
import json, sys, importlib.util
if importlib.util.find_spec("huggingface_hub") is None:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "huggingface_hub"])
from huggingface_hub import hf_hub_download

files = ${FILES[@]@Q}
manifest = {}
for f in files:
    # hf_hub_download verifies SHA256 == repo LFS oid before returning.
    path = hf_hub_download(repo_id="$MODEL", filename=f, repo_type="model")
    manifest[f.replace(".gguf", "")] = path
    print(f"verified: {f} -> {path}")
with open("manifest.json", "w") as fh:
    json.dump(manifest, fh, indent=2)
print("manifest written:", "manifest.json")
PY

# Arch-probe: confirm llama.cpp can LOAD the qwen35 hybrid before the big pulls.
PROBE="Qwen3.8-27B-UD-IQ2_XXS.gguf"
if command -v llama-cli >/dev/null 2>&1; then
  echo "arch-probe: loading $PROBE (expects no 'unknown architecture qwen35')"
  timeout 180 llama-cli -m "$PROBE" -p "hi" -n 1 --no-warmup >/dev/null 2>&1 \
    && echo "arch-probe OK: qwen35 loadable" \
    || echo "arch-probe WARN: load failed — bump LLAMA_COMMIT (see configs/vast-build.sh)"
else
  echo "arch-probe skipped: llama-cli not on PATH (run after configs/vast-build.sh)"
fi
