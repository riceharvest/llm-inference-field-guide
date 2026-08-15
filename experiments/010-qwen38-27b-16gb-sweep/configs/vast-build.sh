#!/bin/bash
# Build llama.cpp on a fresh Vast RTX 5060 Ti 16GB instance (Blackwell, sm_120).
# Run as the non-root user after SSH. Produces ./build/bin/llama-server.
#
# Why this build:
#   - CUDA + Flash Attention: required by the Unsloth Dynamic (UD) quants'
#     V-cache quantization ("-fa on"), and benefits the standard quants too.
#   - sm_120: RTX 5060 Ti is Blackwell; older CUDA toolkit binaries won't JIT
#     the right SASS. CUDA 12.4+ is required. Most Vast CUDA images are 12.x.
#   - No FORK needed: the arch string on the card is "qwen35", and upstream
#     ggml-org llama.cpp carries src/models/qwen35.cpp (incl. the GDN/ATT
#     hybrid). Pin a recent commit; verify on first load (download-verify.sh
#     arch-probes the smallest quant before fetching the rest).

set -euo pipefail
cd "${HOME}"
BUILD_DIR="${HOME}/llama.cpp"
COMMIT="${LLAMA_COMMIT:-master}"

sudo apt-get update -qq
sudo apt-get install -y -qq build-essential cmake curl git jq python3-venv

if [[ ! -d "$BUILD_DIR" ]]; then
  git clone https://github.com/ggml-org/llama.cpp "$BUILD_DIR"
fi
cd "$BUILD_DIR"
git fetch --all --tags -q
git checkout "$COMMIT"

cmake -B build -S . \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES="120" \
  -DGGML_CUDA_FA_ALL_QUANTS=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release -j"$(nproc)"

echo "=== build done ==="
./build/bin/llama-server --version
echo "Add ./build/bin to PATH, then run scripts/download-verify.sh and scripts/sweep.sh"
