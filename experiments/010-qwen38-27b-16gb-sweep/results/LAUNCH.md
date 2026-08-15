# Experiment 010 — launch checklist (Vast RTX 5060 Ti 16 GB)

Run on the rented box, in order. Each step is a real command; no placeholders.

## 1. Rent + SSH
- Vast: RTX 5060 Ti, 16 GB VRAM, CUDA 12.4+ image, ~30 GB disk, openSSH.
- SSH in as the non-root user.

## 2. Clone field guide
```
git clone https://github.com/riceharvest/llm-inference-field-guide ~/fg
cd ~/fg/experiments/010-qwen38-27b-16gb-sweep
```

## 3. Build llama.cpp (sm_120, CUDA, Flash Attention)
```
bash configs/vast-build.sh
export PATH="$HOME/llama.cpp/build/bin:$PATH"
llama-server --version        # confirm CUDA backend present
```

## 4. Download + verify GGUFs (SHA256 vs HF LFS oid)
```
python3 scripts/budget.py            # print the fit matrix for reference
bash scripts/download-verify.sh      # downloads 6 quants + arch-probes smallest
```
The arch-probe confirms the build loads the qwen35 hybrid BEFORE pulling
the remaining ~70 GB. If it fails, stop — the pinned llama.cpp commit lacks
qwen35 support; bump LLAMA_COMMIT and rebuild.

## 5. Run the PERF sweep (real-prompt decode/prefill)
```
MODE=perf bash scripts/sweep.sh
# outputs: results/<quant>-<kv>-<ctx>/{perf.json,vram.csv,server-*.log}
```

## 6. Wire Hermes provider + run AGENT sweep
```
hermes config set providers.qwen38-local.api http://127.0.0.1:1235/v1
hermes config set providers.qwen38-local.default_model qwen3.8-27b
hermes config set providers.qwen38-local.api_key dummy
hermes config set providers.qwen38-local.context_length 32768
hermes config check
MODE=agent bash scripts/sweep.sh
# outputs: results/<quant>-<kv>-<ctx>/ run JSON + score report
```

## 7. Collect + commit
```
bash ../../tools/collect-system.sh > results/system-5060ti.txt
# copy results/*.json into results/ for the dated matrix
# git add, commit, push from the local workstation
```
