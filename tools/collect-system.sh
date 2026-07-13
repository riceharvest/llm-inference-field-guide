#!/usr/bin/env bash
set -u

section() { printf '
## %s
' "$1"; }
run() { printf '$ %s
' "$*"; "$@" 2>&1 || true; }

section timestamp
run date --iso-8601=seconds
section os
run uname -a
if command -v lsb_release >/dev/null; then run lsb_release -a; fi
section cpu
run lscpu
section memory
run free -h
section accelerator
if command -v nvidia-smi >/dev/null; then
  run nvidia-smi --query-gpu=name,driver_version,memory.total,memory.used,temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,utilization.memory --format=csv
  run nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
fi
section listeners
run ss -tlnp
section llama
if command -v llama-server >/dev/null; then run llama-server --version; fi

printf '
Collector intentionally omits environment variables and command histories. Review before sharing.
'
