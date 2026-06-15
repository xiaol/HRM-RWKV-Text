#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"

echo "run_rwkv_qkv_vs_delta_mem_200.sh is kept as a compatibility wrapper."
echo "Launching the fair q/k/v/o RWKV-memory vs delta-rule comparison."
exec bash "${ROOT_DIR}/scripts/run_rwkv_qkv_mem_vs_delta_mem_200.sh"
