#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

MAX_STEPS="${MAX_STEPS:-200}"
RUN_MMLU="${RUN_MMLU:-1}"

run_one() {
  local run_id="$1"
  local mem_mode="$2"

  echo "=== ${run_id}: mode=${mem_mode} heads=[q,k,v] ==="
  RUN_ID="${run_id}" \
  RWKV_MEM_MODE="${mem_mode}" \
  RWKV_MEM_DELTA_HEADS="[q,k,v]" \
  MAX_STEPS="${MAX_STEPS}" \
  RUN_MMLU="${RUN_MMLU}" \
  bash "${ROOT_DIR}/scripts/run_rwkv_mem_posttrain_mmlu.sh"
}

run_one "hrm_h_delta_mem_qkv_s${MAX_STEPS}_${TIMESTAMP}" "delta_rule"
run_one "hrm_h_rwkv7_mem_qkv_s${MAX_STEPS}_${TIMESTAMP}" "rwkv7"
