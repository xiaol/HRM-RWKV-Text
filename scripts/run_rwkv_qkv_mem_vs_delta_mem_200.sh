#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

MAX_STEPS="${MAX_STEPS:-200}"
RUN_MMLU="${RUN_MMLU:-1}"

run_one() {
  local run_id="$1"
  local mem_mode="$2"
  local output_init="$3"

  echo "=== ${run_id}: mode=${mem_mode} heads=[q,k,v] ==="
  RUN_ID="${run_id}" \
  RWKV_MEM_MODE="${mem_mode}" \
  RWKV_MEM_DELTA_HEADS="[q,k,v]" \
  RWKV_MEM_OUTPUT_INIT="${output_init}" \
  RWKV_MEM_LOSS_MODE="${RWKV_MEM_LOSS_MODE:-ce_kl}" \
  RWKV_MEM_KL_WEIGHT="${RWKV_MEM_KL_WEIGHT:-0.02}" \
  RWKV_MEM_KL_TEMPERATURE="${RWKV_MEM_KL_TEMPERATURE:-2.0}" \
  MAX_STEPS="${MAX_STEPS}" \
  RUN_MMLU="${RUN_MMLU}" \
  bash "${ROOT_DIR}/scripts/run_rwkv_mem_posttrain_mmlu.sh"
}

run_one "hrm_h_delta_mem_qkv_s${MAX_STEPS}_${TIMESTAMP}" "delta_rule" "${DELTA_RULE_OUTPUT_INIT:-base_slice_fixed}"
run_one "hrm_h_rwkv7_mem_qkv_s${MAX_STEPS}_${TIMESTAMP}" "rwkv7" "${RWKV7_MEM_OUTPUT_INIT:-zero}"
