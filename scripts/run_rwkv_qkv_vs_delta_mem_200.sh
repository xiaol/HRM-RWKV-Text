#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
TIMESTAMP="${TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"

MAX_STEPS="${MAX_STEPS:-200}"
RUN_MMLU="${RUN_MMLU:-1}"

run_one() {
  local run_id="$1"
  local mem_mode="$2"
  local delta_heads="$3"
  shift 3

  echo "=== ${run_id}: mode=${mem_mode} heads=${delta_heads} ==="
  RUN_ID="${run_id}" \
  RWKV_MEM_MODE="${mem_mode}" \
  RWKV_MEM_DELTA_HEADS="${delta_heads}" \
  MAX_STEPS="${MAX_STEPS}" \
  RUN_MMLU="${RUN_MMLU}" \
  bash "${ROOT_DIR}/scripts/run_rwkv_mem_posttrain_mmlu.sh"
}

run_one "hrm_h_delta_mem_qkvo_s${MAX_STEPS}_${TIMESTAMP}" "delta_rule" "[q,k,v,o]"
run_one "hrm_h_delta_mem_qo_s${MAX_STEPS}_${TIMESTAMP}" "delta_rule" "[q,o]"
run_one "hrm_h_rwkv7_mem_qkvo_s${MAX_STEPS}_${TIMESTAMP}" "rwkv7" "[q,k,v,o]"
