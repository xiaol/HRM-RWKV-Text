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

  echo "=== ${run_id}: mode=${mem_mode} heads=[q,o] release-style TSW ==="
  RUN_ID="${run_id}" \
  RWKV_MEM_MODE="${mem_mode}" \
  RWKV_MEM_DELTA_HEADS="[q,o]" \
  RWKV_MEM_RANK="${RWKV_MEM_RANK:-8}" \
  RWKV_MEM_NUM_STATE_HEADS="${RWKV_MEM_NUM_STATE_HEADS:-1}" \
  RWKV_MEM_ALPHA="${RWKV_MEM_ALPHA:-16.0}" \
  RWKV_MEM_BETA_BIAS_INIT="${RWKV_MEM_BETA_BIAS_INIT:--1.5}" \
  RWKV_MEM_STATE_UPDATE_MODE="${RWKV_MEM_STATE_UPDATE_MODE:-standard}" \
  RWKV_MEM_OUTPUT_INIT="${output_init}" \
  RWKV_MEM_BASE_SLICE_REF_WIDTH="${RWKV_MEM_BASE_SLICE_REF_WIDTH:-8}" \
  RWKV_MEM_ONLINE_GAIN="${RWKV_MEM_ONLINE_GAIN:-0.05}" \
  RWKV_MEM_MEMORY_WRITE_GRANULARITY="${RWKV_MEM_MEMORY_WRITE_GRANULARITY:-token}" \
  RWKV_MEM_LOSS_MODE="${RWKV_MEM_LOSS_MODE:-ce_kl}" \
  RWKV_MEM_KL_WEIGHT="${RWKV_MEM_KL_WEIGHT:-0.02}" \
  RWKV_MEM_KL_TEMPERATURE="${RWKV_MEM_KL_TEMPERATURE:-2.0}" \
  MAX_STEPS="${MAX_STEPS}" \
  RUN_MMLU="${RUN_MMLU}" \
  bash "${ROOT_DIR}/scripts/run_rwkv_mem_posttrain_mmlu.sh"
}

run_one "hrm_h_delta_mem_recipe_qo_s${MAX_STEPS}_${TIMESTAMP}" "delta_rule" "${DELTA_RULE_OUTPUT_INIT:-base_slice_fixed}"
run_one "hrm_h_rwkv7_mem_recipe_qo_s${MAX_STEPS}_${TIMESTAMP}" "rwkv7" "${RWKV7_MEM_OUTPUT_INIT:-zero}"
