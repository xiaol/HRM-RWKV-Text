#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
DISK="${DISK:-/run/media/xiaol/B214449214445C0B}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"
DELTA_MEM_REPO="${DELTA_MEM_REPO:-/home/xiaol/X/delta-Mem}"

if ! findmnt "${DISK}" >/dev/null 2>&1; then
  udisksctl mount -b /dev/nvme1n1p2 >/dev/null || true
fi
if ! findmnt "${DISK}" >/dev/null 2>&1; then
  echo "missing mounted SSD at ${DISK}" >&2
  exit 1
fi

DATA_PATH="${DATA_PATH:-${DISK}/hrm_text_full_v1}"
if [[ ! -d "${DATA_PATH}" ]]; then
  echo "missing DATA_PATH=${DATA_PATH}" >&2
  exit 1
fi

resolve_init_safetensors() {
  if [[ -n "${INIT_SAFETENSORS:-}" ]]; then
    printf '%s\n' "${INIT_SAFETENSORS}"
    return
  fi

  local candidates=(
    "${DISK}/hrm_text_eval_checkpoints/hrm_text_1b_teacher/model.safetensors"
    "${DISK}/hf_models/sapientinc/HRM-Text-1B/model.safetensors"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [[ -s "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return
    fi
  done

  local found
  found="$(find "${DISK}/hf_cache/models--sapientinc--HRM-Text-1B" -path '*/snapshots/*/model.safetensors' -type f -print -quit 2>/dev/null || true)"
  if [[ -n "${found}" ]]; then
    printf '%s\n' "${found}"
    return
  fi

  echo "could not find original HRM-Text-1B model.safetensors; set INIT_SAFETENSORS" >&2
  exit 1
}

INIT_SAFETENSORS="$(resolve_init_safetensors)"

RUN_ID="${RUN_ID:-hrm_h_delta_mem_qkvo_posttrain_mmlu_$(date +%Y%m%d_%H%M%S)}"
CKPT_ROOT="${CKPT_ROOT:-${DISK}/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain}"
LOG_ROOT="${LOG_ROOT:-${DISK}/hrm_text_pretrain_logs/rwkv_mem_posttrain}"
CKPT_DIR="${CKPT_DIR:-${CKPT_ROOT}/${RUN_ID}}"
TRAIN_LOG="${TRAIN_LOG:-${LOG_ROOT}/${RUN_ID}.train.log}"
MMLU_LOG="${MMLU_LOG:-${LOG_ROOT}/${RUN_ID}.mmlu.log}"
LOSS_HISTORY="${LOSS_HISTORY:-${LOG_ROOT}/${RUN_ID}.loss.jsonl}"
MMLU_CONFIG="${MMLU_CONFIG:-evaluation/config/hrm_mmlu_only.yaml}"
CKPT_TAG="${CKPT_TAG:-step_${MAX_STEPS:-200}}"

MAX_STEPS="${MAX_STEPS:-200}"
ARCH_SIZE="${ARCH_SIZE:-XL}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-196608}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-512}"
LR="${LR:-2e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"
LOG_INTERVAL="${LOG_INTERVAL:-1}"
RUN_MMLU="${RUN_MMLU:-1}"
TRAINABLE_PARAM_SUBSTRINGS="${TRAINABLE_PARAM_SUBSTRINGS:-[rwkv_mem]}"
RWKV_MEM_DELTA_HEADS="${RWKV_MEM_DELTA_HEADS:-[q,k,v,o]}"
RWKV_MEM_MODE="${RWKV_MEM_MODE:-delta_rule}"
RWKV_MEM_RANK="${RWKV_MEM_RANK:-8}"
RWKV_MEM_NUM_STATE_HEADS="${RWKV_MEM_NUM_STATE_HEADS:-1}"
RWKV_MEM_ALPHA="${RWKV_MEM_ALPHA:-16.0}"
RWKV_MEM_BETA_BIAS_INIT="${RWKV_MEM_BETA_BIAS_INIT:--1.5}"
RWKV_MEM_STATE_UPDATE_MODE="${RWKV_MEM_STATE_UPDATE_MODE:-standard}"
RWKV_MEM_SEPARATE_DELTA_PROJECTIONS="${RWKV_MEM_SEPARATE_DELTA_PROJECTIONS:-false}"
RWKV_MEM_OUTPUT_INIT="${RWKV_MEM_OUTPUT_INIT:-zero}"
RWKV_MEM_BASE_SLICE_REF_WIDTH="${RWKV_MEM_BASE_SLICE_REF_WIDTH:-8}"
RWKV_MEM_ONLINE_GAIN="${RWKV_MEM_ONLINE_GAIN:-0.05}"
RWKV_MEM_MEMORY_WRITE_GRANULARITY="${RWKV_MEM_MEMORY_WRITE_GRANULARITY:-token}"

mkdir -p "${CKPT_DIR}" "${LOG_ROOT}"

cd "${ROOT_DIR}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
if [[ -d "${DELTA_MEM_REPO}/deltamem" ]]; then
  export PYTHONPATH="${DELTA_MEM_REPO}${PYTHONPATH:+:${PYTHONPATH}}"
fi

{
  echo "run_id=${RUN_ID}"
  echo "data_path=${DATA_PATH}"
  echo "init_safetensors=${INIT_SAFETENSORS}"
  echo "checkpoint_dir=${CKPT_DIR}"
  echo "max_steps=${MAX_STEPS}"
  echo "arch_size=${ARCH_SIZE}"
  echo "global_batch_size=${GLOBAL_BATCH_SIZE}"
  echo "micro_batch_size=${MICRO_BATCH_SIZE}"
  echo "trainable_param_substrings=${TRAINABLE_PARAM_SUBSTRINGS}"
  echo "rwkv_mem_mode=${RWKV_MEM_MODE}"
  echo "rwkv_mem_delta_heads=${RWKV_MEM_DELTA_HEADS}"
  echo "rwkv_mem_rank=${RWKV_MEM_RANK}"
  echo "rwkv_mem_num_state_heads=${RWKV_MEM_NUM_STATE_HEADS}"
  echo "rwkv_mem_alpha=${RWKV_MEM_ALPHA}"
  echo "rwkv_mem_beta_bias_init=${RWKV_MEM_BETA_BIAS_INIT}"
  echo "rwkv_mem_state_update_mode=${RWKV_MEM_STATE_UPDATE_MODE}"
  echo "rwkv_mem_separate_delta_projections=${RWKV_MEM_SEPARATE_DELTA_PROJECTIONS}"
  echo "rwkv_mem_output_init=${RWKV_MEM_OUTPUT_INIT}"
  echo "rwkv_mem_base_slice_ref_width=${RWKV_MEM_BASE_SLICE_REF_WIDTH}"
  echo "rwkv_mem_online_gain=${RWKV_MEM_ONLINE_GAIN}"
  echo "rwkv_mem_memory_write_granularity=${RWKV_MEM_MEMORY_WRITE_GRANULARITY}"
  echo "delta_mem_repo=${DELTA_MEM_REPO}"
} | tee "${LOG_ROOT}/${RUN_ID}.manifest"

"${PYTHON_BIN}" pretrain.py \
  arch/net@arch=hrm_h_rwkv_mem \
  arch/size@arch="${ARCH_SIZE}" \
  data.path="${DATA_PATH}" \
  init_from_safetensors="${INIT_SAFETENSORS}" \
  trainable_param_substrings="${TRAINABLE_PARAM_SUBSTRINGS}" \
  global_batch_size="${GLOBAL_BATCH_SIZE}" \
  micro_batch_size="${MICRO_BATCH_SIZE}" \
  epochs=1 \
  max_steps="${MAX_STEPS}" \
  lr="${LR}" \
  lr_min_ratio=1.0 \
  lr_warmup_steps="${LR_WARMUP_STEPS}" \
  weight_decay=0.0 \
  ema=null \
  compile_train=false \
  checkpoint_interval=999 \
  save_checkpoints=true \
  checkpoint_path="${CKPT_DIR}" \
  run_name="${RUN_ID}" \
  loss_history_path="${LOSS_HISTORY}" \
  log_interval="${LOG_INTERVAL}" \
  arch.H_override.rwkv_mem_mode="${RWKV_MEM_MODE}" \
  arch.H_override.rwkv_mem_backend=cuda \
  arch.H_override.rwkv_mem_output_init="${RWKV_MEM_OUTPUT_INIT}" \
  arch.H_override.rwkv_mem_delta_heads="${RWKV_MEM_DELTA_HEADS}" \
  arch.H_override.rwkv_mem_rank="${RWKV_MEM_RANK}" \
  arch.H_override.rwkv_mem_num_state_heads="${RWKV_MEM_NUM_STATE_HEADS}" \
  arch.H_override.rwkv_mem_alpha="${RWKV_MEM_ALPHA}" \
  arch.H_override.rwkv_mem_beta_bias_init="${RWKV_MEM_BETA_BIAS_INIT}" \
  arch.H_override.rwkv_mem_state_update_mode="${RWKV_MEM_STATE_UPDATE_MODE}" \
  arch.H_override.rwkv_mem_separate_delta_projections="${RWKV_MEM_SEPARATE_DELTA_PROJECTIONS}" \
  arch.H_override.rwkv_mem_base_slice_ref_width="${RWKV_MEM_BASE_SLICE_REF_WIDTH}" \
  arch.H_override.rwkv_mem_online_gain="${RWKV_MEM_ONLINE_GAIN}" \
  arch.H_override.rwkv_mem_memory_write_granularity="${RWKV_MEM_MEMORY_WRITE_GRANULARITY}" \
  2>&1 | tee "${TRAIN_LOG}"

if [[ "${RUN_MMLU}" == "1" ]]; then
  MMLU_LOG="${MMLU_LOG}" LOG_ROOT="${LOG_ROOT}" MMLU_CONFIG="${MMLU_CONFIG}" \
    bash scripts/eval_rwkv_mem_mmlu.sh "${CKPT_DIR}" "${CKPT_TAG}"
  sed -n '/EVALUATION SUMMARY/,$p' "${MMLU_LOG}" | tee -a "${LOG_ROOT}/${RUN_ID}.manifest" || true
fi

echo "done run_id=${RUN_ID}"
