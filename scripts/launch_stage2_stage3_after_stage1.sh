#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/home/xiaol/X/HRM-Text"
RUN_DIR="/run/media/xiaol/B214449214445C0B/hrm_text_migrations"
EVAL_ROOT="/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints"
EVAL_TEMPLATE_DIR="${EVAL_ROOT}/hrm_h_rwkv7_cekl_79980"
HF_CACHE_DIR="/run/media/xiaol/B214449214445C0B/hf_cache"
HF_TOKEN_FILE="/home/xiaol/.cache/huggingface/token"
TEACHER_DIR="/run/media/xiaol/B214449214445C0B/hf_models/sapientinc/HRM-Text-1B"
DATASET_PATH="/home/xiaol/X/hrm_text_subset_1B"
MMLU_CONFIG="evaluation/config/hrm_mmlu_only.yaml"

STAGE1_ID="${1:?stage1 run id is required}"
STAGE1_PID_FILE="${RUN_DIR}/${STAGE1_ID}.pid"
PIPELINE_ID="hrm_text_1b_to_hrm_h_rwkv7_stage2_stage3_cuda_$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${RUN_DIR}/${PIPELINE_ID}.log"

cd "${REPO_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${PIPELINE_LOG}"
}

wait_for_pid_file() {
  local pid_file="$1"
  while [[ ! -f "${pid_file}" ]]; do
    log "waiting for pid file ${pid_file}"
    sleep 30
  done
}

wait_for_run_exit() {
  local pid_file="$1"
  local pid
  pid="$(cat "${pid_file}")"
  while kill -0 "${pid}" 2>/dev/null; do
    log "waiting for stage1 pid=${pid}"
    sleep 60
  done
}

select_checkpoint() {
  local run_id="$1"
  local final_ckpt="${RUN_DIR}/${run_id}.safetensors"
  if [[ -s "${final_ckpt}" ]]; then
    printf '%s\n' "${final_ckpt}"
    return 0
  fi
  local latest_step
  latest_step="$(ls -1t "${RUN_DIR}/${run_id}"_step*.safetensors 2>/dev/null | head -1 || true)"
  if [[ -n "${latest_step}" && -s "${latest_step}" ]]; then
    printf '%s\n' "${latest_step}"
    return 0
  fi
  return 1
}

run_stage() {
  local stage="$1"
  local init_ckpt="$2"
  local run_id="$3"
  local lr="$4"
  local skip_batches="$5"
  local train_scope="$6"

  log "starting stage=${stage} run_id=${run_id} init=${init_ckpt}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/align_hrm_h_rwkv7_to_hrm_text_teacher.py \
      --stage "${stage}" \
      --train-scope "${train_scope}" \
      --teacher-dir "${TEACHER_DIR}" \
      --student-init "${init_ckpt}" \
      --dataset-path "${DATASET_PATH}" \
      --output "${RUN_DIR}/${run_id}.safetensors" \
      --history "${RUN_DIR}/${run_id}.jsonl" \
      --summary "${RUN_DIR}/${run_id}.summary.json" \
      --device cuda --dtype bf16 --rwkv7-backend cuda \
      --steps 10000 --skip-batches "${skip_batches}" \
      --batch-tokens 256 --grad-accum-steps 2 --bp-steps 5 \
      --lr "${lr}" --grad-clip 1.0 --log-every 100 \
      --val-batches 4 --val-skip-batches 10000 --val-every 500 \
      --save-every 2000 --keep-last-checkpoints 2 \
      >> "${RUN_DIR}/${run_id}.log" 2>&1
  log "finished stage=${stage} run_id=${run_id}"
}

prepare_eval_checkpoint() {
  local run_id="$1"
  local model_ckpt="$2"
  local eval_dir="${EVAL_ROOT}/${run_id}"

  if [[ ! -f "${EVAL_TEMPLATE_DIR}/all_config.yaml" || ! -f "${EVAL_TEMPLATE_DIR}/train_metadata.yaml" ]]; then
    log "missing eval template files in ${EVAL_TEMPLATE_DIR}"
    return 1
  fi

  mkdir -p "${eval_dir}"
  cp "${EVAL_TEMPLATE_DIR}/all_config.yaml" "${eval_dir}/all_config.yaml"
  cp "${EVAL_TEMPLATE_DIR}/train_metadata.yaml" "${eval_dir}/train_metadata.yaml"
  ln -sfn "${model_ckpt}" "${eval_dir}/model.safetensors"

  sed -i \
    -e "s|^checkpoint_path: .*|checkpoint_path: ${eval_dir}|" \
    -e "s|^run_name: .*|run_name: ${run_id}_eval|" \
    -e "s|^  rwkv7_backend: .*|  rwkv7_backend: cuda|" \
    "${eval_dir}/all_config.yaml"

  log "prepared eval checkpoint ${eval_dir}"
}

run_mmlu_benchmark() {
  local eval_dir="$1"
  local log_path="${eval_dir}/mmlu.log"
  local token=""

  if [[ -r "${HF_TOKEN_FILE}" ]]; then
    token="$(cat "${HF_TOKEN_FILE}")"
  fi

  log "starting MMLU benchmark eval_dir=${eval_dir} log=${log_path}"
  HF_HOME="${HF_CACHE_DIR}" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    HF_TOKEN="${token}" \
    .venv/bin/python -m evaluation.main \
      config="${MMLU_CONFIG}" \
      ckpt_path="${eval_dir}" \
      > "${log_path}" 2>&1

  log "finished MMLU benchmark eval_dir=${eval_dir}"
  sed -n '/EVALUATION SUMMARY/,$p' "${log_path}" | tee -a "${PIPELINE_LOG}"
}

main() {
  log "pipeline=${PIPELINE_ID}"
  wait_for_pid_file "${STAGE1_PID_FILE}"
  wait_for_run_exit "${STAGE1_PID_FILE}"

  local stage1_ckpt
  stage1_ckpt="$(select_checkpoint "${STAGE1_ID}")"
  log "selected stage1 checkpoint ${stage1_ckpt}"

  local stage2_id="${STAGE1_ID/stage1/stage2}_from_stage1_$(date +%Y%m%d_%H%M%S)"
  run_stage "2" "${stage1_ckpt}" "${stage2_id}" "1e-5" "20000" "h"

  local stage2_ckpt
  stage2_ckpt="$(select_checkpoint "${stage2_id}")"
  log "selected stage2 checkpoint ${stage2_ckpt}"

  local stage3_id="${stage2_id/stage2/stage3}_from_stage2_$(date +%Y%m%d_%H%M%S)"
  run_stage "3" "${stage2_ckpt}" "${stage3_id}" "7e-6" "40000" "h_lm_head"

  local stage3_ckpt
  stage3_ckpt="$(select_checkpoint "${stage3_id}")"
  log "selected stage3 checkpoint ${stage3_ckpt}"

  local stage3_eval_dir="${EVAL_ROOT}/${stage3_id}"
  prepare_eval_checkpoint "${stage3_id}" "${stage3_ckpt}"
  run_mmlu_benchmark "${stage3_eval_dir}"

  log "pipeline complete stage3=${stage3_id}"
}

main "$@"
