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

RESUME_CKPT="${1:?resume checkpoint is required}"
RESUME_STEP="${2:-}"

if [[ -z "${RESUME_STEP}" ]]; then
  base="$(basename "${RESUME_CKPT}")"
  if [[ "${base}" =~ _step([0-9]+)\.safetensors$ ]]; then
    RESUME_STEP="${BASH_REMATCH[1]}"
  else
    echo "could not infer resume step from checkpoint name: ${RESUME_CKPT}" >&2
    exit 1
  fi
fi

if [[ ! -s "${RESUME_CKPT}" ]]; then
  echo "missing resume checkpoint: ${RESUME_CKPT}" >&2
  exit 1
fi

TOTAL_STAGE2_STEPS=10000
STAGE2_BASE_SKIP_BATCHES=20000
GRAD_ACCUM_STEPS=2
REMAINING_STAGE2_STEPS=$((TOTAL_STAGE2_STEPS - RESUME_STEP))
RESUME_SKIP_BATCHES=$((STAGE2_BASE_SKIP_BATCHES + RESUME_STEP * GRAD_ACCUM_STEPS))

if (( REMAINING_STAGE2_STEPS <= 0 )); then
  echo "resume step ${RESUME_STEP} already reaches total stage2 steps ${TOTAL_STAGE2_STEPS}" >&2
  exit 1
fi

PIPELINE_ID="hrm_text_1b_to_hrm_h_rwkv7_corrected_resume${RESUME_STEP}_$(date +%Y%m%d_%H%M%S)"
PIPELINE_LOG="${RUN_DIR}/${PIPELINE_ID}.log"
STAGE2_ID="hrm_text_1b_to_hrm_h_rwkv7_stage2_corrected_resume${RESUME_STEP}_$(date +%Y%m%d_%H%M%S)"

cd "${REPO_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${PIPELINE_LOG}"
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
  local steps="$6"
  local train_scope="$7"

  log "starting stage=${stage} run_id=${run_id} init=${init_ckpt} skip_batches=${skip_batches} steps=${steps}"
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
      --steps "${steps}" --skip-batches "${skip_batches}" \
      --batch-tokens 256 --grad-accum-steps "${GRAD_ACCUM_STEPS}" --bp-steps 5 \
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
  log "resume_ckpt=${RESUME_CKPT} resume_step=${RESUME_STEP} remaining_stage2_steps=${REMAINING_STAGE2_STEPS}"

  run_stage "2" "${RESUME_CKPT}" "${STAGE2_ID}" "1e-5" "${RESUME_SKIP_BATCHES}" "${REMAINING_STAGE2_STEPS}" "h"

  local stage2_ckpt
  stage2_ckpt="$(select_checkpoint "${STAGE2_ID}")"
  log "selected resumed stage2 checkpoint ${stage2_ckpt}"

  local stage3_id="${STAGE2_ID/stage2/stage3}_from_stage2_$(date +%Y%m%d_%H%M%S)"
  run_stage "3" "${stage2_ckpt}" "${stage3_id}" "7e-6" "40000" "10000" "h_lm_head"

  local stage3_ckpt
  stage3_ckpt="$(select_checkpoint "${stage3_id}")"
  log "selected stage3 checkpoint ${stage3_ckpt}"

  local stage3_eval_dir="${EVAL_ROOT}/${stage3_id}"
  prepare_eval_checkpoint "${stage3_id}" "${stage3_ckpt}"
  run_mmlu_benchmark "${stage3_eval_dir}"

  log "pipeline complete stage3=${stage3_id}"
}

main "$@"
