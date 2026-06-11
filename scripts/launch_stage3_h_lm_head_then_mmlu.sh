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

DEFAULT_STAGE2_CKPT="${RUN_DIR}/hrm_text_1b_to_hrm_h_rwkv7_stage2_teacherforced_layerloss_10000_b256_acc2_cuda_20260609_120703_from_stage1_20260609_130509.safetensors"
STAGE2_CKPT="${1:-${DEFAULT_STAGE2_CKPT}}"
RUN_ID="hrm_text_1b_to_hrm_h_rwkv7_stage3_h_lm_head_10000_b256_acc2_cuda_$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${RUN_DIR}/${RUN_ID}.log"

cd "${REPO_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S')" "$*" | tee -a "${LOG_PATH}"
}

prepare_eval_checkpoint() {
  local model_ckpt="$1"
  local eval_dir="${EVAL_ROOT}/${RUN_ID}"

  mkdir -p "${eval_dir}"
  cp "${EVAL_TEMPLATE_DIR}/all_config.yaml" "${eval_dir}/all_config.yaml"
  cp "${EVAL_TEMPLATE_DIR}/train_metadata.yaml" "${eval_dir}/train_metadata.yaml"
  ln -sfn "${model_ckpt}" "${eval_dir}/model.safetensors"

  sed -i \
    -e "s|^checkpoint_path: .*|checkpoint_path: ${eval_dir}|" \
    -e "s|^run_name: .*|run_name: ${RUN_ID}_eval|" \
    -e "s|^  rwkv7_backend: .*|  rwkv7_backend: cuda|" \
    "${eval_dir}/all_config.yaml"

  log "prepared eval checkpoint ${eval_dir}"
}

run_mmlu_benchmark() {
  local eval_dir="${EVAL_ROOT}/${RUN_ID}"
  local mmlu_log="${eval_dir}/mmlu.log"
  local token=""

  if [[ -r "${HF_TOKEN_FILE}" ]]; then
    token="$(cat "${HF_TOKEN_FILE}")"
  fi

  log "starting MMLU benchmark eval_dir=${eval_dir} log=${mmlu_log}"
  HF_HOME="${HF_CACHE_DIR}" \
    HF_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}" \
    HF_TOKEN="${token}" \
    .venv/bin/python -m evaluation.main \
      config="${MMLU_CONFIG}" \
      ckpt_path="${eval_dir}" \
      > "${mmlu_log}" 2>&1

  log "finished MMLU benchmark eval_dir=${eval_dir}"
  sed -n '/EVALUATION SUMMARY/,$p' "${mmlu_log}" | tee -a "${LOG_PATH}"
}

main() {
  if [[ ! -s "${STAGE2_CKPT}" ]]; then
    log "missing stage2 checkpoint ${STAGE2_CKPT}"
    return 1
  fi

  log "starting corrected stage3 run_id=${RUN_ID} init=${STAGE2_CKPT}"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    .venv/bin/python scripts/align_hrm_h_rwkv7_to_hrm_text_teacher.py \
      --stage 3 \
      --train-scope h_lm_head \
      --teacher-dir "${TEACHER_DIR}" \
      --student-init "${STAGE2_CKPT}" \
      --dataset-path "${DATASET_PATH}" \
      --output "${RUN_DIR}/${RUN_ID}.safetensors" \
      --history "${RUN_DIR}/${RUN_ID}.jsonl" \
      --summary "${RUN_DIR}/${RUN_ID}.summary.json" \
      --device cuda --dtype bf16 --rwkv7-backend cuda \
      --steps 10000 --skip-batches 40000 \
      --batch-tokens 256 --grad-accum-steps 2 --bp-steps 5 \
      --lr 7e-6 --grad-clip 1.0 --log-every 100 \
      --val-batches 4 --val-skip-batches 10000 --val-every 500 \
      --save-every 2000 --keep-last-checkpoints 2 \
      >> "${LOG_PATH}" 2>&1

  log "finished corrected stage3 run_id=${RUN_ID}"
  prepare_eval_checkpoint "${RUN_DIR}/${RUN_ID}.safetensors"
  run_mmlu_benchmark
  log "stage3 plus MMLU pipeline complete run_id=${RUN_ID}"
}

main "$@"
