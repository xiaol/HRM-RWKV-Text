#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
DISK="${DISK:-/run/media/xiaol/B214449214445C0B}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

DATA_PATH="${DATA_PATH:-${DISK}/hrm_text_full_v1}"
TEACHER_SAFETENSORS="${TEACHER_SAFETENSORS:-${DISK}/hrm_text_eval_checkpoints/hrm_text_1b_teacher/model.safetensors}"
BOOTSTRAP_DIR="${BOOTSTRAP_DIR:-${DISK}/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851}"
RUN_ID="${RUN_ID:-rwkv_mem_qo_sep_spare_600m}"
CKPT_DIR="${CKPT_DIR:-${DISK}/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/${RUN_ID}}"
LOG_DIR="${LOG_DIR:-${DISK}/hrm_text_pretrain_logs/rwkv_mem_posttrain/${RUN_ID}}"
PID_FILE="${PID_FILE:-${LOG_DIR}/train.pid}"
LOSS_HISTORY="${LOSS_HISTORY:-${LOG_DIR}/loss.jsonl}"

GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-196608}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-512}"
TARGET_TOKENS="${TARGET_TOKENS:-600000000}"
TARGET_STEPS="${TARGET_STEPS:-$(((TARGET_TOKENS + GLOBAL_BATCH_SIZE - 1) / GLOBAL_BATCH_SIZE))}"
SESSION_STEPS="${SESSION_STEPS:-100}"
SECONDS_PER_STEP="${SECONDS_PER_STEP:-46.5}"
LR="${LR:-2e-4}"
LR_WARMUP_STEPS="${LR_WARMUP_STEPS:-20}"

latest_step_in() {
  local directory="$1"
  if [[ ! -d "${directory}" ]]; then
    return 0
  fi
  find "${directory}" -maxdepth 1 -type d -name 'fsdp2_step_*' -printf '%f\n' 2>/dev/null \
    | sed 's/^fsdp2_step_//' | sort -n | tail -1 || true
}

is_running() {
  [[ -s "${PID_FILE}" ]] || return 1
  local pid
  pid="$(cat "${PID_FILE}")"
  kill -0 "${pid}" 2>/dev/null
}

current_step() {
  local step
  step="$(latest_step_in "${CKPT_DIR}")"
  if [[ -n "${step}" ]]; then
    printf '%s\n' "${step}"
    return
  fi
  step="$(latest_step_in "${BOOTSTRAP_DIR}")"
  printf '%s\n' "${step:-0}"
}

latest_log() {
  if [[ ! -d "${LOG_DIR}" ]]; then
    return 0
  fi
  find "${LOG_DIR}" -maxdepth 1 -type f -name 'session_*.log' -printf '%p\n' 2>/dev/null \
    | sort | tail -1 || true
}

show_status() {
  local step trained_tokens remaining_steps remaining_seconds log_path
  step="$(current_step)"
  trained_tokens=$((step * GLOBAL_BATCH_SIZE))
  remaining_steps=$((TARGET_STEPS > step ? TARGET_STEPS - step : 0))
  remaining_seconds="$("${PYTHON_BIN}" -c "print(round(${remaining_steps} * ${SECONDS_PER_STEP}))")"
  log_path="$(latest_log)"

  if is_running; then
    echo "state=running pid=$(cat "${PID_FILE}")"
  else
    echo "state=stopped"
  fi
  echo "checkpoint_dir=${CKPT_DIR}"
  echo "step=${step}/${TARGET_STEPS}"
  echo "trained_tokens=${trained_tokens}/${TARGET_TOKENS}"
  echo "remaining_steps=${remaining_steps}"
  echo "estimated_remaining_seconds=${remaining_seconds}"
  if [[ -n "${log_path}" ]]; then
    echo "latest_log=${log_path}"
  fi
}

start_training() {
  if is_running; then
    echo "training is already running with pid=$(cat "${PID_FILE}")" >&2
    exit 1
  fi

  mkdir -p "${CKPT_DIR}" "${LOG_DIR}"
  rm -f "${PID_FILE}"

  local step session_end resume_dir resume_tag timestamp log_path manifest_path
  step="$(current_step)"
  if (( step >= TARGET_STEPS )); then
    echo "target already reached: step=${step} target=${TARGET_STEPS}"
    exit 0
  fi

  session_end=$((step + SESSION_STEPS))
  if (( session_end > TARGET_STEPS )); then
    session_end="${TARGET_STEPS}"
  fi

  resume_dir=""
  resume_tag=""
  if [[ -n "$(latest_step_in "${CKPT_DIR}")" ]]; then
    resume_dir="${CKPT_DIR}"
    resume_tag="step_${step}"
  elif [[ -d "${BOOTSTRAP_DIR}/fsdp2_step_${step}" && "${step}" -gt 0 ]]; then
    resume_dir="${BOOTSTRAP_DIR}"
    resume_tag="step_${step}"
  fi

  timestamp="$(date +%Y%m%d_%H%M%S)"
  log_path="${LOG_DIR}/session_${timestamp}_step${step}_to${session_end}.log"
  manifest_path="${LOG_DIR}/session_${timestamp}_step${step}_to${session_end}.manifest"

  local -a args=(
    pretrain.py
    arch/net@arch=hrm_h_rwkv_mem
    arch/size@arch=XL
    "data.path=${DATA_PATH}"
    "trainable_param_substrings=[rwkv_mem]"
    "global_batch_size=${GLOBAL_BATCH_SIZE}"
    "micro_batch_size=${MICRO_BATCH_SIZE}"
    epochs=1
    "max_steps=${session_end}"
    "lr=${LR}"
    lr_min_ratio=1.0
    "lr_warmup_steps=${LR_WARMUP_STEPS}"
    weight_decay=0.0
    ema=null
    compile_train=false
    checkpoint_interval=999
    save_checkpoints=true
    "checkpoint_path=${CKPT_DIR}"
    "run_name=${RUN_ID}_${timestamp}"
    "loss_history_path=${LOSS_HISTORY}"
    log_interval=1
    resume_skip_data=true
    arch.H_override.rwkv_mem_backend=cuda
    "arch.H_override.rwkv_mem_delta_heads=[q,o]"
    arch.H_override.rwkv_mem_separate_delta_projections=true
  )

  if [[ -n "${resume_dir}" ]]; then
    args+=("resume_from=${resume_dir}" "resume_tag=${resume_tag}")
  else
    if [[ ! -s "${TEACHER_SAFETENSORS}" ]]; then
      echo "missing teacher checkpoint: ${TEACHER_SAFETENSORS}" >&2
      exit 1
    fi
    args+=("init_from_safetensors=${TEACHER_SAFETENSORS}")
  fi

  {
    echo "run_id=${RUN_ID}"
    echo "start_step=${step}"
    echo "session_end_step=${session_end}"
    echo "target_steps=${TARGET_STEPS}"
    echo "target_tokens=${TARGET_TOKENS}"
    echo "resume_from=${resume_dir:-teacher_safetensors}"
    echo "resume_tag=${resume_tag:-none}"
    echo "checkpoint_dir=${CKPT_DIR}"
    echo "log=${log_path}"
  } > "${manifest_path}"

  cd "${ROOT_DIR}"
  nohup setsid env \
    WANDB_MODE="${WANDB_MODE:-offline}" \
    PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    "${PYTHON_BIN}" "${args[@]}" >> "${log_path}" 2>&1 &
  echo "$!" > "${PID_FILE}"

  echo "started pid=$(cat "${PID_FILE}") step=${step}->${session_end}"
  echo "log=${log_path}"
  echo "stop safely: bash scripts/rwkv_mem_spare_train.sh stop"
}

stop_training() {
  if ! is_running; then
    rm -f "${PID_FILE}"
    echo "training is not running"
    show_status
    return
  fi

  local pid
  pid="$(cat "${PID_FILE}")"
  echo "requesting safe stop for pid=${pid}; waiting for checkpoint..."
  kill -TERM "${pid}"

  local waited=0
  while kill -0 "${pid}" 2>/dev/null; do
    sleep 2
    waited=$((waited + 2))
    if (( waited >= 240 )); then
      echo "still stopping after ${waited}s; do not reboot until status reports stopped" >&2
      exit 1
    fi
  done
  rm -f "${PID_FILE}"
  echo "safe stop complete"
  show_status
}

evaluate_latest() {
  if is_running; then
    echo "stop training before evaluation" >&2
    exit 1
  fi
  local step
  step="$(latest_step_in "${CKPT_DIR}")"
  if [[ -z "${step}" ]]; then
    echo "no scaled checkpoint found in ${CKPT_DIR}" >&2
    exit 1
  fi
  cd "${ROOT_DIR}"
  bash scripts/eval_rwkv_mem_mmlu.sh "${CKPT_DIR}" "step_${step}"
}

command="${1:-status}"
case "${command}" in
  start|resume)
    start_training
    ;;
  stop)
    stop_training
    ;;
  status)
    if [[ -s "${PID_FILE}" ]] && ! is_running; then
      rm -f "${PID_FILE}"
    fi
    show_status
    ;;
  log)
    log_path="$(latest_log)"
    if [[ -z "${log_path}" ]]; then
      echo "no session log found" >&2
      exit 1
    fi
    tail -n "${LINES:-40}" "${log_path}"
    ;;
  eval)
    evaluate_latest
    ;;
  *)
    echo "usage: $0 {start|resume|stop|status|log|eval}" >&2
    exit 2
    ;;
esac
