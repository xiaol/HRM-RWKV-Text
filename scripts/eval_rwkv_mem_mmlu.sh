#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." >/dev/null 2>&1 && pwd)}"
DISK="${DISK:-/run/media/xiaol/B214449214445C0B}"
PYTHON_BIN="${PYTHON_BIN:-${ROOT_DIR}/.venv/bin/python}"

CKPT_DIR="${1:?usage: scripts/eval_rwkv_mem_mmlu.sh CKPT_DIR [CKPT_TAG]}"
CKPT_TAG="${2:-}"
if [[ -z "${CKPT_TAG}" ]]; then
  latest_step="$(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'fsdp2_step_*' -printf '%f\n' 2>/dev/null | sed 's/^fsdp2_//' | sort -V | tail -1)"
  latest_epoch="$(find "${CKPT_DIR}" -maxdepth 1 -type d -name 'fsdp2_epoch_*' -printf '%f\n' 2>/dev/null | sed 's/^fsdp2_//' | sort -V | tail -1)"
  CKPT_TAG="${latest_step:-${latest_epoch:-}}"
fi
if [[ -z "${CKPT_TAG}" ]]; then
  echo "could not infer checkpoint tag from ${CKPT_DIR}" >&2
  exit 1
fi

RUN_ID="$(basename "${CKPT_DIR}")_${CKPT_TAG}"
LOG_ROOT="${LOG_ROOT:-${DISK}/hrm_text_eval_runs/rwkv_mem_posttrain}"
MMLU_CONFIG="${MMLU_CONFIG:-evaluation/config/hrm_mmlu_only.yaml}"
MMLU_LOG="${MMLU_LOG:-${LOG_ROOT}/${RUN_ID}.mmlu.log}"
MMLU_JSON="${MMLU_JSON:-${LOG_ROOT}/${RUN_ID}.mmlu.json}"

mkdir -p "${LOG_ROOT}"
cd "${ROOT_DIR}"
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false

"${PYTHON_BIN}" -m evaluation.main \
  config="${MMLU_CONFIG}" \
  ckpt_path="${CKPT_DIR}" \
  ckpt_tag="${CKPT_TAG}" \
  ckpt_use_ema=false \
  2>&1 | tee "${MMLU_LOG}"

"${PYTHON_BIN}" scripts/parse_mmlu_log.py "${MMLU_LOG}" --json-out "${MMLU_JSON}"
echo "mmlu_json=${MMLU_JSON}"
