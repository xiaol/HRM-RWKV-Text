#!/usr/bin/env bash
set -euo pipefail

REPO=${REPO:-/home/xiaol/X/HRM-Text}
DISK=${DISK:-/run/media/xiaol/B214449214445C0B}
DATA=${DATA:-$DISK/hrm_text_10b_v1}
LOG_DIR=${LOG_DIR:-$DISK/hrm_text_pretrain_logs}
CKPT_DIR=${CKPT_DIR:-$DISK/hrm_text_pretrain_checkpoints/10b_compare}
WANDB_ROOT=${WANDB_ROOT:-$DISK/wandb}
LT2_PATH=${LT2_PATH:-$DISK/X_bak/LT2_upstream}
TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-$DISK/torch_extensions}

mkdir -p "$LOG_DIR" "$CKPT_DIR" "$WANDB_ROOT" "$TORCH_EXTENSIONS_DIR"

cd "$REPO"

export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_DIR="$WANDB_ROOT"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export TORCH_EXTENSIONS_DIR
export PYTHONPATH="$LT2_PATH${PYTHONPATH:+:$PYTHONPATH}"

STAMP=$(date +%Y%m%d_%H%M%S)
QUEUE_LOG="$LOG_DIR/10b_compare_queue_$STAMP.log"

run_one() {
  local arch_cfg=$1
  local run_name=$2
  local port=$3
  shift 3

  local log="$LOG_DIR/${run_name}_$STAMP.log"
  local ckpt="$CKPT_DIR/$run_name"

  {
    echo "=== START arch_cfg=${arch_cfg} run=${run_name} $(date -Is) ==="
    echo "log=${log}"
    echo "checkpoint=${ckpt}"
  } | tee -a "$QUEUE_LOG"

  local start
  start=$(date +%s)

  MASTER_PORT="$port" /usr/bin/time -p .venv/bin/python pretrain.py \
    arch/net@arch="$arch_cfg" \
    arch/size@arch=L \
    data.path="$DATA" \
    global_batch_size=172032 \
    micro_batch_size=512 \
    epochs=1 \
    checkpoint_interval=1 \
    compile_train=false \
    ema=null \
    save_checkpoints=true \
    log_interval=5 \
    run_name="$run_name" \
    checkpoint_path="$ckpt" \
    "$@" > "$log" 2>&1

  local rc=$?
  local end
  end=$(date +%s)
  echo "=== END arch_cfg=${arch_cfg} run=${run_name} rc=${rc} seconds=$((end - start)) $(date -Is) ===" | tee -a "$QUEUE_LOG"
  return "$rc"
}

run_one hrm l06_10b_transformer 29701
run_one hrm_rwkv7 l06_10b_rwkv7 29702 arch.expansion=1.0 arch.rwkv7_backend=cuda
run_one hrm_h_rwkv7 l06_10b_hybrid_h_rwkv7 29703 +arch.transformer_expansion=4.0 +arch.rwkv7_expansion=1.0 arch.rwkv7_backend=cuda
run_one hrm_l_rwkv7 l06_10b_hybrid_l_rwkv7 29704 +arch.transformer_expansion=4.0 +arch.rwkv7_expansion=1.0 arch.rwkv7_backend=cuda

echo "=== ALL DONE $(date -Is) ===" | tee -a "$QUEUE_LOG"
