# RWKV-Memory Post-Train Plan

Objective: start from the original HRM-Text-1B Transformer checkpoint, add H-level delta-rule online memory, post-train on the prepared HRM dataset, and use MMLU as the primary acceptance benchmark.

## Baseline

Original teacher checkpoint:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_text_1b_teacher
```

Known MMLU baseline:

```text
MMLU acc: 0.6088
invalid: 0.0005
log: /run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/MMLU_teacher_20260605_082100.log
```

Acceptance target for this direction:

```text
new hrm_h_rwkv_mem MMLU > 0.6088
invalid <= 0.0005 preferred, invalid < 0.02 required
```

## Data

Default post-train data:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_full_v1
tokens: 176,241,004,232
```

Fast iteration slice:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_10b_v1
tokens: 10,000,000,064
```

Smoke/debug slice:

```text
/home/xiaol/X/hrm_text_subset_1B
tokens: 1,000,000,035
```

## Training Recipe

Stage A trains only the new RWKV-memory adapter:

```text
arch: hrm_h_rwkv_mem
size: XL
init: original HRM-Text-1B model.safetensors
trainable_param_substrings: [rwkv_mem]
rwkv_mem_mode: delta_rule
rwkv_mem_rank: 8
rwkv_mem_alpha: 16.0
rwkv_mem_beta_bias_init: -1.5
rwkv_mem_state_update_mode: standard
rwkv_mem_output_init: zero
rwkv_mem_delta_heads: [q, k, v, o]
rwkv_mem_separate_delta_projections: false
rwkv_mem_backend: cuda
global_batch_size: 196608
micro_batch_size: 512
lr: 2e-4
lr_warmup_steps: 20
weight_decay: 0
ema: null
compile_train: false
```

Why adapter-only first: zero-init makes the starting model identical to the original checkpoint, and adapter-only updates limit MMLU regression risk. The full active head set is `q,k,v,o`; use `q,o` as a separate release-compatible delta-Mem comparison because the public Qwen adapter is described as Q/O. The implementation has the full memory-side recipe: learned memory q/k/v, delta-rule online state update, and q/k/v/o attention deltas. With zero delta heads, memory q/k/v gradients begin after the delta heads move off zero; use `rwkv_mem_output_init: small` only for experiments that do not require exact step-0 teacher equivalence. If MMLU improves or stays close while validation CE improves, Stage B can unfreeze more H-level parameters at a lower LR.

## Launch

Default full-data post-train for 200 optimizer steps plus MMLU:

```bash
bash scripts/run_rwkv_mem_posttrain_mmlu.sh
```

Fast 10B-slice iteration:

```bash
DATA_PATH=/run/media/xiaol/B214449214445C0B/hrm_text_10b_v1 \
MAX_STEPS=200 \
bash scripts/run_rwkv_mem_posttrain_mmlu.sh
```

Comparison launch:

```bash
bash scripts/run_rwkv_qkv_vs_delta_mem_200.sh
```

This runs full delta-Mem `[q,k,v,o]`, release-compatible delta-Mem `[q,o]`, and RWKV-state memory `[q,k,v]`. The RWKV-state path is not parameter-matched yet; it trains the RWKV reader plus projection heads and is much larger than the delta-rule adapter.

Smoke without MMLU:

```bash
DATA_PATH=/home/xiaol/X/hrm_text_subset_1B \
MAX_STEPS=1 \
GLOBAL_BATCH_SIZE=512 \
MICRO_BATCH_SIZE=512 \
RUN_MMLU=0 \
bash scripts/run_rwkv_mem_posttrain_mmlu.sh
```

## Benchmark

The launch script runs:

```bash
bash scripts/eval_rwkv_mem_mmlu.sh "$CKPT_DIR" "step_$MAX_STEPS"
```

`simple_inference_engine.py` now supports `ckpt_tag=step_N`, so max-step training checkpoints can be evaluated directly without converting to `model.safetensors`.
`scripts/parse_mmlu_log.py` writes a compact JSON summary with `acc` and `invalid` for comparison against the `0.6088` teacher baseline.

MMLU uses `max_tokens=1`. The memory adapter is active during cached full-prompt prefill, so the one-token MCQ answer logits include the post-trained adapter. Single-token autoregressive decode still skips memory updates until persistent memory state caching is added; this is acceptable for MMLU but not enough for long-form generation benchmarks.

Validated smoke:

```text
run: rwkv_mem_posttrain_smoke_20260611_103842
init load: teacher checkpoint, strict=False
missing keys: rwkv_mem only
trainable: 170,188,800 / 1,352,982,528 params
checkpoint load: ckpt_tag=step_1
one-token generation: passed
next recipe default at that time: q,o legacy memory injection with separate delta projections
```

Updated implementation note:

```text
current default: rwkv_mem_mode=delta_rule
memory state: learned low-rank q/k/v with keep/erase/write delta-rule scan
active delta heads: q,o by default; k,v are supported for ablations
legacy reproduction: set rwkv_mem_mode=rwkv7_legacy for the older RWKV-7 state-reader adapter
performance caveat: the native HRM scan is currently a PyTorch loop, not the upstream Triton affine scan
```

Completed baseline check:

```text
run: rwkv_mem_posttrain_10b_s5_20260611_104539
checkpoint: step_5
adapter path: earlier shared/output-compatible RWKV memory config, before separate delta_q/delta_o default
MMLU acc: 0.6088
invalid: 0.0005
result: matched teacher baseline, did not improve it
json: /run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/rwkv_mem_posttrain/rwkv_mem_posttrain_10b_s5_20260611_104539_step_5.mmlu.json
```

Completed corrected legacy run:

```text
run: rwkv_mem_qo_sep_full_s200_20260611_111851
data: /run/media/xiaol/B214449214445C0B/hrm_text_full_v1
checkpoint_dir: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851
train_log: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_logs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851.train.log
nohup_log: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_logs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851.nohup.log
loss_history: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_logs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851.loss.jsonl
max_steps: 200
rwkv_mem_delta_heads: [q, o]
rwkv_mem_separate_delta_projections: true
trainable params: 245,686,272 / 1,428,480,000
final checkpoint: fsdp2_step_200
final train loss: 0.2211
MMLU acc: 0.6092
MMLU invalid: 0.0006
teacher MMLU acc: 0.6088
delta vs teacher: +0.0004
result: primary MMLU target met for the legacy RWKV-state adapter; margin is small
mmlu_log: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_logs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851.mmlu.log
mmlu_json: /run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851_step_200.mmlu.json
```

## Decision Rules

Continue adapter-only training if:

- validation CE improves;
- MMLU is equal to or above teacher;
- invalid rate stays below `0.02`.

Stop or change recipe if:

- MMLU drops below the teacher by more than noise after multiple checkpoints;
- invalid rate rises;
- loss improves but MMLU is flat or worse, repeating the archived H-RWKV failure mode.
