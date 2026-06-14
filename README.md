# HRM-RWKV-Text

This repository is an experimental fork of HRM-Text that replaces or mixes the HRM recurrent Transformer cores with RWKV-7.

Original upstream project:

```text
https://github.com/sapientinc/HRM-Text
```

Refer to upstream HRM-Text for the original paper, architecture, training framework, evaluation stack, and license context. This fork is focused on local architecture validation and speed/loss comparison for RWKV-7 inside HRM-Text.

Fork repository:

```text
https://github.com/xiaol/HRM-RWKV-Text
```

## Status

The direct HRM-H Transformer to RWKV-7 migration direction is archived as of 2026-06-11.

Best MMLU result from the early hidden-state alignment checkpoint was `0.3048` with `0.1175` invalid. The later corrected PrefixLM-masked CE/KL stage3 run improved invalid answers to `0.0126`, but MMLU dropped to `0.2945`. The HRM-Text-1B teacher reference was `0.6088`.

Conclusion: continuing the same stage3 recipe is not recommended. The implementation and logs remain in this repo for reference, but this direction is no longer the active path.

Detailed archive note:

```text
docs/archive_hrm_h_rwkv7_alignment.md
```

Current accepted direction:

```text
docs/next_delta_mem_direction.md
docs/rwkv_mem_posttrain_mmlu_plan.md
```

## What Changed

Added HRM core variants:

| config / benchmark name | H core | L core |
| --- | --- | --- |
| `transformer` | Transformer | Transformer |
| `rwkv7` / `hrm_rwkv7` | RWKV-7 | RWKV-7 |
| `hybrid_h_rwkv7` / `hrm_h_rwkv7` | RWKV-7 | Transformer |
| `hybrid_l_rwkv7` / `hrm_l_rwkv7` | Transformer | RWKV-7 |
| `rwkv_mem` / `hrm_h_rwkv_mem` | Transformer + delta-rule memory | Transformer |
| `hrm_l_rwkv_mem` | Transformer | Transformer + delta-rule memory |
| `hrm_hl_rwkv_mem` | Transformer + delta-rule memory | Transformer + delta-rule memory |

Key files:

```text
models/rwkv7.py
models/baselines/hrm_rwkv7_nocarry_bp_warmup.py
models/baselines/hrm_hybrid_rwkv7_nocarry_bp_warmup.py
models/rwkv_memory.py
config/arch/net/hrm_rwkv7.yaml
config/arch/net/hrm_h_rwkv7.yaml
config/arch/net/hrm_l_rwkv7.yaml
config/arch/net/hrm_h_rwkv_mem.yaml
scripts/benchmark_hrm_rwkv7.py
scripts/prepare_hf_subset_data.py
```

The RWKV-7 implementation can use LT2 CUDA kernels via:

```bash
PYTHONPATH=/path/to/LT2_upstream
```

For the full RWKV-7 CUDA path, use:

```text
dtype=bf16
rwkv7_backend=cuda
rwkv7_head_size=64
rwkv7_expansion=1.0
```

The RWKV path calls the LT2 kernels for time mix, recurrence, layernorm/RKV residual/gate, and channel mix.

## Official 1B-Token Subset

The full HRM-Text cleaned pretraining dataset is large, so this fork includes a compact subset builder. It streams rows from:

```text
sapientinc/HRM-Text-data-io-cleaned-20260515
```

and writes HRM `V1Dataset` format using compact `uint16` token storage.

Example command:

```bash
HF_HOME=/home/xiaol/.cache/huggingface \
.venv/bin/python scripts/prepare_hf_subset_data.py \
  --hf-dataset sapientinc/HRM-Text-data-io-cleaned-20260515 \
  --split train \
  --streaming \
  --tokenizer outputs/hrm_official_assets/tokenizer.json \
  --output /home/xiaol/X/hrm_text_subset_1B \
  --epochs 1 \
  --context-size 4097 \
  --target-tokens 1000000000 \
  --compact-uint16
```

Local generated subset used for the benchmark:

```text
tokens: 1,000,000,035
rows: 26,620,178
disk: 2.7 GB
```

Generated datasets and benchmark outputs are intentionally not committed.

## Benchmark

Small 40M validation benchmark command:

```bash
PYTHONPATH=/home/xiaol/X/LT2_upstream \
.venv/bin/python scripts/benchmark_hrm_rwkv7.py \
  --mode v1 \
  --device cuda \
  --dtype bf16 \
  --archs transformer,rwkv7,hybrid_h_rwkv7,hybrid_l_rwkv7 \
  --warmup-steps 3 \
  --steps 30 \
  --v1-batch-tokens 4096 \
  --v1-eval-batch-tokens 4096 \
  --v1-val-batches 10 \
  --seq-len 4096 \
  --hidden-size 256 \
  --n-layers 4 \
  --num-heads 4 \
  --transformer-expansion 4.0 \
  --rwkv7-expansion 1.0 \
  --h-cycles 2 \
  --l-cycles 2 \
  --bp-steps 3 \
  --vocab-size 65536 \
  --rwkv7-head-size 64 \
  --rwkv7-backend cuda \
  --json-out outputs/hrm_official_1b_v1_compare_4090_h256_l4_s30_packed_rwkv.json
```

Current result after batching packed RWKV sequences:

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 40.89M | 8,578 | 1,132 | 5.3036 | 3.6450 | 3.7159 | 3.96 GB |
| `rwkv7` | 40.53M | 76,873 | 10,149 | 4.9837 | 3.5006 | 3.5359 | 6.39 GB |
| `hybrid_h_rwkv7` | 40.71M | 16,010 | 2,114 | 4.8157 | 3.3210 | 3.4045 | 5.60 GB |
| `hybrid_l_rwkv7` | 40.71M | 15,242 | 2,012 | 4.9637 | 3.4560 | 3.5877 | 4.77 GB |

This is a short training-process validation run, not a final model-quality result.

## Delta-Rule Memory Adapter

The active post-archive direction keeps the HRM Transformer H/L cores and adds an optional delta-rule online memory path inside `Attention`:

```text
memory_q, memory_k, memory_v = learned_memory_projections(hidden)
read_t = S_{t-1} memory_q_t
S_t = delta_rule_update(S_{t-1}, memory_k_t, memory_v_t)
query = query + delta_q(read)
key = key + delta_k(read)      # optional
value = value + delta_v(read)  # optional
output = attention_out + delta_o(read)
```

For `rwkv_mem_output_init=zero`, the adapter starts as an exact Transformer baseline: all delta heads are zero and adapter construction restores RNG state so later Transformer layers initialize identically. New full-recipe runs use `rwkv_mem_mode: delta_rule` and `rwkv_mem_delta_heads: [q, k, v, o]`. For release-compatible delta-Mem comparison, set `rwkv_mem_delta_heads: [q, o]`, matching the Q/O public Qwen adapter.

The RWKV-state comparison has two modes. `rwkv_mem_mode: rwkv7` reads from an RWKV-7 state path and projects that readout into q/k/v/o injection heads. `rwkv_mem_mode: rwkv7_legacy` keeps the older minimal q/o-compatible path for reproducing the accepted `step_200` run. Neither RWKV-state mode is the same as delta-Mem: the full delta-Mem recipe is the `delta_rule` associative state with learned memory q/k/v writes.

### Method Details

The adapter is intentionally not a replacement of the HRM Transformer core. The direct H-level Transformer-to-RWKV migration was archived because hidden-state alignment and CE/KL continuation did not preserve MMLU. This method keeps the original HRM-Text-1B Transformer weights frozen and adds a trainable online memory path only inside H-level attention.

The forward pass is:

```text
x_norm = norm(hidden)
gate, q0, k, v = transformer_qkv(x_norm)
mem_q, mem_k, mem_v = memory_projections(x_norm)
read = associative_state_read_before_write(mem_q)
state = delta_rule_update(state, mem_k, mem_v)
q = q0 + delta_q(read)
k = k + delta_k(read)     # if enabled
v = v + delta_v(read)     # if enabled
attn = attention(q, k, v)
out = o_proj(gate * attn) + delta_o(read)
```

Important implementation choices:

```text
rwkv_mem_mode: delta_rule
rwkv_mem_rank: 8
rwkv_mem_alpha: 16.0
rwkv_mem_beta_bias_init: -1.5
rwkv_mem_output_init: zero
rwkv_mem_delta_heads: [q, k, v, o]
rwkv_mem_separate_delta_projections: false
rwkv_mem_backend: cuda
trainable_param_substrings: [rwkv_mem]
```

Zero initialization makes the first forward pass exactly match the teacher checkpoint. The delta heads begin from no-op outputs; memory q/k/v and gate gradients start flowing after the delta heads move off zero. Use `rwkv_mem_output_init: small` only when immediate memory-gradient flow is worth losing exact teacher equivalence.

The `q` path is the main difference from an output-only residual adapter. Adding `delta_q(read)` before RoPE and attention changes the attention distribution itself, so the online memory state can steer which historical tokens the Transformer attends to. Full `[q,k,v,o]` additionally changes the written key/value content that attention consumes, then adds `delta_o(read)` after attention as a direct memory residual.

The current implementation runs memory during full-prompt prefill. That is enough for MMLU in this repo because MMLU is a one-token multiple-choice generation benchmark: the answer logits come from the prompt prefill. Long-form autoregressive generation still needs persistent memory decode state before it should be treated as a target benchmark.

Performance caveat: the HRM-native delta-rule path can use the upstream Triton affine-scan kernel if the local `deltamem` package is importable; otherwise it falls back to a PyTorch CUDA token loop. RWKV7 memory uses the repo's LT2 RWKV kernels when `rwkv_mem_backend=cuda`.

First H-only V1 comparison on RTX 4090:

```bash
.venv/bin/python scripts/benchmark_hrm_rwkv7.py \
  --mode v1 \
  --archs transformer,rwkv_mem \
  --device cuda \
  --dtype bf16 \
  --warmup-steps 2 \
  --steps 20 \
  --v1-batch-tokens 2048 \
  --v1-eval-batch-tokens 2048 \
  --v1-val-batches 10 \
  --seq-len 4096 \
  --hidden-size 256 \
  --n-layers 4 \
  --half-layers \
  --num-heads 4 \
  --expansion 4.0 \
  --h-cycles 2 \
  --l-cycles 3 \
  --bp-steps 5 \
  --vocab-size 65536 \
  --rwkv-mem-mode delta_rule \
  --rwkv-mem-rank 8 \
  --rwkv-mem-output-init zero \
  --json-out outputs/rwkv_mem_v1_compare_zero.json
```

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 37.22M | 10,299 | 1,429 | 6.6394 | 4.8207 | 4.8887 | 2.04 GB |
| `rwkv_mem` | 37.89M | 10,523 | 1,460 | 6.6272 | 4.8087 | 4.8785 | 2.41 GB |

This is the earlier legacy RWKV-state-reader validation run. The current delta-rule memory implementation supersedes it; rerun this benchmark before comparing speed/loss against the new default.

### Post-Train To MMLU Target

The current target is no longer just validation CE. Start from the original HRM-Text-1B checkpoint, post-train the new H-level delta-rule memory adapter on the prepared HRM dataset, and use MMLU as the acceptance metric.

Baseline:

```text
teacher checkpoint: /run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_text_1b_teacher
teacher MMLU: 0.6088
```

Best legacy post-train result:

| model | MMLU | invalid | delta vs teacher |
| --- | ---: | ---: | ---: |
| HRM-Text-1B teacher | 0.6088 | 0.0005 | - |
| H legacy RWKV-state memory `step_200` | 0.6092 | 0.0006 | +0.0004 |

```text
checkpoint: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851
ckpt_tag: step_200
```

Result artifacts:

```text
MMLU log: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_logs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851.mmlu.log
MMLU JSON: /run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851_step_200.mmlu.json
```

Primary launch:

```bash
bash scripts/run_rwkv_mem_posttrain_mmlu.sh
```

Default recipe:

```text
data: /run/media/xiaol/B214449214445C0B/hrm_text_full_v1
init: /run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_text_1b_teacher/model.safetensors
trainable params: rwkv_mem only
arch size: XL / HRM-Text-1B shape
rwkv_mem_delta_heads: [q, k, v, o]
rwkv_mem_mode: delta_rule
rwkv_mem_rank: 8
rwkv_mem_alpha: 16.0
rwkv_mem_beta_bias_init: -1.5
rwkv_mem_separate_delta_projections: false
global_batch_size: 196608
micro_batch_size: 512
gradient accumulation: 384
optimizer steps: 200
training-token exposure: 39,321,600
trainable adapter params: 1,572,992
total params: 1,184,366,720
lr: 2e-4
MMLU target: > 0.6088
```

Current 200-step comparison launcher:

```bash
bash scripts/run_rwkv_qkv_vs_delta_mem_200.sh
```

It runs:

```text
delta_rule [q,k,v,o]   full HRM delta-Mem recipe, about 1.57M trainable params
delta_rule [q,o]       release-compatible delta-Mem comparison, same memory q/k/v state
rwkv7 [q,k,v,o]        RWKV-state memory comparison, about 321M trainable params
```

The RWKV-state comparison is intentionally not parameter-matched with delta-Mem yet. It trains the RWKV7 reader plus q/k/v/o projections, so it answers a different question: whether an RWKV recurrent state can act as a stronger memory adapter. A later fair-size comparison should reduce or freeze the RWKV reader.

The run reads from the complete prepared `176.24B`-token corpus, but the 200-step experiment is only a `39.3M`-token continuation, not a full epoch over that corpus.

For quick iteration:

```bash
DATA_PATH=/run/media/xiaol/B214449214445C0B/hrm_text_10b_v1 \
MAX_STEPS=200 \
bash scripts/run_rwkv_mem_posttrain_mmlu.sh
```

MMLU is a one-token MCQ generation benchmark in this repo. `rwkv_mem` is active during cached full-prompt prefill, so MMLU logits use the adapter. Long-form generation still needs persistent memory state for decode before it should be used as a target.

### Reflection

The useful lesson is that preserving the Transformer backbone matters more than forcing a full architectural swap. Directly replacing H with RWKV-7 created a much larger distribution shift and did not recover MMLU, even when hidden-state losses looked good. The memory-adapter route is smaller and more conservative: it starts from exactly the teacher behavior, then lets online memory learn a side-channel transformation.

The legacy RWKV-state result is technically positive but small. It improved MMLU from `0.6088` to `0.6092`, with invalid rate `0.0006`. This clears the stated benchmark target, but it is not yet evidence for the new full delta-rule memory path. Treat it as a working proof that the training path, checkpoint loading, adapter-only freezing, and MMLU gate are valid.

The next serious iteration should focus on robustness rather than only chasing a single score: repeat the `step_200` recipe with another seed, try lower LR such as `1e-4`, and evaluate intermediate checkpoints if MMLU begins to regress after CE improves. Persistent decode state is also required before using this adapter for long-form generation metrics.

### Spare-Time Resumable Scaling

Use the spare-time controller to train the current delta-rule memory run toward `0.6B` training tokens. It runs in the background and uses the 2 TB SSD for checkpoints and logs:

```bash
bash scripts/rwkv_mem_spare_train.sh start
bash scripts/rwkv_mem_spare_train.sh status
bash scripts/rwkv_mem_spare_train.sh log
bash scripts/rwkv_mem_spare_train.sh stop
bash scripts/rwkv_mem_spare_train.sh resume
```

`stop` sends a graceful termination request. The trainer finishes the active optimizer step, writes `fsdp2_step_N` with model and optimizer state, and exits. Do not reboot until `stop` reports `safe stop complete`. `resume` loads the newest step checkpoint and skips the already-consumed data batches, so it continues the same training stream instead of replaying the beginning.

Defaults:

```text
bootstrap: original HRM-Text-1B teacher checkpoint by default
target: 600,000,000 tokens
target optimizer step: 3,052
session length: 100 optimizer steps; re-estimate after a delta-rule scan speed benchmark
checkpoint/log storage: /run/media/xiaol/B214449214445C0B
```

Each `start` or `resume` runs one session. A session can end naturally or be stopped early. Change its planned duration without changing the final target:

```bash
SESSION_STEPS=25 bash scripts/rwkv_mem_spare_train.sh resume
```

Change the final scaling target:

```bash
TARGET_TOKENS=1000000000 bash scripts/rwkv_mem_spare_train.sh resume
```

Run MMLU manually on the latest completed checkpoint:

```bash
bash scripts/rwkv_mem_spare_train.sh eval
```

Step checkpoints are large, about `3.6 GB` for the accepted run, so old non-milestone checkpoints should be removed periodically only after a newer checkpoint and its MMLU result have been verified.

## Original HRM-Text Evaluation

The upstream evaluation entrypoint is kept. For this fork, package a checkpoint directory with:

```text
all_config.yaml
train_metadata.yaml
model.safetensors
```

Prepared local aligned H-RWKV checkpoint:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_h_rwkv7_aligned_5000
```

4090 smoke run:

```bash
.venv/bin/python -m evaluation.main \
  config=evaluation/config/hrm_boolq_smoke.yaml \
  ckpt_path="/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_h_rwkv7_aligned_5000"
```

Original benchmark command style:

```bash
.venv/bin/python -m evaluation.main \
  ckpt_path="/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_h_rwkv7_aligned_5000"
```

The default upstream evaluation config targets a much larger GPU than a 4090 for the CoT/freeform groups. On 4090, use `generation_config.batch_size=1` or the smoke config first, then run the full suite only if memory and runtime are acceptable.

Completed local original-evaluator BoolQ run:

```bash
.venv/bin/python -m evaluation.main \
  ckpt_path="/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_h_rwkv7_aligned_5000" \
  run_only=[BoolQ]
```

| benchmark | n | acc | invalid | generation time |
| --- | ---: | ---: | ---: | ---: |
| BoolQ | 3270 | 0.6873 | 0.1734 | 5m50s |

## Local 0.6B-Size Baseline

For the upstream HRM-Text L/0.6B shape on a local RTX 4090, use:

```text
hidden_size=1280
n_layers=24
half_layers=true
num_heads=10
H_cycles=2
L_cycles=3
bp_steps=5
```

The local 4090 comparison uses the 1B-token official subset and a common `1024` packed-token microbatch. Pure RWKV-7 needs `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and OOMs at `2048` packed tokens in this L-size training benchmark.

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 694.68M | 1,691 | 233 | 4.2500 | 3.6556 | 3.2188 | 8.57 GB |
| `rwkv7` | 667.62M | 3,662 | 504 | 4.0606 | 3.5916 | 3.2192 | 21.28 GB |
| `hybrid_h_rwkv7` | 681.15M | 2,037 | 280 | 3.8253 | 3.5945 | 3.1869 | 13.64 GB |
| `hybrid_l_rwkv7` | 681.15M | 2,643 | 364 | 4.3985 | 3.6666 | 3.2319 | 16.21 GB |

This is still a short validation run, not full 1B-token pretraining.

For actual local training with the upstream L effective batch, keep `global_batch_size=172032` and use gradient accumulation with the 4090-safe pretrain microbatch. For RWKV-size matching, keep RWKV channel expansion at `1.0`; using the L-size default expansion `4.0` makes RWKV much larger and disables the LT2 channel kernel eligibility.

```bash
WANDB_MODE=offline \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/run/media/xiaol/B214449214445C0B/X_bak/LT2_upstream \
.venv/bin/python pretrain.py \
  arch/net@arch=hrm_rwkv7 \
  arch/size@arch=L \
  data.path=/home/xiaol/X/hrm_text_subset_1B \
  global_batch_size=172032 \
  micro_batch_size=512 \
  epochs=1 \
  max_steps=1 \
  checkpoint_interval=999 \
  compile_train=false \
  ema=null \
  arch.expansion=1.0 \
  arch.rwkv7_backend=cuda \
  run_name=hrm_rwkv7_l_1b_subset_b172k_micro512
```

That gives `172032 / 512 = 336` gradient-accumulation microsteps per optimizer step on one 4090. `ema=null` is used for the local 4090 comparison to keep optimizer state within 24 GB.

## Local 10B Comparison

A lightweight 10B subset is prepared at:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_10b_v1
```

It reuses the full `tokens.bin` by symlink and stores only sliced V1 index arrays:

```text
rows: 29,095,087
tokens: 10,000,000,064
disk: 1.1 GB plus symlinked tokens.bin
```

Launch the sequential 10B comparison queue with:

```bash
scripts/run_10b_pretrain_compare.sh
```

The queue runs:

| arch | params | important overrides |
| --- | ---: | --- |
| `transformer` | 694.68M | L default expansion `4.0` |
| `rwkv7` | 667.62M | `arch.expansion=1.0`, `arch.rwkv7_backend=cuda` |
| `hybrid_h_rwkv7` | 681.15M | `+arch.transformer_expansion=4.0`, `+arch.rwkv7_expansion=1.0` |
| `hybrid_l_rwkv7` | 681.15M | `+arch.transformer_expansion=4.0`, `+arch.rwkv7_expansion=1.0` |

Measured one-step `bp_steps=5` local 4090 calibration:

| arch | step time | rough 10B time |
| --- | ---: | ---: |
| `transformer` | 41.8s | 27 days with BP warmup |
| `rwkv7` | 56.8s | 37 days with BP warmup |
| `hybrid_h_rwkv7` | 45.8s | 30 days with BP warmup |
| `hybrid_l_rwkv7` | 52.3s | 34 days with BP warmup |

## Speed Notes

The first RWKV implementation looped over PrefixLM-packed sequences one by one, which caused many tiny kernel launches. `RWKV7Stack` now pads packed `[T, C]` batches into `[numseqs, max_seq_len, C]`, runs the RWKV stack once, and scatters back to `[T, C]`.

Speedup on the same benchmark:

| arch | speedup |
| --- | ---: |
| `rwkv7` | 37.5x |
| `hybrid_h_rwkv7` | 5.0x |
| `hybrid_l_rwkv7` | 4.5x |

On RTX 4090, Transformer PrefixLM uses the local PyTorch fallback rather than FlashAttention 3 because FA3 targets Hopper. Hopper behavior will differ.

More detailed notes are in:

```text
RWKV7_BENCHMARK_README.md
```
