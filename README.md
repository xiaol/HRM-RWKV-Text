# HRM-RWKV-Text

This repository is an experimental fork of HRM-Text that replaces or mixes the HRM recurrent Transformer cores with RWKV-7.

Original upstream project:

```text
https://github.com/sapientinc/HRM-Text
```

Refer to upstream HRM-Text for the original paper, architecture, training framework, evaluation stack, and license context. This fork is focused on local architecture validation and speed/loss comparison for RWKV-7 inside HRM-Text.

## What Changed

Added HRM core variants:

| config / benchmark name | H core | L core |
| --- | --- | --- |
| `transformer` | Transformer | Transformer |
| `rwkv7` / `hrm_rwkv7` | RWKV-7 | RWKV-7 |
| `hybrid_h_rwkv7` / `hrm_h_rwkv7` | RWKV-7 | Transformer |
| `hybrid_l_rwkv7` / `hrm_l_rwkv7` | Transformer | RWKV-7 |

Key files:

```text
models/rwkv7.py
models/baselines/hrm_rwkv7_nocarry_bp_warmup.py
models/baselines/hrm_hybrid_rwkv7_nocarry_bp_warmup.py
config/arch/net/hrm_rwkv7.yaml
config/arch/net/hrm_h_rwkv7.yaml
config/arch/net/hrm_l_rwkv7.yaml
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

For actual local training with the upstream L effective batch, keep `global_batch_size=172032` and use gradient accumulation with the 4090-safe microbatch:

```bash
WANDB_MODE=offline \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/xiaol/X/LT2_upstream \
.venv/bin/python -m torch.distributed.run --nproc_per_node=1 pretrain.py \
  arch/net@arch=hrm_rwkv7 \
  arch/size@arch=L \
  data.path=/home/xiaol/X/hrm_text_subset_1B \
  global_batch_size=172032 \
  micro_batch_size=1024 \
  epochs=1 \
  compile_train=false \
  arch.rwkv7_backend=cuda \
  run_name=hrm_rwkv7_l_1b_subset_b172k_micro1024
```

That gives `172032 / 1024 = 168` gradient-accumulation microsteps per optimizer step on one 4090.

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
