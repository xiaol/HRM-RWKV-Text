# HRM-Text RWKV-7 Comparison

This branch adds RWKV-7 and H/L hybrid cores to HRM-Text for local training-process validation.

## Architectures

The benchmark supports four HRM core layouts:

| name | H core | L core |
| --- | --- | --- |
| `transformer` | Transformer | Transformer |
| `rwkv7` | RWKV-7 | RWKV-7 |
| `hybrid_h_rwkv7` | RWKV-7 | Transformer |
| `hybrid_l_rwkv7` | Transformer | RWKV-7 |

RWKV-7 uses the LT2 CUDA kernels when available:

- `tmix_mix6`
- `tmix_a_gate`
- `tmix_vres_gate`
- `tmix_kk_pre`
- `rwkv7_recurrence_cuda_bf16`
- `tmix_lnx_rkvres_xg`
- `cmix_layer`

For full RWKV-7 CUDA eligibility, use `bf16`, `rwkv7_head_size=64`, `rwkv7_backend=cuda`, and `rwkv7_expansion=1.0`.

## Official Subset

The full HRM-Text pretraining dataset is too large for this machine. I generated a compact 1B-token subset from the official cleaned dataset using the official tokenizer:

```text
dataset: sapientinc/HRM-Text-data-io-cleaned-20260515
tokenizer: sapientinc/HRM-Text-1B tokenizer.json
local path: /home/xiaol/X/hrm_text_subset_1B
tokens: 1,000,000,035
rows: 26,620,178
storage: uint16 tokens.bin + V1Dataset index arrays
disk: 2.7 GB
```

Generation command:

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

The generated dataset is intentionally not committed.

## 4090 Benchmark

Command:

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
  --json-out outputs/hrm_official_1b_v1_compare_4090_h256_l4_s30.json
```

Initial results before packed RWKV batching:

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 40.89M | 8,724 | 1,152 | 5.3036 | 3.6450 | 3.7159 | 3.96 GB |
| `rwkv7` | 40.53M | 2,052 | 271 | 4.9836 | 3.5009 | 3.5364 | 4.44 GB |
| `hybrid_h_rwkv7` | 40.71M | 3,219 | 425 | 4.8156 | 3.3221 | 3.4045 | 4.28 GB |
| `hybrid_l_rwkv7` | 40.71M | 3,400 | 449 | 4.9633 | 3.4557 | 3.5851 | 4.12 GB |

This is a short training-process validation run, not a final model-quality result.

## Speed Notes

The RWKV-7 path does call the LT2 kernels, but it is still slower in this HRM-Text benchmark because the official dataset is PrefixLM-packed into many short sequences. In this run each 4096-token packed batch had roughly 112 sequences and max sequence length around 92 tokens. The current `RWKV7Stack` loops over packed sequences and launches RWKV kernels per sequence, which creates heavy kernel-launch overhead.

The next speed fix is to batch packed sequences inside `RWKV7Stack` instead of looping sequence-by-sequence.

On RTX 4090, HRM-Text Transformer PrefixLM also uses the local PyTorch fallback rather than FlashAttention 3, because FA3 is Hopper-targeted. Hopper speed behavior will differ.

## Packed RWKV Update

`RWKV7Stack` now batches packed PrefixLM sequences by padding the packed `[T, C]` tensor into `[numseqs, max_seq_len, C]`, running the RWKV stack once, then scattering back to `[T, C]`. This removes the per-sequence RWKV kernel-launch loop.

Same benchmark command, output:

```text
outputs/hrm_official_1b_v1_compare_4090_h256_l4_s30_packed_rwkv.json
```

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 40.89M | 8,578 | 1,132 | 5.3036 | 3.6450 | 3.7159 | 3.96 GB |
| `rwkv7` | 40.53M | 76,873 | 10,149 | 4.9837 | 3.5006 | 3.5359 | 6.39 GB |
| `hybrid_h_rwkv7` | 40.71M | 16,010 | 2,114 | 4.8157 | 3.3210 | 3.4045 | 5.60 GB |
| `hybrid_l_rwkv7` | 40.71M | 15,242 | 2,012 | 4.9637 | 3.4560 | 3.5877 | 4.77 GB |

Speedup versus the initial looped RWKV path:

| arch | speedup |
| --- | ---: |
| `rwkv7` | 37.5x |
| `hybrid_h_rwkv7` | 5.0x |
| `hybrid_l_rwkv7` | 4.5x |

## Local 0.6B-Size Baseline

To match the upstream HRM-Text L/0.6B shape locally, the benchmark must use `--half-layers`. Without it, `n_layers=24` builds 24 layers per H/L level and produces a roughly 1.2B-parameter model in this helper. With `--half-layers`, each H/L level gets 12 layers, matching the upstream L config.

This run uses the 1B-token official subset above, not the full pretraining corpus. It is intended as a local training-process and architecture comparison on a single RTX 4090.

Command:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
PYTHONPATH=/home/xiaol/X/LT2_upstream \
.venv/bin/python scripts/benchmark_hrm_rwkv7.py \
  --mode v1 \
  --device cuda \
  --dtype bf16 \
  --archs transformer,rwkv7,hybrid_h_rwkv7,hybrid_l_rwkv7 \
  --warmup-steps 3 \
  --steps 30 \
  --v1-batch-tokens 1024 \
  --v1-eval-batch-tokens 1024 \
  --v1-val-batches 10 \
  --seq-len 4096 \
  --hidden-size 1280 \
  --n-layers 24 \
  --half-layers \
  --num-heads 10 \
  --transformer-expansion 4.0 \
  --rwkv7-expansion 1.0 \
  --h-cycles 2 \
  --l-cycles 3 \
  --bp-steps 5 \
  --vocab-size 65536 \
  --rwkv7-head-size 64 \
  --rwkv7-backend cuda \
  --json-out outputs/l06_official_1b_v1_compare_4090_b1024_s30.json
```

Results:

| arch | params | tok/s | supervised tok/s | train mean CE | last CE | val CE | VRAM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `transformer` | 694.68M | 1,691 | 233 | 4.2500 | 3.6556 | 3.2188 | 8.57 GB |
| `rwkv7` | 667.62M | 3,662 | 504 | 4.0606 | 3.5916 | 3.2192 | 21.28 GB |
| `hybrid_h_rwkv7` | 681.15M | 2,037 | 280 | 3.8253 | 3.5945 | 3.1869 | 13.64 GB |
| `hybrid_l_rwkv7` | 681.15M | 2,643 | 364 | 4.3985 | 3.6666 | 3.2319 | 16.21 GB |

On this 4090, pure RWKV-7 fits reliably at `1024` packed tokens with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; it OOMs at `2048` in this L-size training benchmark. The Transformer baseline can fit larger microbatches, but `1024` is the common local setting for same-batch architecture comparison.

The upstream L reference run uses `global_batch_size=172032` tokens on 8 H100s. For the real pretrain path on a single 4090, use `micro_batch_size=512` to keep that effective batch through 336 gradient-accumulation microsteps. A one-step 1B-subset smoke test passed in about 99 seconds; `micro_batch_size=1024` OOMed during CE/loss allocation in the FSDP pretrain path.
