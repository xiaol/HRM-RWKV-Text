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

Results:

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
