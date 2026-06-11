# Next Direction: RWKV-State Memory For HRM-Text

Started on 2026-06-11.

Reference repo:

```text
https://github.com/declare-lab/delta-Mem
local: /home/xiaol/X/delta-Mem
commit: 5cd5d91
```

## Why This Direction

The archived H-RWKV direction changed the H-level transition operator too aggressively. MMLU stayed near random-choice quality even after fixing the PrefixLM token-mask bug.

Delta-Mem is a better conceptual direction because it keeps the full-attention backbone and adds a small online associative memory. For HRM-Text, that means we can preserve the Transformer H/L computation that gives the teacher its benchmark behavior, then test whether a stateful memory adapter improves or preserves reasoning while adding online state.

The first implemented variant uses the repo's existing RWKV-7 state recurrence as that online memory. This avoids copying the upstream Delta-Mem implementation and lets the adapter use the local LT2 CUDA kernels.

## What Delta-Mem Provides

The upstream implementation wraps Hugging Face Qwen3 / SmolLM3 attention modules. Its active path:

- projects hidden states to low-rank memory `q/k/v`;
- keeps a compact online state with shape roughly `[batch, rank, rank]`;
- updates the state using a delta-rule affine scan;
- reads from the state and injects low-rank deltas into attention `q/k/v/o`;
- supports token, message-mean, and sentence-mean write granularity.

The public repo currently targets Qwen3-4B/8B and SmolLM3-3B. It is not directly compatible with HRM-Text because HRM uses a custom packed `gqkv_proj` attention layer in `models/layers.py`.

## HRM Integration Plan

Use the idea, not the HF wrapper.

First implementation is native HRM:

```text
models/rwkv_memory.py
models/layers.py
config/arch/net/hrm_h_rwkv_mem.yaml
```

Implemented adapter:

- keep Transformer attention unchanged;
- run a per-layer RWKV-7 state reader over the same hidden states;
- inject the RWKV memory output into the attention query before RoPE/attention;
- add the RWKV memory output as an attention-output residual after attention;
- use separate identity-initialized `delta_q` and `delta_o` projections for new runs;
- support packed PrefixLM batches by padding `[T, C]` into `[numseqs, max_len, C]`, running the RWKV state once, and scattering back;
- H-level only first through `H_override`.

Recommended first config shape:

```yaml
name: baselines.hrm_nocarry_bp_warmup@HierarchicalReasoningModel
head: lm_head@LMHead
half_layers: true
H_cycles: 2
L_cycles: 3
H_override:
  rwkv_mem_enabled: true
  rwkv_mem_head_size: 64
  rwkv_mem_backend: auto
  rwkv_mem_chunk_len: 16
  rwkv_mem_scale: 1.0
  rwkv_mem_output_init: zero
  rwkv_mem_delta_heads: [q, o]
  rwkv_mem_separate_delta_projections: true
bp_warmup_ratio: 0.2
bp_max_steps: 5
```

With `rwkv_mem_output_init=zero`, the model starts exactly as the Transformer baseline. Adapter construction restores RNG state after initializing RWKV parameters, so later Transformer layers keep identical initialization.

## First Experiments

1. Smoke-test tiny model with H-only RWKV-state memory.
2. Run the existing short 40M validation benchmark against:
   - Transformer baseline
   - H-only RWKV-state memory
   - H+L RWKV-state memory
3. Load HRM-Text-1B teacher weights with `strict=False` only for new RWKV-memory params, freeze the base Transformer, and train only memory parameters on the 1B subset.
4. Evaluate MMLU before any long continuation.

## First Result

H-only zero-init adapter on the 1B V1 subset, 20 timed steps, `hidden_size=256`, `n_layers=4`, `H_cycles=2`, `L_cycles=3`, `bp_steps=5`:

| arch | train mean CE | last CE | val CE |
| --- | ---: | ---: | ---: |
| `transformer` | 6.6394 | 4.8207 | 4.8887 |
| `rwkv_mem` | 6.6272 | 4.8087 | 4.8785 |

## Accepted Post-Train Result

The H-level `q,o` RWKV-memory adapter was post-trained from the HRM-Text-1B teacher checkpoint for 200 optimizer steps on the prepared full HRM-Text corpus:

```text
checkpoint: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851
ckpt_tag: step_200
training-token exposure: 39,321,600
MMLU: 0.6092
invalid: 0.0006
teacher MMLU: 0.6088
```

This clears the target but with a small margin. Treat it as validation that the method is trainable and benchmark-compatible, not as a robust quality claim.

## Stop Criteria

Stop early if:

- MMLU does not beat the old hidden-alignment H-RWKV result `0.3048`;
- invalid rate rises above `0.02`;
- delta-memory validation CE improves but MMLU stays flat, which would repeat the archived H-RWKV failure mode.

## Notes

Do not copy the upstream implementation directly unless licensing is clarified. The cloned repo does not include a top-level LICENSE file in the current checkout; it only advertises CC-BY-4.0 in the README badge.

The HRM implementation should stay small and local. The current accepted path is `q,o` injection in the existing `Attention` module. Full Q/K/V/O delta-memory injection can be tested later, but it is more likely to disturb the pretrained Transformer path.
