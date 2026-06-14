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

The first accepted experiment used the repo's existing RWKV-7 state recurrence as a q/o memory adapter. That was useful as a minimal proof of concept, but it was not the full delta-Mem mechanism.

The current default implementation is now `rwkv_mem_mode: delta_rule`: a native HRM delta-rule associative memory that learns memory q/k/v projections, updates a compact online state, reads from that state, and injects q/k/v/o deltas into HRM attention. `rwkv_mem_mode: rwkv7` is the separate comparison path: it uses an RWKV-7 state reader and projects that readout into q/k/v/o attention deltas. The old RWKV-7 reader remains available as `rwkv_mem_mode: rwkv7_legacy` for reproducing earlier checkpoints.

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

- keep the pretrained Transformer attention backbone;
- project hidden states to low-rank memory `q/k/v`;
- read from an online associative state before writing the current token;
- update the state with delta-rule keep/erase/write coefficients;
- inject the memory readout into attention `q/k/v/o`, with `[q, k, v, o]` as the full-recipe default and `[q, o]` kept for the released delta-Mem adapter comparison;
- support packed PrefixLM batches by padding `[T, C]` into `[numseqs, max_len, C]`, scanning once, and scattering deltas back;
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
  rwkv_mem_mode: delta_rule
  rwkv_mem_rank: 8
  rwkv_mem_alpha: 16.0
  rwkv_mem_beta_bias_init: -1.5
  rwkv_mem_state_update_mode: standard
  rwkv_mem_output_init: zero
  rwkv_mem_delta_heads: [q, k, v, o]
  rwkv_mem_separate_delta_projections: false
bp_warmup_ratio: 0.2
bp_max_steps: 5
```

With `rwkv_mem_output_init=zero`, the model starts exactly as the Transformer baseline because all delta heads output zero. Adapter construction restores RNG state after initializing memory parameters, so later Transformer layers keep identical initialization.

Current limitation: the HRM-native delta-rule scan uses the upstream Triton affine-scan kernel only when the local `deltamem` package is importable; otherwise it falls back to a PyTorch CUDA token loop. Vendoring or locally packaging that kernel is the next speed task.

## First Experiments

1. Smoke-test tiny model with H-only delta-rule memory.
2. Run the existing short 40M validation benchmark against:
   - Transformer baseline
   - H-only delta-rule memory
   - H+L delta-rule memory
3. Load HRM-Text-1B teacher weights with `strict=False` only for new RWKV-memory params, freeze the base Transformer, and train only memory parameters on the 1B subset.
4. Evaluate MMLU before any long continuation.

## First Result

Legacy H-only zero-init RWKV-state-reader adapter on the 1B V1 subset, 20 timed steps, `hidden_size=256`, `n_layers=4`, `H_cycles=2`, `L_cycles=3`, `bp_steps=5`:

| arch | train mean CE | last CE | val CE |
| --- | ---: | ---: | ---: |
| `transformer` | 6.6394 | 4.8207 | 4.8887 |
| `rwkv_mem` | 6.6272 | 4.8087 | 4.8785 |

## Legacy Accepted Post-Train Result

The H-level `q,o` legacy RWKV-state-reader adapter was post-trained from the HRM-Text-1B teacher checkpoint for 200 optimizer steps on the prepared full HRM-Text corpus:

```text
checkpoint: /run/media/xiaol/B214449214445C0B/hrm_text_pretrain_checkpoints/rwkv_mem_posttrain/rwkv_mem_qo_sep_full_s200_20260611_111851
ckpt_tag: step_200
training-token exposure: 39,321,600
MMLU: 0.6092
invalid: 0.0006
teacher MMLU: 0.6088
```

This clears the target but with a small margin. Treat it as validation that the adapter-only training/evaluation path is benchmark-compatible, not as a result for the new full delta-rule memory implementation.

## Stop Criteria

Stop early if:

- MMLU does not beat the old hidden-alignment H-RWKV result `0.3048`;
- invalid rate rises above `0.02`;
- delta-memory validation CE improves but MMLU stays flat, which would repeat the archived H-RWKV failure mode.

## Notes

Do not copy the upstream implementation directly unless licensing is clarified. The cloned repo does not include a top-level LICENSE file in the current checkout; it only advertises CC-BY-4.0 in the README badge.

The HRM implementation should stay small and local. The current full default uses active heads `q,k,v,o`; use `q,o` only for the release-compatible delta-Mem comparison.
