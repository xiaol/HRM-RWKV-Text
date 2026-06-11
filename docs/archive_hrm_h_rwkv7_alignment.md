# Archived Direction: HRM-H RWKV-7 Alignment

Archived on 2026-06-11.

This note records the local attempt to migrate the HRM-Text-1B H-level Transformer core to RWKV-7 while keeping the L-level Transformer core. The direction is archived because the best local MMLU results stayed near random-choice quality and far below the teacher, even after fixing the PrefixLM masking bug in the staged CE/KL alignment.

## Decision

Do not continue the same stage3 training recipe.

The corrected CE/KL continuation reduced invalid answers, but did not recover teacher benchmark quality. More stage3 training of the same objective is unlikely to close the gap without a new architecture or alignment method.

## Result Summary

| run | checkpoint / log | MMLU acc | invalid |
| --- | --- | ---: | ---: |
| HRM-Text-1B teacher | `/run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/MMLU_teacher_20260605_082100.log` | `0.6088` | `0.0005` |
| Early hidden-state alignment | `/run/media/xiaol/B214449214445C0B/hrm_text_eval_runs/MMLU_h_rwkv_20260605_083832.log` | `0.3048` | `0.1175` |
| Corrected stage2 plus stage3 CE/KL | `/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_text_1b_to_hrm_h_rwkv7_stage3_corrected_resume4000_20260610_135906_from_stage2_20260610_150352/mmlu.log` | `0.2945` | `0.0126` |

The corrected run made the output format much cleaner, but the MMLU score did not improve.

## Key Checkpoints

Teacher model:

```text
/run/media/xiaol/B214449214445C0B/hf_models/sapientinc/HRM-Text-1B
```

Early hidden-state alignment checkpoint used by the first H-RWKV MMLU run:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_aligned_continue_5000_b512_skip1000_20260605_001407.safetensors
```

Eval wrapper for that checkpoint:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_eval_checkpoints/hrm_h_rwkv7_aligned_5000
```

Final corrected stage3 checkpoint:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_stage3_corrected_resume4000_20260610_135906_from_stage2_20260610_150352.safetensors
```

## What Was Fixed

The first staged CE/KL attempts trained hidden and KL losses on all valid packed tokens. That was wrong for HRM-Text PrefixLM data:

- HRM-Text masks prompt labels with `IGNORE_LABEL_ID` and supervises response tokens only.
- The Transformer teacher uses PrefixLM attention over prompt and response.
- The RWKV H stack only receives sequence boundaries, not bidirectional prefix attention.

The alignment script was corrected so stage2/stage3 default to supervised target-token masks for hidden and KL losses. Stage1 still uses valid-token local teacher-forced alignment.

Relevant script:

```text
scripts/align_hrm_h_rwkv7_to_hrm_text_teacher.py
```

## Corrected Stage Results

Corrected resumed stage2 summary:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_stage2_corrected_resume4000_20260610_135906.summary.json
```

Final corrected stage2 validation:

| metric | value |
| --- | ---: |
| val loss | `0.8379` |
| val CE | `0.2264` |
| val hidden | `0.6898` |
| val KL | `0.4182` |
| val acc | `0.9257` |

Corrected stage3 summary:

```text
/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_stage3_corrected_resume4000_20260610_135906_from_stage2_20260610_150352.summary.json
```

Final corrected stage3 validation:

| metric | value |
| --- | ---: |
| val loss | `0.3672` |
| val CE | `0.1414` |
| val hidden | `0.0738` |
| val KL | `0.3038` |
| val acc | `0.9459` |

Despite the low validation CE and low invalid MMLU rate, benchmark accuracy remained poor.

## Interpretation

The PrefixLM masking bug was real, but not the main blocker. The H Transformer to RWKV-7 replacement still loses behavior needed for the HRM-Text benchmark tasks.

The likely problem is architectural mismatch, not only loss saturation:

- Stage1 teacher-forced hidden alignment can look good while the free-running H stack remains behaviorally different.
- Target-token CE/KL improves local next-token metrics and answer validity, but does not restore task-level reasoning.
- PrefixLM bidirectional prompt behavior is naturally represented by attention but not by the current causal RWKV H replacement.

## Files Kept For Reference

Implementation and launch scripts remain useful as reference material:

```text
apps/LT2/rwkv7_cuda.py
models/rwkv7.py
models/baselines/hrm_hybrid_rwkv7_nocarry_bp_warmup.py
scripts/align_hrm_h_rwkv7_to_hrm_text_teacher.py
scripts/launch_stage2_stage3_after_stage1.sh
scripts/resume_corrected_stage2_stage3_from_checkpoint.sh
evaluation/config/hrm_mmlu_only.yaml
```

Large checkpoints and logs remain on the 2TB SSD under:

```text
/run/media/xiaol/B214449214445C0B
```

They are intentionally not committed.
