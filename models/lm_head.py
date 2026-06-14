from typing import Tuple

import torch
from torch import nn
from torch import Tensor
import torch.distributed as dist
import torch.nn.functional as F
from pydantic import BaseModel

from models.layers import LinearInit, ScaledEmbeddingInit, Carry
from models.common import IGNORE_LABEL_ID, packing_sequence_sum


class LMHeadConfig(BaseModel):
    vocab_size: int


class LMHead(nn.Module):
    def __init__(self, model: nn.Module, config_dict: dict) -> None:
        super().__init__()
        self.model = model
        # Create cache function
        self.create_cache = self.model.create_cache
        # Train extra args function
        self.compute_train_extra_args = self.model.compute_train_extra_args

        config = LMHeadConfig(**config_dict)
        head_hint: dict = self.model.head_hint  # pyright: ignore[reportAssignmentType]

        # LMHead input and output
        self.embed_tokens = ScaledEmbeddingInit(config.vocab_size, head_hint["in"]["dim"], init_std=head_hint["in"]["init_std"])  # pyright: ignore[reportArgumentType]
        self.lm_head = LinearInit(head_hint["out"]["dim"], config.vocab_size, bias=False, init_std=head_hint["out"]["init_std"])  # pyright: ignore[reportArgumentType]

    def forward(
        self,
        carry: Carry,
        batch: dict[str, Tensor],
        loss_divisor_override: Tensor | None = None,
        rwkv_mem_loss_mode: str = "ce",
        rwkv_mem_kl_weight: float = 0.0,
        rwkv_mem_kl_temperature: float = 2.0,
        rwkv_mem_reset_each_forward: bool = True,
        **kwargs,
    ) -> Tuple[Carry, Tensor] | Tuple[Carry, Tensor, dict[str, Tuple[Tensor, Tensor]]]:
        # Token embedding
        input_embedding = self.embed_tokens(batch["inputs"])

        if rwkv_mem_reset_each_forward:
            from models.rwkv_memory import reset_rwkv_mem_states

            reset_rwkv_mem_states(self.model)

        # Model forward
        new_carry, logits = self.model(carry,
                                       input_embedding,
                                       **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                       **kwargs)
        logits = self.lm_head(logits)

        # Loss & Metrics
        if "labels" in batch:
            # Masks & labels
            labels = batch["labels"]
            masks = labels != IGNORE_LABEL_ID
            local_valid_counts = masks.sum()

            # Loss (CE in F32)
            ce_loss_sum = F.cross_entropy(
                logits.to(torch.float32).flatten(0, -2),
                labels.to(torch.long).flatten(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="sum",
            )
            # AllReduce loss divisor. Divide by mean of valid tokens across all processes, as gradient will be averaged.
            if loss_divisor_override is None:
                loss_divisor = masks.sum().to(torch.float32)
                if dist.is_available() and dist.is_initialized():
                    dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)
            else:
                loss_divisor = loss_divisor_override.to(device=ce_loss_sum.device, dtype=torch.float32)

            kl_weight = float(rwkv_mem_kl_weight)
            normalized_loss = ce_loss_sum / loss_divisor
            kl_loss_sum = logits.new_zeros(())
            kl_loss_mean = logits.new_zeros(())
            if rwkv_mem_loss_mode != "ce":
                if rwkv_mem_loss_mode != "ce_kl":
                    raise ValueError(f"Unsupported rwkv_mem_loss_mode={rwkv_mem_loss_mode!r}")
                if kl_weight > 0.0:
                    from models.rwkv_memory import reset_rwkv_mem_states, set_rwkv_mem_runtime_enabled

                    try:
                        set_rwkv_mem_runtime_enabled(self.model, False)
                        reset_rwkv_mem_states(self.model)
                        with torch.no_grad():
                            _teacher_carry, teacher_hidden = self.model(
                                carry,
                                input_embedding,
                                **{k: v for k, v in batch.items() if k not in ("inputs", "labels")},
                                **kwargs,
                            )
                            teacher_logits = self.lm_head(teacher_hidden)
                    finally:
                        reset_rwkv_mem_states(self.model)
                        set_rwkv_mem_runtime_enabled(self.model, True)

                    if masks.any():
                        temperature = float(rwkv_mem_kl_temperature)
                        student_logp = F.log_softmax(logits[masks].float() / temperature, dim=-1)
                        teacher_prob = F.softmax(teacher_logits[masks].float() / temperature, dim=-1)
                        kl_per_token = F.kl_div(student_logp, teacher_prob, reduction="none").sum(dim=-1) * (temperature**2)
                        kl_loss_sum = kl_per_token.sum().to(dtype=normalized_loss.dtype)
                        kl_loss_mean = kl_loss_sum / local_valid_counts.clamp_min(1).to(device=kl_loss_sum.device, dtype=kl_loss_sum.dtype)
                        normalized_loss = normalized_loss + kl_weight * (kl_loss_sum / loss_divisor)

            # Accuracy
            with torch.no_grad():
                is_correct = torch.argmax(logits, dim=-1) == labels
                total_loss_sum = ce_loss_sum + kl_weight * kl_loss_sum
                # Sequence-level statistics
                if "cu_seqlens" in batch:
                    seq_num_tokens_correct = packing_sequence_sum(is_correct, batch["cu_seqlens"])
                    seq_num_valid_tokens = packing_sequence_sum(masks, batch["cu_seqlens"])
                    seq_is_valid = seq_num_valid_tokens > 0
                    exact_accuracy = (((seq_num_tokens_correct == seq_num_valid_tokens) & seq_is_valid).sum(), seq_is_valid.sum())
                else:
                    seq_is_valid = masks.any(dim=-1) if masks.dim() > 1 else masks.any().view(1)
                    exact_accuracy = (((is_correct | ~masks).all(dim=-1) & seq_is_valid).sum(), seq_is_valid.sum()) if masks.dim() > 1 else ((is_correct[masks].all() & seq_is_valid[0]).to(torch.int64), seq_is_valid.sum())
                # Metrics
                metrics = {
                    "loss": (total_loss_sum.detach(), local_valid_counts),
                    "ce_loss": (ce_loss_sum.detach(), local_valid_counts),
                    "delta_kl_loss": (kl_loss_sum.detach(), local_valid_counts),
                    "delta_kl_loss_mean": (kl_loss_mean.detach() * local_valid_counts, local_valid_counts),
                    "delta_kl_weighted_loss": ((kl_weight * kl_loss_sum).detach(), local_valid_counts),
                    "delta_total_loss": (total_loss_sum.detach(), local_valid_counts),
                    "accuracy": (is_correct.sum(), local_valid_counts),
                    "exact_accuracy": exact_accuracy,
                }

            return new_carry, normalized_loss, metrics

        return new_carry, logits
