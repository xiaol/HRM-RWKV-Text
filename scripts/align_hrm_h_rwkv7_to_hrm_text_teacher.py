from __future__ import annotations

import argparse
import json
import sys
import time
from collections import deque
from math import prod
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn.functional as F
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset_new import V1Dataset, V1DatasetConfig
from models.baselines.hrm_hybrid_rwkv7_nocarry_bp_warmup import HierarchicalHybridRWKV7Model
from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.common import IGNORE_LABEL_ID, unwrap_tensor, wrap_tensor
from models.lm_head import LMHead


def expansion_from_hf_config(hf_config: dict[str, Any]) -> float:
    return float(hf_config["intermediate_size"]) * 3.0 / (float(hf_config["hidden_size"]) * 2.0)


def base_config_from_hf(hf_config: dict[str, Any]) -> dict[str, Any]:
    return {
        "max_seq_len": int(hf_config["max_position_embeddings"]),
        "n_layers": int(hf_config["num_hidden_layers"]) * 2,
        "hidden_size": int(hf_config["hidden_size"]),
        "num_heads": int(hf_config["num_attention_heads"]),
        "expansion": expansion_from_hf_config(hf_config),
        "norm_type": "pre",
        "norm_eps": float(hf_config["rms_norm_eps"]),
        "rope_theta": float(hf_config["rope_theta"]),
        "pos_emb_type": "rope",
        "init_type": "lecun_normal",
        "half_layers": True,
        "H_cycles": int(hf_config["H_cycles"]),
        "L_cycles": int(hf_config["L_cycles"]),
        "H_override": {},
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": 2,
        "bp_max_steps": 5,
        "vocab_size": int(hf_config["vocab_size"]),
        "target_only": True,
    }


def student_config_from_hf(hf_config: dict[str, Any], rwkv7_expansion: float, rwkv7_backend: str) -> dict[str, Any]:
    cfg = base_config_from_hf(hf_config)
    cfg.update(
        {
            "H_arch": "rwkv7",
            "L_arch": "transformer",
            "transformer_expansion": expansion_from_hf_config(hf_config),
            "rwkv7_expansion": rwkv7_expansion,
            "L_override": {},
            "rwkv7_head_size": 64,
            "rwkv7_backend": rwkv7_backend,
            "rwkv7_chunk_len": 16,
            "rwkv7_enable_v_first_mix": True,
        }
    )
    return cfg


def map_hf_key_to_teacher(key: str) -> str | None:
    if key == "lm_head.weight":
        return "lm_head.weight"
    if key == "model.embed_tokens.weight":
        return "embed_tokens.embedding_weight"
    if key == "model.z_L_init":
        return "model.zL_init"
    if key.startswith("model.H_module."):
        return "model.H_level.core." + key.removeprefix("model.H_module.")
    if key.startswith("model.L_module."):
        return "model.L_level.core." + key.removeprefix("model.L_module.")
    return None


def load_hf_teacher_state(source_dir: Path) -> dict[str, Tensor]:
    source_path = source_dir / "model.safetensors"
    tensors = {}
    with safe_open(source_path, framework="pt", device="cpu") as source:
        for source_key in source.keys():
            target_key = map_hf_key_to_teacher(source_key)
            if target_key is not None:
                tensors[target_key] = source.get_tensor(source_key)
    return tensors


def build_teacher(source_dir: Path, device: torch.device, dtype: torch.dtype) -> LMHead:
    hf_config = json.loads((source_dir / "config.json").read_text())
    cfg = base_config_from_hf(hf_config)
    model = LMHead(HierarchicalReasoningModel(cfg), cfg).to(device=device, dtype=dtype)
    incompatible = model.load_state_dict(load_hf_teacher_state(source_dir), strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(f"Teacher load mismatch: {incompatible}")
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model


def build_student(
    source_dir: Path,
    init_path: Path,
    device: torch.device,
    dtype: torch.dtype,
    rwkv7_expansion: float,
    rwkv7_backend: str,
    strict_init: bool,
) -> LMHead:
    hf_config = json.loads((source_dir / "config.json").read_text())
    cfg = student_config_from_hf(hf_config, rwkv7_expansion=rwkv7_expansion, rwkv7_backend=rwkv7_backend)
    model = LMHead(HierarchicalHybridRWKV7Model(cfg), cfg).to(device=device, dtype=dtype)
    tensors = load_file(init_path, device="cpu")
    incompatible = model.load_state_dict(tensors, strict=strict_init)
    print(
        f"[Student init] strict={strict_init} loaded={len(tensors)} missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected student init keys: {incompatible.unexpected_keys[:20]}")
    return model


def make_loader(dataset_path: str, batch_tokens: int, seed: int = 0) -> DataLoader:
    dataset = V1Dataset(
        V1DatasetConfig(
            seed=seed,
            dataset_path=dataset_path,
            batch_max_length=batch_tokens,
            drop_last_batch=True,
            target_only=True,
            rank=0,
            num_replicas=1,
        )
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


def move_v1_batch_to_device(batch: dict[str, Tensor], info: dict[str, Any], device: torch.device) -> dict[str, Tensor]:
    moved = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    moved.update({key: wrap_tensor(torch.tensor(value, device="cpu")) for key, value in info.items()})
    return moved


def forward_hidden(model: LMHead, batch: dict[str, Tensor], bp_steps: int) -> Tensor:
    input_embedding = model.embed_tokens(batch["inputs"])
    _, hidden = model.model(
        None,
        input_embedding,
        **{key: value for key, value in batch.items() if key not in ("inputs", "labels")},
        bp_steps=bp_steps,
    )
    return hidden


def forward_hidden_logits(model: LMHead, batch: dict[str, Tensor], bp_steps: int) -> tuple[Tensor, Tensor]:
    hidden = forward_hidden(model, batch, bp_steps)
    return hidden, model.lm_head(hidden)


def batch_seq_info(batch: dict[str, Tensor]) -> dict[str, Any]:
    return {key: value for key, value in batch.items() if key not in ("inputs", "labels")}


def valid_token_mask(batch: dict[str, Tensor], info: dict[str, Any]) -> Tensor:
    return torch.arange(batch["inputs"].shape[0], device=batch["inputs"].device) < int(info["total_seqlen"])


def as_int(value: Any) -> int:
    if isinstance(value, Tensor):
        return int(value.detach().cpu().item())
    if hasattr(value, "item"):
        return int(value.item())
    return int(value)


def batch_token_counts(args: argparse.Namespace, batch: dict[str, Tensor], info: dict[str, Any]) -> dict[str, int]:
    supervised_tokens = int((batch["labels"] != IGNORE_LABEL_ID).sum().item())
    valid_tokens = as_int(info["total_seqlen"])
    return {
        "supervised_tokens": supervised_tokens,
        "hidden_tokens": valid_tokens if args.hidden_mask == "valid" else supervised_tokens,
        "kl_tokens": valid_tokens if args.kl_mask == "valid" else supervised_tokens,
        "tokens": valid_tokens,
    }


def should_skip_counts(args: argparse.Namespace, counts: dict[str, int]) -> bool:
    return (
        counts["hidden_tokens"] == 0
        or (args.ce_weight > 0 and counts["supervised_tokens"] == 0)
        or (args.kl_weight > 0 and counts["kl_tokens"] == 0)
    )


def supervised_hidden_loss(student_hidden: Tensor, teacher_hidden: Tensor, mask: Tensor, loss_type: str) -> Tensor:
    if loss_type == "l2":
        return torch.linalg.vector_norm(teacher_hidden[mask].float() - student_hidden[mask].float(), dim=-1).mean() * (
            teacher_hidden.shape[-1] ** -0.5
        )
    if loss_type != "cosine":
        raise ValueError(f"Unknown hidden loss type: {loss_type}")
    student_norm = F.normalize(student_hidden.float(), dim=-1)
    teacher_norm = F.normalize(teacher_hidden.float(), dim=-1)
    cosine = (student_norm[mask] * teacher_norm[mask]).sum(dim=-1)
    return (1.0 - cosine).mean()


def supervised_kl_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
    student_logp = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits[mask].float() / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (temperature**2)


def _loss_over_states(student_states: list[Tensor], teacher_states: list[Tensor], mask: Tensor, loss_type: str) -> Tensor:
    if not student_states or not teacher_states:
        raise ValueError("No hidden states were collected for staged alignment.")
    if len(student_states) != len(teacher_states):
        raise ValueError(f"State count mismatch: student={len(student_states)} teacher={len(teacher_states)}")
    losses = [supervised_hidden_loss(s, t, mask, loss_type) for s, t in zip(student_states, teacher_states)]
    return torch.stack(losses).mean()


def _transformer_core_trace(core: torch.nn.Module, x: Tensor, seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor], list[Tensor]]:
    local_seq_info = dict(seq_info)
    local_seq_info["cos_sin"] = core.rotary_emb(local_seq_info.pop("position_ids", None)) if hasattr(core, "rotary_emb") else None
    inputs = []
    states = []
    for layer_id, layer in enumerate(core.layers):
        inputs.append(x)
        x = layer(x, **local_seq_info, cache=None)
        states.append(x)
    return core.norm_f(x), states, inputs


def _transformer_core_with_states(core: torch.nn.Module, x: Tensor, seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor]]:
    final, states, _inputs = _transformer_core_trace(core, x, seq_info)
    return final, states


def _rwkv7_batched_with_states(core: torch.nn.Module, x: Tensor) -> tuple[Tensor, list[Tensor]]:
    v_first = None
    states = []
    for idx, layer in enumerate(core.layers):
        x, v_first = layer(x, v_first=v_first, reset_v_first=(idx == 0))
        states.append(x)
    return core.norm_f(x), states


def _rwkv7_core_with_states(core: torch.nn.Module, x: Tensor, seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor]]:
    if x.dim() == 3:
        return _rwkv7_batched_with_states(core, x)
    if x.dim() != 2:
        raise ValueError(f"RWKV7Stack expects [B,T,C] or [T,C], got {tuple(x.shape)}")
    if "cu_seqlens" not in seq_info or "numseqs" not in seq_info:
        final, states = _rwkv7_batched_with_states(core, x.unsqueeze(0))
        return final.squeeze(0), [state.squeeze(0) for state in states]

    numseqs = unwrap_tensor(seq_info["numseqs"])
    n = int(numseqs.item()) if isinstance(numseqs, Tensor) else int(numseqs)
    if n <= 0:
        zeros = torch.zeros_like(x)
        return zeros, [zeros for _ in core.layers]

    cu_seqlens = unwrap_tensor(seq_info["cu_seqlens"])[: n + 1].to(device=x.device, dtype=torch.long)
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    max_len = int(lengths.max().item())
    if max_len <= 0:
        zeros = torch.zeros_like(x)
        return zeros, [zeros for _ in core.layers]

    pos = torch.arange(max_len, device=x.device, dtype=torch.long).unsqueeze(0)
    mask = pos < lengths.unsqueeze(1)
    src = starts.unsqueeze(1) + pos

    padded = x.new_zeros((n, max_len, x.shape[-1]))
    padded[mask] = x[src[mask]]
    padded_final, padded_states = _rwkv7_batched_with_states(core, padded)

    def unpad(padded_tensor: Tensor) -> Tensor:
        out = torch.zeros_like(x)
        out[src[mask]] = padded_tensor[mask]
        return out

    return unpad(padded_final), [unpad(state) for state in padded_states]


def _packed_to_padded(x: Tensor, seq_info: dict[str, Any]) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    if "cu_seqlens" not in seq_info or "numseqs" not in seq_info:
        mask = torch.ones((1, x.shape[0]), device=x.device, dtype=torch.bool)
        src = torch.arange(x.shape[0], device=x.device, dtype=torch.long).unsqueeze(0)
        return x.unsqueeze(0), mask, src, torch.tensor([0], device=x.device, dtype=torch.long)

    numseqs = unwrap_tensor(seq_info["numseqs"])
    n = int(numseqs.item()) if isinstance(numseqs, Tensor) else int(numseqs)
    if n <= 0:
        empty_mask = torch.zeros((0, 0), device=x.device, dtype=torch.bool)
        empty_src = torch.zeros((0, 0), device=x.device, dtype=torch.long)
        return x.new_zeros((0, 0, x.shape[-1])), empty_mask, empty_src, torch.zeros(0, device=x.device, dtype=torch.long)

    cu_seqlens = unwrap_tensor(seq_info["cu_seqlens"])[: n + 1].to(device=x.device, dtype=torch.long)
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    max_len = int(lengths.max().item())
    pos = torch.arange(max_len, device=x.device, dtype=torch.long).unsqueeze(0)
    mask = pos < lengths.unsqueeze(1)
    src = starts.unsqueeze(1) + pos
    padded = x.new_zeros((n, max_len, x.shape[-1]))
    padded[mask] = x[src[mask]]
    return padded, mask, src, starts


def _unpad_like(padded: Tensor, like: Tensor, mask: Tensor, src: Tensor) -> Tensor:
    out = torch.zeros_like(like)
    if mask.numel() > 0:
        out[src[mask]] = padded[mask]
    return out


def _rwkv7_forced_with_states(core: torch.nn.Module, layer_inputs: list[Tensor], seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor]]:
    if len(layer_inputs) != len(core.layers):
        raise ValueError(f"Forced H input count mismatch: inputs={len(layer_inputs)} layers={len(core.layers)}")
    if not layer_inputs:
        raise ValueError("No layer inputs were provided for forced RWKV-7 alignment.")

    if layer_inputs[0].dim() == 3:
        v_first = None
        states = []
        for idx, (layer, x_in) in enumerate(zip(core.layers, layer_inputs)):
            x_out, v_first = layer(x_in, v_first=v_first, reset_v_first=(idx == 0))
            states.append(x_out)
        return core.norm_f(states[-1]), states

    if layer_inputs[0].dim() != 2:
        raise ValueError(f"RWKV7Stack expects forced inputs as [B,T,C] or [T,C], got {tuple(layer_inputs[0].shape)}")

    padded_inputs = []
    mask = src = None
    for x_in in layer_inputs:
        padded, layer_mask, layer_src, _starts = _packed_to_padded(x_in, seq_info)
        padded_inputs.append(padded)
        if mask is None:
            mask, src = layer_mask, layer_src

    assert mask is not None and src is not None
    v_first = None
    padded_states = []
    for idx, (layer, x_in) in enumerate(zip(core.layers, padded_inputs)):
        x_out, v_first = layer(x_in, v_first=v_first, reset_v_first=(idx == 0))
        padded_states.append(x_out)

    states = [_unpad_like(state, layer_inputs[0], mask, src) for state in padded_states]
    return core.norm_f(states[-1]), states


def h_core_with_states(h_level: torch.nn.Module, h_input: Tensor, seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor]]:
    core = h_level.core
    if hasattr(core, "rotary_emb"):
        return _transformer_core_with_states(core, h_input, seq_info)
    if hasattr(core, "layers") and core.__class__.__name__ == "RWKV7Stack":
        return _rwkv7_core_with_states(core, h_input, seq_info)
    raise TypeError(f"Unsupported H core for staged alignment: {core.__class__.__name__}")


def h_core_forced_with_states(h_level: torch.nn.Module, layer_inputs: list[Tensor], seq_info: dict[str, Any]) -> tuple[Tensor, list[Tensor]]:
    core = h_level.core
    if hasattr(core, "layers") and core.__class__.__name__ == "RWKV7Stack":
        return _rwkv7_forced_with_states(core, layer_inputs, seq_info)
    raise TypeError(f"Forced stage-1 alignment expects RWKV7Stack student H core, got {core.__class__.__name__}")


def trace_hrm_h_states(model: LMHead, batch: dict[str, Tensor], bp_steps: int) -> tuple[Tensor, Tensor, list[Tensor], list[Tensor]]:
    seq_info = batch_seq_info(batch)
    x = model.embed_tokens(batch["inputs"])
    hrm = model.model
    x = x.to(dtype=hrm.zL_init.dtype)
    z_H, z_L = x, hrm.zL_init

    H_bp_steps = min(hrm.H_cycles, bp_steps - 1)
    L_bp_steps = bp_steps - H_bp_steps

    cycle_states = []
    layer_states = []
    for i in range(hrm.H_cycles):
        for k in range(i * hrm.L_cycles, (i + 1) * hrm.L_cycles):
            with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= hrm.H_cycles * hrm.L_cycles - L_bp_steps)):
                z_L = hrm.L_level(z_L, z_H, **seq_info, cache=None)

        with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= hrm.H_cycles - H_bp_steps)):
            z_H, h_layers = h_core_with_states(hrm.H_level, z_H + z_L, seq_info)
        cycle_states.append(z_H)
        layer_states.extend(h_layers)

    logits = model.lm_head(z_H)
    return z_H, logits, cycle_states, layer_states


def trace_local_h_alignment_states(
    teacher: LMHead,
    student: LMHead,
    batch: dict[str, Tensor],
    bp_steps: int,
) -> tuple[list[Tensor], list[Tensor], list[Tensor], list[Tensor]]:
    seq_info = batch_seq_info(batch)
    teacher_hrm = teacher.model
    student_hrm = student.model

    with torch.no_grad():
        x = teacher.embed_tokens(batch["inputs"]).to(dtype=teacher_hrm.zL_init.dtype)
        teacher_z_H, teacher_z_L = x, teacher_hrm.zL_init

    teacher_cycles = []
    student_cycles = []
    teacher_layers = []
    student_layers = []

    for i in range(teacher_hrm.H_cycles):
        with torch.no_grad():
            for k in range(i * teacher_hrm.L_cycles, (i + 1) * teacher_hrm.L_cycles):
                teacher_z_L = teacher_hrm.L_level(teacher_z_L, teacher_z_H, **seq_info, cache=None)
            h_input = (teacher_z_H + teacher_z_L).detach()
            teacher_z_H, cycle_teacher_layers, teacher_layer_inputs = _transformer_core_trace(teacher_hrm.H_level.core, h_input, seq_info)
            teacher_cycles.append(teacher_z_H.detach())
            teacher_layers.extend([state.detach() for state in cycle_teacher_layers])

        forced_inputs = [state.detach().to(dtype=student_hrm.zL_init.dtype) for state in teacher_layer_inputs]
        student_z_H, cycle_student_layers = h_core_forced_with_states(student_hrm.H_level, forced_inputs, seq_info)
        student_cycles.append(student_z_H)
        student_layers.extend(cycle_student_layers)

    return student_cycles, teacher_cycles, student_layers, teacher_layers


def choose_token_mask(args: argparse.Namespace, batch: dict[str, Tensor], info: dict[str, Any], kind: Literal["hidden", "kl"]) -> Tensor:
    requested = args.hidden_mask if kind == "hidden" else args.kl_mask
    if requested == "valid":
        return valid_token_mask(batch, info)
    if requested == "supervised":
        return batch["labels"] != IGNORE_LABEL_ID
    raise ValueError(f"Unknown {kind} mask: {requested}")


def save_student_state(model: LMHead, output_path: Path, metadata: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(output_path), metadata=metadata)


def checkpoint_path_for_step(output_path: Path, step: int) -> Path:
    return output_path.with_name(f"{output_path.stem}_step{step}{output_path.suffix}")


def prune_step_checkpoints(output_path: Path, keep_last: int) -> None:
    if keep_last <= 0:
        return
    checkpoints = sorted(output_path.parent.glob(f"{output_path.stem}_step*{output_path.suffix}"), key=lambda p: p.stat().st_mtime)
    for checkpoint in checkpoints[:-keep_last]:
        checkpoint.unlink(missing_ok=True)


def configure_trainable_params(student: LMHead, stage: str, train_scope: str) -> None:
    if train_scope == "auto":
        train_scope = "h_lm_head" if stage == "3" else "h"

    for param in student.parameters():
        param.requires_grad_(False)

    if train_scope in {"h", "h_lm_head", "h_lm_head_embed"}:
        for param in student.model.H_level.parameters():
            param.requires_grad_(True)
    if train_scope in {"h_lm_head", "h_lm_head_embed"}:
        for param in student.lm_head.parameters():
            param.requires_grad_(True)
    if train_scope == "h_lm_head_embed":
        for param in student.embed_tokens.parameters():
            param.requires_grad_(True)


def zero_like_loss(student: LMHead) -> Tensor:
    return next(student.parameters()).new_zeros(())


def next_loader_batch(iterator: Any, loader: DataLoader) -> tuple[dict[str, Tensor], dict[str, Any], Any]:
    try:
        batch, info = next(iterator)
    except StopIteration:
        iterator = iter(loader)
        batch, info = next(iterator)
    return batch, info, iterator


def collect_eval_batches(loader: DataLoader, skip_batches: int, val_batches: int) -> list[tuple[dict[str, Tensor], dict[str, Any]]]:
    if val_batches <= 0:
        return []
    iterator = iter(loader)
    for _ in range(skip_batches):
        _batch, _info, iterator = next_loader_batch(iterator, loader)
    batches = []
    for _ in range(val_batches):
        batch, info, iterator = next_loader_batch(iterator, loader)
        batches.append((batch, info))
    return batches


def compute_alignment_tensors(
    args: argparse.Namespace,
    teacher: LMHead,
    student: LMHead,
    batch: dict[str, Tensor],
    info: dict[str, Any],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    batch = move_v1_batch_to_device(batch, info, device)
    supervised_mask = batch["labels"] != IGNORE_LABEL_ID
    hidden_mask = choose_token_mask(args, batch, info, "hidden")
    kl_mask = choose_token_mask(args, batch, info, "kl")
    supervised_tokens = int(supervised_mask.sum().detach().cpu())
    hidden_tokens = int(hidden_mask.sum().detach().cpu())
    kl_tokens = int(kl_mask.sum().detach().cpu())
    if hidden_tokens == 0 or (args.ce_weight > 0 and supervised_tokens == 0) or (args.kl_weight > 0 and kl_tokens == 0):
        return None

    teacher_logits = None
    with torch.autocast(device_type=device.type, dtype=dtype, enabled=(args.dtype == "bf16" and device.type == "cuda")):
        if args.stage == "legacy":
            with torch.no_grad():
                if args.kl_weight > 0:
                    teacher_hidden, teacher_logits = forward_hidden_logits(teacher, batch, args.bp_steps)
                else:
                    teacher_hidden = forward_hidden(teacher, batch, args.bp_steps)

            student_hidden, student_logits = forward_hidden_logits(student, batch, args.bp_steps)
            final_hidden_loss = supervised_hidden_loss(student_hidden, teacher_hidden, hidden_mask, args.hidden_loss_type)
            cycle_hidden_loss = zero_like_loss(student)
            layer_hidden_loss = zero_like_loss(student)

        elif args.stage == "1":
            student_cycles, teacher_cycles, student_layers, teacher_layers = trace_local_h_alignment_states(
                teacher, student, batch, args.bp_steps
            )
            student_logits = None
            final_hidden_loss = supervised_hidden_loss(student_cycles[-1], teacher_cycles[-1], hidden_mask, args.hidden_loss_type)
            cycle_hidden_loss = _loss_over_states(student_cycles, teacher_cycles, hidden_mask, args.hidden_loss_type)
            layer_hidden_loss = _loss_over_states(student_layers, teacher_layers, hidden_mask, args.hidden_loss_type)

        else:
            with torch.no_grad():
                teacher_hidden, teacher_logits, teacher_cycles, teacher_layers = trace_hrm_h_states(teacher, batch, args.bp_steps)
            student_hidden, student_logits, student_cycles, student_layers = trace_hrm_h_states(student, batch, args.bp_steps)
            final_hidden_loss = supervised_hidden_loss(student_hidden, teacher_hidden, hidden_mask, args.hidden_loss_type)
            cycle_hidden_loss = _loss_over_states(student_cycles, teacher_cycles, hidden_mask, args.hidden_loss_type)
            layer_hidden_loss = _loss_over_states(student_layers, teacher_layers, hidden_mask, args.hidden_loss_type)

        if student_logits is not None and args.ce_weight > 0:
            ce_loss = F.cross_entropy(
                student_logits.float().flatten(0, -2),
                batch["labels"].long().flatten(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="mean",
            )
        else:
            ce_loss = zero_like_loss(student)

        if student_logits is not None and teacher_logits is not None and args.kl_weight > 0:
            kl_loss = supervised_kl_loss(student_logits, teacher_logits, kl_mask, args.kl_temperature)
        else:
            kl_loss = zero_like_loss(student)

        hidden_loss = (
            args.hidden_weight * final_hidden_loss
            + args.cycle_hidden_weight * cycle_hidden_loss
            + args.layer_hidden_weight * layer_hidden_loss
        )
        loss = hidden_loss + args.ce_weight * ce_loss + args.kl_weight * kl_loss

    with torch.no_grad():
        if student_logits is not None and supervised_tokens > 0:
            is_correct = torch.argmax(student_logits, dim=-1) == batch["labels"]
            accuracy = float((is_correct[supervised_mask].sum() / supervised_mask.sum()).detach().cpu())
        else:
            accuracy = 0.0

    return {
        "loss": loss,
        "ce_loss": ce_loss,
        "hidden_loss": hidden_loss,
        "final_hidden_loss": final_hidden_loss,
        "cycle_hidden_loss": cycle_hidden_loss,
        "layer_hidden_loss": layer_hidden_loss,
        "kl_loss": kl_loss,
        "accuracy": accuracy,
        "supervised_tokens": supervised_tokens,
        "hidden_tokens": hidden_tokens,
        "kl_tokens": kl_tokens,
        "tokens": int(info["total_seqlen"]),
    }


def detach_metric_row(metrics: dict[str, Any]) -> dict[str, Any]:
    row = dict(metrics)
    for key in ("loss", "ce_loss", "hidden_loss", "final_hidden_loss", "cycle_hidden_loss", "layer_hidden_loss", "kl_loss"):
        value = row[key]
        row[key] = float(value.detach().cpu()) if isinstance(value, Tensor) else float(value)
    return row


def mean_metric_rows(rows: list[dict[str, Any]], args: argparse.Namespace | None = None) -> dict[str, Any]:
    if not rows:
        return {}

    def weighted_mean(key: str, weight_key: str) -> float:
        total_weight = sum(max(int(row[weight_key]), 0) for row in rows)
        if total_weight <= 0:
            return sum(float(row[key]) for row in rows) / len(rows)
        return sum(float(row[key]) * max(int(row[weight_key]), 0) for row in rows) / total_weight

    out: dict[str, Any] = {}
    out["ce_loss"] = weighted_mean("ce_loss", "supervised_tokens")
    out["hidden_loss"] = weighted_mean("hidden_loss", "hidden_tokens")
    out["final_hidden_loss"] = weighted_mean("final_hidden_loss", "hidden_tokens")
    out["cycle_hidden_loss"] = weighted_mean("cycle_hidden_loss", "hidden_tokens")
    out["layer_hidden_loss"] = weighted_mean("layer_hidden_loss", "hidden_tokens")
    out["kl_loss"] = weighted_mean("kl_loss", "kl_tokens")
    out["accuracy"] = weighted_mean("accuracy", "supervised_tokens")
    if args is None:
        out["loss"] = weighted_mean("loss", "tokens")
    else:
        out["loss"] = out["hidden_loss"] + args.ce_weight * out["ce_loss"] + args.kl_weight * out["kl_loss"]
    for key in ("supervised_tokens", "hidden_tokens", "kl_tokens", "tokens"):
        out[key] = sum(int(row[key]) for row in rows)
    out["batches"] = len(rows)
    return out


def accumulated_loss(metrics: dict[str, Any], totals: dict[str, int], args: argparse.Namespace) -> Tensor:
    hidden_scale = metrics["hidden_tokens"] / max(totals["hidden_tokens"], 1)
    supervised_scale = metrics["supervised_tokens"] / max(totals["supervised_tokens"], 1)
    kl_scale = metrics["kl_tokens"] / max(totals["kl_tokens"], 1)
    return (
        metrics["hidden_loss"] * hidden_scale
        + args.ce_weight * metrics["ce_loss"] * supervised_scale
        + args.kl_weight * metrics["kl_loss"] * kl_scale
    )


@torch.no_grad()
def run_validation(
    args: argparse.Namespace,
    teacher: LMHead,
    student: LMHead,
    val_batches: list[tuple[dict[str, Tensor], dict[str, Any]]],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any] | None:
    if not val_batches:
        return None
    was_training = student.training
    student.eval()
    rows = []
    for batch, info in val_batches:
        metrics = compute_alignment_tensors(args, teacher, student, batch, info, device, dtype)
        if metrics is not None:
            rows.append(detach_metric_row(metrics))
    if was_training:
        student.train()
    if not rows:
        return None
    return mean_metric_rows(rows, args)


def train_alignment(args: argparse.Namespace) -> dict[str, Any]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32

    source_dir = Path(args.teacher_dir)
    init_path = Path(args.student_init)
    output_path = Path(args.output)
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text("")
    strict_init = args.student_init_strict == "true" or (
        args.student_init_strict == "auto" and args.stage in {"2", "3"}
    )

    teacher = build_teacher(source_dir, device=device, dtype=dtype)
    student = build_student(
        source_dir,
        init_path,
        device=device,
        dtype=dtype,
        rwkv7_expansion=args.rwkv7_expansion,
        rwkv7_backend=args.rwkv7_backend,
        strict_init=strict_init,
    )

    configure_trainable_params(student, args.stage, args.train_scope)

    trainable_params = sum(param.numel() for param in student.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in student.parameters())
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(args.dataset_path, args.batch_tokens, seed=args.seed)
    val_loader = make_loader(args.dataset_path, args.batch_tokens, seed=args.val_seed)
    val_batches = collect_eval_batches(val_loader, args.val_skip_batches, args.val_batches)
    iterator = iter(loader)
    for _ in range(args.skip_batches):
        _batch, _info, iterator = next_loader_batch(iterator, loader)

    fixed_batch: tuple[dict[str, Tensor], dict[str, Any]] | None = None
    if args.fixed_batch:
        batch, info, iterator = next_loader_batch(iterator, loader)
        fixed_batch = (batch, info)

    rows = []
    val_rows = []
    recent_hidden_losses = deque(maxlen=max(1, args.target_window))
    started = time.perf_counter()
    student.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        train_micro_batches = []
        micro_totals = {"supervised_tokens": 0, "hidden_tokens": 0, "kl_tokens": 0, "tokens": 0}
        for _micro_step in range(args.grad_accum_steps):
            if fixed_batch is not None:
                batch, info = fixed_batch
            else:
                batch, info, iterator = next_loader_batch(iterator, loader)

            counts = batch_token_counts(args, batch, info)
            if should_skip_counts(args, counts):
                continue
            train_micro_batches.append((batch, info))
            for key in micro_totals:
                micro_totals[key] += counts[key]

        train_micro_rows = []
        for batch, info in train_micro_batches:
            metrics = compute_alignment_tensors(args, teacher, student, batch, info, device, dtype)
            if metrics is None:
                continue

            loss = accumulated_loss(metrics, micro_totals, args)
            loss.backward()
            train_micro_rows.append(detach_metric_row(metrics))

        if not train_micro_rows:
            continue

        torch.nn.utils.clip_grad_norm_([param for param in student.parameters() if param.requires_grad], args.grad_clip)
        optimizer.step()

        row = mean_metric_rows(train_micro_rows, args) | {
            "step": step,
            "mode": "train",
            "stage": args.stage,
            "grad_accum_steps": len(train_micro_rows),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        recent_hidden_losses.append(row["hidden_loss"])
        with history_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step} loss={row['loss']:.6f} ce={row['ce_loss']:.6f} "
                f"hidden={row['hidden_loss']:.6f} final={row['final_hidden_loss']:.6f} "
                f"cycle={row['cycle_hidden_loss']:.6f} layer={row['layer_hidden_loss']:.6f} "
                f"kl={row['kl_loss']:.6f} acc={row['accuracy']:.4f} "
                f"tokens={row['tokens']} hidden_tokens={row['hidden_tokens']} supervised={row['supervised_tokens']}",
                flush=True,
            )
        if val_batches and args.val_every > 0 and (step == 1 or step % args.val_every == 0):
            val_row = run_validation(args, teacher, student, val_batches, device, dtype)
            if val_row is not None:
                val_row = val_row | {
                    "step": step,
                    "mode": "val",
                    "stage": args.stage,
                    "elapsed_seconds": time.perf_counter() - started,
                }
                val_rows.append(val_row)
                with history_path.open("a") as f:
                    f.write(json.dumps(val_row, sort_keys=True) + "\n")
                print(
                    f"val step={step} loss={val_row['loss']:.6f} ce={val_row['ce_loss']:.6f} "
                    f"hidden={val_row['hidden_loss']:.6f} final={val_row['final_hidden_loss']:.6f} "
                    f"cycle={val_row['cycle_hidden_loss']:.6f} layer={val_row['layer_hidden_loss']:.6f} "
                    f"kl={val_row['kl_loss']:.6f} acc={val_row['accuracy']:.4f} "
                    f"tokens={val_row['tokens']} batches={val_row['batches']}",
                    flush=True,
                )
        if args.save_every > 0 and step % args.save_every == 0:
            save_student_state(
                student,
                checkpoint_path_for_step(output_path, step),
                {
                    "source_teacher": str(source_dir),
                    "student_init": str(init_path),
                    "alignment_steps": str(step),
                    "trainable_params": str(trainable_params),
                },
            )
            prune_step_checkpoints(output_path, args.keep_last_checkpoints)
        if args.target_hidden_loss > 0 and len(recent_hidden_losses) == recent_hidden_losses.maxlen:
            rolling_hidden = sum(recent_hidden_losses) / len(recent_hidden_losses)
            if rolling_hidden <= args.target_hidden_loss:
                print(
                    f"target reached step={step} rolling_hidden={rolling_hidden:.6f} "
                    f"window={len(recent_hidden_losses)}",
                    flush=True,
                )
                break

    if args.output and not args.no_save_final:
        save_student_state(
            student,
            output_path,
            {
                "source_teacher": str(source_dir),
                "student_init": str(init_path),
                "alignment_steps": str(args.steps),
                "trainable_params": str(trainable_params),
            },
        )

    summary = {
        "teacher_dir": str(source_dir),
        "student_init": str(init_path),
        "output": str(output_path) if args.output else "",
        "history": str(history_path),
        "steps": len(rows),
        "total_params": total_params,
        "trainable_params": trainable_params,
        "first": rows[0] if rows else None,
        "last": rows[-1] if rows else None,
        "val_last": val_rows[-1] if val_rows else None,
    }
    if args.summary:
        summary_path = Path(args.summary)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Align H-RWKV hidden states to an HRM-Text Transformer teacher.")
    parser.add_argument("--teacher-dir", default="/run/media/xiaol/B214449214445C0B/hf_models/sapientinc/HRM-Text-1B")
    parser.add_argument("--student-init", default="/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_init.safetensors")
    parser.add_argument("--dataset-path", default="/home/xiaol/X/hrm_text_subset_1B")
    parser.add_argument("--output", default="/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_aligned.safetensors")
    parser.add_argument("--history", default="/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_alignment.jsonl")
    parser.add_argument("--summary", default="/run/media/xiaol/B214449214445C0B/hrm_text_migrations/hrm_text_1b_to_hrm_h_rwkv7_alignment_summary.json")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--skip-batches", type=int, default=0)
    parser.add_argument("--batch-tokens", type=int, default=128)
    parser.add_argument("--bp-steps", type=int, default=5)
    parser.add_argument("--stage", default="legacy", choices=["legacy", "1", "2", "3"])
    parser.add_argument("--train-scope", default="auto", choices=["auto", "h", "h_lm_head", "h_lm_head_embed"])
    parser.add_argument("--fixed-batch", action="store_true")
    parser.add_argument("--hidden-mask", default="supervised", choices=["supervised", "valid"])
    parser.add_argument("--hidden-loss-type", default="l2", choices=["l2", "cosine"])
    parser.add_argument("--kl-mask", default="supervised", choices=["supervised", "valid"])
    parser.add_argument("--student-init-strict", default="auto", choices=["auto", "true", "false"])
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--cycle-hidden-weight", type=float, default=0.0)
    parser.add_argument("--layer-hidden-weight", type=float, default=0.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--rwkv7-expansion", type=float, default=1.0)
    parser.add_argument("--rwkv7-backend", default="cuda", choices=["auto", "cuda", "torch"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--keep-last-checkpoints", type=int, default=0)
    parser.add_argument("--no-save-final", action="store_true")
    parser.add_argument("--val-batches", type=int, default=4)
    parser.add_argument("--val-skip-batches", type=int, default=10000)
    parser.add_argument("--val-every", type=int, default=500)
    parser.add_argument("--val-seed", type=int, default=4321)
    parser.add_argument("--target-hidden-loss", type=float, default=0.0)
    parser.add_argument("--target-window", type=int, default=100)
    args = parser.parse_args()

    if args.stage == "1":
        args.hidden_mask = "valid" if args.hidden_mask == "supervised" else args.hidden_mask
        args.ce_weight = 0.0 if args.ce_weight == 1.0 else args.ce_weight
        args.hidden_weight = 0.0 if args.hidden_weight == 1.0 else args.hidden_weight
        args.cycle_hidden_weight = 0.0 if args.cycle_hidden_weight == 0.0 else args.cycle_hidden_weight
        args.layer_hidden_weight = 1.0 if args.layer_hidden_weight == 0.0 else args.layer_hidden_weight
        args.kl_weight = 0.0
    elif args.stage == "2":
        args.ce_weight = 0.1 if args.ce_weight == 1.0 else args.ce_weight
        args.hidden_weight = 1.0 if args.hidden_weight == 1.0 else args.hidden_weight
        args.cycle_hidden_weight = 0.5 if args.cycle_hidden_weight == 0.0 else args.cycle_hidden_weight
        args.layer_hidden_weight = 0.2 if args.layer_hidden_weight == 0.0 else args.layer_hidden_weight
        args.kl_weight = 0.3 if args.kl_weight == 0.0 else args.kl_weight
    elif args.stage == "3":
        args.ce_weight = 1.0 if args.ce_weight == 1.0 else args.ce_weight
        args.hidden_weight = 0.1 if args.hidden_weight == 1.0 else args.hidden_weight
        args.cycle_hidden_weight = 0.05 if args.cycle_hidden_weight == 0.0 else args.cycle_hidden_weight
        args.layer_hidden_weight = 0.02 if args.layer_hidden_weight == 0.0 else args.layer_hidden_weight
        args.kl_weight = 0.5 if args.kl_weight == 0.0 else args.kl_weight

    summary = train_alignment(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
