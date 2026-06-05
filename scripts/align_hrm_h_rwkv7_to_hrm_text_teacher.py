from __future__ import annotations

import argparse
import json
import sys
import time
from math import prod
from pathlib import Path
from typing import Any

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
from models.common import IGNORE_LABEL_ID, wrap_tensor
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
) -> LMHead:
    hf_config = json.loads((source_dir / "config.json").read_text())
    cfg = student_config_from_hf(hf_config, rwkv7_expansion=rwkv7_expansion, rwkv7_backend=rwkv7_backend)
    model = LMHead(HierarchicalHybridRWKV7Model(cfg), cfg).to(device=device, dtype=dtype)
    tensors = load_file(init_path, device="cpu")
    incompatible = model.load_state_dict(tensors, strict=False)
    print(
        f"[Student init] loaded={len(tensors)} missing={len(incompatible.missing_keys)} "
        f"unexpected={len(incompatible.unexpected_keys)}",
        flush=True,
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(f"Unexpected student init keys: {incompatible.unexpected_keys[:20]}")
    return model


def make_loader(dataset_path: str, batch_tokens: int) -> DataLoader:
    dataset = V1Dataset(
        V1DatasetConfig(
            seed=0,
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


def supervised_hidden_loss(student_hidden: Tensor, teacher_hidden: Tensor, mask: Tensor) -> Tensor:
    student_norm = F.normalize(student_hidden.float(), dim=-1)
    teacher_norm = F.normalize(teacher_hidden.float(), dim=-1)
    cosine = (student_norm[mask] * teacher_norm[mask]).sum(dim=-1)
    return (1.0 - cosine).mean()


def supervised_kl_loss(student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, temperature: float) -> Tensor:
    student_logp = F.log_softmax(student_logits[mask].float() / temperature, dim=-1)
    teacher_prob = F.softmax(teacher_logits[mask].float() / temperature, dim=-1)
    return F.kl_div(student_logp, teacher_prob, reduction="batchmean") * (temperature**2)


def save_student_state(model: LMHead, output_path: Path, metadata: dict[str, str]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    state = {key: value.detach().cpu().contiguous() for key, value in model.state_dict().items()}
    save_file(state, str(output_path), metadata=metadata)


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

    teacher = build_teacher(source_dir, device=device, dtype=dtype)
    student = build_student(
        source_dir,
        init_path,
        device=device,
        dtype=dtype,
        rwkv7_expansion=args.rwkv7_expansion,
        rwkv7_backend=args.rwkv7_backend,
    )

    for param in student.parameters():
        param.requires_grad_(False)
    for param in student.model.H_level.parameters():
        param.requires_grad_(True)

    trainable_params = sum(param.numel() for param in student.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in student.parameters())
    optimizer = torch.optim.AdamW(
        [param for param in student.parameters() if param.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    loader = make_loader(args.dataset_path, args.batch_tokens)
    iterator = iter(loader)
    for _ in range(args.skip_batches):
        try:
            next(iterator)
        except StopIteration:
            iterator = iter(loader)
            next(iterator)

    rows = []
    started = time.perf_counter()
    student.train()
    for step in range(1, args.steps + 1):
        try:
            batch, info = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, info = next(iterator)
        batch = move_v1_batch_to_device(batch, info, device)
        mask = batch["labels"] != IGNORE_LABEL_ID
        supervised_tokens = int(mask.sum().detach().cpu())
        if supervised_tokens == 0:
            continue

        optimizer.zero_grad(set_to_none=True)
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=dtype, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            if args.kl_weight > 0:
                teacher_hidden, teacher_logits = forward_hidden_logits(teacher, batch, args.bp_steps)
            else:
                teacher_hidden = forward_hidden(teacher, batch, args.bp_steps)
                teacher_logits = None

        with torch.autocast(device_type=device.type, dtype=dtype, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            student_hidden, student_logits = forward_hidden_logits(student, batch, args.bp_steps)
            ce_loss = F.cross_entropy(
                student_logits.float().flatten(0, -2),
                batch["labels"].long().flatten(),
                ignore_index=IGNORE_LABEL_ID,
                reduction="mean",
            )
            hidden_loss = supervised_hidden_loss(student_hidden, teacher_hidden, mask)
            if args.kl_weight > 0:
                assert teacher_logits is not None
                kl_loss = supervised_kl_loss(student_logits, teacher_logits, mask, args.kl_temperature)
            else:
                kl_loss = student_logits.new_zeros(())
            loss = args.ce_weight * ce_loss + args.hidden_weight * hidden_loss + args.kl_weight * kl_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_([param for param in student.parameters() if param.requires_grad], args.grad_clip)
        optimizer.step()

        with torch.no_grad():
            is_correct = torch.argmax(student_logits, dim=-1) == batch["labels"]
            accuracy = float((is_correct[mask].sum() / mask.sum()).detach().cpu())
        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "ce_loss": float(ce_loss.detach().cpu()),
            "hidden_loss": float(hidden_loss.detach().cpu()),
            "kl_loss": float(kl_loss.detach().cpu()),
            "accuracy": accuracy,
            "supervised_tokens": supervised_tokens,
            "tokens": int(info["total_seqlen"]),
            "elapsed_seconds": time.perf_counter() - started,
        }
        rows.append(row)
        with history_path.open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
        if step == 1 or step % args.log_every == 0:
            print(
                f"step={step} loss={row['loss']:.6f} ce={row['ce_loss']:.6f} "
                f"hidden={row['hidden_loss']:.6f} acc={row['accuracy']:.4f} "
                f"tokens={row['tokens']} supervised={row['supervised_tokens']}",
                flush=True,
            )

    if args.output:
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
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ce-weight", type=float, default=1.0)
    parser.add_argument("--hidden-weight", type=float, default=1.0)
    parser.add_argument("--kl-weight", type=float, default=0.0)
    parser.add_argument("--kl-temperature", type=float, default=2.0)
    parser.add_argument("--rwkv7-expansion", type=float, default=1.0)
    parser.add_argument("--rwkv7-backend", default="cuda", choices=["auto", "cuda", "torch"])
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()

    summary = train_alignment(args)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
