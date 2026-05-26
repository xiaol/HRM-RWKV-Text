from __future__ import annotations

import argparse
import gc
import glob
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModel
from models.baselines.hrm_hybrid_rwkv7_nocarry_bp_warmup import HierarchicalHybridRWKV7Model
from models.baselines.hrm_rwkv7_nocarry_bp_warmup import HierarchicalRWKV7Model
from models.common import wrap_tensor
from models.lm_head import LMHead
from dataset_new import V1Dataset, V1DatasetConfig


HYBRID_ARCHS = {
    "hybrid_h_rwkv7": ("rwkv7", "transformer"),
    "hybrid_l_rwkv7": ("transformer", "rwkv7"),
}


ARCH_ALIASES = {
    "hybrid_h": "hybrid_h_rwkv7",
    "h_rwkv7": "hybrid_h_rwkv7",
    "rwkv7_h": "hybrid_h_rwkv7",
    "hybrid_l": "hybrid_l_rwkv7",
    "l_rwkv7": "hybrid_l_rwkv7",
    "rwkv7_l": "hybrid_l_rwkv7",
}


def load_data_shard(path: Path) -> torch.Tensor:
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(path, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header: {path}")
    num_tokens = int(header[2])
    expected_size = header_bytes + num_tokens * token_bytes
    if path.stat().st_size != expected_size:
        raise ValueError(f"Shard size mismatch for {path}: expected {expected_size}")
    tokens_np = np.fromfile(path, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))


def resolve_files(pattern: str, max_shards: int) -> List[Path]:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    return files[:max_shards] if max_shards > 0 else files


class TokenStream:
    def __init__(self, files: Iterable[Path]):
        self.files = list(files)
        if not self.files:
            raise ValueError("TokenStream requires files")
        self.file_idx = 0
        self.tokens = load_data_shard(self.files[0])
        self.pos = 0

    def _advance_file(self) -> None:
        self.file_idx = (self.file_idx + 1) % len(self.files)
        self.tokens = load_data_shard(self.files[self.file_idx])
        self.pos = 0

    def take(self, n_tokens: int) -> torch.Tensor:
        chunks = []
        remaining = n_tokens
        while remaining > 0:
            available = self.tokens.numel() - self.pos
            if available <= 0:
                self._advance_file()
                continue
            n = min(remaining, available)
            chunks.append(self.tokens[self.pos:self.pos + n])
            self.pos += n
            remaining -= n
        return chunks[0] if len(chunks) == 1 else torch.cat(chunks)


def parse_named_patterns(raw: str) -> Dict[str, str]:
    patterns = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" in item:
            name, pattern = item.split("=", 1)
        else:
            pattern = item
            name = Path(pattern).parent.name or Path(pattern).stem
        name = name.strip()
        pattern = pattern.strip()
        if not name or not pattern:
            raise ValueError(f"Invalid named pattern: {item!r}")
        patterns[name] = pattern
    return patterns


def next_batch(stream: TokenStream, batch_size: int, seq_len: int, device: torch.device) -> Dict[str, torch.Tensor]:
    tokens = stream.take(batch_size * seq_len + 1).to(dtype=torch.int64)
    x = tokens[:-1].reshape(batch_size, seq_len).to(device, non_blocking=True)
    y = tokens[1:].reshape(batch_size, seq_len).to(device, non_blocking=True)
    return {"inputs": x, "labels": y}


def _arch_expansion(args: argparse.Namespace, arch: str) -> float:
    if arch == "transformer" and args.transformer_expansion is not None:
        return args.transformer_expansion
    if arch == "rwkv7" and args.rwkv7_expansion is not None:
        return args.rwkv7_expansion
    return args.expansion


def normalize_arch(arch: str) -> str:
    arch = arch.strip().lower()
    return ARCH_ALIASES.get(arch, arch)


def _uses_rwkv7(arch: str) -> bool:
    return arch == "rwkv7" or arch in HYBRID_ARCHS


def _rwkv7_full_channel_cuda_eligible(args: argparse.Namespace, arch: str) -> bool:
    return (
        _uses_rwkv7(arch)
        and args.rwkv7_backend != "torch"
        and args.dtype == "bf16"
        and args.rwkv7_head_size == 64
        and math.isclose(args.rwkv7_expansion if args.rwkv7_expansion is not None else args.expansion, 1.0)
    )


def common_config(args: argparse.Namespace, arch: str) -> dict:
    cfg = {
        "max_seq_len": args.seq_len,
        "n_layers": args.n_layers,
        "hidden_size": args.hidden_size,
        "num_heads": args.num_heads,
        "expansion": _arch_expansion(args, arch),
        "norm_type": "pre",
        "norm_eps": 1e-6,
        "pos_emb_type": "rope",
        "rope_theta": 10000.0,
        "init_type": "lecun_normal",
        "half_layers": args.half_layers,
        "H_cycles": args.h_cycles,
        "L_cycles": args.l_cycles,
        "H_override": {},
        "bp_warmup_ratio": 0.0,
        "bp_min_steps": args.bp_steps,
        "bp_max_steps": args.bp_steps,
        "vocab_size": args.vocab_size,
        "target_only": False,
    }
    if arch in HYBRID_ARCHS:
        h_arch, l_arch = HYBRID_ARCHS[arch]
        cfg.update(
            {
                "H_arch": h_arch,
                "L_arch": l_arch,
                "transformer_expansion": args.transformer_expansion if args.transformer_expansion is not None else args.expansion,
                "rwkv7_expansion": args.rwkv7_expansion if args.rwkv7_expansion is not None else args.expansion,
            }
        )
    if _uses_rwkv7(arch):
        cfg.update(
            {
                "rwkv7_head_size": args.rwkv7_head_size,
                "rwkv7_backend": args.rwkv7_backend,
                "rwkv7_chunk_len": args.rwkv7_chunk_len,
                "rwkv7_enable_v_first_mix": True,
            }
        )
    return cfg


def build_model(args: argparse.Namespace, arch: str, device: torch.device) -> torch.nn.Module:
    cfg = common_config(args, arch)
    if arch == "rwkv7":
        model_cls = HierarchicalRWKV7Model
    elif arch in HYBRID_ARCHS:
        model_cls = HierarchicalHybridRWKV7Model
    elif arch == "transformer":
        model_cls = HierarchicalReasoningModel
    else:
        raise ValueError(f"Unknown arch: {arch}")
    model = LMHead(model_cls(cfg), cfg).to(device)
    if args.dtype == "bf16":
        model = model.to(dtype=torch.bfloat16)
    elif args.dtype == "fp32":
        model = model.to(dtype=torch.float32)
    else:
        raise ValueError(args.dtype)
    return model


def clear_cuda_cache(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def move_v1_batch_to_device(batch: dict[str, torch.Tensor], info: dict, device: torch.device) -> dict:
    moved = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
    moved.update({k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in info.items()})
    return moved


def make_v1_loader(dataset_path: str, batch_tokens: int, target_only: bool, drop_last_batch: bool) -> DataLoader:
    dataset = V1Dataset(
        V1DatasetConfig(
            seed=0,
            dataset_path=dataset_path,
            batch_max_length=batch_tokens,
            drop_last_batch=drop_last_batch,
            target_only=target_only,
            rank=0,
            num_replicas=1,
        )
    )
    return DataLoader(dataset, batch_size=None, num_workers=0)


@torch.no_grad()
def evaluate_loss(
    args: argparse.Namespace,
    model: torch.nn.Module,
    files: List[Path],
    device: torch.device,
    batches: int,
) -> dict:
    was_training = model.training
    model.eval()
    stream = TokenStream(files)
    losses = []
    for _ in range(batches):
        batch = next_batch(stream, args.eval_batch_size, args.seq_len, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            _, loss, _metrics = model(None, batch, bp_steps=args.bp_steps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite eval loss: {loss.item()}")
        losses.append(float(loss.detach().cpu()))
    if was_training:
        model.train()
    tokens = batches * args.eval_batch_size * args.seq_len
    return {
        "batches": batches,
        "tokens": tokens,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def evaluate_named_sets(args: argparse.Namespace, model: torch.nn.Module, device: torch.device) -> dict[str, dict]:
    if not args.val_patterns:
        return {}
    val_results = {}
    for name, pattern in parse_named_patterns(args.val_patterns).items():
        files = resolve_files(pattern, args.max_val_shards)
        result = evaluate_loss(args, model, files, device, args.val_batches)
        result["files"] = [str(p) for p in files]
        val_results[name] = result
    return val_results


@torch.no_grad()
def evaluate_v1_loss(args: argparse.Namespace, model: torch.nn.Module, device: torch.device, batches: int) -> dict:
    was_training = model.training
    model.eval()
    loader = make_v1_loader(args.v1_dataset_path, args.v1_eval_batch_tokens, args.v1_target_only, drop_last_batch=False)
    losses = []
    supervised_tokens = 0
    total_tokens = 0
    for idx, (batch, info) in enumerate(loader):
        if idx >= batches:
            break
        batch = move_v1_batch_to_device(batch, info, device)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            _, loss, metrics = model(None, batch, bp_steps=args.bp_steps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite V1 eval loss: {loss.item()}")
        losses.append(float(loss.detach().cpu()))
        supervised_tokens += int(metrics["loss"][1].detach().cpu())
        total_tokens += int(info["total_seqlen"])
    if was_training:
        model.train()
    return {
        "batches": len(losses),
        "tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "mean_loss": sum(losses) / len(losses),
        "first_loss": losses[0],
        "last_loss": losses[-1],
    }


def run_arch(args: argparse.Namespace, arch: str, files: List[Path], device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    model = build_model(args, arch, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    stream = TokenStream(files)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    losses = []
    periodic_val = []
    timed_start = None
    for step in range(args.warmup_steps + args.steps):
        batch = next_batch(stream, args.batch_size, args.seq_len, device)
        if step == args.warmup_steps:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            timed_start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            _, loss, metrics = model(None, batch, bp_steps=args.bp_steps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arch}: non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        if step >= args.warmup_steps:
            losses.append(float(loss.detach().cpu()))
        if args.verbose:
            phase = "warmup" if step < args.warmup_steps else "timed"
            print(f"{arch} {phase}_step={step} loss={float(loss.detach().cpu()):.6f}", flush=True)
        timed_step = step - args.warmup_steps + 1
        if (
            args.val_patterns
            and args.val_every > 0
            and step >= args.warmup_steps
            and timed_step % args.val_every == 0
        ):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            periodic_val.append(
                {
                    "step": timed_step,
                    "sets": evaluate_named_sets(args, model, device),
                }
            )
            if device.type == "cuda":
                torch.cuda.synchronize(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory_mb = None
    elapsed = time.perf_counter() - (timed_start or time.perf_counter())
    tokens = args.steps * args.batch_size * args.seq_len
    final_val = evaluate_named_sets(args, model, device) if args.val_patterns and args.final_val else {}
    return {
        "arch": arch,
        "H_arch": HYBRID_ARCHS.get(arch, (arch, arch))[0],
        "L_arch": HYBRID_ARCHS.get(arch, (arch, arch))[1],
        "params": count_params(model),
        "expansion": _arch_expansion(args, arch),
        "transformer_expansion": args.transformer_expansion if args.transformer_expansion is not None else args.expansion,
        "rwkv7_expansion": args.rwkv7_expansion if args.rwkv7_expansion is not None else args.expansion,
        "rwkv7_full_channel_cuda_eligible": _rwkv7_full_channel_cuda_eligible(args, arch),
        "steps": args.steps,
        "tokens": tokens,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "mean_loss": sum(losses) / len(losses),
        "seconds": elapsed,
        "tokens_per_second": tokens / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "val_loss": final_val,
        "periodic_val_loss": periodic_val,
    }


def run_v1_arch(args: argparse.Namespace, arch: str, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    model = build_model(args, arch, device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loader = make_v1_loader(args.v1_dataset_path, args.v1_batch_tokens, args.v1_target_only, drop_last_batch=True)
    iterator = iter(loader)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    losses = []
    supervised_counts = []
    token_counts = []
    timed_start = None
    total_loops = args.warmup_steps + args.steps
    for step in range(total_loops):
        try:
            batch, info = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch, info = next(iterator)
        batch = move_v1_batch_to_device(batch, info, device)
        if step == args.warmup_steps:
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                torch.cuda.synchronize(device)
            timed_start = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.dtype == "bf16" and device.type == "cuda")):
            _, loss, metrics = model(None, batch, bp_steps=args.bp_steps)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{arch}: non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        optimizer.step()

        if step >= args.warmup_steps:
            losses.append(float(loss.detach().cpu()))
            supervised_counts.append(int(metrics["loss"][1].detach().cpu()))
            token_counts.append(int(info["total_seqlen"]))
        if args.verbose:
            phase = "warmup" if step < args.warmup_steps else "timed"
            print(
                f"{arch} {phase}_step={step} loss={float(loss.detach().cpu()):.6f} "
                f"supervised_tokens={int(metrics['loss'][1].detach().cpu())} total_tokens={int(info['total_seqlen'])}",
                flush=True,
            )

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak_memory_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    else:
        peak_memory_mb = None
    elapsed = time.perf_counter() - (timed_start or time.perf_counter())
    total_tokens = sum(token_counts)
    supervised_tokens = sum(supervised_counts)
    v1_val_loss = evaluate_v1_loss(args, model, device, args.v1_val_batches) if args.v1_val_batches > 0 else {}
    return {
        "arch": arch,
        "H_arch": HYBRID_ARCHS.get(arch, (arch, arch))[0],
        "L_arch": HYBRID_ARCHS.get(arch, (arch, arch))[1],
        "params": count_params(model),
        "expansion": _arch_expansion(args, arch),
        "transformer_expansion": args.transformer_expansion if args.transformer_expansion is not None else args.expansion,
        "rwkv7_expansion": args.rwkv7_expansion if args.rwkv7_expansion is not None else args.expansion,
        "rwkv7_full_channel_cuda_eligible": _rwkv7_full_channel_cuda_eligible(args, arch),
        "steps": args.steps,
        "tokens": total_tokens,
        "supervised_tokens": supervised_tokens,
        "first_loss": losses[0],
        "last_loss": losses[-1],
        "loss_delta": losses[-1] - losses[0],
        "mean_loss": sum(losses) / len(losses),
        "seconds": elapsed,
        "tokens_per_second": total_tokens / elapsed,
        "supervised_tokens_per_second": supervised_tokens / elapsed,
        "peak_memory_mb": peak_memory_mb,
        "v1_val_loss": v1_val_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark HRM-Text Transformer, RWKV-7, and H/L hybrid cores.")
    parser.add_argument("--mode", default="causal-bin", choices=["causal-bin", "v1"])
    parser.add_argument("--data-dir", default="/home/xiaol/X/parameter-golf/data/datasets/fineweb10B_sp1024")
    parser.add_argument("--train-pattern", default="")
    parser.add_argument("--max-train-shards", type=int, default=1)
    parser.add_argument("--v1-dataset-path", default="/home/xiaol/X/hrm_text_subset_1B")
    parser.add_argument("--v1-batch-tokens", type=int, default=4096)
    parser.add_argument("--v1-eval-batch-tokens", type=int, default=None)
    parser.add_argument("--v1-val-batches", type=int, default=10)
    parser.add_argument("--v1-target-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--val-patterns", default="", help="Comma-separated name=glob validation sets in Parameter Golf .bin format.")
    parser.add_argument("--max-val-shards", type=int, default=1)
    parser.add_argument("--val-batches", type=int, default=10)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--val-every", type=int, default=0, help="Run validation every N timed train steps; 0 disables periodic validation.")
    parser.add_argument("--no-final-val", action="store_true")
    parser.add_argument("--archs", default="transformer,rwkv7,hybrid_h_rwkv7,hybrid_l_rwkv7")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--dtype", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--half-layers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--expansion", type=float, default=0.5)
    parser.add_argument("--transformer-expansion", type=float, default=None)
    parser.add_argument("--rwkv7-expansion", type=float, default=None)
    parser.add_argument("--h-cycles", type=int, default=1)
    parser.add_argument("--l-cycles", type=int, default=1)
    parser.add_argument("--bp-steps", type=int, default=2)
    parser.add_argument("--rwkv7-head-size", type=int, default=64)
    parser.add_argument("--rwkv7-backend", default="auto", choices=["auto", "cuda", "torch"])
    parser.add_argument("--rwkv7-chunk-len", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", default="")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    args.final_val = not args.no_final_val
    if args.eval_batch_size is None:
        args.eval_batch_size = args.batch_size
    if args.v1_eval_batch_tokens is None:
        args.v1_eval_batch_tokens = args.v1_batch_tokens

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)

    results = []
    archs = [normalize_arch(x) for x in args.archs.split(",") if x.strip()]
    if args.mode == "causal-bin":
        train_pattern = args.train_pattern or str(Path(args.data_dir) / "fineweb_train_*.bin")
        files = resolve_files(train_pattern, args.max_train_shards)
        for arch in archs:
            clear_cuda_cache(device)
            results.append(run_arch(args, arch, files, device))
            clear_cuda_cache(device)

        print("arch,H_arch,L_arch,params,expansion,transformer_expansion,rwkv7_expansion,rwkv7_full_channel_cuda_eligible,tokens_per_second,first_loss,last_loss,loss_delta,mean_loss,peak_memory_mb,val_loss")
        for result in results:
            peak = "" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}"
            val = ";".join(f"{name}:{value['mean_loss']:.6f}" for name, value in result["val_loss"].items())
            print(
                f"{result['arch']},{result['H_arch']},{result['L_arch']},{result['params']},{result['expansion']:.6f},"
                f"{result['transformer_expansion']:.6f},{result['rwkv7_expansion']:.6f},{result['rwkv7_full_channel_cuda_eligible']},{result['tokens_per_second']:.2f},"
                f"{result['first_loss']:.6f},{result['last_loss']:.6f},{result['loss_delta']:.6f},"
                f"{result['mean_loss']:.6f},{peak},{val}"
            )
    else:
        for arch in archs:
            clear_cuda_cache(device)
            results.append(run_v1_arch(args, arch, device))
            clear_cuda_cache(device)

        print("arch,H_arch,L_arch,params,rwkv7_full_channel_cuda_eligible,tokens_per_second,supervised_tokens_per_second,first_loss,last_loss,loss_delta,mean_loss,val_loss,peak_memory_mb")
        for result in results:
            peak = "" if result["peak_memory_mb"] is None else f"{result['peak_memory_mb']:.2f}"
            val = "" if not result["v1_val_loss"] else f"{result['v1_val_loss']['mean_loss']:.6f}"
            print(
                f"{result['arch']},{result['H_arch']},{result['L_arch']},{result['params']},"
                f"{result['rwkv7_full_channel_cuda_eligible']},{result['tokens_per_second']:.2f},"
                f"{result['supervised_tokens_per_second']:.2f},{result['first_loss']:.6f},{result['last_loss']:.6f},"
                f"{result['loss_delta']:.6f},{result['mean_loss']:.6f},{val},{peak}"
            )

    if args.json_out:
        payload = {"args": vars(args), "results": results}
        if args.mode == "causal-bin":
            payload["train_files"] = [str(p) for p in files]
        Path(args.json_out).write_text(json.dumps(payload, indent=2) + "\n")


if __name__ == "__main__":
    main()
