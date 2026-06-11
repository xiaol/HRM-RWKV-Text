from typing import Any, Optional
from dataclasses import dataclass
from pathlib import Path
from glob import glob
import math
import os
import json
import signal
import yaml
import shutil

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import get_optimizer_state_dict, set_optimizer_state_dict
from torch.distributed.fsdp import fully_shard, FSDPModule, MixedPrecisionPolicy
from torch import Tensor, nn
from torch.utils.data import DataLoader

import tqdm
import wandb
import coolname
import hydra
import pydantic
from omegaconf import DictConfig, OmegaConf

from models.layers import Carry
from models.common import IGNORE_LABEL_ID, wrap_tensor
from models.transformer import TransformerBlock
from models.adam_atan2 import AdamATan2
from utils.functions import load_model_class, get_model_source_path
from dataset_new import V1Dataset, V1DatasetConfig, V1DatasetMeta


class ArchConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    name: str
    head: str


class DataConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra='allow')

    path: str
    target_only: bool = True  # Only supervise Answer.


class PretrainConfig(pydantic.BaseModel):
    # Config
    arch: ArchConfig
    data: DataConfig

    # Hyperparams
    global_batch_size: int
    micro_batch_size: Optional[int] = None
    epochs: int

    lr: float
    lr_min_ratio: float
    lr_warmup_steps: int

    weight_decay: float
    beta1: float
    beta2: float
    ema: Optional[float] = None
    fwd_bwd_dtype: str = "bfloat16"
    compile_train: bool = True

    # Names
    project_name: Optional[str] = None
    run_name: Optional[str] = None
    checkpoint_path: Optional[str] = None

    # Resume / fine-tune from checkpoint
    resume_from: Optional[str] = None
    resume_tag: Optional[str] = None
    resume_epoch: Optional[int] = None
    resume_skip_data: bool = True
    init_from_safetensors: Optional[str] = None
    weights_only_resume_from_ema: bool = False  # Swap EMA into model + reset optim

    # Extras
    seed: int = 0
    max_steps: Optional[int] = None
    checkpoint_interval: int = 1
    save_checkpoints: bool = True
    log_interval: int = 5
    loss_history_path: Optional[str] = None
    trainable_param_substrings: Optional[list[str]] = None


@dataclass
class TrainState:
    model: nn.Module
    carry: Optional[Carry]
    
    optim: AdamATan2

    step: int
    total_steps: int


def create_dataloader(config: PretrainConfig, local_batch_size: int, drop_last_batch: bool, rank: int, world_size: int):
    dataset = V1Dataset(V1DatasetConfig(
        seed=config.seed,

        dataset_path=config.data.path,
        drop_last_batch=drop_last_batch,

        target_only=config.data.target_only,

        batch_max_length=local_batch_size,
        rank=rank,
        num_replicas=world_size,
    ))
    dataloader = DataLoader(
        dataset,
        batch_size=None,

        num_workers=1,
        prefetch_factor=8,

        pin_memory=True,
        persistent_workers=True  # NOTE: Required for correct epoch handling
    )
    return dataloader, dataset.metadata


def apply_fsdp(module: nn.Module, param_dtype: torch.dtype):
    fully_shard(module,
                mp_policy=MixedPrecisionPolicy(param_dtype=param_dtype,
                                               reduce_dtype=torch.get_default_dtype()),  # Use master dtype for reduction
                reshard_after_forward=False)  # Trade off VRAM for less comms
    
    assert isinstance(module, FSDPModule)
    # Disable gradient division. Adams is scale invariant.
    module.set_gradient_divide_factor(1.0)
    module.set_force_sum_reduction_for_comms(True)


def load_initial_safetensors(config: PretrainConfig, model: nn.Module, rank: int) -> None:
    if config.init_from_safetensors is None:
        return

    from safetensors.torch import load_file

    tensors = load_file(config.init_from_safetensors, device="cpu")
    incompatible = model.load_state_dict(tensors, strict=False)
    if rank == 0:
        print(
            f"[Init] Loaded {len(tensors)} tensors from {config.init_from_safetensors}; "
            f"missing={len(incompatible.missing_keys)} unexpected={len(incompatible.unexpected_keys)}"
        )
        if incompatible.unexpected_keys:
            print(f"[Init] Unexpected keys: {incompatible.unexpected_keys[:20]}")


def apply_trainable_filter(config: PretrainConfig, model: nn.Module, rank: int) -> None:
    patterns = config.trainable_param_substrings
    if not patterns:
        return

    trainable_names: list[str] = []
    trainable_count = 0
    for name, param in model.named_parameters():
        trainable = any(pattern in name for pattern in patterns)
        param.requires_grad_(trainable)
        if trainable:
            trainable_count += param.numel()
            trainable_names.append(name)

    if trainable_count == 0:
        raise ValueError(f"No trainable parameters matched trainable_param_substrings={patterns}")

    if rank == 0:
        total_params = sum(p.numel() for p in model.parameters())
        print(
            f"[Trainable] patterns={patterns} trainable={trainable_count:,} "
            f"total={total_params:,} ratio={trainable_count / total_params:.6f}"
        )
        print(f"[Trainable] first matched names: {trainable_names[:20]}")


def create_model_and_carry(config: PretrainConfig, train_metadata: V1DatasetMeta, local_batch_size: int, world_size: int, rank: int):
    model_cfg = config.arch.model_dump() | train_metadata.model_dump() | config.data.model_dump()
    fwd_bwd_dtype = getattr(torch, config.fwd_bwd_dtype)

    # Instantiate model with head
    model_cls = load_model_class(config.arch.name)
    head_cls = load_model_class(config.arch.head)

    with torch.device("cuda"):
        model: nn.Module = model_cls(model_cfg)
        carry = model.initial_carry(local_batch_size, dtype=fwd_bwd_dtype)  # pyright: ignore[reportCallIssue]
        # Attach loss head
        model = head_cls(model, model_cfg)
        if world_size == 1 and fwd_bwd_dtype != torch.float32:
            model = model.to(dtype=fwd_bwd_dtype)
        load_initial_safetensors(config, model, rank)
        apply_trainable_filter(config, model, rank)

    # ----FSDP----
    # FSDP only helps when parameters are actually sharded across ranks. On a
    # single 4090 it adds all-gather/reduce buffers without reducing model size.
    if world_size > 1:
        # Broadcast buffers
        for buffer in model.buffers():
            dist.broadcast(buffer, src=0)

        # Detect TransformerBlock recursively and apply FSDP
        for module in model.modules():
            if isinstance(module, TransformerBlock):
                apply_fsdp(module, fwd_bwd_dtype)

        apply_fsdp(model, fwd_bwd_dtype)

    # ----Create optimizer----
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise ValueError("No trainable parameters available for optimizer")
    optim = AdamATan2(trainable_params,
                      lr=torch.tensor(0.0, dtype=torch.get_default_dtype(), device="cpu"),
                      betas=(config.beta1, config.beta2),
                      weight_decay=config.weight_decay,
                      ema=config.ema)

    return model, carry, optim


def get_local_micro_batch_size(config: PretrainConfig, world_size: int) -> int:
    assert config.global_batch_size % world_size == 0, f"Global batch size {config.global_batch_size} must be divisible by world size {world_size}."
    local_effective_batch_size = config.global_batch_size // world_size
    if config.micro_batch_size is None:
        return local_effective_batch_size
    assert config.micro_batch_size > 0, "micro_batch_size must be positive."
    assert local_effective_batch_size % config.micro_batch_size == 0, (
        f"Local effective batch size {local_effective_batch_size} must be divisible by "
        f"micro_batch_size {config.micro_batch_size}."
    )
    return config.micro_batch_size


def init_train(config: PretrainConfig, rank: int, world_size: int):
    local_micro_batch_size = get_local_micro_batch_size(config, world_size)

    # Dataset
    train_loader, train_metadata = create_dataloader(config, local_micro_batch_size, drop_last_batch=True,  rank=rank, world_size=world_size)

    # Model
    model, carry, optim = create_model_and_carry(config, train_metadata, local_micro_batch_size, world_size, rank)

    # Train state
    # Estimated total training steps
    total_steps = int(config.epochs * train_metadata.total_length // config.global_batch_size)
    train_state = TrainState(
        model=model,
        carry=carry,
        optim=optim,
        
        step=0,
        total_steps=total_steps
    )
    return train_state, train_loader, train_metadata


def update_lr(config: PretrainConfig, train_state: TrainState) -> float:
    # Linear warmup cosine schedule
    if train_state.step < config.lr_warmup_steps:
        lr = config.lr * min(1.0, train_state.step / config.lr_warmup_steps)
    else:
        progress = (train_state.step - config.lr_warmup_steps) / (train_state.total_steps - config.lr_warmup_steps)
        lr = config.lr * (config.lr_min_ratio + max(0.0, (1 - config.lr_min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))))

    tensor_lr = torch.tensor(lr, dtype=torch.get_default_dtype(), device="cpu")
    for param_group in train_state.optim.param_groups:
        param_group["lr"] = tensor_lr

    return lr


@torch.compile(dynamic=False)
def train_batch(train_state: TrainState, batch: dict[str, Tensor], **kwargs):
    train_state.carry, loss, metrics = train_state.model(batch=batch, carry=train_state.carry, **kwargs)
    loss.backward()
    train_state.optim.step()
    train_state.optim.zero_grad()
    return metrics


def train_microbatch(train_state: TrainState, batch: dict[str, Tensor], loss_divisor: Tensor, autocast_dtype: Optional[torch.dtype] = None, **kwargs):
    autocast_enabled = autocast_dtype is not None and batch["inputs"].is_cuda
    with torch.autocast(device_type="cuda", dtype=autocast_dtype or torch.bfloat16, enabled=autocast_enabled):
        train_state.carry, loss, metrics = train_state.model(
            batch=batch,
            carry=train_state.carry,
            loss_divisor_override=loss_divisor,
            **kwargs,
        )
    loss.backward()
    return metrics


train_microbatch_compiled = torch.compile(train_microbatch, dynamic=False)


def move_batch_to_device(batch: dict[str, Tensor], device: torch.device) -> dict[str, Tensor]:
    moved = {}
    for k, v in batch.items():
        if isinstance(v, Tensor):
            moved[k] = v.to(device, non_blocking=True)
        else:
            moved[k] = v
    return moved


def move_batch_info_to_device(batch_info: dict[str, Any]) -> dict[str, Tensor]:
    return {k: wrap_tensor(torch.tensor(v, device="cpu")) for k, v in batch_info.items()}


@torch.inference_mode()
def reduce_metrics(local_metrics: dict[str, Tensor], prefix: str):
    metric_keys = list(sorted(local_metrics.keys()))  # Sort keys to guarantee all processes use the same order.
    # Reduce and reconstruct
    metric_values = torch.stack([local_metrics[k][0] for k in metric_keys] + [local_metrics[k][1] for k in metric_keys])
    dist.reduce(metric_values, dst=0)
    # Split and normalize
    metrics, metrics_div = metric_values.chunk(2, dim=-1)
    metrics = (metrics / metrics_div).cpu().numpy().tolist()
    return {prefix + name: metrics[idx] for idx, name in enumerate(metric_keys)}


def append_loss_history(config: PretrainConfig, epoch: int, step: int, metrics: dict[str, Any]) -> None:
    if config.loss_history_path is None:
        return
    history_dir = os.path.dirname(config.loss_history_path)
    if history_dir:
        os.makedirs(history_dir, exist_ok=True)
    record = {"epoch": epoch, "step": step}
    record.update(metrics)
    with open(config.loss_history_path, "a") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")


def _checkpoint_step(config: PretrainConfig, tag: str, rank: int) -> tuple[int, int]:
    state_path = os.path.join(config.resume_from or "", f"train_state_{tag}.{rank}.pt")
    if os.path.isfile(state_path):
        state = torch.load(state_path, map_location="cpu", weights_only=True)
        return int(state.get("epoch", 1)), int(state.get("step", 0))

    if tag.startswith("step_"):
        return 1, int(tag.removeprefix("step_"))
    if tag.startswith("epoch_"):
        return int(tag.removeprefix("epoch_")), 0
    raise ValueError(f"Cannot infer training state from checkpoint tag: {tag}")


def _resolve_resume_tag(config: PretrainConfig, rank: int) -> tuple[str, int, int]:
    assert config.resume_from is not None
    if config.resume_tag is not None:
        tag = config.resume_tag.removeprefix("fsdp2_")
        checkpoint_id = os.path.join(config.resume_from, f"fsdp2_{tag}")
        if not os.path.isdir(checkpoint_id):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_id}")
        epoch, step = _checkpoint_step(config, tag, rank)
        return tag, epoch, step

    candidates = []
    for checkpoint_id in glob(os.path.join(config.resume_from, "fsdp2_*")):
        if not os.path.isdir(checkpoint_id):
            continue
        tag = Path(checkpoint_id).name.removeprefix("fsdp2_")
        try:
            epoch, step = _checkpoint_step(config, tag, rank)
        except (FileNotFoundError, ValueError):
            continue
        candidates.append((step, epoch, tag))

    if config.resume_epoch is not None:
        tag = f"epoch_{config.resume_epoch}"
        epoch, step = _checkpoint_step(config, tag, rank)
        return tag, epoch, step
    if not candidates:
        raise FileNotFoundError(f"No checkpoint found in {config.resume_from}")

    step, epoch, tag = max(candidates)
    return tag, epoch, step


def load_checkpoint(config: PretrainConfig, train_state: TrainState, rank: int):
    """Resume from a saved checkpoint.

    Loads both model weights and optimizer state (which carries EMA in
    AdamATan2). When weights_only_resume_from_ema=True, swaps the EMA buffer
    into the model and resets the optimizer state — typical for fine-tuning
    off a pretrain run with EMA-smoothed weights.
    """
    if config.resume_from is None:
        return

    tag, epoch, step = _resolve_resume_tag(config, rank)
    checkpoint_id = os.path.join(config.resume_from, f"fsdp2_{tag}")
    print(f"[Resume] Loading model + optimizer from {checkpoint_id} (epoch={epoch}, step={step})")
    optim_state = get_optimizer_state_dict(train_state.model, train_state.optim)
    dcp.load(
        {"model": train_state.model.state_dict(), "optim": optim_state},
        checkpoint_id=checkpoint_id,
    )
    set_optimizer_state_dict(train_state.model, train_state.optim, optim_state)

    # set_optimizer_state_dict silently overwrites param_groups with the pretrain hyperparams
    # (lr, betas, weight_decay, ema). Restore the SFT cfg values so that overrides take effect.
    # (lr is also restored every step by update_lr() — these three are not.)
    for param_group in train_state.optim.param_groups:
        param_group["betas"] = (config.beta1, config.beta2)
        param_group["weight_decay"] = config.weight_decay
        param_group["ema"] = config.ema

    train_state.step = step
    carry_path = os.path.join(config.resume_from, f"carry_{tag}.{rank}.pt")
    if os.path.isfile(carry_path):
        train_state.carry = torch.load(carry_path, map_location="cpu", weights_only=False)

    if config.weights_only_resume_from_ema:
        print("[Resume] Swapping EMA into model and resetting optimizer state")
        train_state.optim.swap_ema()
        train_state.optim._init_state()
        train_state.step = 0

    print(f"[Resume] Done at step {train_state.step}.")


def save_code_and_config(config: PretrainConfig, train_metadata: V1DatasetMeta):
    if (not config.save_checkpoints) or config.checkpoint_path is None or wandb.run is None:
        return

    os.makedirs(config.checkpoint_path, exist_ok=True)

    # Copy code
    code_list = [
        get_model_source_path(config.arch.name)
    ]
    for code_file in code_list:
        if code_file is not None:
            code_name = os.path.basename(code_file)

            shutil.copy(code_file, os.path.join(config.checkpoint_path, code_name))

    # Dump config as yaml
    with open(os.path.join(config.checkpoint_path, "all_config.yaml"), "wt") as f:
        yaml.dump(config.model_dump(), f)
    with open(os.path.join(config.checkpoint_path, "train_metadata.yaml"), "wt") as f:
        yaml.dump(train_metadata.model_dump(), f)

    # Log code
    wandb.run.log_code(config.checkpoint_path)


def save_training_checkpoint(config: PretrainConfig, train_state: TrainState, epoch: int, rank: int, tag: str):
    if (not config.save_checkpoints) or config.checkpoint_path is None:
        return

    checkpoint_id = os.path.join(config.checkpoint_path, f"fsdp2_{tag}")
    dcp.save(
        {"model": train_state.model.state_dict(), "optim": get_optimizer_state_dict(train_state.model, train_state.optim)},  # pyright: ignore[reportPrivateImportUsage]
        checkpoint_id=checkpoint_id,
    )
    torch.save(train_state.carry, os.path.join(config.checkpoint_path, f"carry_{tag}.{rank}.pt"))
    torch.save({"epoch": epoch, "step": train_state.step}, os.path.join(config.checkpoint_path, f"train_state_{tag}.{rank}.pt"))


def load_synced_config(hydra_config: DictConfig, rank: int) -> PretrainConfig:
    objects = [None]
    if rank == 0:
        config = PretrainConfig(**OmegaConf.to_container(hydra_config, resolve=True))  # type: ignore

        # Naming
        if config.project_name is None:
            config.project_name = f"{Path(config.data.path).stem.capitalize()} HLM-torch"
        if config.run_name is None:
            config.run_name = os.environ.get("MLP_TASK_NAME", f"{config.arch.name.split('@')[-1]} {coolname.generate_slug(2)}")  # pyright: ignore[reportPrivateImportUsage]
        if config.checkpoint_path is None:
            config.checkpoint_path = os.path.join("checkpoints", config.project_name, config.run_name)

        objects = [config]

    dist.broadcast_object_list(objects, src=0)
    return objects[0]  # type: ignore


@hydra.main(config_path="config", config_name="cfg_pretrain", version_base=None)
def launch(hydra_config: DictConfig):
    WORLD_SIZE = 1
    RANK = 0
    DEVICE_ID = 0

    # Initialize distributed training if in distributed environment (e.g. torchrun)
    if "LOCAL_RANK" in os.environ:
        # Initialize distributed, default device and dtype
        dist.init_process_group(backend="nccl")

        WORLD_SIZE = dist.get_world_size()
        RANK = dist.get_rank()
        DEVICE_ID = int(os.environ["LOCAL_RANK"])

        torch.cuda.set_device(DEVICE_ID)
    else:
        torch.cuda.set_device(DEVICE_ID)
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29500")
        dist.init_process_group(backend="nccl", rank=0, world_size=1)

    device = torch.device("cuda", DEVICE_ID)

    # Load sync'ed config
    config = load_synced_config(hydra_config, rank=RANK)
    if config.resume_from is not None and config.init_from_safetensors is not None:
        raise ValueError("Use either resume_from or init_from_safetensors, not both.")

    # Seed RNGs to ensure consistency
    torch.random.manual_seed(config.seed + RANK)

    # --- Training
    train_state, train_loader, train_metadata = init_train(config, rank=RANK, world_size=WORLD_SIZE)
    load_checkpoint(config, train_state, rank=RANK)
    local_effective_batch_size = config.global_batch_size // WORLD_SIZE
    local_micro_batch_size = get_local_micro_batch_size(config, WORLD_SIZE)
    grad_accum_steps = local_effective_batch_size // local_micro_batch_size
    if config.resume_from is not None and config.resume_skip_data and train_state.step > 0:
        skip_batches = train_state.step * grad_accum_steps
        dataset = train_loader.dataset
        if not isinstance(dataset, V1Dataset):
            raise TypeError("resume_skip_data requires V1Dataset")
        dataset.set_skip_batches(skip_batches)
        print(f"[Resume] Skipping {skip_batches:,} consumed microbatches.")

    stop_requested = False

    def request_stop(signum, _frame):
        nonlocal stop_requested
        stop_requested = True
        if RANK == 0:
            print(
                f"\n[Signal] Received {signal.Signals(signum).name}; "
                "will checkpoint after the current optimizer step.",
                flush=True,
            )

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    train_microbatch_fn = train_microbatch_compiled if config.compile_train else train_microbatch
    autocast_dtype = getattr(torch, config.fwd_bwd_dtype) if WORLD_SIZE == 1 else None

    # Progress bar and logger
    progress_bar = None
    if RANK == 0:
        progress_bar = tqdm.tqdm(total=train_state.total_steps, initial=train_state.step)

        wandb.init(project=config.project_name, name=config.run_name, config=config.model_dump() | {"train_metadata": train_metadata.model_dump()},
                   settings=wandb.Settings(_disable_stats=True))  # type: ignore
        wandb.log({"num_params": sum(x.numel() for x in train_state.model.parameters())}, step=0)
        save_code_and_config(config, train_metadata)

    # Training Loop
    for epoch in range(1, config.epochs + 1):
        print (f"[Rank {RANK}, World Size {WORLD_SIZE}]: Epoch {epoch}")

        # ############ Train Iter
        train_state.model.train()
        microbatch_iter = iter(train_loader)
        while True:
            microbatches: list[dict[str, Tensor]] = []
            microbatch_infos: list[dict[str, Tensor]] = []
            local_valid_tokens = torch.tensor(0, dtype=torch.float32, device="cpu")
            for _i in range(grad_accum_steps):
                try:
                    batch, batch_info = next(microbatch_iter)
                except StopIteration:
                    break

                microbatches.append(batch)
                microbatch_infos.append(move_batch_info_to_device(batch_info))
                local_valid_tokens += (batch["labels"] != IGNORE_LABEL_ID).sum(dtype=torch.float32)

            if len(microbatches) != grad_accum_steps:
                break

            train_state.step += 1            
            lr = update_lr(config, train_state)
            # Extra train arguments (such as BP warmup etc.)
            train_extra_args = train_state.model.compute_train_extra_args(train_state)  # pyright: ignore[reportCallIssue]
            loss_divisor = local_valid_tokens.detach().to(device)
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(loss_divisor, op=dist.ReduceOp.AVG)

            train_state.optim.zero_grad()
            metrics_accum: Optional[dict[str, tuple[Tensor, Tensor]]] = None
            for batch, batch_info in zip(microbatches, microbatch_infos):
                batch = move_batch_to_device(batch, device)
                metrics = train_microbatch_fn(train_state, batch | batch_info, loss_divisor=loss_divisor, autocast_dtype=autocast_dtype, **train_extra_args)
                if metrics_accum is None:
                    metrics_accum = {k: (v[0].detach(), v[1].detach()) for k, v in metrics.items()}
                else:
                    metrics_accum = {
                        k: (metrics_accum[k][0] + v[0].detach(), metrics_accum[k][1] + v[1].detach())
                        for k, v in metrics.items()
                    }
                del metrics
            train_state.optim.step()
            metrics = metrics_accum
            assert metrics is not None

            if train_state.step % config.log_interval == 0:
                metrics = reduce_metrics(metrics, prefix="train/")
                if RANK == 0:
                    progress_bar.update(train_state.step - progress_bar.n)  # type: ignore
                    log_metrics = metrics | train_extra_args | {"train/lr": lr, "train/grad_accum_steps": grad_accum_steps}
                    append_loss_history(config, epoch, train_state.step, log_metrics)
                    wandb.log(
                        log_metrics,
                        step=train_state.step,
                    )

            del metrics

            if stop_requested:
                break
            if config.max_steps is not None and train_state.step >= config.max_steps:
                break

        if stop_requested or (config.max_steps is not None and train_state.step >= config.max_steps):
            save_training_checkpoint(config, train_state, epoch, RANK, f"step_{train_state.step}")
            break

        ############ EVAL STACK: TBD TODO

        ############ Checkpointing
        if (epoch % config.checkpoint_interval == 0) or (epoch == config.epochs):
            save_training_checkpoint(config, train_state, epoch, RANK, f"epoch_{epoch}")

    # finalize
    if dist.is_initialized():
        dist.destroy_process_group()
    wandb.finish()
    if stop_requested and RANK == 0:
        print(f"[Signal] Safe stop complete at step {train_state.step}.", flush=True)


if __name__ == "__main__":
    launch()
