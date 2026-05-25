from typing import Any, Dict, Literal, Optional, Tuple

import torch
from torch import Tensor, nn
from pydantic import Field

from models.baselines.hrm_nocarry_bp_warmup import HierarchicalReasoningModelRecurrentBlock
from models.common import trunc_normal_init_
from models.rwkv7 import RWKV7Config, RWKV7ReasoningBlock
from models.transformer import Cache, TransformerConfig


CoreArch = Literal["transformer", "rwkv7"]


class HierarchicalHybridRWKV7ModelConfig(TransformerConfig):
    half_layers: bool = False

    H_cycles: int
    L_cycles: int

    H_arch: CoreArch = "rwkv7"
    L_arch: CoreArch = "transformer"

    transformer_expansion: Optional[float] = None
    rwkv7_expansion: Optional[float] = None

    bp_warmup_ratio: float = 0.0
    bp_min_steps: int = 2
    bp_max_steps: int = 5

    H_override: Dict[str, Any] = Field(default_factory=dict)
    L_override: Dict[str, Any] = Field(default_factory=dict)

    rwkv7_head_size: int = 64
    rwkv7_backend: str = "auto"
    rwkv7_chunk_len: int = 16
    rwkv7_enable_v_first_mix: bool = True


def _level_config(config: HierarchicalHybridRWKV7ModelConfig, arch: CoreArch, override: Dict[str, Any]) -> dict:
    level_config = config.model_dump()
    if arch == "transformer" and config.transformer_expansion is not None:
        level_config["expansion"] = config.transformer_expansion
    elif arch == "rwkv7" and config.rwkv7_expansion is not None:
        level_config["expansion"] = config.rwkv7_expansion
    level_config.update(override)
    return level_config


def _make_level(config: HierarchicalHybridRWKV7ModelConfig, arch: CoreArch, override: Dict[str, Any]) -> nn.Module:
    level_config = _level_config(config, arch, override)
    if arch == "transformer":
        return HierarchicalReasoningModelRecurrentBlock(TransformerConfig(**level_config))
    if arch == "rwkv7":
        return RWKV7ReasoningBlock(RWKV7Config(**level_config))
    raise ValueError(f"Unknown HRM hybrid core architecture: {arch}")


class HierarchicalHybridRWKV7Model(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = HierarchicalHybridRWKV7ModelConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        self.H_arch = config.H_arch
        self.L_arch = config.L_arch
        self.H_level = _make_level(config, config.H_arch, config.H_override)
        self.L_level = _make_level(config, config.L_arch, config.L_override)

        self.H_cycles = config.H_cycles
        self.L_cycles = config.L_cycles
        self.bp_warmup_ratio = config.bp_warmup_ratio
        self.bp_min_steps = config.bp_min_steps
        self.bp_max_steps = config.bp_max_steps

        self.hidden_size = config.hidden_size
        self.head_hint = self.H_level.core.head_hint
        self.zL_init = nn.Buffer(
            trunc_normal_init_(torch.empty(config.hidden_size, dtype=torch.bfloat16), std=1.0),
            persistent=True,
        )

        self.create_cache = lambda **kwargs: dict(
            H=[self.H_level.create_cache(**kwargs) for _i in range(self.H_cycles)],
            L=[self.L_level.create_cache(**kwargs) for _i in range(self.H_cycles * self.L_cycles)],
        )

    def forward(
        self,
        carry: None,
        x: Tensor,
        cache: Optional[dict[str, list[list[Cache] | None]]] = None,
        bp_steps: int = 2,
        **seq_info,
    ) -> Tuple[None, Tensor]:
        z_H, z_L = x, self.zL_init

        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)):
                    z_L = self.L_level(z_L, z_H, **seq_info, cache=cache["L"][k] if cache is not None else None)

            with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)):
                z_H = self.H_level(z_H, z_L, **seq_info, cache=cache["H"][i] if cache is not None else None)

        return None, z_H

    def compute_train_extra_args(self, train_state: Any) -> dict[str, Any]:
        warmup_steps = train_state.total_steps * self.bp_warmup_ratio
        progress = min(1.0, train_state.step / warmup_steps) if warmup_steps > 0 else 1.0
        return dict(bp_steps=self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps)))

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
