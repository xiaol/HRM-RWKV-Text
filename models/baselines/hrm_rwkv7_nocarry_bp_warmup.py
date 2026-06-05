from typing import Any, Dict, Optional, Tuple

import torch
from torch import Tensor, nn
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.rwkv7 import RWKV7Config, RWKV7ReasoningBlock


class HierarchicalRWKV7ModelConfig(RWKV7Config):
    half_layers: bool = False

    H_cycles: int
    L_cycles: int

    bp_warmup_ratio: float = 0.0
    bp_min_steps: int = 2
    bp_max_steps: int = 5

    H_override: Dict[str, Any] = {}


class HierarchicalRWKV7Model(nn.Module):
    def __init__(self, config_dict: dict) -> None:
        super().__init__()
        config = HierarchicalRWKV7ModelConfig(**config_dict)
        if config.half_layers:
            assert config.n_layers % 2 == 0, "n_layers must be divisible by 2."
            config.n_layers //= 2

        h_config = RWKV7Config(**(config.model_dump() | config.H_override))
        l_config = RWKV7Config(**config.model_dump())
        self.H_level = RWKV7ReasoningBlock(h_config)
        self.L_level = RWKV7ReasoningBlock(l_config)

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
        self.create_cache = lambda **kwargs: None

    def forward(
        self,
        carry: None,
        x: Tensor,
        cache: Optional[dict] = None,
        bp_steps: int = 2,
        **seq_info,
    ) -> Tuple[None, Tensor]:
        x = x.to(dtype=self.zL_init.dtype)
        z_H, z_L = x, self.zL_init

        H_bp_steps = min(self.H_cycles, bp_steps - 1)
        L_bp_steps = bp_steps - H_bp_steps

        for i in range(self.H_cycles):
            for k in range(i * self.L_cycles, (i + 1) * self.L_cycles):
                with torch.set_grad_enabled(torch.is_grad_enabled() and (k >= self.H_cycles * self.L_cycles - L_bp_steps)):
                    z_L = self.L_level(z_L, z_H, **seq_info)

            with torch.set_grad_enabled(torch.is_grad_enabled() and (i >= self.H_cycles - H_bp_steps)):
                z_H = self.H_level(z_H, z_L, **seq_info)

        return None, z_H

    def compute_train_extra_args(self, train_state: Any) -> dict[str, Any]:
        warmup_steps = train_state.total_steps * self.bp_warmup_ratio
        progress = min(1.0, train_state.step / warmup_steps) if warmup_steps > 0 else 1.0
        return dict(bp_steps=self.bp_min_steps + int(progress * (self.bp_max_steps - self.bp_min_steps)))

    def initial_carry(self, batch_size: int, dtype: torch.dtype) -> None:
        return None
