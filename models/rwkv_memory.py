from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn

from models.common import unwrap_tensor
from models.rwkv7 import RWKV7Config, RWKV7TimeMix


class RWKVStateMemory(nn.Module):
    """RWKV-7 state reader used to produce Transformer attention deltas."""

    def __init__(
        self,
        *,
        max_seq_len: int,
        hidden_size: int,
        head_size: int = 64,
        backend: str = "auto",
        chunk_len: int = 16,
        scale: float = 1.0,
        output_init: str = "zero",
        output_init_scale: float = 0.02,
    ) -> None:
        super().__init__()
        self.scale = scale
        config = RWKV7Config(
            max_seq_len=max_seq_len,
            n_layers=1,
            hidden_size=hidden_size,
            expansion=1.0,
            rwkv7_head_size=head_size,
            rwkv7_backend=backend,
            rwkv7_chunk_len=chunk_len,
            rwkv7_enable_v_first_mix=False,
        )
        self.time_mix = RWKV7TimeMix(config, layer_id=0)
        self._reset_output(output_init, output_init_scale)

    def _reset_output(self, output_init: str, output_init_scale: float) -> None:
        if output_init == "zero":
            nn.init.zeros_(self.time_mix.output.weight)
        elif output_init == "small":
            nn.init.trunc_normal_(
                self.time_mix.output.weight,
                mean=0.0,
                std=output_init_scale,
                a=-3 * output_init_scale,
                b=3 * output_init_scale,
            )
        else:
            raise ValueError(f"Unknown RWKV memory output init: {output_init}")

    def _forward_batched(self, x: Tensor) -> Tensor:
        y, _v_first = self.time_mix(x, v_first=None, reset_v_first=True)
        return y * self.scale

    def _forward_packed(self, x: Tensor, cu_seqlens: Tensor, numseqs: Tensor | int) -> Tensor:
        n = int(numseqs.item()) if isinstance(numseqs, Tensor) else int(numseqs)
        if n <= 0:
            return torch.zeros_like(x)

        cu = cu_seqlens[: n + 1].to(device=x.device, dtype=torch.long)
        starts = cu[:-1]
        lengths = cu[1:] - starts
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return torch.zeros_like(x)

        pos = torch.arange(max_len, device=x.device, dtype=torch.long).unsqueeze(0)
        mask = pos < lengths.unsqueeze(1)
        src = starts.unsqueeze(1) + pos

        padded = x.new_zeros((n, max_len, x.shape[-1]))
        padded[mask] = x[src[mask]]

        padded_out = self._forward_batched(padded)
        out = torch.zeros_like(x)
        out[src[mask]] = padded_out[mask]
        return out

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        numseqs: Optional[Tensor] = None,
        **_seq_info,
    ) -> Tensor:
        if x.dim() == 3:
            return self._forward_batched(x)
        if x.dim() != 2:
            raise ValueError(f"RWKVStateMemory expects [B,T,C] or [T,C], got {tuple(x.shape)}")
        if cu_seqlens is None or numseqs is None:
            return self._forward_batched(x.unsqueeze(0)).squeeze(0)

        cu_seqlens = unwrap_tensor(cu_seqlens)
        numseqs = unwrap_tensor(numseqs)
        return self._forward_packed(x, cu_seqlens, numseqs)
