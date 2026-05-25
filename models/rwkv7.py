from __future__ import annotations

from typing import Optional, Tuple
import math
import warnings

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from pydantic import BaseModel

from models.common import trunc_normal_init_
from models.common import unwrap_tensor
from models.layers import find_multiple


def _ortho_init_(x: Tensor, scale: float = 1.0) -> Tensor:
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1.0
            nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1.0
            for i in range(shape[0]):
                nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            raise ValueError(f"Unsupported tensor shape for RWKV-7 orthogonal init: {shape}")
    return x


def _time_shift_delta(x: Tensor) -> Tensor:
    xx = torch.empty_like(x)
    xx[:, 0] = -x[:, 0]
    if x.size(1) > 1:
        xx[:, 1:] = x[:, :-1] - x[:, 1:]
    return xx


def rwkv7_recurrence_torch(r: Tensor, w: Tensor, k: Tensor, v: Tensor, a: Tensor, b: Tensor, head_size: int) -> Tensor:
    B, T, C = r.shape
    H = C // head_size
    dtype = r.dtype
    r = r.reshape(B, T, H, head_size)
    k = k.reshape(B, T, H, head_size)
    v = v.reshape(B, T, H, head_size)
    a = a.reshape(B, T, H, head_size)
    b = b.reshape(B, T, H, head_size)
    decay = torch.exp(-torch.exp(w.float())).reshape(B, T, H, head_size)
    state = torch.zeros(B, H, head_size, head_size, device=r.device, dtype=torch.float32)
    out = []
    for t in range(T):
        rt = r[:, t].float()
        kt = k[:, t].float()
        vt = v[:, t].float()
        at = a[:, t].float()
        bt = b[:, t].float()
        wt = decay[:, t]
        sa = (state * at.unsqueeze(-2)).sum(dim=-1)
        state = state * wt.unsqueeze(-2) + vt.unsqueeze(-1) * kt.unsqueeze(-2) + sa.unsqueeze(-1) * bt.unsqueeze(-2)
        out.append((state * rt.unsqueeze(-2)).sum(dim=-1).to(dtype))
    return torch.stack(out, dim=1).reshape(B, T, C)


class RWKV7Config(BaseModel):
    max_seq_len: int
    n_layers: int
    hidden_size: int
    expansion: float = 1.0
    norm_eps: float = 1e-6
    rwkv7_head_size: int = 64
    rwkv7_backend: str = "auto"
    rwkv7_chunk_len: int = 16
    rwkv7_enable_v_first_mix: bool = True

    @property
    def init_std(self) -> float:
        return self.hidden_size ** -0.5

    @property
    def channel_size(self) -> int:
        return find_multiple(round(4 * self.expansion * self.hidden_size), 32)


class RWKV7TimeMix(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        dim = config.hidden_size
        head_size = config.rwkv7_head_size
        if dim % head_size != 0:
            raise ValueError(f"hidden_size ({dim}) must be divisible by rwkv7_head_size ({head_size})")
        if config.rwkv7_backend not in {"auto", "cuda", "torch"}:
            raise ValueError(f"Unknown RWKV-7 backend: {config.rwkv7_backend}")

        self.dim = dim
        self.depth = config.n_layers
        self.layer_id = layer_id
        self.head_size = head_size
        self.n_head = dim // head_size
        self.backend = config.rwkv7_backend
        self.chunk_len = config.rwkv7_chunk_len
        self.enable_v_first_mix = config.rwkv7_enable_v_first_mix

        decay_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        aaa_lora_dim = max(32, int(round((2.5 * (dim**0.5)) / 32) * 32))
        gate_lora_dim = max(32, int(round((5.0 * (dim**0.5)) / 32) * 32))
        mv_lora_dim = max(32, int(round((1.7 * (dim**0.5)) / 32) * 32))

        self.x_r = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_w = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_v = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.x_g = nn.Parameter(torch.empty(dim, dtype=torch.float32))

        self.w1 = nn.Parameter(torch.empty(dim, decay_lora_dim, dtype=torch.float32))
        self.w2 = nn.Parameter(torch.empty(decay_lora_dim, dim, dtype=torch.float32))
        self.w0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.a1 = nn.Parameter(torch.empty(dim, aaa_lora_dim, dtype=torch.float32))
        self.a2 = nn.Parameter(torch.empty(aaa_lora_dim, dim, dtype=torch.float32))
        self.a0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        if self.enable_v_first_mix:
            self.v1 = nn.Parameter(torch.empty(dim, mv_lora_dim, dtype=torch.float32))
            self.v2 = nn.Parameter(torch.empty(mv_lora_dim, dim, dtype=torch.float32))
            self.v0 = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.g1 = nn.Parameter(torch.empty(dim, gate_lora_dim, dtype=torch.float32))
        self.g2 = nn.Parameter(torch.empty(gate_lora_dim, dim, dtype=torch.float32))
        self.k_k = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.k_a = nn.Parameter(torch.empty(dim, dtype=torch.float32))
        self.r_k = nn.Parameter(torch.empty(self.n_head, head_size, dtype=torch.float32))

        self.receptance = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, dim, eps=64e-5)
        self.reset_parameters(config.init_std)

    def reset_parameters(self, init_std: float) -> None:
        device = self.x_r.device
        dim = self.dim
        ratio_0_to_1 = self.layer_id / max(self.depth - 1, 1)
        ratio_1_to_almost0 = 1.0 - (self.layer_id / max(self.depth, 1))
        ddd = torch.arange(dim, device=device, dtype=torch.float32) / dim
        linear = torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1) - 0.5
        zigzag = torch.arange(dim, device=device, dtype=torch.float32) % self.head_size
        zigzag = (zigzag - ((self.head_size - 1) / 2)) / max((self.head_size - 1) / 2, 1.0)
        zigzag = zigzag * zigzag.abs()
        decay = -6 + 6 * (torch.arange(dim, device=device, dtype=torch.float32) / max(dim - 1, 1)) ** (1 + ratio_0_to_1**0.3)
        with torch.no_grad():
            self.x_r.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.x_w.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_k.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_v.copy_(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
            self.x_a.copy_(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
            self.x_g.copy_(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
            self.w0.copy_(decay + 0.5 + zigzag * 2.5)
            self.a0.copy_(torch.zeros_like(linear) - 0.19 + zigzag * 0.3 + linear * 0.4)
            if self.enable_v_first_mix:
                self.v0.copy_(torch.zeros_like(linear) + 0.73 - linear * 0.4)
            self.k_k.copy_(torch.zeros_like(linear) + 0.71 - linear * 0.1)
            self.k_a.fill_(1.02)
            self.r_k.fill_(-0.04)
            self.w1.zero_()
            _ortho_init_(self.w2, 0.1)
            self.a1.zero_()
            _ortho_init_(self.a2, 0.1)
            if self.enable_v_first_mix:
                self.v1.zero_()
                _ortho_init_(self.v2, 0.1)
            self.g1.zero_()
            _ortho_init_(self.g2, 0.1)
        for proj in (self.receptance, self.key, self.value):
            nn.init.trunc_normal_(proj.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.zeros_(self.output.weight)
        self.ln_x.reset_parameters()

    def _can_use_cuda(self, x: Tensor) -> bool:
        return self.backend != "torch" and x.is_cuda and x.dtype == torch.bfloat16 and self.head_size == 64

    def _forward_cuda(self, x: Tensor, v_first: Optional[Tensor], reset_v_first: bool) -> Tuple[Tensor, Tensor]:
        from apps.LT2 import rwkv7_cuda

        xr, xw, xk, xv, xa, xg = rwkv7_cuda.tmix_mix6(
            x,
            self.x_r.to(device=x.device, dtype=x.dtype),
            self.x_w.to(device=x.device, dtype=x.dtype),
            self.x_k.to(device=x.device, dtype=x.dtype),
            self.x_v.to(device=x.device, dtype=x.dtype),
            self.x_a.to(device=x.device, dtype=x.dtype),
            self.x_g.to(device=x.device, dtype=x.dtype),
        )
        r = self.receptance(xr)
        w = self.w0.to(dtype=x.dtype).view(1, 1, -1) + (torch.tanh(xw @ self.w1.to(dtype=x.dtype)) @ self.w2.to(dtype=x.dtype))
        k = self.key(xk)
        v = self.value(xv)
        if reset_v_first or v_first is None:
            v_first = v
        elif self.enable_v_first_mix:
            v12 = (xv @ self.v1.to(dtype=x.dtype)) @ self.v2.to(dtype=x.dtype)
            v = rwkv7_cuda.tmix_vres_gate(v, v_first, self.v0.to(device=x.device, dtype=x.dtype), v12)
        a = rwkv7_cuda.tmix_a_gate(self.a0.to(device=x.device, dtype=x.dtype), (xa @ self.a1.to(dtype=x.dtype)) @ self.a2.to(dtype=x.dtype))
        g = torch.sigmoid(xg @ self.g1.to(dtype=x.dtype)) @ self.g2.to(dtype=x.dtype)
        k, neg_kk, kka = rwkv7_cuda.tmix_kk_pre(k, self.k_k.to(device=x.device, dtype=x.dtype), a, self.k_a.to(device=x.device, dtype=x.dtype), self.head_size)
        y = rwkv7_cuda.rwkv7_recurrence_cuda_bf16(r, w, k, v, neg_kk, kka, self.head_size, self.chunk_len)
        y = rwkv7_cuda.tmix_lnx_rkvres_xg(
            y,
            r,
            k,
            v,
            self.r_k.to(device=x.device, dtype=x.dtype),
            self.ln_x.weight.to(device=x.device, dtype=x.dtype),
            self.ln_x.bias.to(device=x.device, dtype=x.dtype),
            g,
        )
        return self.output(y), v_first

    def forward(self, x: Tensor, v_first: Optional[Tensor] = None, reset_v_first: bool = False) -> Tuple[Tensor, Tensor]:
        if self._can_use_cuda(x):
            try:
                return self._forward_cuda(x, v_first, reset_v_first)
            except Exception as exc:
                if self.backend == "cuda":
                    raise
                warnings.warn(f"Falling back to PyTorch RWKV-7 backend: {exc}", RuntimeWarning, stacklevel=2)

        B, T, C = x.shape
        xx = _time_shift_delta(x)
        xr = x + xx * self.x_r.to(dtype=x.dtype).view(1, 1, -1)
        xw = x + xx * self.x_w.to(dtype=x.dtype).view(1, 1, -1)
        xk = x + xx * self.x_k.to(dtype=x.dtype).view(1, 1, -1)
        xv = x + xx * self.x_v.to(dtype=x.dtype).view(1, 1, -1)
        xa = x + xx * self.x_a.to(dtype=x.dtype).view(1, 1, -1)
        xg = x + xx * self.x_g.to(dtype=x.dtype).view(1, 1, -1)

        r = self.receptance(xr)
        w = self.w0.to(dtype=x.dtype).view(1, 1, -1) + (torch.tanh(xw @ self.w1.to(dtype=x.dtype)) @ self.w2.to(dtype=x.dtype))
        k = self.key(xk)
        v = self.value(xv)
        if reset_v_first or v_first is None:
            v_first = v
        elif self.enable_v_first_mix:
            v_mix = torch.sigmoid(self.v0.to(dtype=x.dtype).view(1, 1, -1) + (xv @ self.v1.to(dtype=x.dtype)) @ self.v2.to(dtype=x.dtype))
            v = v + (v_first - v) * v_mix
        w = -F.softplus(-w.float()).to(dtype=x.dtype) - 0.5
        a = torch.sigmoid(self.a0.to(dtype=x.dtype).view(1, 1, -1) + (xa @ self.a1.to(dtype=x.dtype)) @ self.a2.to(dtype=x.dtype))
        g = torch.sigmoid(xg @ self.g1.to(dtype=x.dtype)) @ self.g2.to(dtype=x.dtype)
        kk = k * self.k_k.to(dtype=x.dtype).view(1, 1, -1)
        kk = F.normalize(kk.reshape(B, T, self.n_head, self.head_size), dim=-1, p=2.0).reshape(B, T, C)
        k = k * (1 + (a - 1) * self.k_a.to(dtype=x.dtype).view(1, 1, -1))
        y = rwkv7_recurrence_torch(r, w, k, v, -kk, kk * a, self.head_size)
        y = self.ln_x(y.reshape(B * T, C)).reshape(B, T, C)
        y = y + (
            (
                r.reshape(B, T, self.n_head, self.head_size)
                * k.reshape(B, T, self.n_head, self.head_size)
                * self.r_k.to(dtype=x.dtype)
            ).sum(dim=-1, keepdim=True)
            * v.reshape(B, T, self.n_head, self.head_size)
        ).reshape(B, T, C)
        return self.output(y * g), v_first


class RWKV7ChannelMix(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        dim = config.hidden_size
        self.dim = dim
        self.backend = config.rwkv7_backend
        self.ffn_dim = config.channel_size
        ratio_1_to_almost0 = 1.0 - (layer_id / max(config.n_layers, 1))
        ddd = torch.arange(dim, dtype=torch.float32) / dim
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))
        self.key = nn.Linear(dim, self.ffn_dim, bias=False)
        self.value = nn.Linear(self.ffn_dim, dim, bias=False)
        self.reset_parameters(config.init_std)

    def reset_parameters(self, init_std: float) -> None:
        nn.init.trunc_normal_(self.key.weight, mean=0.0, std=init_std, a=-3 * init_std, b=3 * init_std)
        nn.init.zeros_(self.value.weight)

    def forward(self, x: Tensor) -> Tensor:
        if self.backend != "torch" and x.is_cuda and x.dtype == torch.bfloat16 and self.ffn_dim == self.dim * 4:
            try:
                from apps.LT2 import rwkv7_cuda

                return rwkv7_cuda.cmix_layer(
                    x,
                    self.x_k.to(device=x.device, dtype=x.dtype),
                    self.key.weight.to(dtype=x.dtype),
                    self.value.weight.to(dtype=x.dtype),
                )
            except Exception as exc:
                if self.backend == "cuda":
                    raise
                warnings.warn(f"Falling back to PyTorch RWKV-7 channel mix: {exc}", RuntimeWarning, stacklevel=2)
        xx = _time_shift_delta(x)
        k = x + xx * self.x_k.to(dtype=x.dtype).view(1, 1, -1)
        return self.value(F.relu(self.key(k)).square())


class RWKV7Block(nn.Module):
    def __init__(self, config: RWKV7Config, layer_id: int) -> None:
        super().__init__()
        self.time_mix = RWKV7TimeMix(config, layer_id)
        self.channel_mix = RWKV7ChannelMix(config, layer_id)
        self.time_norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)
        self.channel_norm = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)

    def forward(self, x: Tensor, v_first: Optional[Tensor], reset_v_first: bool) -> Tuple[Tensor, Tensor]:
        y, v_first = self.time_mix(self.time_norm(x), v_first=v_first, reset_v_first=reset_v_first)
        x = x + y
        return x + self.channel_mix(self.channel_norm(x)), v_first


class RWKV7Stack(nn.Module):
    def __init__(self, config: RWKV7Config) -> None:
        super().__init__()
        self.head_hint = {"in": {"dim": config.hidden_size, "init_std": config.init_std}, "out": {"dim": config.hidden_size, "init_std": config.init_std}}
        self.layers = nn.ModuleList([RWKV7Block(config, i) for i in range(config.n_layers)])
        self.norm_f = lambda x: F.rms_norm(x, (x.shape[-1],), eps=config.norm_eps)
        self.create_cache = lambda **kwargs: None

    def _forward_batched(self, x: Tensor) -> Tensor:
        v_first = None
        for idx, layer in enumerate(self.layers):
            x, v_first = layer(x, v_first=v_first, reset_v_first=(idx == 0))
        return self.norm_f(x)

    def forward(self, x: Tensor, cu_seqlens: Optional[Tensor] = None, numseqs: Optional[Tensor] = None, **_seq_info) -> Tensor:
        if x.dim() == 3:
            return self._forward_batched(x)
        if x.dim() != 2:
            raise ValueError(f"RWKV7Stack expects [B,T,C] or [T,C], got {tuple(x.shape)}")
        if cu_seqlens is None or numseqs is None:
            return self._forward_batched(x.unsqueeze(0)).squeeze(0)

        cu_seqlens = unwrap_tensor(cu_seqlens)
        numseqs = unwrap_tensor(numseqs)
        n = int(numseqs.item()) if isinstance(numseqs, Tensor) else int(numseqs)
        cu = cu_seqlens.detach().cpu()
        out = torch.zeros_like(x)
        for i in range(n):
            start = int(cu[i].item())
            end = int(cu[i + 1].item())
            if end > start:
                out[start:end] = self._forward_batched(x[start:end].unsqueeze(0)).squeeze(0)
        return out


class RWKV7ReasoningBlock(nn.Module):
    def __init__(self, config: RWKV7Config) -> None:
        super().__init__()
        self.core = RWKV7Stack(config)
        self.create_cache = self.core.create_cache

    def forward(self, hidden_states: Tensor, input_injection: Tensor, **kwargs) -> Tensor:
        return self.core(hidden_states + input_injection, **kwargs)
