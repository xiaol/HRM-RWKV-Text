from __future__ import annotations

import math
import warnings
from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.common import unwrap_tensor
from models.rwkv7 import RWKV7Config, RWKV7TimeMix, _time_shift_delta, rwkv7_recurrence_read_before_write_torch

try:
    from deltamem.kernels.affine_scan import triton_affine_scan, triton_scan_support
except Exception:  # pragma: no cover - optional local reference repo dependency
    triton_affine_scan = None
    triton_scan_support = None


VALID_DELTA_MEM_HEADS = ("q", "k", "v", "o")
VALID_DELTA_MEM_STATE_UPDATE_MODES = ("standard", "lambda_outside", "no_lambda")
VALID_DELTA_MEM_OUTPUT_INITS = ("zero", "small", "random", "base_slice", "base_slice_fixed")
VALID_DELTA_MEM_WRITE_GRANULARITIES = ("token", "message_mean", "sentence_mean")
VALID_DELTA_MEM_SCALE_GRANULARITIES = ("layer", "head")


def normalize_delta_mem_heads(heads: Sequence[str] | str) -> tuple[str, ...]:
    if isinstance(heads, str):
        items = tuple(part.strip().lower() for part in heads.split(",") if part.strip())
    else:
        items = tuple(str(part).strip().lower() for part in heads if str(part).strip())
    if not items or items == ("none",):
        return ()
    invalid = sorted(set(items) - set(VALID_DELTA_MEM_HEADS))
    if invalid:
        raise ValueError(f"Unsupported delta-memory heads: {invalid}; expected subset of {VALID_DELTA_MEM_HEADS}")
    out: list[str] = []
    for item in items:
        if item not in out:
            out.append(item)
    return tuple(out)


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
        read_before_write: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.scale = scale
        self.read_before_write = bool(read_before_write)
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
        if kwargs:
            self.time_mix.to(**kwargs)

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

    def _normalize_read_state(self, y: Tensor) -> Tensor:
        tm = self.time_mix
        B, T, C = y.shape
        # No bias here: an empty state read must stay zero even after training.
        return F.group_norm(
            y.reshape(B * T, C),
            num_groups=tm.ln_x.num_groups,
            weight=tm.ln_x.weight.to(device=y.device, dtype=y.dtype),
            bias=None,
            eps=tm.ln_x.eps,
        ).reshape(B, T, C)

    def _forward_read_before_write_torch(self, x: Tensor) -> Tensor:
        tm = self.time_mix
        B, T, C = x.shape
        if T == 0:
            return x.new_zeros((B, T, C))
        xx = _time_shift_delta(x)
        xr = x + xx * tm.x_r.to(dtype=x.dtype).view(1, 1, -1)
        xw = x + xx * tm.x_w.to(dtype=x.dtype).view(1, 1, -1)
        xk = x + xx * tm.x_k.to(dtype=x.dtype).view(1, 1, -1)
        xv = x + xx * tm.x_v.to(dtype=x.dtype).view(1, 1, -1)
        xa = x + xx * tm.x_a.to(dtype=x.dtype).view(1, 1, -1)
        xg = x + xx * tm.x_g.to(dtype=x.dtype).view(1, 1, -1)

        r = tm.receptance(xr)
        w = tm.w0.to(dtype=x.dtype).view(1, 1, -1) + (
            torch.tanh(xw @ tm.w1.to(dtype=x.dtype)) @ tm.w2.to(dtype=x.dtype)
        )
        k = tm.key(xk)
        v = tm.value(xv)
        w = -F.softplus(-w.float()).to(dtype=x.dtype) - 0.5
        a = torch.sigmoid(
            tm.a0.to(dtype=x.dtype).view(1, 1, -1) + (xa @ tm.a1.to(dtype=x.dtype)) @ tm.a2.to(dtype=x.dtype)
        )
        g = torch.sigmoid(xg @ tm.g1.to(dtype=x.dtype)) @ tm.g2.to(dtype=x.dtype)
        kk = k * tm.k_k.to(dtype=x.dtype).view(1, 1, -1)
        kk = F.normalize(kk.reshape(B, T, tm.n_head, tm.head_size), dim=-1, p=2.0).reshape(B, T, C)
        k = k * (1 + (a - 1) * tm.k_a.to(dtype=x.dtype).view(1, 1, -1))

        y = rwkv7_recurrence_read_before_write_torch(r, w, k, v, -kk, kk * a, tm.head_size)
        y = self._normalize_read_state(y)
        return tm.output(y * g)

    def _can_use_read_before_write_cuda(self, x: Tensor) -> bool:
        tm = self.time_mix
        return tm.backend != "torch" and x.is_cuda and x.dtype == torch.bfloat16 and tm.head_size == 64 and tm.chunk_len == 16

    def _forward_read_before_write_cuda(self, x: Tensor) -> Tensor:
        from apps.LT2 import rwkv7_cuda

        tm = self.time_mix
        x = x.contiguous()
        B, T, C = x.shape
        if T == 0:
            return x.new_zeros((B, T, C))

        xr, xw, xk, xv, xa, xg = rwkv7_cuda.tmix_mix6(
            x,
            tm.x_r.to(device=x.device, dtype=x.dtype),
            tm.x_w.to(device=x.device, dtype=x.dtype),
            tm.x_k.to(device=x.device, dtype=x.dtype),
            tm.x_v.to(device=x.device, dtype=x.dtype),
            tm.x_a.to(device=x.device, dtype=x.dtype),
            tm.x_g.to(device=x.device, dtype=x.dtype),
        )
        r = tm.receptance(xr)
        w = tm.w0.to(device=x.device, dtype=x.dtype).view(1, 1, -1) + (
            torch.tanh(xw @ tm.w1.to(device=x.device, dtype=x.dtype)) @ tm.w2.to(device=x.device, dtype=x.dtype)
        )
        k = tm.key(xk)
        v = tm.value(xv)
        a = rwkv7_cuda.tmix_a_gate(
            tm.a0.to(device=x.device, dtype=x.dtype),
            (xa @ tm.a1.to(device=x.device, dtype=x.dtype)) @ tm.a2.to(device=x.device, dtype=x.dtype),
        )
        g = torch.sigmoid(xg @ tm.g1.to(device=x.device, dtype=x.dtype)) @ tm.g2.to(device=x.device, dtype=x.dtype)
        k, neg_kk, kka = rwkv7_cuda.tmix_kk_pre(
            k,
            tm.k_k.to(device=x.device, dtype=x.dtype),
            a,
            tm.k_a.to(device=x.device, dtype=x.dtype),
            tm.head_size,
        )

        # The LT2 kernel emits write-before-read y_t = S_t r_t.  Passing r_{t+1}
        # and shifting the output back gives read-before-write y_t = S_{t-1} r_t.
        r_next = torch.cat((r[:, 1:], torch.zeros_like(r[:, :1])), dim=1).contiguous()
        y_shifted = rwkv7_cuda.rwkv7_recurrence_cuda_bf16(
            r_next,
            w,
            k,
            v,
            neg_kk,
            kka,
            tm.head_size,
            tm.chunk_len,
        )
        y = torch.cat((torch.zeros_like(y_shifted[:, :1]), y_shifted[:, :-1]), dim=1).contiguous()
        y = self._normalize_read_state(y)
        return tm.output(y * g)

    def _forward_batched(self, x: Tensor) -> Tensor:
        if self.read_before_write:
            if self._can_use_read_before_write_cuda(x):
                try:
                    return self._forward_read_before_write_cuda(x) * self.scale
                except Exception as exc:
                    if self.time_mix.backend == "cuda":
                        raise
                    warnings.warn(
                        f"Falling back to PyTorch RWKV-memory read-before-write backend: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            elif self.time_mix.backend == "cuda":
                raise RuntimeError(
                    "RWKV-memory read-before-write CUDA requires a CUDA bfloat16 tensor, "
                    "head_size=64, and chunk_len=16; "
                    f"got device={x.device}, dtype={x.dtype}, "
                    f"head_size={self.time_mix.head_size}, chunk_len={self.time_mix.chunk_len}"
                )
            return self._forward_read_before_write_torch(x) * self.scale
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


class DeltaRuleStateMemory(nn.Module):
    """Delta-Mem-style online associative memory for HRM attention.

    Hidden states are projected to low-rank memory q/k/v. The state is read
    before each token write, then projected into attention q/k/v/o deltas.
    This mirrors the active delta-Mem path while keeping HRM's native packed
    PrefixLM attention.
    """

    def __init__(
        self,
        *,
        hidden_size: int,
        query_size: int,
        key_size: int,
        value_size: int,
        output_size: int,
        rank: int = 8,
        num_state_heads: int = 1,
        alpha: float = 16.0,
        beta_bias_init: float = -1.5,
        normalize_qk: bool = True,
        couple_lambda: bool = True,
        state_update_mode: str = "standard",
        rankwise_gates: bool = True,
        delta_heads: Sequence[str] | str = ("q", "k", "v", "o"),
        output_init: str = "zero",
        output_init_scale: float = 0.02,
        base_slice_ref_width: int = 8,
        online_gain: float = 0.05,
        memory_write_granularity: str = "token",
        backend: str = "auto",
        stateful: bool = False,
        trainable_delta_scale: bool = False,
        delta_scale_init: float = 1.0,
        delta_scale_max: float = 2.0,
        delta_scale_granularity: str = "layer",
        delta_o_rmsnorm: bool = False,
        delta_o_rmsnorm_eps: float = 1e-6,
        base_q_weight: Optional[Tensor] = None,
        base_k_weight: Optional[Tensor] = None,
        base_v_weight: Optional[Tensor] = None,
        base_o_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if num_state_heads < 1:
            raise ValueError("num_state_heads must be >= 1")
        if state_update_mode not in VALID_DELTA_MEM_STATE_UPDATE_MODES:
            raise ValueError(
                f"Unsupported state_update_mode={state_update_mode!r}; "
                f"expected one of {VALID_DELTA_MEM_STATE_UPDATE_MODES}"
            )
        if output_init not in VALID_DELTA_MEM_OUTPUT_INITS:
            raise ValueError(
                f"Unsupported output_init={output_init!r}; "
                f"expected one of {VALID_DELTA_MEM_OUTPUT_INITS}"
            )
        if memory_write_granularity not in VALID_DELTA_MEM_WRITE_GRANULARITIES:
            raise ValueError(
                f"Unsupported memory_write_granularity={memory_write_granularity!r}; "
                f"expected one of {VALID_DELTA_MEM_WRITE_GRANULARITIES}"
            )
        if delta_scale_granularity not in VALID_DELTA_MEM_SCALE_GRANULARITIES:
            raise ValueError(
                f"Unsupported delta_scale_granularity={delta_scale_granularity!r}; "
                f"expected one of {VALID_DELTA_MEM_SCALE_GRANULARITIES}"
            )
        if delta_scale_init <= 0.0 or delta_scale_max <= 0.0 or delta_scale_init >= delta_scale_max:
            raise ValueError("delta_scale_init and delta_scale_max must satisfy 0 < init < max")
        if delta_o_rmsnorm_eps <= 0.0:
            raise ValueError("delta_o_rmsnorm_eps must be > 0")

        self.hidden_size = hidden_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.rank = int(rank)
        self.num_state_heads = int(num_state_heads)
        self.state_read_dim = self.rank * self.num_state_heads
        self.alpha = float(alpha)
        self.delta_scaling = self.alpha / self.rank
        self.beta_bias_init = float(beta_bias_init)
        self.normalize_qk = bool(normalize_qk)
        self.couple_lambda = bool(couple_lambda)
        self.state_update_mode = state_update_mode
        self.rankwise_gates = bool(rankwise_gates)
        self.delta_heads = normalize_delta_mem_heads(delta_heads)
        self.base_slice_ref_width = int(base_slice_ref_width)
        self.online_gain = float(online_gain)
        self.memory_write_granularity = memory_write_granularity
        self.stateful = bool(stateful)
        self.write_enabled = True
        self.delta_state: Optional[Tensor] = None
        self.read_context_mask: Optional[Tensor] = None
        self.trainable_delta_scale = bool(trainable_delta_scale)
        self.delta_scale_max = float(delta_scale_max)
        self.delta_scale_granularity = delta_scale_granularity
        self.delta_o_rmsnorm = bool(delta_o_rmsnorm)
        self.delta_o_rmsnorm_eps = float(delta_o_rmsnorm_eps)
        if backend not in {"auto", "cuda", "torch"}:
            raise ValueError("delta-rule memory backend must be 'auto', 'cuda', or 'torch'")
        self.backend = backend

        gate_dim_per_head = self.rank if self.rankwise_gates else 1
        self.gate_dim = gate_dim_per_head * self.num_state_heads
        self.memory_q_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False, **kwargs)
        self.memory_k_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False, **kwargs)
        self.memory_v_proj = nn.Linear(hidden_size, self.state_read_dim, bias=False, **kwargs)
        self.beta_proj = nn.Linear(hidden_size, self.gate_dim, bias=False, **kwargs)
        self.beta_bias = nn.Parameter(torch.full((self.gate_dim,), self.beta_bias_init, **kwargs))
        if self.couple_lambda:
            self.lambda_proj = None
            self.lambda_bias = None
        else:
            self.lambda_proj = nn.Linear(hidden_size, self.gate_dim, bias=False, **kwargs)
            self.lambda_bias = nn.Parameter(torch.full((self.gate_dim,), -self.beta_bias_init, **kwargs))

        self.delta_q_proj = nn.Linear(self.state_read_dim, query_size, bias=False, **kwargs)
        self.delta_k_proj = nn.Linear(self.state_read_dim, key_size, bias=False, **kwargs)
        self.delta_v_proj = nn.Linear(self.state_read_dim, value_size, bias=False, **kwargs)
        self.delta_o_proj = nn.Linear(self.state_read_dim, output_size, bias=False, **kwargs)
        if self.trainable_delta_scale:
            scale_shape = (len(VALID_DELTA_MEM_HEADS),) if self.delta_scale_granularity == "head" else (1,)
            init_raw = self._inverse_bounded_sigmoid(delta_scale_init, self.delta_scale_max)
            self.delta_scale_raw = nn.Parameter(torch.full(scale_shape, init_raw, **kwargs))
        else:
            self.delta_scale_raw = None
        if self.delta_o_rmsnorm:
            self.delta_o_rmsnorm_weight = nn.Parameter(torch.ones(output_size, **kwargs))
        else:
            self.delta_o_rmsnorm_weight = None
        self.reset_parameters(
            output_init,
            output_init_scale,
            base_q_weight=base_q_weight,
            base_k_weight=base_k_weight,
            base_v_weight=base_v_weight,
            base_o_weight=base_o_weight,
        )

    @staticmethod
    def _inverse_bounded_sigmoid(value: float, max_value: float) -> float:
        clipped = min(max(value / max_value, 1e-4), 1.0 - 1e-4)
        return math.log(clipped / (1.0 - clipped))

    def reset_state(self) -> None:
        self.delta_state = None
        self.read_context_mask = None

    def set_write_enabled(self, enabled: bool) -> None:
        if enabled:
            self.read_context_mask = None
        self.write_enabled = bool(enabled)

    def set_read_context_mask(self, mask: Optional[Tensor]) -> None:
        self.read_context_mask = mask

    def _init_delta_head(
        self,
        *,
        head_name: str,
        proj: nn.Linear,
        output_init: str,
        output_init_scale: float,
        base_weight: Optional[Tensor],
    ) -> None:
        if head_name not in self.delta_heads or output_init == "zero":
            nn.init.zeros_(proj.weight)
            return
        if output_init == "small":
            nn.init.trunc_normal_(
                proj.weight,
                mean=0.0,
                std=output_init_scale,
                a=-3 * output_init_scale,
                b=3 * output_init_scale,
            )
            return
        if output_init == "random":
            nn.init.kaiming_uniform_(proj.weight, a=5**0.5)
            with torch.no_grad():
                proj.weight.mul_(self.online_gain)
            return
        if output_init not in {"base_slice", "base_slice_fixed"}:
            raise ValueError(f"Unknown delta-memory output init: {output_init}")
        if base_weight is None:
            raise ValueError(f"output_init={output_init!r} requires base_{head_name}_weight")

        with torch.no_grad():
            proj.weight.zero_()
            if output_init == "base_slice":
                slice_width = min(self.rank, self.state_read_dim, base_weight.shape[1])
            else:
                slice_width = min(self.base_slice_ref_width, self.rank, self.state_read_dim, base_weight.shape[1])
            if slice_width <= 0:
                return
            copy_rows = min(proj.weight.shape[0], base_weight.shape[0])
            base_slice = base_weight[:copy_rows, :slice_width].detach().float()
            base_slice = F.normalize(base_slice, dim=0, eps=1e-6)
            proj.weight[:copy_rows, :slice_width].copy_((base_slice * self.online_gain).to(dtype=proj.weight.dtype))

    def reset_parameters(
        self,
        output_init: str,
        output_init_scale: float,
        *,
        base_q_weight: Optional[Tensor] = None,
        base_k_weight: Optional[Tensor] = None,
        base_v_weight: Optional[Tensor] = None,
        base_o_weight: Optional[Tensor] = None,
    ) -> None:
        for proj in (self.memory_q_proj, self.memory_k_proj, self.memory_v_proj):
            nn.init.kaiming_uniform_(proj.weight, a=5**0.5)
        nn.init.zeros_(self.beta_proj.weight)
        if self.lambda_proj is not None:
            nn.init.zeros_(self.lambda_proj.weight)
        if self.delta_o_rmsnorm_weight is not None:
            nn.init.ones_(self.delta_o_rmsnorm_weight)

        for name, proj, base_weight in (
            ("q", self.delta_q_proj, base_q_weight),
            ("k", self.delta_k_proj, base_k_weight),
            ("v", self.delta_v_proj, base_v_weight),
            ("o", self.delta_o_proj, base_o_weight),
        ):
            self._init_delta_head(
                head_name=name,
                proj=proj,
                output_init=output_init,
                output_init_scale=output_init_scale,
                base_weight=base_weight,
            )

    def _reshape_gate(self, gate: Tensor) -> Tensor:
        if self.rankwise_gates:
            return gate.view(*gate.shape[:-1], self.num_state_heads, self.rank)
        return gate.view(*gate.shape[:-1], self.num_state_heads, 1).expand(
            *gate.shape[:-1],
            self.num_state_heads,
            self.rank,
        )

    def _normalize_memory_projection(self, projected: Tensor) -> Tensor:
        if self.normalize_qk:
            original_dtype = projected.dtype
            projected = projected.float().view(*projected.shape[:-1], self.num_state_heads, self.rank)
            projected = F.normalize(torch.tanh(projected), dim=-1, eps=1e-6)
            projected = projected.reshape(*projected.shape[:-2], self.state_read_dim).to(dtype=original_dtype)
        return projected

    def _project_memory(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        memory_q = self._normalize_memory_projection(self.memory_q_proj(x))
        memory_k = self._normalize_memory_projection(self.memory_k_proj(x))
        memory_v = self.memory_v_proj(x)

        beta = torch.sigmoid(self.beta_proj(x) + self.beta_bias.to(device=x.device, dtype=x.dtype))
        beta = self._reshape_gate(beta)

        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            assert self.lambda_proj is not None and self.lambda_bias is not None
            lam = torch.sigmoid(self.lambda_proj(x) + self.lambda_bias.to(device=x.device, dtype=x.dtype))
            lam = self._reshape_gate(lam)

        return memory_q, memory_k, memory_v, beta, lam

    def is_trainable_parameter(self, sub_name: str) -> bool:
        if sub_name.startswith("delta_q_proj."):
            return "q" in self.delta_heads
        if sub_name.startswith("delta_k_proj."):
            return "k" in self.delta_heads
        if sub_name.startswith("delta_v_proj."):
            return "v" in self.delta_heads
        if sub_name.startswith("delta_o_proj."):
            return "o" in self.delta_heads
        if sub_name == "delta_scale_raw":
            return self.trainable_delta_scale
        if sub_name == "delta_o_rmsnorm_weight":
            return self.delta_o_rmsnorm and "o" in self.delta_heads
        return True

    def _update_coefficients(self, beta: Tensor, lam: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if self.state_update_mode == "standard":
            return lam, beta, beta
        if self.state_update_mode == "lambda_outside":
            return lam, lam * beta, beta
        if self.state_update_mode == "no_lambda":
            return torch.ones_like(beta), beta, beta
        raise ValueError(f"Unsupported state_update_mode: {self.state_update_mode}")

    def _scan_batched_torch(
        self,
        memory_q: Tensor,
        memory_k: Tensor,
        memory_v: Tensor,
        keep: Tensor,
        erase: Tensor,
        write: Tensor,
        token_mask: Optional[Tensor],
        dtype: torch.dtype,
        initial_state: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        batch_size, seq_len, _ = memory_q.shape
        if initial_state is None:
            state = torch.zeros(batch_size, self.rank, self.rank, device=memory_q.device, dtype=torch.float32)
        else:
            state = initial_state.float()
        reads: list[Tensor] = []
        q_seq = memory_q.float()
        k_seq = memory_k.float()
        v_seq = memory_v.float()

        for token_idx in range(seq_len):
            q_t = q_seq[:, token_idx, :]
            k_t = k_seq[:, token_idx, :]
            v_t = v_seq[:, token_idx, :]
            keep_t = keep[:, token_idx, :].unsqueeze(-1)
            erase_t = erase[:, token_idx, :].unsqueeze(-1)
            write_t = write[:, token_idx, :].unsqueeze(-1)

            read_t = torch.einsum("bij,bj->bi", state, q_t)
            pred_t = torch.einsum("bij,bj->bi", state, k_t)
            next_state = (
                keep_t * state
                - erase_t * pred_t.unsqueeze(-1) * k_t.unsqueeze(1)
                + write_t * v_t.unsqueeze(-1) * k_t.unsqueeze(1)
            )

            if token_mask is not None:
                valid = token_mask[:, token_idx].view(batch_size, 1, 1).to(dtype=state.dtype)
                state = next_state * valid + state * (1.0 - valid)
                read_t = read_t * valid.squeeze(-1)
            else:
                state = next_state
            reads.append(read_t.to(dtype=dtype))

        return state, torch.stack(reads, dim=1)

    def _initial_state_for_scan(self, batch_size: int, device: torch.device) -> Tensor:
        flat_batch = batch_size * self.num_state_heads
        if self.stateful and self.delta_state is not None:
            state = self.delta_state
            if state.shape[0] == batch_size and state.device == device:
                return state.reshape(flat_batch, self.rank, self.rank).float()
        return torch.zeros(flat_batch, self.rank, self.rank, device=device, dtype=torch.float32)

    def _store_final_state(self, final_state: Tensor, batch_size: int) -> None:
        if not self.stateful:
            return
        self.delta_state = final_state.reshape(batch_size, self.num_state_heads, self.rank, self.rank).detach()

    def _state_reads_only(self, state: Tensor, q_seq: Tensor, token_mask: Optional[Tensor], dtype: torch.dtype) -> Tensor:
        reads = torch.einsum("bij,btj->bti", state.float(), q_seq.float())
        if token_mask is not None:
            reads = reads * token_mask.unsqueeze(-1).to(dtype=reads.dtype)
        return reads.to(dtype=dtype)

    def _scan_batched(self, x: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        memory_q, memory_k, memory_v, beta, lam = self._project_memory(x)
        keep, erase, write = self._update_coefficients(beta.float(), lam.float())

        batch_size, seq_len, _ = x.shape
        flat_batch = batch_size * self.num_state_heads
        state = self._initial_state_for_scan(batch_size, x.device)
        q_seq = memory_q.float().view(batch_size, seq_len, self.num_state_heads, self.rank)
        k_seq = memory_k.float().view(batch_size, seq_len, self.num_state_heads, self.rank)
        v_seq = memory_v.float().view(batch_size, seq_len, self.num_state_heads, self.rank)
        q_seq = q_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        k_seq = k_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        v_seq = v_seq.permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        keep_seq = keep.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        erase_seq = erase.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        write_seq = write.float().permute(0, 2, 1, 3).reshape(flat_batch, seq_len, self.rank)
        token_mask_for_scan = None
        if token_mask is not None:
            token_mask_for_scan = (
                token_mask.unsqueeze(1)
                .expand(batch_size, self.num_state_heads, seq_len)
                .reshape(flat_batch, seq_len)
            )

        if not self.write_enabled:
            reads_flat = self._state_reads_only(state, q_seq, token_mask_for_scan, x.dtype)
            reads = reads_flat.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
            return reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim)

        if self.backend in {"auto", "cuda"} and triton_affine_scan is not None and triton_scan_support is not None:
            support = triton_scan_support(state, q_seq, k_seq, v_seq, keep_seq, erase_seq, write_seq)
            if support.supported:
                state_out, reads_flat = triton_affine_scan(
                    state,
                    q_seq,
                    k_seq,
                    v_seq,
                    keep_seq,
                    erase_seq,
                    write_seq,
                    token_mask_for_scan,
                )
                self._store_final_state(state_out, batch_size)
                reads = reads_flat.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
                return reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim).to(dtype=x.dtype)

        state_out, reads_flat = self._scan_batched_torch(
            memory_q=q_seq,
            memory_k=k_seq,
            memory_v=v_seq,
            keep=keep_seq,
            erase=erase_seq,
            write=write_seq,
            token_mask=token_mask_for_scan,
            dtype=x.dtype,
            initial_state=state,
        )
        self._store_final_state(state_out, batch_size)
        reads = reads_flat.reshape(batch_size, self.num_state_heads, seq_len, self.rank)
        return reads.permute(0, 2, 1, 3).reshape(batch_size, seq_len, self.state_read_dim)

    def _delta_scale_multiplier(self, head: str, dtype: torch.dtype, device: torch.device) -> Tensor:
        if self.delta_scale_raw is None:
            return torch.ones((), dtype=dtype, device=device)
        if self.delta_scale_granularity == "head":
            raw = self.delta_scale_raw[VALID_DELTA_MEM_HEADS.index(head)]
        else:
            raw = self.delta_scale_raw[0]
        return (torch.sigmoid(raw) * self.delta_scale_max).to(device=device, dtype=dtype)

    def _apply_delta_o_rmsnorm(self, delta: Tensor) -> Tensor:
        if self.delta_o_rmsnorm_weight is None:
            return delta
        normalized = F.rms_norm(
            delta.float(),
            (delta.shape[-1],),
            weight=self.delta_o_rmsnorm_weight.float(),
            eps=self.delta_o_rmsnorm_eps,
        )
        return normalized.to(dtype=delta.dtype)

    def _project_delta(self, reads: Tensor, head: str, proj: nn.Linear) -> Optional[Tensor]:
        if head not in self.delta_heads:
            return None
        delta = proj(reads) * self.delta_scaling * self._delta_scale_multiplier(head, reads.dtype, reads.device)
        if head == "o":
            delta = self._apply_delta_o_rmsnorm(delta)
        return delta

    def _deltas_from_reads(self, reads: Tensor) -> dict[str, Tensor]:
        deltas = {
            "q": self._project_delta(reads, "q", self.delta_q_proj),
            "k": self._project_delta(reads, "k", self.delta_k_proj),
            "v": self._project_delta(reads, "v", self.delta_v_proj),
            "o": self._project_delta(reads, "o", self.delta_o_proj),
        }
        return {name: delta for name, delta in deltas.items() if delta is not None}

    def _forward_batched(self, x: Tensor, token_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
        if self.read_context_mask is not None and self.read_context_mask.shape[:2] == x.shape[:2]:
            token_mask = self.read_context_mask.to(device=x.device, dtype=torch.bool)
        reads = self._scan_batched(x, token_mask=token_mask)
        return self._deltas_from_reads(reads)

    def _forward_packed(self, x: Tensor, cu_seqlens: Tensor, numseqs: Tensor | int) -> dict[str, Tensor]:
        n = int(numseqs.item()) if isinstance(numseqs, Tensor) else int(numseqs)
        empty = {
            "q": x.new_zeros((x.shape[0], self.query_size)),
            "k": x.new_zeros((x.shape[0], self.key_size)),
            "v": x.new_zeros((x.shape[0], self.value_size)),
            "o": x.new_zeros((x.shape[0], self.output_size)),
        }
        if n <= 0:
            return {name: value for name, value in empty.items() if name in self.delta_heads}

        cu = cu_seqlens[: n + 1].to(device=x.device, dtype=torch.long)
        starts = cu[:-1]
        lengths = cu[1:] - starts
        max_len = int(lengths.max().item())
        if max_len <= 0:
            return {name: value for name, value in empty.items() if name in self.delta_heads}

        pos = torch.arange(max_len, device=x.device, dtype=torch.long).unsqueeze(0)
        mask = pos < lengths.unsqueeze(1)
        src = starts.unsqueeze(1) + pos

        padded = x.new_zeros((n, max_len, x.shape[-1]))
        padded[mask] = x[src[mask]]
        read_mask = mask
        if self.read_context_mask is not None and self.read_context_mask.numel() == x.shape[0]:
            flat_read_mask = self.read_context_mask.to(device=x.device, dtype=torch.bool)
            read_mask = torch.zeros_like(mask)
            read_mask[mask] = flat_read_mask[src[mask]]
        padded_deltas = self._forward_batched(padded, token_mask=read_mask)

        out: dict[str, Tensor] = {}
        for name, padded_delta in padded_deltas.items():
            flat = empty[name]
            flat[src[mask]] = padded_delta[mask]
            out[name] = flat
        return out

    def forward(
        self,
        x: Tensor,
        cu_seqlens: Optional[Tensor] = None,
        numseqs: Optional[Tensor] = None,
        **_seq_info,
    ) -> dict[str, Tensor]:
        if self.memory_write_granularity != "token":
            raise NotImplementedError(
                "HRM delta-rule memory currently supports token-granularity TSW writes only. "
                "message_mean/sentence_mean require an episode/chat collator that supplies write span IDs "
                "and explicit state priming, matching upstream delta-Mem SSW/MSW training."
            )
        if x.dim() == 3:
            return self._forward_batched(x)
        if x.dim() != 2:
            raise ValueError(f"DeltaRuleStateMemory expects [B,T,C] or [T,C], got {tuple(x.shape)}")
        if cu_seqlens is None or numseqs is None:
            return {name: delta.squeeze(0) for name, delta in self._forward_batched(x.unsqueeze(0)).items()}

        cu_seqlens = unwrap_tensor(cu_seqlens)
        numseqs = unwrap_tensor(numseqs)
        return self._forward_packed(x, cu_seqlens, numseqs)


def iter_rwkv_mem_modules(module: nn.Module):
    for name, child in module.named_modules():
        if isinstance(child, DeltaRuleStateMemory):
            yield name, child


def reset_rwkv_mem_states(module: nn.Module) -> None:
    for _name, child in iter_rwkv_mem_modules(module):
        child.reset_state()


def set_rwkv_mem_write_enabled(module: nn.Module, enabled: bool) -> None:
    for _name, child in iter_rwkv_mem_modules(module):
        child.set_write_enabled(enabled)


def set_rwkv_mem_read_context_mask(module: nn.Module, mask: Optional[Tensor]) -> None:
    for _name, child in iter_rwkv_mem_modules(module):
        child.set_read_context_mask(mask)


def set_rwkv_mem_runtime_enabled(module: nn.Module, enabled: bool) -> None:
    for child in module.modules():
        if hasattr(child, "rwkv_mem_runtime_enabled"):
            child.rwkv_mem_runtime_enabled = bool(enabled)
