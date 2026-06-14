from __future__ import annotations

from typing import Optional, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from models.common import unwrap_tensor
from models.rwkv7 import RWKV7Config, RWKV7TimeMix

try:
    from deltamem.kernels.affine_scan import triton_affine_scan, triton_scan_support
except Exception:  # pragma: no cover - optional local reference repo dependency
    triton_affine_scan = None
    triton_scan_support = None


VALID_DELTA_MEM_HEADS = ("q", "k", "v", "o")
VALID_DELTA_MEM_STATE_UPDATE_MODES = ("standard", "lambda_outside", "no_lambda")


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
        **kwargs,
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
        alpha: float = 16.0,
        beta_bias_init: float = -1.5,
        normalize_qk: bool = True,
        couple_lambda: bool = True,
        state_update_mode: str = "standard",
        rankwise_gates: bool = True,
        delta_heads: Sequence[str] | str = ("q", "k", "v", "o"),
        output_init: str = "zero",
        output_init_scale: float = 0.02,
        backend: str = "auto",
        **kwargs,
    ) -> None:
        super().__init__()
        if rank < 1:
            raise ValueError("rank must be >= 1")
        if state_update_mode not in VALID_DELTA_MEM_STATE_UPDATE_MODES:
            raise ValueError(
                f"Unsupported state_update_mode={state_update_mode!r}; "
                f"expected one of {VALID_DELTA_MEM_STATE_UPDATE_MODES}"
            )

        self.hidden_size = hidden_size
        self.query_size = query_size
        self.key_size = key_size
        self.value_size = value_size
        self.output_size = output_size
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.delta_scaling = self.alpha / self.rank
        self.beta_bias_init = float(beta_bias_init)
        self.normalize_qk = bool(normalize_qk)
        self.couple_lambda = bool(couple_lambda)
        self.state_update_mode = state_update_mode
        self.rankwise_gates = bool(rankwise_gates)
        self.delta_heads = normalize_delta_mem_heads(delta_heads)
        if backend not in {"auto", "cuda", "torch"}:
            raise ValueError("delta-rule memory backend must be 'auto', 'cuda', or 'torch'")
        self.backend = backend

        gate_dim = self.rank if self.rankwise_gates else 1
        self.memory_q_proj = nn.Linear(hidden_size, self.rank, bias=False, **kwargs)
        self.memory_k_proj = nn.Linear(hidden_size, self.rank, bias=False, **kwargs)
        self.memory_v_proj = nn.Linear(hidden_size, self.rank, bias=False, **kwargs)
        self.beta_proj = nn.Linear(hidden_size, gate_dim, bias=False, **kwargs)
        self.beta_bias = nn.Parameter(torch.full((gate_dim,), self.beta_bias_init, **kwargs))
        if self.couple_lambda:
            self.lambda_proj = None
            self.lambda_bias = None
        else:
            self.lambda_proj = nn.Linear(hidden_size, gate_dim, bias=False, **kwargs)
            self.lambda_bias = nn.Parameter(torch.full((gate_dim,), -self.beta_bias_init, **kwargs))

        self.delta_q_proj = nn.Linear(self.rank, query_size, bias=False, **kwargs)
        self.delta_k_proj = nn.Linear(self.rank, key_size, bias=False, **kwargs)
        self.delta_v_proj = nn.Linear(self.rank, value_size, bias=False, **kwargs)
        self.delta_o_proj = nn.Linear(self.rank, output_size, bias=False, **kwargs)
        self.reset_parameters(output_init, output_init_scale)

    def reset_parameters(self, output_init: str, output_init_scale: float) -> None:
        for proj in (self.memory_q_proj, self.memory_k_proj, self.memory_v_proj):
            nn.init.kaiming_uniform_(proj.weight, a=5**0.5)
        nn.init.zeros_(self.beta_proj.weight)
        if self.lambda_proj is not None:
            nn.init.zeros_(self.lambda_proj.weight)

        for name, proj in (
            ("q", self.delta_q_proj),
            ("k", self.delta_k_proj),
            ("v", self.delta_v_proj),
            ("o", self.delta_o_proj),
        ):
            if output_init == "zero" or name not in self.delta_heads:
                nn.init.zeros_(proj.weight)
            elif output_init == "small":
                nn.init.trunc_normal_(
                    proj.weight,
                    mean=0.0,
                    std=output_init_scale,
                    a=-3 * output_init_scale,
                    b=3 * output_init_scale,
                )
            else:
                raise ValueError(f"Unknown delta-memory output init: {output_init}")

    def _project_memory(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        memory_q = self.memory_q_proj(x)
        memory_k = self.memory_k_proj(x)
        memory_v = self.memory_v_proj(x)
        if self.normalize_qk:
            memory_q = F.normalize(torch.tanh(memory_q.float()), dim=-1, eps=1e-6).to(dtype=x.dtype)
            memory_k = F.normalize(torch.tanh(memory_k.float()), dim=-1, eps=1e-6).to(dtype=x.dtype)

        beta = torch.sigmoid(self.beta_proj(x) + self.beta_bias.to(device=x.device, dtype=x.dtype))
        if self.rankwise_gates:
            beta = beta.unsqueeze(-1)
        else:
            beta = beta.expand(*beta.shape[:-1], self.rank).unsqueeze(-1)

        if self.state_update_mode == "no_lambda":
            lam = torch.ones_like(beta)
        elif self.couple_lambda:
            lam = 1.0 - beta
        else:
            assert self.lambda_proj is not None and self.lambda_bias is not None
            lam = torch.sigmoid(self.lambda_proj(x) + self.lambda_bias.to(device=x.device, dtype=x.dtype))
            if self.rankwise_gates:
                lam = lam.unsqueeze(-1)
            else:
                lam = lam.expand(*lam.shape[:-1], self.rank).unsqueeze(-1)

        return memory_q, memory_k, memory_v, beta, lam

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
    ) -> Tensor:
        batch_size, seq_len, _ = memory_q.shape
        state = torch.zeros(batch_size, self.rank, self.rank, device=memory_q.device, dtype=torch.float32)
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

        return torch.stack(reads, dim=1)

    def _scan_batched(self, x: Tensor, token_mask: Optional[Tensor] = None) -> Tensor:
        memory_q, memory_k, memory_v, beta, lam = self._project_memory(x)
        keep, erase, write = self._update_coefficients(beta.float(), lam.float())

        batch_size, seq_len, _ = x.shape
        state = torch.zeros(batch_size, self.rank, self.rank, device=x.device, dtype=torch.float32)
        q_seq = memory_q.float()
        k_seq = memory_k.float()
        v_seq = memory_v.float()
        keep_seq = keep.squeeze(-1).float()
        erase_seq = erase.squeeze(-1).float()
        write_seq = write.squeeze(-1).float()

        if self.backend in {"auto", "cuda"} and triton_affine_scan is not None and triton_scan_support is not None:
            support = triton_scan_support(state, q_seq, k_seq, v_seq, keep_seq, erase_seq, write_seq)
            if support.supported:
                _state_out, reads = triton_affine_scan(
                    state,
                    q_seq,
                    k_seq,
                    v_seq,
                    keep_seq,
                    erase_seq,
                    write_seq,
                    token_mask,
                )
                return reads.to(dtype=x.dtype)

        return self._scan_batched_torch(
            memory_q=q_seq,
            memory_k=k_seq,
            memory_v=v_seq,
            keep=keep_seq,
            erase=erase_seq,
            write=write_seq,
            token_mask=token_mask,
            dtype=x.dtype,
        )

    def _project_delta(self, reads: Tensor, head: str, proj: nn.Linear) -> Optional[Tensor]:
        if head not in self.delta_heads:
            return None
        return proj(reads) * self.delta_scaling

    def _deltas_from_reads(self, reads: Tensor) -> dict[str, Tensor]:
        deltas = {
            "q": self._project_delta(reads, "q", self.delta_q_proj),
            "k": self._project_delta(reads, "k", self.delta_k_proj),
            "v": self._project_delta(reads, "v", self.delta_v_proj),
            "o": self._project_delta(reads, "o", self.delta_o_proj),
        }
        return {name: delta for name, delta in deltas.items() if delta is not None}

    def _forward_batched(self, x: Tensor, token_mask: Optional[Tensor] = None) -> dict[str, Tensor]:
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
        padded_deltas = self._forward_batched(padded, token_mask=mask)

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
        if x.dim() == 3:
            return self._forward_batched(x)
        if x.dim() != 2:
            raise ValueError(f"DeltaRuleStateMemory expects [B,T,C] or [T,C], got {tuple(x.shape)}")
        if cu_seqlens is None or numseqs is None:
            return {name: delta.squeeze(0) for name, delta in self._forward_batched(x.unsqueeze(0)).items()}

        cu_seqlens = unwrap_tensor(cu_seqlens)
        numseqs = unwrap_tensor(numseqs)
        return self._forward_packed(x, cu_seqlens, numseqs)
