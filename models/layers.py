from typing import Tuple, Optional, Sequence, Any, NamedTuple, Literal
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from einops import rearrange

from models.common import trunc_normal_init_, unwrap_tensor
from models.flash_attention_prefixlm_v2 import flash_attn_varlen_prefixlm
try:
    from flash_attn_interface import flash_attn_with_kvcache
except ModuleNotFoundError:
    def flash_attn_with_kvcache(q: Tensor, k: Tensor, v: Tensor, k_cache: Tensor, v_cache: Tensor, cache_seqlens, causal: bool = False, **_kwargs):
        """PyTorch fallback for 4090/local eval where FA3 KV-cache kernels are unavailable."""
        batch, seq_len = q.shape[:2]
        if isinstance(cache_seqlens, Tensor):
            if cache_seqlens.dim() == 0:
                lengths = torch.full((batch,), int(cache_seqlens.item()), device=k_cache.device, dtype=torch.long)
            else:
                lengths = cache_seqlens.to(device=k_cache.device, dtype=torch.long)
        else:
            lengths = torch.full((batch,), int(cache_seqlens), device=k_cache.device, dtype=torch.long)

        out = torch.empty_like(q)
        for b in range(batch):
            start = int(lengths[b].item())
            end = start + seq_len
            k_cache[b, start:end].copy_(k[b])
            v_cache[b, start:end].copy_(v[b])

            qb = q[b:b + 1].transpose(1, 2)
            kb = k_cache[b:b + 1, :end].transpose(1, 2)
            vb = v_cache[b:b + 1, :end].transpose(1, 2)
            attn_mask = None
            if causal and seq_len > 1:
                q_pos = torch.arange(start, end, device=q.device).unsqueeze(1)
                k_pos = torch.arange(end, device=q.device).unsqueeze(0)
                attn_mask = k_pos <= q_pos
            out[b:b + 1] = F.scaled_dot_product_attention(qb, kb, vb, attn_mask=attn_mask).transpose(1, 2)
        return out


Carry = dict[str, Any]
CosSin = Tuple[Tensor, Tensor]
AttnType = Literal["causal", "prefixlm"]


def find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x: Tensor, cos_sin: CosSin):
    # x:   [..., seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim] OR [..., seq_len, head_dim]
    # Use FP32 RoPE, as in Transformers OLMo and FlashAttention
    # 
    # https://github.com/huggingface/transformers/blob/v4.55.4/src/transformers/models/olmo/modular_olmo.py#L139-L152
    # https://github.com/Dao-AILab/flash-attention/blob/v2.8.3/csrc/flash_attn/src/rotary.h#L126-L133
    cos, sin = cos_sin
    return ((x * cos.unsqueeze(-2)) + (rotate_half(x) * sin.unsqueeze(-2))).to(x.dtype)


class RotaryEmbedding(torch.nn.Module):
    def __init__(self, dim, max_seq_len, base, **kwargs):
        super().__init__()
        # RoPE
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, **kwargs) / dim))
        t = torch.arange(max_seq_len, dtype=torch.float32, **kwargs)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self, position_ids: Tensor):
        if position_ids is not None:
            return self.cos_cached[position_ids], self.sin_cached[position_ids]

        return self.cos_cached, self.sin_cached


class LinearInit(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool,
                 batch_out_features: Sequence[int] = (),
                 init_std: Optional[float] = None,
                 **kwargs):
        super().__init__()
        self.in_features = in_features
        # Truncated LeCun normal init
        if init_std is None:
            init_std = 1.0 / (in_features ** 0.5)

        # Parameters
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((math.prod(batch_out_features) * out_features, in_features), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((math.prod(batch_out_features) * out_features, ), **kwargs))

    def forward(self, input: Tensor) -> Tensor:
        return F.linear(input, self.weight, self.bias)


class ScaledEmbeddingInit(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 **kwargs):
        super().__init__()
        self.scale = 1.0 / init_std

        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim), **kwargs), std=init_std)  # pyright: ignore[reportArgumentType]
        )

    def forward(self, input: Tensor) -> Tensor:
        return self.scale * F.embedding(input, self.embedding_weight)


class Cache(NamedTuple):
    """A static cache layer that stores the key and value states as static tensors. Built for `torch.compile` support."""
    keys: Tensor
    values: Tensor

    @classmethod
    def create(cls, max_batch_size: int, max_seq_len: int, num_heads: int, head_dim: int, **kwargs):
        return cls(keys=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs),
                   values=torch.zeros((max_batch_size, max_seq_len, num_heads, head_dim), **kwargs))


class Attention(nn.Module):
    def __init__(
        self,
        hidden_size,
        head_dim,
        num_heads,
        num_key_value_heads,
        attn_type,
        max_seq_len,
        init_std_in=None,
        init_std_out=None,
        rwkv_mem_enabled: bool = False,
        rwkv_mem_mode: str = "delta_rule",
        rwkv_mem_head_size: int = 64,
        rwkv_mem_backend: str = "auto",
        rwkv_mem_chunk_len: int = 16,
        rwkv_mem_scale: float = 1.0,
        rwkv_mem_output_init: str = "zero",
        rwkv_mem_output_init_scale: float = 0.02,
        rwkv_mem_delta_heads: Sequence[str] = ("q", "k", "v", "o"),
        rwkv_mem_separate_delta_projections: bool = False,
        rwkv_mem_rank: int = 8,
        rwkv_mem_alpha: float = 16.0,
        rwkv_mem_beta_bias_init: float = -1.5,
        rwkv_mem_normalize_qk: bool = True,
        rwkv_mem_couple_lambda: bool = True,
        rwkv_mem_state_update_mode: str = "standard",
        rwkv_mem_rankwise_gates: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.attn_type = attn_type
        self.rwkv_mem_mode = rwkv_mem_mode
        self.rwkv_mem_delta_heads = tuple(rwkv_mem_delta_heads)
        self.rwkv_mem_separate_delta_projections = rwkv_mem_separate_delta_projections
        unknown_delta_heads = set(self.rwkv_mem_delta_heads) - {"q", "k", "v", "o"}
        if unknown_delta_heads:
            raise ValueError(f"Unknown RWKV memory delta heads: {sorted(unknown_delta_heads)}")
        if rwkv_mem_mode not in {"delta_rule", "rwkv7", "rwkv7_legacy"}:
            raise ValueError("rwkv_mem_mode must be 'delta_rule', 'rwkv7', or 'rwkv7_legacy'")

        self.gqkv_proj = LinearInit(hidden_size, self.head_dim, batch_out_features=(2 * self.num_heads + 2 * self.num_key_value_heads, ),
                                   bias=False, init_std=init_std_in, **kwargs)
        self.o_proj = LinearInit(head_dim * num_heads, hidden_size,
                                 bias=False, init_std=init_std_out, **kwargs)
        self.rwkv_mem = None
        self.rwkv_mem_q_proj = None
        self.rwkv_mem_k_proj = None
        self.rwkv_mem_v_proj = None
        self.rwkv_mem_o_proj = None
        if rwkv_mem_enabled:
            from models.rwkv_memory import DeltaRuleStateMemory, RWKVStateMemory

            cpu_rng_state = torch.random.get_rng_state()
            cuda_rng_states = torch.cuda.get_rng_state_all() if torch.cuda.is_available() and torch.cuda.is_initialized() else None
            try:
                if rwkv_mem_mode == "delta_rule":
                    self.rwkv_mem = DeltaRuleStateMemory(
                        hidden_size=hidden_size,
                        query_size=head_dim * num_heads,
                        key_size=head_dim * num_key_value_heads,
                        value_size=head_dim * num_key_value_heads,
                        output_size=hidden_size,
                        rank=rwkv_mem_rank,
                        alpha=rwkv_mem_alpha,
                        beta_bias_init=rwkv_mem_beta_bias_init,
                        normalize_qk=rwkv_mem_normalize_qk,
                        couple_lambda=rwkv_mem_couple_lambda,
                        state_update_mode=rwkv_mem_state_update_mode,
                        rankwise_gates=rwkv_mem_rankwise_gates,
                        delta_heads=rwkv_mem_delta_heads,
                        output_init=rwkv_mem_output_init,
                        output_init_scale=rwkv_mem_output_init_scale,
                        backend=rwkv_mem_backend,
                        **kwargs,
                    )
                else:
                    self.rwkv_mem = RWKVStateMemory(
                        max_seq_len=max_seq_len,
                        hidden_size=hidden_size,
                        head_size=rwkv_mem_head_size,
                        backend=rwkv_mem_backend,
                        chunk_len=rwkv_mem_chunk_len,
                        scale=rwkv_mem_scale,
                        output_init=rwkv_mem_output_init,
                        output_init_scale=rwkv_mem_output_init_scale,
                        **kwargs,
                    )
                    force_projections = rwkv_mem_mode == "rwkv7"
                    if force_projections or rwkv_mem_separate_delta_projections:
                        if "q" in self.rwkv_mem_delta_heads:
                            self.rwkv_mem_q_proj = nn.Linear(hidden_size, head_dim * num_heads, bias=False, **kwargs)
                            self._init_rwkv_mem_projection(self.rwkv_mem_q_proj)
                        if "k" in self.rwkv_mem_delta_heads:
                            self.rwkv_mem_k_proj = nn.Linear(hidden_size, head_dim * num_key_value_heads, bias=False, **kwargs)
                            self._init_rwkv_mem_projection(self.rwkv_mem_k_proj)
                        if "v" in self.rwkv_mem_delta_heads:
                            self.rwkv_mem_v_proj = nn.Linear(hidden_size, head_dim * num_key_value_heads, bias=False, **kwargs)
                            self._init_rwkv_mem_projection(self.rwkv_mem_v_proj)
                        if "o" in self.rwkv_mem_delta_heads:
                            self.rwkv_mem_o_proj = nn.Linear(hidden_size, hidden_size, bias=False, **kwargs)
                            self._init_rwkv_mem_projection(self.rwkv_mem_o_proj)
            finally:
                torch.random.set_rng_state(cpu_rng_state)
                if cuda_rng_states is not None:
                    torch.cuda.set_rng_state_all(cuda_rng_states)

    @staticmethod
    def _init_rwkv_mem_projection(proj: nn.Linear) -> None:
        nn.init.zeros_(proj.weight)
        diag = min(proj.weight.shape)
        with torch.no_grad():
            proj.weight[:diag, :diag].copy_(torch.eye(diag, device=proj.weight.device, dtype=proj.weight.dtype))

    @staticmethod
    def _project_rwkv_mem_delta(delta: Tensor, proj: Optional[nn.Linear]) -> Tensor:
        return proj(delta) if proj is not None else delta

    def _should_run_rwkv_mem(self, hidden_states: Tensor, cache: Optional[Cache]) -> bool:
        # Cached generation prefill passes the full prompt with cache allocated.
        # Single-token decode still needs persistent RWKV state, so skip it.
        return self.rwkv_mem is not None and (cache is None or hidden_states.shape[-2] > 1)

    def forward(self, hidden_states: Tensor, cos_sin: Optional[CosSin], cache: Optional[Cache] = None, cache_lengths: Optional[Tensor] = None, **seq_info) -> Tensor:
        # hidden_states, gqkv: [..., seq_len, hidden_size]
        gqkv = self.gqkv_proj(hidden_states)

        # Split head (last dimension of projected qkv)
        gqkv = rearrange(gqkv, "... (h hd) -> ... h hd", h=2 * self.num_heads + 2 * self.num_key_value_heads)
        gate, query, key, value = gqkv.split((self.num_heads, self.num_heads, self.num_key_value_heads, self.num_key_value_heads), dim=-2)

        rwkv_mem_delta = None
        rwkv_mem_deltas = None
        if self._should_run_rwkv_mem(hidden_states, cache):
            if self.rwkv_mem_mode == "delta_rule":
                rwkv_mem_deltas = self.rwkv_mem(hidden_states, **seq_info)  # type: ignore[operator]
                if "q" in rwkv_mem_deltas:
                    query = query + rearrange(rwkv_mem_deltas["q"], "... (h hd) -> ... h hd", h=self.num_heads)
                if "k" in rwkv_mem_deltas:
                    key = key + rearrange(rwkv_mem_deltas["k"], "... (h hd) -> ... h hd", h=self.num_key_value_heads)
                if "v" in rwkv_mem_deltas:
                    value = value + rearrange(rwkv_mem_deltas["v"], "... (h hd) -> ... h hd", h=self.num_key_value_heads)
            else:
                rwkv_mem_delta = self.rwkv_mem(hidden_states, **seq_info)  # type: ignore[operator]
                if "q" in self.rwkv_mem_delta_heads:
                    q_delta = self._project_rwkv_mem_delta(rwkv_mem_delta, self.rwkv_mem_q_proj)
                    query = query + rearrange(q_delta, "... (h hd) -> ... h hd", h=self.num_heads)
                if "k" in self.rwkv_mem_delta_heads:
                    k_delta = self._project_rwkv_mem_delta(rwkv_mem_delta, self.rwkv_mem_k_proj)
                    key = key + rearrange(k_delta, "... (h hd) -> ... h hd", h=self.num_key_value_heads)
                if "v" in self.rwkv_mem_delta_heads:
                    v_delta = self._project_rwkv_mem_delta(rwkv_mem_delta, self.rwkv_mem_v_proj)
                    value = value + rearrange(v_delta, "... (h hd) -> ... h hd", h=self.num_key_value_heads)

        # query, key, value: [..., seq_len, num_heads, head_dim]
        # RoPE
        if cos_sin is not None:
            query = apply_rotary_pos_emb(query, cos_sin)
            key = apply_rotary_pos_emb(key, cos_sin)

        is_causal = self.attn_type == "causal"
        if cache is None and "prefix_lens" not in seq_info:
            q = query.transpose(-3, -2)
            k = key.transpose(-3, -2)
            v = value.transpose(-3, -2)
            attn_output = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(-3, -2)
        elif cache is None:
            # flash attn (training)
            attn_output = flash_attn_varlen_prefixlm(query, key, value, is_causal, **{name: unwrap_tensor(tensor) for name, tensor in seq_info.items()})
        else:
            # Regardless of auto / non-autoregressive, apply attention based on current concatenated with cache.
            attn_output = flash_attn_with_kvcache(q=query, k=key, v=value,
                                                  k_cache=cache.keys, v_cache=cache.values, cache_seqlens=cache_lengths,
                                                  num_splits=1,  # Must set to support torch.compile tracing.
                                                  causal=is_causal)  # causal can always be False for PrefixLM. during AR generation seqlen is 1, so causal masking won't matter.

        # attn_output: [..., seq_len, num_heads, head_dim]
        attn_output = rearrange(torch.sigmoid(gate) * attn_output, "... h hd -> ... (h hd)")  # type: ignore
        out = self.o_proj(attn_output)
        if rwkv_mem_deltas is not None and "o" in rwkv_mem_deltas:
            out = out + rwkv_mem_deltas["o"]
        elif rwkv_mem_delta is not None and "o" in self.rwkv_mem_delta_heads:
            o_delta = self._project_rwkv_mem_delta(rwkv_mem_delta, self.rwkv_mem_o_proj)
            out = out + o_delta
        return out


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int, init_std_in=None, init_std_out=None, **kwargs):
        super().__init__()
        self.gate_up_proj = LinearInit(hidden_size, intermediate_size, batch_out_features=(2, ),
                                       bias=False, init_std=init_std_in, **kwargs)
        self.down_proj    = LinearInit(intermediate_size, hidden_size,
                                       bias=False, init_std=init_std_out, **kwargs)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)
