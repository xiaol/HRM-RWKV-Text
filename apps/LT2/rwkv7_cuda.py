from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load


_CUDA_DIR = Path(os.environ.get("LT2_RWKV7_CUDA_DIR", "/home/xiaol/X/LT2_upstream/apps/LT2/cuda/rwkv7"))
_LOADED = False
_HEAD_SIZE = 64
_CHUNK_LEN = 16


def _has_op(namespace: str, op: str = "forward") -> bool:
    try:
        getattr(getattr(torch.ops, namespace), op)
        return True
    except (AttributeError, RuntimeError):
        return False


def _load_extension(name: str, namespace: str, sources: list[str], extra_cuda_cflags: list[str] | None = None) -> None:
    if _has_op(namespace):
        return
    load(
        name=name,
        sources=[str(_CUDA_DIR / source) for source in sources],
        extra_cflags=["-O3"],
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "-Xptxas", "-O3",
            "--extra-device-vectorization",
            *(extra_cuda_cflags or []),
        ],
        is_python_module=False,
        verbose=False,
    )


def ensure_loaded(head_size: int = _HEAD_SIZE, chunk_len: int = _CHUNK_LEN) -> None:
    global _LOADED
    if head_size != _HEAD_SIZE:
        raise RuntimeError(f"LT2 RWKV7 wrapper supports head_size={_HEAD_SIZE}, got {head_size}")
    if chunk_len != _CHUNK_LEN:
        raise RuntimeError(f"LT2 RWKV7 wrapper supports chunk_len={_CHUNK_LEN}, got {chunk_len}")
    if _LOADED:
        return
    if not _CUDA_DIR.exists():
        raise FileNotFoundError(f"LT2 RWKV7 CUDA source directory not found: {_CUDA_DIR}")

    _load_extension(
        "lt2_rwkv7_clampw_h64_c16",
        "rwkv7_clampw",
        ["rwkv7_clampw.cpp", "rwkv7_clampw.cu"],
        [f"-D_N_={head_size}", f"-D_CHUNK_LEN_={chunk_len}"],
    )
    _load_extension(
        "lt2_rwkv7_cmix_bf16_v5",
        "rwkv7_cmix_bf16_v5",
        ["rwkv7_cmix_bf16_v5.cpp", "rwkv7_cmix_bf16_v5.cu"],
    )
    _load_extension(
        "lt2_rwkv7_tmix_mix6_bf16_v5",
        "rwkv7_tmix_mix6_bf16_v5",
        ["rwkv7_tmix_mix6_bf16_v5.cpp", "rwkv7_tmix_mix6_bf16_v5.cu"],
    )
    _load_extension(
        "lt2_rwkv7_tmix_kk_pre_bf16_v5",
        "rwkv7_tmix_kk_pre_bf16_v5",
        ["rwkv7_tmix_kk_pre_bf16_v5.cpp", "rwkv7_tmix_kk_pre_bf16_v5.cu"],
    )
    _load_extension(
        "lt2_rwkv7_tmix_lnx_rkvres_xg_bf16_v1",
        "rwkv7_tmix_lnx_rkvres_xg_bf16_v1",
        ["rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cpp", "rwkv7_tmix_lnx_rkvres_xg_bf16_v1.cu"],
    )
    _load_extension(
        "lt2_rwkv7_tmix_a_gate_bf16",
        "rwkv7_tmix_a_gate_bf16",
        ["rwkv7_tmix_a_gate_bf16.cpp", "rwkv7_tmix_a_gate_bf16.cu"],
    )
    _load_extension(
        "lt2_rwkv7_tmix_vres_gate_bf16_v1",
        "rwkv7_tmix_vres_gate_bf16_v1",
        ["rwkv7_tmix_vres_gate_bf16_v1.cpp", "rwkv7_tmix_vres_gate_bf16_v1.cu"],
    )
    _LOADED = True


def tmix_mix6(x, x_r, x_w, x_k, x_v, x_a, x_g):
    ensure_loaded()
    return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(
        x.contiguous(),
        x_r.contiguous(),
        x_w.contiguous(),
        x_k.contiguous(),
        x_v.contiguous(),
        x_a.contiguous(),
        x_g.contiguous(),
    ))


def tmix_vres_gate(v, v_first, v0, v12):
    ensure_loaded()
    return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(
        v.contiguous(),
        v_first.contiguous(),
        v0.contiguous(),
        v12.contiguous(),
    )


def tmix_a_gate(a0, a12):
    ensure_loaded()
    return torch.ops.rwkv7_tmix_a_gate_bf16.forward(a0.contiguous(), a12.contiguous())


def tmix_kk_pre(k, k_k, a, k_a, head_size: int):
    ensure_loaded(head_size=head_size)
    outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(
        k.contiguous(),
        k_k.contiguous(),
        a.contiguous(),
        k_a.contiguous(),
        head_size,
    )
    return outs[0], outs[1], outs[2]


def tmix_lnx_rkvres_xg(x, r, k, v, r_k, weight, bias, g):
    ensure_loaded()
    outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(
        x.contiguous(),
        r.contiguous(),
        k.contiguous(),
        v.contiguous(),
        r_k.contiguous(),
        weight.contiguous(),
        bias.contiguous(),
        g.contiguous(),
    )
    return outs[0]


def cmix_layer(x, x_k, key_weight, value_weight):
    ensure_loaded()
    outs = torch.ops.rwkv7_cmix_bf16_v5.forward(
        x.contiguous(),
        x_k.contiguous(),
        key_weight.contiguous(),
        value_weight.contiguous(),
    )
    return outs[0]


def rwkv7_recurrence_cuda_bf16(r, w, k, v, a, b, head_size: int, chunk_len: int):
    ensure_loaded(head_size=head_size, chunk_len=chunk_len)
    batch, seq_len, channels = r.shape
    pad_len = (-seq_len) % chunk_len
    if pad_len:
        r, w, k, v, a, b = [F.pad(t, (0, 0, 0, pad_len)).contiguous() for t in (r, w, k, v, a, b)]

    padded_len = r.shape[1]
    heads = channels // head_size
    rv, wv, kv, vv, av, bv = [
        t.view(batch, padded_len, heads, head_size).contiguous()
        for t in (r, w, k, v, a, b)
    ]
    y = torch.empty_like(vv)
    s = torch.empty(batch, heads, padded_len // chunk_len, head_size, head_size, dtype=torch.float32, device=w.device)
    sa = torch.empty(batch, padded_len, heads, head_size, dtype=torch.float32, device=w.device)
    torch.ops.rwkv7_clampw.forward(rv, wv, kv, vv, av, bv, y, s, sa)
    return y.view(batch, padded_len, channels)[:, :seq_len]
