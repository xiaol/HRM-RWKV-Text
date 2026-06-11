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


def _none_like_unused(*_args):
    return None


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
    class _TmixMix6(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, x_r, x_w, x_k, x_v, x_a, x_g):
            ensure_loaded()
            x = x.contiguous()
            x_r = x_r.contiguous()
            x_w = x_w.contiguous()
            x_k = x_k.contiguous()
            x_v = x_v.contiguous()
            x_a = x_a.contiguous()
            x_g = x_g.contiguous()
            ctx.save_for_backward(x, x_r, x_w, x_k, x_v, x_a, x_g)
            return tuple(torch.ops.rwkv7_tmix_mix6_bf16_v5.forward(x, x_r, x_w, x_k, x_v, x_a, x_g))

        @staticmethod
        def backward(ctx, grad_r, grad_w, grad_k, grad_v, grad_a, grad_g):
            x, x_r, x_w, x_k, x_v, x_a, x_g = ctx.saved_tensors
            grads = torch.ops.rwkv7_tmix_mix6_bf16_v5.backward(
                grad_r.contiguous(),
                grad_w.contiguous(),
                grad_k.contiguous(),
                grad_v.contiguous(),
                grad_a.contiguous(),
                grad_g.contiguous(),
                x,
                x_r,
                x_w,
                x_k,
                x_v,
                x_a,
                x_g,
            )
            return tuple(grads)

    return _TmixMix6.apply(x, x_r, x_w, x_k, x_v, x_a, x_g)


def tmix_vres_gate(v, v_first, v0, v12):
    class _TmixVresGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, v, v_first, v0, v12):
            ensure_loaded()
            v = v.contiguous()
            v_first = v_first.contiguous()
            v0 = v0.contiguous()
            v12 = v12.contiguous()
            ctx.save_for_backward(v, v_first, v0, v12)
            return torch.ops.rwkv7_tmix_vres_gate_bf16_v1.forward(v, v_first, v0, v12)

        @staticmethod
        def backward(ctx, grad_out):
            v, v_first, v0, v12 = ctx.saved_tensors
            grad_v, grad_v_first, grad_pre = torch.ops.rwkv7_tmix_vres_gate_bf16_v1.backward(
                grad_out.contiguous(), v, v_first, v0, v12
            )
            grad_v12 = grad_pre.to(dtype=v12.dtype)
            grad_v0 = grad_pre.float().sum(dim=(0, 1)).to(dtype=v0.dtype)
            return grad_v, grad_v_first, grad_v0, grad_v12

    return _TmixVresGate.apply(v, v_first, v0, v12)


def tmix_a_gate(a0, a12):
    class _TmixAGate(torch.autograd.Function):
        @staticmethod
        def forward(ctx, a0, a12):
            ensure_loaded()
            a0 = a0.contiguous()
            a12 = a12.contiguous()
            ctx.save_for_backward(a0, a12)
            return torch.ops.rwkv7_tmix_a_gate_bf16.forward(a0, a12)

        @staticmethod
        def backward(ctx, grad_out):
            a0, a12 = ctx.saved_tensors
            grad_a0, grad_a12 = torch.ops.rwkv7_tmix_a_gate_bf16.backward(grad_out.contiguous(), a0, a12)
            return grad_a0, grad_a12

    return _TmixAGate.apply(a0, a12)


def tmix_kk_pre(k, k_k, a, k_a, head_size: int):
    class _TmixKkPre(torch.autograd.Function):
        @staticmethod
        def forward(ctx, k, k_k, a, k_a, head_size: int):
            ensure_loaded(head_size=head_size)
            k = k.contiguous()
            k_k = k_k.contiguous()
            a = a.contiguous()
            k_a = k_a.contiguous()
            outs = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.forward(k, k_k, a, k_a, head_size)
            ctx.save_for_backward(k, k_k, a, k_a, outs[3])
            ctx.head_size = head_size
            return outs[0], outs[1], outs[2]

        @staticmethod
        def backward(ctx, grad_new_k, grad_neg_kk, grad_kka):
            k, k_k, a, k_a, inv_d = ctx.saved_tensors
            grads = torch.ops.rwkv7_tmix_kk_pre_bf16_v5.backward(
                grad_new_k.contiguous(),
                grad_neg_kk.contiguous(),
                grad_kka.contiguous(),
                k,
                k_k,
                a,
                k_a,
                inv_d,
                ctx.head_size,
            )
            return grads[0], grads[1], grads[2], grads[3], None

    return _TmixKkPre.apply(k, k_k, a, k_a, head_size)


def tmix_lnx_rkvres_xg(x, r, k, v, r_k, weight, bias, g):
    class _TmixLnxRkvresXg(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, r, k, v, r_k, weight, bias, g):
            ensure_loaded()
            x = x.contiguous()
            r = r.contiguous()
            k = k.contiguous()
            v = v.contiguous()
            r_k = r_k.contiguous()
            weight = weight.contiguous()
            bias = bias.contiguous()
            g = g.contiguous()
            outs = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.forward(x, r, k, v, r_k, weight, bias, g)
            ctx.save_for_backward(x, r, k, v, r_k, weight, bias, g, outs[1], outs[2])
            return outs[0]

        @staticmethod
        def backward(ctx, grad_out):
            x, r, k, v, r_k, weight, bias, g, mean, rstd = ctx.saved_tensors
            grads = torch.ops.rwkv7_tmix_lnx_rkvres_xg_bf16_v1.backward(
                grad_out.contiguous(), x, r, k, v, r_k, weight, bias, g, mean, rstd
            )
            return tuple(grads)

    return _TmixLnxRkvresXg.apply(x, r, k, v, r_k, weight, bias, g)


def cmix_layer(x, x_k, key_weight, value_weight):
    class _CmixLayer(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, x_k, key_weight, value_weight):
            ensure_loaded()
            x = x.contiguous()
            x_k = x_k.contiguous()
            key_weight = key_weight.contiguous()
            value_weight = value_weight.contiguous()
            outs = torch.ops.rwkv7_cmix_bf16_v5.forward(x, x_k, key_weight, value_weight)
            ctx.save_for_backward(x, x_k, key_weight, value_weight, outs[1], outs[2])
            return outs[0]

        @staticmethod
        def backward(ctx, grad_out):
            x, x_k, key_weight, value_weight, mixed, act = ctx.saved_tensors
            grads = torch.ops.rwkv7_cmix_bf16_v5.backward(
                grad_out.contiguous(), x, x_k, key_weight, value_weight, mixed, act
            )
            return tuple(grads)

    return _CmixLayer.apply(x, x_k, key_weight, value_weight)


def rwkv7_recurrence_cuda_bf16(r, w, k, v, a, b, head_size: int, chunk_len: int):
    class _Rwkv7Recurrence(torch.autograd.Function):
        @staticmethod
        def forward(ctx, r, w, k, v, a, b, head_size: int, chunk_len: int):
            ensure_loaded(head_size=head_size, chunk_len=chunk_len)
            batch, seq_len, channels = r.shape
            pad_len = (-seq_len) % chunk_len
            if pad_len:
                r, w, k, v, a, b = [F.pad(t, (0, 0, 0, pad_len)).contiguous() for t in (r, w, k, v, a, b)]
            else:
                r, w, k, v, a, b = [t.contiguous() for t in (r, w, k, v, a, b)]

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
            ctx.save_for_backward(rv, wv, kv, vv, av, bv, s, sa)
            ctx.seq_len = seq_len
            ctx.padded_len = padded_len
            ctx.channels = channels
            return y.view(batch, padded_len, channels)[:, :seq_len]

        @staticmethod
        def backward(ctx, grad_out):
            rv, wv, kv, vv, av, bv, s, sa = ctx.saved_tensors
            batch = rv.shape[0]
            heads = rv.shape[2]
            head_size = rv.shape[3]
            if grad_out.shape[1] != ctx.padded_len:
                grad_out = F.pad(grad_out, (0, 0, 0, ctx.padded_len - grad_out.shape[1]))
            dy = grad_out.contiguous().view(batch, ctx.padded_len, heads, head_size).contiguous()
            dr = torch.empty_like(rv)
            dw = torch.empty_like(wv)
            dk = torch.empty_like(kv)
            dv = torch.empty_like(vv)
            da = torch.empty_like(av)
            db = torch.empty_like(bv)
            torch.ops.rwkv7_clampw.backward(rv, wv, kv, vv, av, bv, dy, s, sa, dr, dw, dk, dv, da, db)

            def flatten_unpad(t):
                return t.view(batch, ctx.padded_len, ctx.channels)[:, : ctx.seq_len].contiguous()

            return flatten_unpad(dr), flatten_unpad(dw), flatten_unpad(dk), flatten_unpad(dv), flatten_unpad(da), flatten_unpad(db), None, None

    return _Rwkv7Recurrence.apply(r, w, k, v, a, b, head_size, chunk_len)
