import torch

from models.layers import Attention
from models.lm_head import LMHead
from models.rwkv_memory import DeltaRuleStateMemory, RWKVStateMemory, set_rwkv_mem_runtime_enabled


def test_delta_rule_qo_outputs_only_active_heads():
    mem = DeltaRuleStateMemory(
        hidden_size=16,
        query_size=16,
        key_size=16,
        value_size=16,
        output_size=16,
        rank=4,
        delta_heads=("q", "o"),
        output_init="base_slice_fixed",
        base_q_weight=torch.randn(16, 16),
        base_o_weight=torch.randn(16, 16),
        backend="torch",
    )

    x = torch.randn(2, 5, 16)
    deltas = mem(x)

    assert set(deltas) == {"q", "o"}
    assert deltas["q"].shape == (2, 5, 16)
    assert deltas["o"].shape == (2, 5, 16)


def test_delta_rule_qo_packed_outputs_scatter_back():
    mem = DeltaRuleStateMemory(
        hidden_size=16,
        query_size=16,
        key_size=16,
        value_size=16,
        output_size=16,
        rank=4,
        delta_heads=("q", "o"),
        output_init="base_slice_fixed",
        base_q_weight=torch.randn(16, 16),
        base_o_weight=torch.randn(16, 16),
        backend="torch",
    )

    x = torch.randn(9, 16)
    deltas = mem(x, cu_seqlens=torch.tensor([0, 4, 9], dtype=torch.int32), numseqs=torch.tensor(2))

    assert set(deltas) == {"q", "o"}
    assert deltas["q"].shape == (9, 16)
    assert deltas["o"].shape == (9, 16)


def test_delta_rule_full_recipe_outputs_all_delta_heads():
    mem = DeltaRuleStateMemory(
        hidden_size=16,
        query_size=16,
        key_size=8,
        value_size=8,
        output_size=16,
        rank=4,
        delta_heads=("q", "k", "v", "o"),
        output_init="base_slice_fixed",
        base_q_weight=torch.randn(16, 16),
        base_k_weight=torch.randn(8, 16),
        base_v_weight=torch.randn(8, 16),
        base_o_weight=torch.randn(16, 8),
        backend="torch",
    )

    x = torch.randn(2, 5, 16)
    deltas = mem(x)

    assert set(deltas) == {"q", "k", "v", "o"}
    assert deltas["q"].shape == (2, 5, 16)
    assert deltas["k"].shape == (2, 5, 8)
    assert deltas["v"].shape == (2, 5, 8)
    assert deltas["o"].shape == (2, 5, 16)
    assert deltas["o"].abs().sum() > 0


def test_rwkv_state_memory_read_before_write_starts_from_empty_state():
    torch.manual_seed(0)
    mem = RWKVStateMemory(
        max_seq_len=8,
        hidden_size=8,
        head_size=4,
        backend="torch",
        output_init="small",
        read_before_write=True,
    )
    with torch.no_grad():
        mem.time_mix.ln_x.bias.fill_(3.0)

    y = mem(torch.randn(2, 4, 8))

    assert torch.allclose(y[:, 0], torch.zeros_like(y[:, 0]), atol=1e-6)
    assert y[:, 1:].abs().sum() > 0


def test_rwkv_state_memory_cuda_read_before_write_starts_from_empty_state():
    if not torch.cuda.is_available():
        return

    torch.manual_seed(0)
    mem = RWKVStateMemory(
        max_seq_len=8,
        hidden_size=64,
        head_size=64,
        backend="cuda",
        output_init="small",
        read_before_write=True,
    ).cuda().bfloat16()
    with torch.no_grad():
        mem.time_mix.ln_x.bias.fill_(3.0)
    x = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    y = mem(x)
    y.float().square().mean().backward()

    assert torch.allclose(y[:, 0], torch.zeros_like(y[:, 0]), atol=0, rtol=0)
    assert y[:, 1:].abs().sum() > 0
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert x.grad.abs().sum() > 0


def test_rwkv_state_memory_legacy_write_before_read_keeps_current_token_path():
    torch.manual_seed(0)
    mem = RWKVStateMemory(
        max_seq_len=8,
        hidden_size=8,
        head_size=4,
        backend="torch",
        output_init="small",
        read_before_write=False,
    )

    y = mem(torch.randn(2, 4, 8))

    assert y[:, 0].abs().sum() > 0


def test_attention_runtime_disable_bypasses_memory():
    attn = Attention(
        hidden_size=16,
        head_dim=4,
        num_heads=4,
        num_key_value_heads=4,
        attn_type="causal",
        max_seq_len=8,
        rwkv_mem_enabled=True,
        rwkv_mem_mode="delta_rule",
        rwkv_mem_delta_heads=("q", "o"),
        rwkv_mem_output_init="base_slice_fixed",
        rwkv_mem_backend="torch",
    )

    x = torch.randn(2, 4, 16)
    y_mem = attn(x, cos_sin=None)
    set_rwkv_mem_runtime_enabled(attn, False)
    y_base = attn(x, cos_sin=None)

    assert y_mem.shape == y_base.shape == (2, 4, 16)
    assert not torch.allclose(y_mem, y_base)


def test_lm_head_rejects_unknown_rwkv_mem_loss_mode():
    class ToyModel(torch.nn.Module):
        head_hint = {
            "in": {"dim": 8, "init_std": 0.02},
            "out": {"dim": 8, "init_std": 0.02},
        }

        def create_cache(self, **_kwargs):
            return None

        def compute_train_extra_args(self, _train_state):
            return {}

        def forward(self, carry, x, **_kwargs):
            return carry, x

    head = LMHead(ToyModel(), {"vocab_size": 32})
    batch = {
        "inputs": torch.tensor([1, 2, 3], dtype=torch.long),
        "labels": torch.tensor([2, 3, 4], dtype=torch.long),
    }

    try:
        head(carry=None, batch=batch, rwkv_mem_loss_mode="not_a_mode")
    except ValueError as exc:
        assert "Unsupported rwkv_mem_loss_mode" in str(exc)
    else:
        raise AssertionError("expected unsupported rwkv_mem_loss_mode to raise")
