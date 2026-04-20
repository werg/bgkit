from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn

from bgkit.models.bidirectional_qwen35 import BidirectionalQwen35
from bgkit.models.projection_block import ProjectionBlock
from bgkit.models.pruned_qwen35 import _run_attn_layer
from bgkit.utils import attention_backend as ab


class _DummyRotary(nn.Module):
    def forward(self, hidden_states, position_ids):
        return (hidden_states, position_ids)


class _DummyAttentionLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.last_kwargs = None

    def forward(self, hidden_states, position_embeddings, attention_mask=None, **kwargs):
        self.last_kwargs = {"attention_mask": attention_mask, **kwargs}
        return hidden_states


def test_resolve_attention_implementation_auto_prefers_bgkit_fa4():
    with patch.object(ab, "install_bgkit_attention_backend", return_value=True):
        assert ab.resolve_attention_implementation("auto") == ab.BGKIT_FA4_ATTENTION_IMPL


def test_resolve_attention_implementation_explicit_fa4_requires_install():
    with patch.object(ab, "install_bgkit_attention_backend", return_value=False):
        try:
            ab.resolve_attention_implementation("flash_attention_4")
        except RuntimeError as exc:
            assert "flash_attn.cute" in str(exc)
        else:  # pragma: no cover
            raise AssertionError("expected explicit FA4 request to fail without FA4")


def test_bgkit_fa4_forward_uses_padding_mask_path():
    class _DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation=ab.BGKIT_FA4_ATTENTION_IMPL)
            self.is_causal = False

    module = _DummyModule()
    query = torch.randn(2, 3, 4, 8)
    key = torch.randn(2, 3, 4, 8)
    value = torch.randn(2, 3, 4, 8)
    pad = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=torch.float32)
    mask = (1.0 - pad[:, None, None, :]) * torch.finfo(torch.float32).min

    with patch(
        "transformers.integrations.flash_attention.get_target_dtype",
        return_value=None,
    ), patch(
        "transformers.modeling_flash_attention_utils._flash_attention_forward",
        return_value=torch.zeros(2, 4, 3, 8),
    ) as flash_mock:
        out, attn = ab.bgkit_flash_attention_4_forward(
            module,
            query,
            key,
            value,
            mask,
            is_causal=False,
        )

    assert attn is None
    assert out.shape == (2, 4, 3, 8)
    args = flash_mock.call_args.args
    kwargs = flash_mock.call_args.kwargs
    assert args[3].dtype == torch.bool
    assert args[3].shape == (2, 4)
    assert kwargs["attn_implementation"] == "flash_attention_4"
    assert kwargs["is_causal"] is False


def test_bgkit_fa4_forward_falls_back_for_query_specific_masks():
    class _DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation=ab.BGKIT_FA4_ATTENTION_IMPL)
            self.is_causal = False

    module = _DummyModule()
    query = torch.randn(1, 2, 3, 4)
    key = torch.randn(1, 2, 3, 4)
    value = torch.randn(1, 2, 3, 4)
    mask = torch.zeros(1, 1, 3, 3)

    with patch.object(
        ab,
        "_qwen_eager_attention_forward",
        return_value=("fallback", "weights"),
    ) as eager_mock:
        out = ab.bgkit_flash_attention_4_forward(module, query, key, value, mask, is_causal=False)

    eager_mock.assert_called_once()
    assert out == ("fallback", "weights")


def test_bgkit_fa4_forward_keeps_true_gqa_when_sm12x_native_ready():
    class _DummyModule(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(_attn_implementation=ab.BGKIT_FA4_ATTENTION_IMPL)
            self.is_causal = False

    module = _DummyModule()
    query = torch.randn(2, 4, 5, 8)
    key = torch.randn(2, 2, 5, 8)
    value = torch.randn(2, 2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    with patch.object(ab, "_sm12x_native_true_gqa_ready", return_value=True), patch(
        "transformers.integrations.flash_attention.get_target_dtype",
        return_value=None,
    ), patch(
        "transformers.modeling_flash_attention_utils._flash_attention_forward",
        return_value=torch.zeros(2, 5, 4, 8),
    ) as flash_mock:
        out, attn = ab.bgkit_flash_attention_4_forward(
            module,
            query,
            key,
            value,
            mask,
            is_causal=False,
        )

    assert attn is None
    assert out.shape == (2, 5, 4, 8)
    args = flash_mock.call_args.args
    assert args[1].shape[-2] == 2
    assert args[2].shape[-2] == 2
    assert args[3].dtype == torch.bool
    assert args[3].shape == (2, 5)


def test_bidirectional_qwen35_forces_noncausal_attention():
    layer = _DummyAttentionLayer()

    class _FakeBaseModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(8, 4)
            self.norm = nn.Identity()
            self.rotary_emb = _DummyRotary()
            self.layers = nn.ModuleList([layer])
            self.config = SimpleNamespace(model_type="qwen3_5")

    model = BidirectionalQwen35(_FakeBaseModel(), bidi_warmup_steps=0)
    hidden = torch.randn(1, 3, 4)
    model(hidden, attention_mask=None)

    assert layer.last_kwargs["is_causal"] is False


def test_pruned_attention_runner_forces_noncausal_attention():
    layer = _DummyAttentionLayer()
    hidden = torch.randn(1, 3, 4)
    pos = (_DummyRotary()(hidden, torch.arange(3).unsqueeze(0)))

    _run_attn_layer(layer, hidden, None, pos, use_ckpt=False)

    assert layer.last_kwargs["is_causal"] is False


def test_projection_block_forces_noncausal_attention():
    layer = _DummyAttentionLayer()
    block = ProjectionBlock(
        transformer_layer=layer,
        output_norm=nn.Identity(),
        rotary_emb=_DummyRotary(),
        hidden_dim=4,
    )
    hidden = torch.randn(1, 3, 4)

    block(hidden)

    assert layer.last_kwargs["is_causal"] is False
