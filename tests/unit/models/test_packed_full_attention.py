"""Unit test: packed ``_packed_full_attention`` matches padded HF attention.

Builds a tiny Qwen3.5 attention module (1 head group, small hidden size),
runs it both through our packed ``_packed_full_attention`` wrapper and
through HF's ``Qwen3_5Attention.forward`` with an explicit padding mask,
and asserts numerical parity to fp32 precision.

Runs on CPU via the CPU-SDPA branch of ``_packed_full_attention``
(``_cpu_sdpa_packed``); GPU production always routes through FA4.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")


def _build_tiny_attention_setup():
    """Return (attn_module, rope_module, cfg) for a miniature Qwen3.5 attention."""
    from transformers.models.qwen3_5.modeling_qwen3_5 import (
        Qwen3_5Attention,
        Qwen3_5TextConfig,
        Qwen3_5TextRotaryEmbedding,
    )

    cfg = Qwen3_5TextConfig(
        hidden_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        rms_norm_eps=1e-6,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
        },
        layer_types=["full_attention"],
        num_hidden_layers=1,
    )
    cfg._attn_implementation = "eager"
    attn = Qwen3_5Attention(cfg, layer_idx=0).eval()
    rope = Qwen3_5TextRotaryEmbedding(cfg)
    return attn, rope, cfg


def test_packed_full_attention_matches_padded_reference():
    """Packed attention on a varlen pack matches padded HF attention per-sample."""
    from bgkit.models.bidirectional_qwen35 import _packed_full_attention

    torch.manual_seed(17)
    attn, rope, _cfg = _build_tiny_attention_setup()

    # Two samples of lengths 8 and 12 — so L_max = 12, mask hides 4 pad positions.
    torch.manual_seed(17)
    hidden_padded = torch.randn(2, 12, 64)
    hidden_padded[0, 8:] = 0.0  # padding zeros
    mask = torch.tensor([[1] * 8 + [0] * 4, [1] * 12], dtype=torch.bool)

    # --- packed path ---
    x0 = hidden_padded[0, :8]
    x1 = hidden_padded[1, :12]
    packed = torch.cat([x0, x1], dim=0).contiguous()
    cu = torch.tensor([0, 8, 20], dtype=torch.int32)
    pos = torch.cat(
        [torch.arange(8, dtype=torch.int64), torch.arange(12, dtype=torch.int64)],
        dim=0,
    )
    cos, sin = rope(packed, pos.unsqueeze(0))
    out_packed = _packed_full_attention(
        attn,
        packed,
        (cos, sin),
        cu_seqlens=cu,
        max_seqlen=12,
        position_ids=pos,
        is_causal=False,
    )
    assert out_packed.shape == (20, 64)

    # --- padded reference via HF ---
    pos_padded = torch.arange(12).unsqueeze(0).expand(2, 12)
    cos_p, sin_p = rope(hidden_padded, pos_padded)
    neg_inf = torch.finfo(hidden_padded.dtype).min
    attn_mask_4d = mask[:, None, None, :].to(hidden_padded.dtype)
    attn_mask_4d = (1.0 - attn_mask_4d) * neg_inf
    out_padded, _ = attn(
        hidden_padded,
        position_embeddings=(cos_p, sin_p),
        attention_mask=attn_mask_4d,
    )
    assert out_padded.shape == (2, 12, 64)

    # --- parity check at valid positions ---
    out_pad_0 = out_padded[0, :8]
    out_pad_1 = out_padded[1, :12]
    out_pac_0 = out_packed[:8]
    out_pac_1 = out_packed[8:20]

    torch.testing.assert_close(out_pac_0, out_pad_0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(out_pac_1, out_pad_1, atol=1e-5, rtol=1e-5)


def test_packed_full_attention_causal_matches_padded_reference():
    """Causal variant: packed causal attention matches padded causal ref."""
    from bgkit.models.bidirectional_qwen35 import _packed_full_attention

    torch.manual_seed(29)
    attn, rope, _cfg = _build_tiny_attention_setup()

    # Single sample to avoid mask-x-causal composition subtleties.
    torch.manual_seed(29)
    hidden = torch.randn(1, 10, 64)
    pos = torch.arange(10, dtype=torch.int64)

    # packed
    packed = hidden.squeeze(0).contiguous()
    cu = torch.tensor([0, 10], dtype=torch.int32)
    cos_pk, sin_pk = rope(packed, pos.unsqueeze(0))
    out_pack = _packed_full_attention(
        attn,
        packed,
        (cos_pk, sin_pk),
        cu_seqlens=cu,
        max_seqlen=10,
        position_ids=pos,
        is_causal=True,
    )

    # padded with causal mask
    pos_pad = torch.arange(10).unsqueeze(0)
    cos_p, sin_p = rope(hidden, pos_pad)
    neg_inf = torch.finfo(hidden.dtype).min
    causal = torch.triu(
        torch.full((10, 10), neg_inf, dtype=hidden.dtype),
        diagonal=1,
    )[None, None, :, :]
    out_pad, _ = attn(hidden, position_embeddings=(cos_p, sin_p), attention_mask=causal)

    torch.testing.assert_close(out_pack, out_pad.squeeze(0), atol=1e-5, rtol=1e-5)
