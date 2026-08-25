from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn

from bgkit.models.projection_block import ProjectionBlock, effective_projection_cu


class _ZeroAttentionLayer(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.input_layernorm = nn.Identity()
        self.post_attention_layernorm = nn.Identity()
        self.self_attn = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(hidden_dim, hidden_dim, bias=False))
        nn.init.zeros_(self.mlp[0].weight)


class _Rotary(nn.Module):
    def forward(self, x, position_ids):
        n = position_ids.shape[-1]
        return (
            torch.ones(1, n, 1, dtype=x.dtype, device=x.device),
            torch.zeros(1, n, 1, dtype=x.dtype, device=x.device),
        )


def _make_block(hidden_dim: int = 4, split: int = 2) -> ProjectionBlock:
    block = ProjectionBlock(
        _ZeroAttentionLayer(hidden_dim),
        nn.Identity(),
        _Rotary(),
        hidden_dim=hidden_dim,
        output_split_factor=split,
    )
    with torch.no_grad():
        block.projection_head.weight.copy_(torch.eye(hidden_dim))
        block.projection_head.bias.zero_()
    return block


def test_split_projection_interleaves_halves(monkeypatch):
    from bgkit.models import projection_block as pb

    monkeypatch.setattr(
        pb,
        "_packed_full_attention",
        lambda *_args, **_kwargs: torch.zeros(2, 4),
    )
    block = _make_block(hidden_dim=4, split=2)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]])
    cu = torch.tensor([0, 2], dtype=torch.int32)
    pos = torch.arange(2, dtype=torch.int64)

    out = block(hidden, cu_seqlens=cu, max_seqlen=2, position_ids=pos)

    assert out.projected_embeddings.tolist() == [
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
    ]
    assert out.survivor_cu_seqlens.tolist() == [0, 4]
    assert out.survivor_counts.tolist() == [4]
    assert effective_projection_cu(out, cu).tolist() == [0, 4]


def test_split_projection_doubles_survivor_boundaries_with_mask(monkeypatch):
    from bgkit.models import projection_block as pb

    monkeypatch.setattr(
        pb,
        "_packed_full_attention",
        lambda *_args, **_kwargs: torch.zeros(3, 4),
    )
    block = _make_block(hidden_dim=4, split=2)
    hidden = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    cu = torch.tensor([0, 2, 3], dtype=torch.int32)
    pos = torch.tensor([0, 1, 0], dtype=torch.int64)
    mask = torch.tensor([True, False, True])

    out = block(hidden, cu_seqlens=cu, max_seqlen=2, position_ids=pos, survivor_mask=mask)

    assert out.projected_embeddings.shape == (4, 2)
    assert out.survivor_cu_seqlens.tolist() == [0, 2, 4]
    assert out.survivor_counts.tolist() == [2, 2]


def test_projection_head_cannot_inflate_decoder_interface_norm():
    block = _make_block(hidden_dim=4, split=1)
    with torch.no_grad():
        block.projection_head.weight.mul_(1000.0)
    hidden = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    projected = block._project(hidden)
    assert torch.allclose(
        projected.norm(dim=-1),
        hidden.norm(dim=-1),
        rtol=1e-5,
    )
