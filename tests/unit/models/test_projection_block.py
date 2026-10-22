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


# ---------------------------------------------------------------------------
# Decoder-interface contract
# ---------------------------------------------------------------------------


def _make_interface_block(hidden_dim: int = 8, split: int = 1, **kw) -> ProjectionBlock:
    block = ProjectionBlock(
        _ZeroAttentionLayer(hidden_dim),
        nn.Identity(),
        _Rotary(),
        hidden_dim=hidden_dim,
        output_split_factor=split,
        interface_norm=True,
        **kw,
    )
    with torch.no_grad():
        block.projection_head.weight.copy_(torch.eye(hidden_dim))
        block.projection_head.bias.zero_()
    return block


def _run(block, hidden, monkeypatch):
    from bgkit.models import projection_block as pb

    monkeypatch.setattr(
        pb, "_packed_full_attention", lambda *_a, **_k: torch.zeros_like(hidden),
    )
    n = hidden.shape[0]
    return block(
        hidden,
        cu_seqlens=torch.tensor([0, n], dtype=torch.int32),
        max_seqlen=n,
        position_ids=torch.arange(n, dtype=torch.int64),
    )


def test_interface_norm_removes_a_corpus_constant_from_the_output(monkeypatch):
    """The measured v8 pathology: 99% of every survivor's energy is one shared
    vector. What leaves the projection must be the part that differs."""
    torch.manual_seed(0)
    # A DENSE shared vector, not a single spiked channel: measured on the real
    # checkpoints the top 8 channels hold only ~6% of the corpus mean's energy,
    # so this is not the "massive activation" shape and a fix that only
    # rebalanced a few big channels would not have touched it. Subtracting a
    # mean removes a spread constant just as well as a spiked one.
    hidden = torch.randn(64, 8) + torch.randn(8) * 300.0
    block = _make_interface_block(8, interface_target_row_norm=0.64)
    block.train()
    out = _run(block, hidden, monkeypatch).projected_embeddings.detach()
    energy = out.pow(2).sum(dim=-1).mean()
    assert float(out.mean(dim=0).pow(2).sum() / energy) < 0.05
    assert float(out.norm(dim=-1).mean()) == pytest.approx(0.64, rel=0.3)


def test_interface_norm_replaces_the_output_rescale(monkeypatch):
    """Both on would fight: the rescale targets the ENCODER-space norm of the
    projection's own input, which the decoder has no stake in."""
    block = _make_interface_block(8)
    assert block.enforce_output_norm is True  # default, deliberately untouched
    torch.manual_seed(1)
    hidden = torch.randn(32, 8) * 50.0
    block.train()
    out = _run(block, hidden, monkeypatch).projected_embeddings.detach()
    # The rescale would have produced rows at the input's norm (~50 * sqrt(8));
    # the contract produces rows at target_row_norm.
    assert float(out.norm(dim=-1).mean()) < 5.0


def test_interface_norm_applies_after_the_split(monkeypatch):
    """A split slices a hidden-width vector into several decoder-width rows;
    standardising before the slice would normalise a width the decoder never
    sees, leaving each emitted row un-normalised."""
    torch.manual_seed(2)
    hidden = torch.randn(64, 8) + 200.0
    block = _make_interface_block(8, split=2, interface_target_row_norm=0.5)
    block.train()
    out = _run(block, hidden, monkeypatch).projected_embeddings.detach()
    assert out.shape == (128, 4)
    assert float(out.norm(dim=-1).mean()) == pytest.approx(0.5, rel=0.3)


def test_interface_norm_is_off_by_default(monkeypatch):
    block = _make_block(hidden_dim=4, split=2)
    assert block.interface_norm is None


def test_interface_norm_statistics_ride_in_the_state_dict(monkeypatch):
    """The EMA reference IS the contract. If it did not survive a save/load,
    every resume would hand the decoder a payload normalised against numbers
    from a different distribution until the reference caught up again."""
    torch.manual_seed(3)
    hidden = torch.randn(64, 8) + torch.randn(8) * 300.0
    src = _make_interface_block(8, interface_target_row_norm=0.64)
    src.train()
    _run(src, hidden, monkeypatch)

    dst = _make_interface_block(8, interface_target_row_norm=0.64)
    dst.load_state_dict(src.state_dict())
    torch.testing.assert_close(
        dst.interface_norm.running_mean, src.interface_norm.running_mean,
    )
    assert int(dst.interface_norm.num_updates.item()) == 1
    dst.eval()
    src.eval()
    torch.testing.assert_close(
        _run(dst, hidden, monkeypatch).projected_embeddings,
        _run(src, hidden, monkeypatch).projected_embeddings,
    )


def test_a_checkpoint_predating_the_contract_loads_and_calibrates(monkeypatch):
    """Turning the contract on for a run resuming an older checkpoint must not
    require a migration -- the buffers are simply absent and get set by the
    first forward."""
    plain = _make_block(hidden_dim=8, split=1)
    with_norm = _make_interface_block(8, interface_target_row_norm=0.64)
    missing, unexpected = with_norm.load_state_dict(plain.state_dict(), strict=False)
    assert not unexpected
    assert all("interface_norm" in k for k in missing), missing
    assert int(with_norm.interface_norm.num_updates.item()) == 0

    torch.manual_seed(4)
    hidden = torch.randn(64, 8) + torch.randn(8) * 300.0
    with_norm.eval()
    out = _run(with_norm, hidden, monkeypatch).projected_embeddings.detach()
    assert int(with_norm.interface_norm.num_updates.item()) == 1
    energy = out.pow(2).sum(dim=-1).mean()
    assert float(out.mean(dim=0).pow(2).sum() / energy) < 0.05
