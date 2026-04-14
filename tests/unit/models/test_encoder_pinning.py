"""Tests for the encoder's pinned_positions + level plumbing."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from torch import nn

from bgkit.models.bgkit_compressor import BgKITCompressor
from bgkit.models.encoder import BgKITEncoder
from bgkit.models.projection_block import ProjectionBlock


class _PassThroughBackbone(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.embed_tokens = nn.Embedding(32, hidden_dim)
        self.layers = nn.ModuleList([nn.Identity()])
        self.norm = nn.Identity()

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        inputs_embeds=None,
        attention_mask=None,
        return_intermediates=False,
        layer_hooks=None,
        **kwargs,
    ):
        hidden = inputs_embeds
        if layer_hooks:
            for idx in sorted(layer_hooks):
                hidden = layer_hooks[idx](hidden)

        class Out:
            last_hidden_state = hidden
            hidden_states = None
        return Out()


class _TrivialProjection(ProjectionBlock):
    def __init__(self, hidden_dim: int):
        nn.Module.__init__(self)
        self.hidden_dim = hidden_dim
        self.output_norm = nn.Identity()
        self.projection_head = nn.Identity()
        object.__setattr__(self, "_rotary_emb", nn.Identity())

    def forward(self, hidden_states, attention_mask=None, survivor_mask=None):
        from bgkit.models.components.drop_flag import extract_survivors, pad_survivors
        from bgkit.models.projection_block import ProjectionOutput

        if survivor_mask is None:
            return ProjectionOutput(hidden_states, None, None)
        survivors = extract_survivors(hidden_states, survivor_mask)
        padded, counts = pad_survivors(survivors)
        max_s = padded.size(1)
        mask = torch.arange(max_s).unsqueeze(0) < counts.unsqueeze(1)
        return ProjectionOutput(padded, counts, mask)


def _make_encoder(hidden_dim: int = 16) -> BgKITEncoder:
    backbone = _PassThroughBackbone(hidden_dim)
    compressor = BgKITCompressor(
        backbone, norm=nn.Identity(), hidden_dim=hidden_dim, survivorship_inner_dim=8,
    )
    projection = _TrivialProjection(hidden_dim)
    return BgKITEncoder(compressor, projection)


def test_pinned_positions_force_survival():
    """Pinned positions must survive even when the head says drop them."""
    encoder = _make_encoder()
    b, seq, d = 1, 6, 16
    embeddings = torch.randn(b, seq, d)
    pinned = torch.zeros(b, seq, dtype=torch.bool)
    pinned[0, 0] = True
    pinned[0, 3] = True

    # Force survivorship head to output very negative logits (all doomed)
    # so only pinned positions survive.
    with torch.no_grad():
        head = encoder.compressor.survivorship_head_l0
        # Set the final linear bias to a large negative value
        head.head[2].bias.fill_(-10.0)
        head.head[2].weight.fill_(0.0)
        head.head[0].weight.fill_(0.0)
        head.head[0].bias.fill_(-10.0)

    out = encoder(
        input_embeddings=embeddings,
        attention_mask=torch.ones(b, seq, dtype=torch.bool),
        pinned_positions=pinned,
        target_ratio=0.1,
        level="l0",
    )
    # After pinning override, both positions must have survived.
    assert int(out.survivor_counts[0].item()) == 2
    assert out.survivor_embeddings.size(1) == 2


def test_level_runs_without_adapters_installed():
    """With no LoRARouter bound, level should be ignored silently."""
    encoder = _make_encoder()
    b, seq, d = 2, 4, 16
    embeddings = torch.randn(b, seq, d)
    out = encoder(
        input_embeddings=embeddings,
        attention_mask=torch.ones(b, seq, dtype=torch.bool),
        level="l1",
    )
    assert out.survivor_embeddings.shape == (b, seq, d)
