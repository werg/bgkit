"""Tests for the encoder's pinned_positions + lora_level plumbing."""

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

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        class Out:
            last_hidden_state = inputs_embeds
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
    compressor = BgKITCompressor(backbone, norm=nn.Identity(), hidden_dim=hidden_dim)
    projection = _TrivialProjection(hidden_dim)
    return BgKITEncoder(compressor, projection)


def test_pinned_positions_force_survival():
    encoder = _make_encoder()
    b, seq, d = 1, 6, 16
    embeddings = torch.randn(b, seq, d)
    # Nobody selected by the ICE-style mask — but positions 0 and 3 are pinned.
    survivor_mask = torch.zeros(b, seq, dtype=torch.bool)
    pinned = torch.zeros(b, seq, dtype=torch.bool)
    pinned[0, 0] = True
    pinned[0, 3] = True
    out = encoder(
        input_embeddings=embeddings,
        survivor_mask=survivor_mask,
        attention_mask=torch.ones(b, seq, dtype=torch.bool),
        pinned_positions=pinned,
    )
    # After pinning merge, both positions must have survived.
    assert int(out.survivor_counts[0].item()) == 2
    assert out.survivor_embeddings.size(1) == 2


def test_lora_level_runs_without_adapters_installed():
    encoder = _make_encoder()
    b, seq, d = 2, 4, 16
    embeddings = torch.randn(b, seq, d)
    # With no LoRARouter bound, lora_level should be ignored silently.
    out = encoder(
        input_embeddings=embeddings,
        attention_mask=torch.ones(b, seq, dtype=torch.bool),
        lora_level="l1",
    )
    assert out.survivor_embeddings.shape == (b, seq, d)
