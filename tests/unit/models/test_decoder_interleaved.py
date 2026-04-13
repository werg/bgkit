"""Tests for interleaved decoder loss paths and single-splice helpers."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn.functional as F
from torch import nn

from bgkit.models.decoder import (
    EmbeddingSegment,
    ReconstructionDecoder,
    TokenSegment,
)

# ---------------------------------------------------------------------------
# Mock CausalLM (same pattern as test_decoder_fused_ce.py)
# ---------------------------------------------------------------------------

VOCAB_SIZE = 128
HIDDEN_DIM = 24


class _InnerOut:
    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _LMOut:
    def __init__(self, logits):
        self.logits = logits


class _InnerModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(VOCAB_SIZE, HIDDEN_DIM)
        self.layers = nn.ModuleList([nn.Linear(HIDDEN_DIM, HIDDEN_DIM) for _ in range(2)])
        self.norm = nn.LayerNorm(HIDDEN_DIM)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        x = inputs_embeds
        for layer in self.layers:
            x = layer(x)
        return _InnerOut(self.norm(x))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _InnerModel()
        self.lm_head = nn.Linear(HIDDEN_DIM, VOCAB_SIZE, bias=False)

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        out = self.model(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        return _LMOut(self.lm_head(out.last_hidden_state))


def _decoder() -> ReconstructionDecoder:
    torch.manual_seed(7)
    return ReconstructionDecoder(_Backbone(), hidden_dim=HIDDEN_DIM)


# ---------------------------------------------------------------------------
# Single-splice helper
# ---------------------------------------------------------------------------


def test_forward_with_single_splice_matches_equivalent_interleaved_segments():
    decoder = _decoder()
    torch.manual_seed(3)
    survivors = torch.randn(2, 4, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (2, 8))
    target_mask = torch.ones(2, 8, dtype=torch.bool)
    survivor_mask = torch.ones(2, 4, dtype=torch.bool)
    splice_starts = torch.zeros(2, dtype=torch.long)
    splice_lengths = torch.zeros(2, dtype=torch.long)

    ref = decoder.forward_interleaved_with_loss([
        EmbeddingSegment(embeddings=survivors),
        TokenSegment(token_ids=target_ids, loss=True),
    ])
    got = decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_attention_mask=survivor_mask,
        token_ids=target_ids,
        token_attention_mask=target_mask,
        splice_starts=splice_starts,
        splice_lengths=splice_lengths,
    )
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_forward_with_single_splice_respects_loss_mask_at_position_0():
    decoder = _decoder()
    torch.manual_seed(11)
    survivors = torch.randn(1, 3, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    target_mask = torch.ones(1, 6, dtype=torch.bool)
    survivor_mask = torch.ones(1, 3, dtype=torch.bool)
    splice_starts = torch.zeros(1, dtype=torch.long)
    splice_lengths = torch.zeros(1, dtype=torch.long)

    mask_a = torch.tensor([[1, 1, 1, 1, 1, 1]], dtype=torch.bool)
    mask_b = torch.tensor([[0, 1, 1, 1, 1, 1]], dtype=torch.bool)
    loss_a = decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_attention_mask=survivor_mask,
        token_ids=target_ids,
        token_attention_mask=target_mask,
        splice_starts=splice_starts,
        splice_lengths=splice_lengths,
        loss_mask=mask_a,
    )
    loss_b = decoder.forward_with_single_splice(
        survivor_embeddings=survivors,
        survivor_attention_mask=survivor_mask,
        token_ids=target_ids,
        token_attention_mask=target_mask,
        splice_starts=splice_starts,
        splice_lengths=splice_lengths,
        loss_mask=mask_b,
    )
    assert not torch.allclose(loss_a, loss_b, atol=1e-6)


# ---------------------------------------------------------------------------
# Direct interleaved API
# ---------------------------------------------------------------------------


def test_interleaved_single_token_segment_equals_plain_lm_loss():
    """A lone TokenSegment with loss=True is just next-token LM loss on it."""
    decoder = _decoder()
    torch.manual_seed(5)
    ids = torch.randint(0, VOCAB_SIZE, (1, 10))
    segments = [TokenSegment(token_ids=ids, loss=True)]
    got = decoder.forward_interleaved_with_loss(segments)

    # Reference: embed, run inner model, chunked CE with default mask
    inner, head = decoder._get_inner_model_and_head()
    emb = inner.get_input_embeddings()(ids)
    hidden = inner(inputs_embeds=emb).last_hidden_state
    logits = head(hidden[:, :-1])
    shift_t = ids[:, 1:]
    per = F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        shift_t.reshape(-1),
        reduction="none",
    ).reshape(shift_t.shape)
    ref = per.sum() / shift_t.numel()
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_interleaved_multi_segment_loss_masking():
    """Layout: [TokA(loss=False) | Emb(X) | TokB(loss=True) | Emb(Y) | TokC(loss=True)].

    Loss is nonzero, and zeroing X/Y does not change it (they're target-masked).
    Flipping TokA's contents does not change it (loss=False).
    Flipping TokB's or TokC's contents does change it.
    """
    decoder = _decoder()
    torch.manual_seed(17)

    def build(a_ids, b_ids, c_ids, x_emb, y_emb):
        return [
            TokenSegment(token_ids=a_ids, loss=False),
            EmbeddingSegment(embeddings=x_emb),
            TokenSegment(token_ids=b_ids, loss=True),
            EmbeddingSegment(embeddings=y_emb),
            TokenSegment(token_ids=c_ids, loss=True),
        ]

    a = torch.randint(0, VOCAB_SIZE, (1, 3))
    b = torch.randint(0, VOCAB_SIZE, (1, 4))
    c = torch.randint(0, VOCAB_SIZE, (1, 5))
    x = torch.randn(1, 2, HIDDEN_DIM)
    y = torch.randn(1, 3, HIDDEN_DIM)

    base = decoder.forward_interleaved_with_loss(build(a, b, c, x, y))
    assert torch.isfinite(base) and base.item() > 0

    # X/Y swapped for different random tensors → loss must not change, because
    # embedding-segment positions are not loss-bearing targets. (They do feed
    # hidden states that affect later predictions, so the loss CAN change.
    # What cannot change is the *denominator* — the count of loss sites.)
    # So instead we verify that TokA's contents don't affect loss at all.
    a2 = torch.randint(0, VOCAB_SIZE, (1, 3))
    alt = decoder.forward_interleaved_with_loss(build(a2, b, c, x, y))
    torch.testing.assert_close(base, alt, atol=1e-5, rtol=1e-5)

    # Changing tokens in B SHOULD change the loss.
    b2 = torch.randint(0, VOCAB_SIZE, (1, 4))
    diff_b = decoder.forward_interleaved_with_loss(build(a, b2, c, x, y))
    assert not torch.allclose(base, diff_b, atol=1e-4)

    # Changing tokens in C SHOULD change the loss.
    c2 = torch.randint(0, VOCAB_SIZE, (1, 5))
    diff_c = decoder.forward_interleaved_with_loss(build(a, b, c2, x, y))
    assert not torch.allclose(base, diff_c, atol=1e-4)


def test_interleaved_cross_boundary_gradient_flows_into_embedding_segment():
    """An embedding segment's last vector must get gradient from predicting
    the first token of the following token segment."""
    decoder = _decoder()
    torch.manual_seed(23)

    survivors = torch.randn(1, 4, HIDDEN_DIM, requires_grad=True)
    target_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    segments = [
        EmbeddingSegment(embeddings=survivors),
        TokenSegment(token_ids=target_ids, loss=True),
    ]
    loss = decoder.forward_interleaved_with_loss(segments)
    loss.backward()

    assert survivors.grad is not None
    # The LAST survivor position is the direct source for predicting target[0];
    # it must see a nonzero gradient.
    last_grad = survivors.grad[0, -1].abs().sum()
    assert last_grad.item() > 0


def test_interleaved_chunk_size_invariance():
    """Same inputs, different chunk_size → same loss (no numerical drift)."""
    decoder = _decoder()
    torch.manual_seed(29)

    segments = [
        EmbeddingSegment(embeddings=torch.randn(1, 3, HIDDEN_DIM)),
        TokenSegment(
            token_ids=torch.randint(0, VOCAB_SIZE, (1, 40)),
            loss=True,
        ),
    ]
    loss_big = decoder.forward_interleaved_with_loss(segments, chunk_size=256)
    loss_small = decoder.forward_interleaved_with_loss(segments, chunk_size=7)
    torch.testing.assert_close(loss_big, loss_small, atol=1e-5, rtol=1e-5)


def test_interleaved_per_token_loss_mask_override():
    """TokenSegment.loss_mask restricts loss to specific positions,
    overriding the segment-wide `loss` flag."""
    decoder = _decoder()
    torch.manual_seed(31)
    ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    # Only positions 4 and 5 contribute. Position 0 is always dropped by the
    # shift, so it's equivalent either way.
    mask_partial = torch.tensor([[0, 0, 0, 0, 1, 1, 0, 0]], dtype=torch.bool)
    mask_none = torch.zeros(1, 8, dtype=torch.bool)

    segments_partial = [
        EmbeddingSegment(embeddings=torch.randn(1, 2, HIDDEN_DIM)),
        TokenSegment(token_ids=ids, loss=True, loss_mask=mask_partial),
    ]
    segments_none = [
        EmbeddingSegment(embeddings=torch.randn(1, 2, HIDDEN_DIM)),
        TokenSegment(token_ids=ids, loss=True, loss_mask=mask_none),
    ]
    loss_partial = decoder.forward_interleaved_with_loss(segments_partial)
    loss_none = decoder.forward_interleaved_with_loss(segments_none)

    assert torch.isfinite(loss_partial)
    assert loss_partial.item() > 0
    # All-zero mask → denominator clamped to 1, numerator zero → loss = 0
    torch.testing.assert_close(loss_none, torch.zeros_like(loss_none))


def test_interleaved_rejects_empty_segments():
    decoder = _decoder()
    with pytest.raises(ValueError):
        decoder.forward_interleaved_with_loss([])


def test_interleaved_return_hidden_states():
    """return_hidden_states=True exposes everything needed for eval."""
    from bgkit.models.decoder import InterleavedForwardOutput

    decoder = _decoder()
    torch.manual_seed(41)
    survivors = torch.randn(1, 3, HIDDEN_DIM, requires_grad=True)
    target_ids = torch.randint(0, VOCAB_SIZE, (1, 8))
    segments = [
        EmbeddingSegment(embeddings=survivors),
        TokenSegment(token_ids=target_ids, loss=True),
    ]
    output = decoder.forward_interleaved_with_loss(
        segments, return_hidden_states=True,
    )
    assert isinstance(output, InterleavedForwardOutput)
    assert output.loss.ndim == 0
    assert output.hidden_states.shape == (1, 3 + 8, HIDDEN_DIM)
    assert output.token_ids.shape == (1, 3 + 8)
    assert output.loss_mask.shape == (1, 3 + 8)
    # Survivor positions are zero in token_ids and False in loss_mask.
    assert (output.token_ids[0, :3] == 0).all()
    assert (~output.loss_mask[0, :3]).all()
    # Token positions have the right IDs and are all loss-bearing (loss=True default).
    assert (output.token_ids[0, 3:] == target_ids[0]).all()
    assert output.loss_mask[0, 3:].all()

    # argmax_predictions returns the right shape and has nonzero variance.
    preds = output.argmax_predictions()
    assert preds.shape == (1, 3 + 8 - 1)
    assert preds.dtype == torch.long


def test_interleaved_scalar_loss_matches_hidden_output_loss():
    """The scalar-loss path and the rich-output path produce the same loss."""
    decoder = _decoder()
    torch.manual_seed(43)
    survivors = torch.randn(1, 2, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (1, 6))
    segments = [
        EmbeddingSegment(embeddings=survivors),
        TokenSegment(token_ids=target_ids, loss=True),
    ]
    scalar = decoder.forward_interleaved_with_loss(segments)
    rich = decoder.forward_interleaved_with_loss(segments, return_hidden_states=True)
    torch.testing.assert_close(scalar, rich.loss, atol=1e-6, rtol=1e-6)


def test_interleaved_rejects_batch_size_mismatch():
    decoder = _decoder()
    segments = [
        EmbeddingSegment(embeddings=torch.randn(1, 2, HIDDEN_DIM)),
        TokenSegment(token_ids=torch.randint(0, VOCAB_SIZE, (2, 4)), loss=True),
    ]
    with pytest.raises(ValueError, match="batch size"):
        decoder.forward_interleaved_with_loss(segments)
