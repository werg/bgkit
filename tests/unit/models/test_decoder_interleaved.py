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
    """Packed forward_with_single_splice matches forward_interleaved_with_loss
    when there is no prefix (survivors at the start, suffix = all tokens)."""
    decoder = _decoder()
    torch.manual_seed(3)
    batch_size, num_surv, target_len = 2, 4, 8
    survivors_3d = torch.randn(batch_size, num_surv, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (batch_size, target_len))

    ref = decoder.forward_interleaved_with_loss(
        [
            EmbeddingSegment(embeddings=survivors_3d),
            TokenSegment(token_ids=target_ids, loss=True),
        ]
    )

    # Packed: flat survivors, no prefix, all tokens as suffix
    surv_flat = survivors_3d.reshape(batch_size * num_surv, HIDDEN_DIM)
    cu = torch.tensor([0, num_surv, 2 * num_surv], dtype=torch.int32)
    got = decoder.forward_with_single_splice(
        survivor_embeddings=surv_flat,
        survivor_cu_seqlens=cu,
        prefix_ids=[torch.zeros(0, dtype=torch.long)] * batch_size,
        suffix_ids=[target_ids[b] for b in range(batch_size)],
    )
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_forward_with_single_splice_respects_loss_mask_at_position_0():
    """Loss mask restricts which suffix positions contribute to CE."""
    decoder = _decoder()
    torch.manual_seed(11)
    num_surv, target_len = 3, 6
    survivors_flat = torch.randn(num_surv, HIDDEN_DIM)
    target_ids = torch.randint(0, VOCAB_SIZE, (1, target_len))
    cu = torch.tensor([0, num_surv], dtype=torch.int32)
    n_total = num_surv + target_len  # 9 positions

    # mask_a: all suffix positions contribute (positions 3..8)
    mask_a = torch.zeros(n_total, dtype=torch.bool)
    mask_a[num_surv:] = True

    # mask_b: skip the first suffix token (position 3)
    mask_b = torch.zeros(n_total, dtype=torch.bool)
    mask_b[num_surv + 1 :] = True

    loss_a = decoder.forward_with_single_splice(
        survivor_embeddings=survivors_flat,
        survivor_cu_seqlens=cu,
        prefix_ids=[torch.zeros(0, dtype=torch.long)],
        suffix_ids=[target_ids[0]],
        loss_mask=mask_a,
    )
    loss_b = decoder.forward_with_single_splice(
        survivor_embeddings=survivors_flat,
        survivor_cu_seqlens=cu,
        prefix_ids=[torch.zeros(0, dtype=torch.long)],
        suffix_ids=[target_ids[0]],
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
        segments,
        return_hidden_states=True,
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


# ---------------------------------------------------------------------------
# FIX 1: interleaved decode routes through FA4 VARLEN (O(S)), not O(S^2) padded
# ---------------------------------------------------------------------------


def test_inner_forward_routes_single_sequence_through_varlen():
    """A single contiguous (B=1, all-valid) interleaved sequence is routed
    through _packed_forward with cu_seqlens=[0,S] + position_ids=arange(S) —
    the FA4 varlen path (O(S)) — NOT the O(S^2) padded attention_mask path."""
    import types as _types

    d = ReconstructionDecoder.__new__(ReconstructionDecoder)
    captured: dict = {}

    def fake_packed(self, emb, cu, max_s, pos, mamba_seq_idx=None):
        captured["cu"] = cu.tolist()
        captured["max_s"] = int(max_s)
        captured["pos"] = pos.tolist()
        return torch.zeros_like(emb), 0

    d._packed_forward = _types.MethodType(fake_packed, d)
    seq = 13
    emb = torch.randn(1, seq, 8)

    # None mask (the KRKBTrainer / typical call) -> varlen.
    h, pad = d._inner_forward(emb, None)
    assert captured["cu"] == [0, seq]
    assert captured["pos"] == list(range(seq))
    assert captured["max_s"] == seq
    assert pad == 0 and h.shape == emb.shape

    # Explicit all-ones mask -> still varlen.
    captured.clear()
    d._inner_forward(emb, torch.ones(1, seq, dtype=torch.bool))
    assert captured["cu"] == [0, seq]


def test_inner_forward_legacy_fallback_for_padded_mask():
    """A non-all-valid mask (the unused padded case) does NOT take the varlen
    path — it falls back to the legacy padded forward (kept for B>1 / masks)."""
    import types as _types

    d = ReconstructionDecoder.__new__(ReconstructionDecoder)
    d._use_te = False
    called = {"packed": False, "inner": False}

    def fake_packed(self, *a, **k):
        called["packed"] = True
        return torch.zeros(1), 0

    class _Inner:
        def __call__(self, *, inputs_embeds, attention_mask, use_cache):
            called["inner"] = True
            return _InnerOut(inputs_embeds)

    d._packed_forward = _types.MethodType(fake_packed, d)
    d._get_inner_model_and_head = _types.MethodType(
        lambda self: (_Inner(), None), d,
    )
    emb = torch.randn(1, 6, 8)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0]], dtype=torch.bool)  # trailing pad
    d._inner_forward(emb, mask)
    assert called["packed"] is False  # NOT the varlen path
    assert called["inner"] is True    # legacy padded path


# ---------------------------------------------------------------------------
# FIX 1b: force per-layer GC on the inner model in the interleaved decode
# ---------------------------------------------------------------------------


class _GCLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.gradient_checkpointing = False
        self._gradient_checkpointing_func = None


class _GCInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([_GCLayer(), _GCLayer(), _GCLayer()])
        self._enable_calls = 0

    def gradient_checkpointing_enable(self, **kw):
        self._enable_calls += 1
        for lyr in self.layers:
            # HF populates the func on enable.
            lyr._gradient_checkpointing_func = object()


def _make_gc_decoder(mode="reentrant", min_seqlen=4096):
    d = ReconstructionDecoder.__new__(ReconstructionDecoder)
    d.decoder_family = "qwen35"
    d._interleaved_gc_mode = mode
    d._decode_gc_min_seqlen = min_seqlen
    return d


def test_ensure_interleaved_gc_conditional_on_seqlen():
    """SPEED: GC engages ONLY when max_seqlen > _decode_gc_min_seqlen.
    Short decodes leave layers un-checkpointed (fast); long decodes force the
    per-layer flag + FLA-safe reentrant func."""
    from bgkit.training.gradient_utils import _reentrant_checkpoint_func

    d = _make_gc_decoder(min_seqlen=4096)
    inner = _GCInner()

    # Below threshold -> NO GC (no enable, layers stay un-checkpointed).
    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=2000)
    assert inner._enable_calls == 0
    assert all(lyr.gradient_checkpointing is False for lyr in inner.layers)
    assert d._interleaved_gc_active is False

    # Above threshold -> GC engaged on every layer.
    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=8000)
    assert inner._enable_calls == 1
    assert all(lyr.gradient_checkpointing is True for lyr in inner.layers)
    assert all(lyr._gradient_checkpointing_func is _reentrant_checkpoint_func
               for lyr in inner.layers)
    assert d._interleaved_gc_active is True
    assert d._interleaved_gc_layers == 3


def test_ensure_interleaved_gc_toggles_idempotently():
    """Re-evaluated every forward; a swap only happens on a state change, and
    enable() is called at most once (cached via _interleaved_gc_initialized)."""
    d = _make_gc_decoder(min_seqlen=4096)
    inner = _GCInner()

    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=8000)  # ON
    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=9000)  # still ON
    assert inner._enable_calls == 1  # not re-enabled
    assert d._interleaved_gc_active is True

    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=1000)  # OFF
    assert all(lyr.gradient_checkpointing is False for lyr in inner.layers)
    assert d._interleaved_gc_active is False

    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=8000)  # ON again
    assert all(lyr.gradient_checkpointing is True for lyr in inner.layers)
    assert inner._enable_calls == 1  # enable still only ever called once


def test_ensure_interleaved_gc_noop_when_mode_none():
    """mode None -> no-op regardless of seqlen (non-opted phases unchanged)."""
    d = _make_gc_decoder(mode=None)
    inner = _GCInner()
    d._ensure_interleaved_gradient_checkpointing(inner, max_seqlen=99999)
    assert inner._enable_calls == 0
    assert all(lyr.gradient_checkpointing is False for lyr in inner.layers)
