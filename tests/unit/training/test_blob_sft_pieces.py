"""Unit tests for the Family-A blob SFT trainer's pure pieces (2026-08-24)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.blob_sft_trainer import (
    BlobSFTTrainer,
    _resolve_shards,
    build_blob_segments,
)


def _fake_sample(n_tokens: int = 20, spans=((5, 8), (12, 15))):
    token_ids = torch.arange(n_tokens, dtype=torch.long)
    loss_mask = torch.zeros(n_tokens, dtype=torch.bool)
    loss_mask[16:] = True
    return token_ids, loss_mask, [tuple(s) for s in spans]


def test_build_blob_segments_interleaves_and_drops_sentinels():
    token_ids, loss_mask, spans = _fake_sample()
    survivors = [torch.randn(4, 8), torch.randn(2, 8)]
    segs = build_blob_segments(token_ids, loss_mask, spans, survivors)
    # tokens[0:5] | emb(4) | tokens[8:12] | emb(2) | tokens[15:20]
    assert len(segs) == 5
    from bgkit.models.decoder import EmbeddingSegment, TokenSegment

    assert isinstance(segs[0], TokenSegment) and segs[0].token_ids.tolist() == list(range(5))
    assert isinstance(segs[1], EmbeddingSegment) and segs[1].embeddings.shape == (4, 8)
    assert segs[2].token_ids.tolist() == list(range(8, 12))
    assert isinstance(segs[3], EmbeddingSegment) and segs[3].embeddings.shape == (2, 8)
    assert segs[4].token_ids.tolist() == list(range(15, 20))
    # Sentinel tokens are gone: token count = L - sum(span lens).
    n_tok = sum(int(s.token_ids.shape[0]) for s in segs if isinstance(s, TokenSegment))
    assert n_tok == 20 - 3 - 3
    # Loss mask travels with its run (only the tail run carries loss).
    assert not segs[0].loss_mask.any()
    assert segs[4].loss_mask.tolist() == [False, True, True, True, True]


def test_build_blob_segments_zero_reps_preserves_shape():
    token_ids, loss_mask, spans = _fake_sample()
    survivors = [torch.randn(4, 8), torch.randn(2, 8)]
    segs = build_blob_segments(token_ids, loss_mask, spans, survivors, zero_reps=True)
    assert (segs[1].embeddings == 0).all() and segs[1].embeddings.shape == (4, 8)
    assert (segs[3].embeddings == 0).all()


def test_build_blob_segments_rejects_overlap_and_count_mismatch():
    token_ids, loss_mask, _ = _fake_sample()
    with pytest.raises(ValueError, match="overlapping"):
        build_blob_segments(
            token_ids, loss_mask, [(5, 10), (8, 12)],
            [torch.randn(1, 8), torch.randn(1, 8)],
        )
    with pytest.raises(ValueError, match="survivor runs"):
        build_blob_segments(token_ids, loss_mask, [(5, 8)], [])


def test_adjacent_spans_and_trailing_span():
    token_ids, loss_mask, _ = _fake_sample()
    segs = build_blob_segments(
        token_ids, loss_mask, [(0, 3), (3, 6)],
        [torch.randn(2, 8), torch.randn(2, 8)],
    )
    # emb | emb | tokens[6:] — no empty token segments in between.
    from bgkit.models.decoder import EmbeddingSegment, TokenSegment

    assert [type(s) for s in segs] == [EmbeddingSegment, EmbeddingSegment, TokenSegment]


def test_forward_backward_skips_none_sample():
    t = BlobSFTTrainer.__new__(BlobSFTTrainer)
    m = t._forward_backward([None])
    assert m == {"loss": 0.0, "skipped_samples": 1.0}


def test_wrap_dataloader_iter_never_prefetches():
    """The resume-replay path wraps its raw iterator via
    _wrap_dataloader_iter directly; the device prefetcher must be bypassed
    there too (2026-08-25 crash: list batches have no .items())."""
    from bgkit.training.base_trainer import _DevicePrefetcher

    t = BlobSFTTrainer.__new__(BlobSFTTrainer)
    t.device = torch.device("cpu")
    src = iter([[{"token_ids": torch.zeros(2, dtype=torch.long)}]])
    wrapped = t._wrap_dataloader_iter(src)
    assert not isinstance(wrapped, _DevicePrefetcher)
    assert next(wrapped)[0]["token_ids"].shape == (2,)


def test_resolve_shards_dedupes_and_excludes(tmp_path):
    a = tmp_path / "s1.parquet"
    b = tmp_path / "s2.parquet"
    a.touch()
    b.touch()
    pat = str(tmp_path / "s*.parquet")
    assert _resolve_shards([pat]) == [str(a), str(b)]
    assert _resolve_shards([pat], exclude={str(b)}) == [str(a)]
    assert _resolve_shards([pat, str(a)]) == [str(a), str(b)]


def test_zeroed_ablation_stands_down_the_rep_norm_guard():
    """The decoder's spliced-rep norm guard raises after a streak of
    degenerate splices. Under the zeroed-rep ablation those splices are the
    experiment — the 2026-08-25 standalone 256-sample eval crashed because
    the flag was never armed (the in-training pass ran short enough streaks
    to survive)."""

    class _Dec:
        def __init__(self):
            self.seen = []

        def forward_interleaved_with_loss(self, segments, return_hidden_states=False):
            self.seen.append(getattr(self, "_rep_norm_guard_expect_degenerate", False))
            return "out"

    t = BlobSFTTrainer.__new__(BlobSFTTrainer)
    t.device = torch.device("cpu")
    t.decoder = _Dec()
    t._encode_blobs = lambda contents: [torch.zeros(2, 4)]
    sample = {
        "token_ids": torch.arange(10, dtype=torch.long),
        "loss_mask": torch.ones(10, dtype=torch.bool),
        "sentinel_spans": [(4, 6)],
        "blob_content_ids": [torch.zeros(3, dtype=torch.long)],
    }
    t._sample_loss(sample, zero_reps=True)
    t._sample_loss(sample, zero_reps=False)
    assert t.decoder.seen == [True, False]
    # Restored afterwards, so a live forward is never left unguarded.
    assert t.decoder._rep_norm_guard_expect_degenerate is False
