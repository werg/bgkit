"""Tests for TopicEmbeddingModule batch gradient averaging + LR scaling."""

from __future__ import annotations

import math

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.taxonomy import TagTaxonomy
from bgkit.models.topic_embeddings import TopicEmbeddingModule


def _toy_taxonomy() -> TagTaxonomy:
    """Build a tiny taxonomy with two top-level domains."""
    return TagTaxonomy.from_tag_counts(
        {
            "global": 100,
            "global/python": 80,
            "global/python/numpy": 30,
            "global/python/flask": 5,
            "global/rust": 10,
            "global/rust/serde": 3,
        },
        min_frequency=1,
    )


def test_get_optimizer_groups_scales_by_sqrt_frequency():
    """Rare tags get a larger LR scale than the global / median tags."""
    tax = _toy_taxonomy()
    mod = TopicEmbeddingModule(tax, positions_per_tag=4, hidden_dim=8)
    groups = mod.get_optimizer_groups(base_lr=1e-3)
    by_tag = {g["tag"]: g["lr"] for g in groups}
    # Median frequency in {100, 80, 30, 5, 10, 3} is 20 (the average of 10,30 — uses index-based median).
    # Specifically: sorted = [3, 5, 10, 30, 80, 100]; index 3 → 30. So scale uses median=30.
    median = 30
    expected_global = 1e-3 * math.sqrt(median / 100)
    expected_flask = 1e-3 * math.sqrt(median / 5)
    assert by_tag["global"] == pytest.approx(expected_global, rel=1e-5)
    assert by_tag["global/python/flask"] == pytest.approx(expected_flask, rel=1e-5)
    # Rare tag should have a strictly larger LR than the most frequent.
    assert by_tag["global/python/flask"] > by_tag["global"]


def test_record_batch_usage_counts_distinct_samples_per_tag():
    """Tags shared across multiple samples in a batch must record the
    correct per-tag count, even when a single sample lists the tag
    multiple times via expansion through the taxonomy."""
    tax = _toy_taxonomy()
    mod = TopicEmbeddingModule(tax, positions_per_tag=2, hidden_dim=4)
    # Three samples; "global" is on all of them, flask is on one.
    mod.record_batch_usage([
        ["global/python/numpy"],
        ["global/python/flask"],
        ["global/rust/serde"],
    ])
    counts = mod._batch_tag_counts
    assert counts["global"] == 3
    assert counts["global/python"] == 2
    assert counts["global/python/flask"] == 1
    assert counts["global/rust"] == 1
    assert counts["global/rust/serde"] == 1


def test_apply_gradient_averaging_divides_by_batch_count():
    """A tag that appeared in N samples gets its gradient divided by N."""
    tax = _toy_taxonomy()
    mod = TopicEmbeddingModule(tax, positions_per_tag=2, hidden_dim=4)
    # Stamp counts manually so we don't depend on the recorder.
    mod._batch_tag_counts = {
        "global": 3,
        "global/python": 2,
        "global/python/numpy": 1,  # count == 1, no-op
    }
    # Set gradients on a few of the parameters.
    for tag in ("global", "global/python", "global/python/numpy"):
        param = mod.embeddings[mod._key(tag)]
        param.grad = torch.ones_like(param)
    mod.apply_gradient_averaging()
    # global: divided by 3
    assert torch.allclose(
        mod.embeddings[mod._key("global")].grad,
        torch.full_like(mod.embeddings[mod._key("global")], 1.0 / 3.0),
    )
    # global/python: divided by 2
    assert torch.allclose(
        mod.embeddings[mod._key("global/python")].grad,
        torch.full_like(mod.embeddings[mod._key("global/python")], 0.5),
    )
    # global/python/numpy: count == 1, untouched
    assert torch.allclose(
        mod.embeddings[mod._key("global/python/numpy")].grad,
        torch.ones_like(mod.embeddings[mod._key("global/python/numpy")]),
    )


def test_apply_gradient_averaging_noop_when_no_batch_recorded():
    """Calling apply_gradient_averaging without a recorded batch is a no-op."""
    tax = _toy_taxonomy()
    mod = TopicEmbeddingModule(tax, positions_per_tag=2, hidden_dim=4)
    # No record_batch_usage call; calling averaging should be safe.
    mod.embeddings[mod._key("global")].grad = torch.ones(2, 4)
    mod.apply_gradient_averaging()
    assert torch.allclose(
        mod.embeddings[mod._key("global")].grad, torch.ones(2, 4),
    )
