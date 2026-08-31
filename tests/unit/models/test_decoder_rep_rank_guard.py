"""The splice guard must see a payload that is k copies of one vector.

The norm guard next door catches reps at the wrong SCALE. It is blind to
reps at the right scale whose vectors are all the same vector -- which is
what widenet v8 was: effective rank 1.01 per document against the Phase-1
base's 12.97, measured 2026-08-31. Every metric-level check missed it for
weeks, and a 33x retention-budget sweep was spent before the shape of the
defect was visible.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from bgkit.models.decoder import _rep_rank_stats  # noqa: E402


def test_rank_one_payload_is_reported_as_rank_one():
    v = torch.randn(64)
    rows = torch.arange(1.0, 21.0).unsqueeze(-1) * v
    st = _rep_rank_stats([rows])
    assert st["eff_rank"] == pytest.approx(1.0, abs=1e-2)
    assert st["n_payloads"] == 1


def test_a_shared_vector_plus_tiny_content_still_reports_the_content_rank():
    """The measured pathology has 99% of the energy in one shared vector.
    Centring is what makes the statistic report the CONTENT's rank instead of
    the mean's -- without it every such payload reads as rank 1 and the guard
    could not tell v8 (1.01) from the healthy base (12.97)."""
    shared = torch.randn(64) * 300.0
    rows = shared + torch.randn(40, 64)
    st = _rep_rank_stats([rows])
    assert st["eff_rank"] > 10.0
    assert st["shared_frac"] > 0.99


def test_spread_payload_reports_a_high_rank():
    st = _rep_rank_stats([torch.randn(40, 64)])
    assert st["eff_rank"] > 20.0
    assert st["shared_frac"] < 0.1


def test_single_row_payloads_are_skipped():
    """A lone vector is trivially rank 1; counting it would drag the average
    toward an alarm on every short document."""
    assert _rep_rank_stats([torch.randn(1, 64)]) is None
    st = _rep_rank_stats([torch.randn(1, 64), torch.randn(40, 64)])
    assert st["n_payloads"] == 1


def test_empty_and_non_tensor_inputs_are_ignored():
    assert _rep_rank_stats([]) is None
    assert _rep_rank_stats([torch.zeros(0, 64), None, "not a tensor"]) is None


def test_payloads_are_averaged_not_concatenated():
    """Concatenating documents would measure ACROSS-document spread and hide a
    per-document collapse -- exactly the failure being guarded against."""
    a = torch.randn(64)
    b = torch.randn(64)
    rank1_a = torch.arange(1.0, 21.0).unsqueeze(-1) * a
    rank1_b = torch.arange(1.0, 21.0).unsqueeze(-1) * b
    st = _rep_rank_stats([rank1_a, rank1_b])
    assert st["n_payloads"] == 2
    assert st["eff_rank"] == pytest.approx(1.0, abs=1e-2)


def test_large_payloads_are_subsampled_not_rejected():
    from bgkit.models.decoder import _REP_RANK_GUARD_MAX_ROWS

    st = _rep_rank_stats([torch.randn(_REP_RANK_GUARD_MAX_ROWS * 4, 64)])
    assert st is not None
    assert st["eff_rank"] > 20.0


def test_rank_is_scale_invariant():
    """A payload's rank must not move when the whole thing is rescaled --
    otherwise the rank guard would just be a second, noisier norm guard."""
    x = torch.randn(40, 64)
    lo = _rep_rank_stats([x])["eff_rank"]
    hi = _rep_rank_stats([x * 1000.0])["eff_rank"]
    assert lo == pytest.approx(hi, rel=1e-3)
