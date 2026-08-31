"""The distinguishability probe's statistics must be right before its verdict is.

This probe is meant to settle "do the reps carry document-specific content?",
and the whole conclusion rests on two small numbers. A participation ratio
that forgets to centre, or a retrieval that reports the rank off by one,
would produce a confident and wrong answer -- the failure mode this
investigation has already hit twice (a mean-vs-max null, a probe that
contradicted the metric it was checking).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "probe_rep_distinguishability",
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "probe_rep_distinguishability.py",
)
probe = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(probe)


def test_participation_ratio_is_one_for_a_single_direction():
    """Every row on one line = one effective direction, whatever the offset."""
    d = torch.randn(1, 8)
    rows = torch.arange(1.0, 33.0).unsqueeze(-1) * d
    assert probe._participation_ratio(rows) == pytest.approx(1.0, abs=1e-3)


def test_participation_ratio_counts_orthogonal_directions():
    rows = torch.eye(5, 16) * 3.0
    # 5 centred orthogonal rows span 4 directions (centring removes one).
    assert probe._participation_ratio(rows) == pytest.approx(4.0, abs=0.05)


def test_participation_ratio_ignores_a_constant_offset():
    """The thing being tested IS 'is this just a shared mean' -- so centring
    must make a constant-plus-noise matrix report its noise rank, not 1."""
    noise = torch.randn(64, 32)
    offset = torch.randn(32) * 100.0
    assert probe._participation_ratio(noise + offset) == pytest.approx(
        probe._participation_ratio(noise), rel=1e-4,
    )


def test_halves_are_disjoint_and_unit_norm():
    x = torch.randn(9, 6)
    a, b = probe._halves(x)
    assert a.norm() == pytest.approx(1.0, abs=1e-5)
    assert b.norm() == pytest.approx(1.0, abs=1e-5)
    ea = x[0::2].float().mean(dim=0)
    torch.testing.assert_close(a, ea / ea.norm(), rtol=1e-5, atol=1e-6)


def test_halves_refuses_too_few_rows():
    assert probe._halves(torch.randn(3, 6)) is None


def test_retrieval_is_perfect_when_halves_match():
    v = torch.randn(12, 32)
    v = v / v.norm(dim=-1, keepdim=True)
    r = probe._retrieval(v, v)
    assert r["top1"] == 1.0
    assert r["mrr"] == 1.0
    assert r["median_rank"] == 1.0
    assert r["chance_top1"] == pytest.approx(1.0 / 12)


def test_retrieval_is_at_chance_for_a_constant_representation():
    """The exact failure this probe exists to detect: every document emits the
    same pooled vector. Rank must then be uninformative, not accidentally 1."""
    m = 16
    const = torch.randn(1, 24)
    const = const / const.norm()
    a = const.repeat(m, 1) + 1e-6 * torch.randn(m, 24)
    b = const.repeat(m, 1) + 1e-6 * torch.randn(m, 24)
    a = a / a.norm(dim=-1, keepdim=True)
    b = b / b.norm(dim=-1, keepdim=True)
    r = probe._retrieval(a, b)
    assert r["top1"] < 0.5
    assert r["mean_offdiag_cos"] > 0.99
    assert probe._participation_ratio(a) > 1.0  # noise only, no shared signal


def test_retrieval_rank_is_one_indexed_for_a_known_permutation():
    """Row i's true match is deliberately the SECOND-best; rank must read 2."""
    m = 6
    b = torch.eye(m, m)
    a = b.clone() * 0.5
    for i in range(m):
        a[i, (i + 1) % m] = 0.9  # a stronger, wrong neighbour
    a = a / a.norm(dim=-1, keepdim=True)
    r = probe._retrieval(a, b)
    assert r["top1"] == 0.0
    assert r["median_rank"] == 2.0
    assert r["mrr"] == pytest.approx(0.5)


def test_shared_energy_is_one_when_every_row_is_the_same_vector():
    v = torch.randn(32)
    assert probe._shared_energy(v.repeat(20, 1))["shared_frac"] == pytest.approx(
        1.0, abs=1e-5,
    )


def test_shared_energy_is_near_zero_for_zero_mean_rows():
    x = torch.randn(4096, 64)
    x = x - x.mean(dim=0, keepdim=True)
    assert probe._shared_energy(x)["shared_frac"] < 1e-6


def test_shared_energy_tracks_the_signal_to_shared_ratio():
    """A tiny content perturbation on a large shared vector is exactly the
    measured pathology; shared_frac must report it as ~1, not as ~0.5."""
    shared = torch.randn(64) * 10.0
    rows = shared.repeat(50, 1) + 0.01 * torch.randn(50, 64)
    assert probe._shared_energy(rows)["shared_frac"] > 0.999


def test_shared_energy_returns_the_mean_vector_it_measured():
    x = torch.randn(9, 12)
    torch.testing.assert_close(
        probe._shared_energy(x)["mean_vec"], x.float().mean(dim=0),
    )
