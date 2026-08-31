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


def test_halves_are_disjoint_and_unnormalised():
    """Unnormalised on purpose: the whitening path estimates a covariance
    from these, and normalising first would discard the scale it needs."""
    x = torch.randn(9, 6)
    a, b = probe._halves(x)
    torch.testing.assert_close(a, x[0::2].float().mean(dim=0))
    torch.testing.assert_close(b, x[1::2].float().mean(dim=0))


def _buried(sig: float, seed: int, m: int = 192, d: int = 128):
    """A corpus-constant vector, a few high-variance per-half nuisance
    directions, and a document signal of size ``sig`` -- the measured
    pathology in miniature. ``sig=0`` is the same setup with no signal."""
    torch.manual_seed(seed)
    shared = torch.randn(d) * 300.0
    nuisance = torch.linalg.qr(torch.randn(d, d))[0][:6]
    content = torch.randn(m, d) * sig

    def half():
        return (
            shared
            + content
            + (torch.randn(m, 6) * 12.0) @ nuisance
            + 0.05 * torch.randn(m, d)
        )

    return probe._reweighted_retrieval(half(), half())


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_noise_whitening_recovers_a_signal_cosine_cannot_see(seed):
    w = _buried(0.3, seed)
    assert w["raw_heldout"]["top1"] < 0.1
    assert w["noise_whitened_heldout"]["top1"] > 0.8


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_reweighting_invents_a_signal_that_is_absent(seed):
    """The other half of the discrimination. Without this the probe could
    report 'the information is in there' for any input at all."""
    w = _buried(0.0, seed)
    assert w["noise_whitened_heldout"]["top1"] < 0.15
    assert w["corpus_whitened_heldout"]["top1"] < 0.15


def test_reweighting_fits_and_scores_on_disjoint_documents():
    w = probe._reweighted_retrieval(torch.randn(64, 32), torch.randn(64, 32))
    assert w["n_heldout"] == 32
    for k in ("raw_heldout", "corpus_whitened_heldout", "noise_whitened_heldout"):
        assert w[k]["n_docs"] == 32


def test_reweighting_stays_inside_a_fitted_subspace():
    """A covariance estimated from a few hundred samples in 1024 dimensions is
    rank deficient; whitening the unfitted directions divides by the
    regulariser rather than by anything measured. The PCA cap is what keeps
    that from being reported as signal."""
    w = probe._reweighted_retrieval(torch.randn(64, 512), torch.randn(64, 512))
    assert w["pca_dims"] <= 32 // 3 + 1
    assert w["pca_dims"] >= 2


def test_reweighting_skips_when_there_are_too_few_documents():
    assert probe._reweighted_retrieval(torch.randn(12, 12), torch.randn(12, 12)) == {}


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


def test_diagonal_whitening_is_reported_and_is_weaker_than_a_rotation():
    """The diag arm must be the SHIPPED transform (centre + per-channel scale),
    not a stand-in for the full whitening. A signal that lives in a rotated
    direction is recoverable by corpusW and not by diagW, and conflating them
    would report the shipped fix as sufficient when it is not."""
    torch.manual_seed(0)
    m, d = 192, 32
    # Signal along a direction that no single channel isolates, buried under
    # per-channel nuisance of equal size in every channel -- so scaling
    # channels cannot separate them and only a rotation can.
    direction = torch.randn(d)
    direction = direction / direction.norm()
    coeff = torch.randn(m, 1)
    shared = torch.randn(d) * 300.0

    def half():
        return shared + coeff * direction * 3.0 + torch.randn(m, d) * 3.0

    w = probe._reweighted_retrieval(half(), half())
    assert "diag_whitened_heldout" in w
    assert w["corpus_whitened_heldout"]["top1"] > w["diag_whitened_heldout"]["top1"]


def test_diagonal_whitening_recovers_a_per_channel_burial():
    """And when the burial IS per-channel, the shipped transform must suffice
    -- otherwise the arm would always understate it."""
    torch.manual_seed(1)
    m, d = 192, 32
    shared = torch.randn(d) * 300.0
    scale = torch.cat([torch.full((4,), 60.0), torch.full((d - 4,), 0.05)])
    content = torch.randn(m, d)

    def half():
        return shared + content + torch.randn(m, d) * scale

    w = probe._reweighted_retrieval(half(), half())
    assert w["raw_heldout"]["top1"] < 0.3
    assert w["diag_whitened_heldout"]["top1"] > 0.7
