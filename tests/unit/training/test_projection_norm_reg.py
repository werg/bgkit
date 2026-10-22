"""Unit tests for the projection-output NORM-BAND regularizer (2026-07-31
collapse fix) in :class:`~bgkit.training.phase2.kr_kb_trainer.KRKBTrainer`.

The regularizer keeps every projected/spliced survivor-rep's L2 norm inside the
active decoder family's readable band and penalizes the inflation runaway that
collapsed the git-repro encoder (readable ~4x embed-norm -> 12-320x). These CPU
tests exercise the loss math + the gradient path WITHOUT a real encoder/decoder
(the trainer method is called on a lightweight stub carrying only the attributes
it reads). End-to-end grad-to-real-backbone is covered by the always-on runtime
guard + the gpu-marked liveness test.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

import torch.nn as nn
import torch.nn.functional as F

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


class _Enc:
    """Minimal encoder stub — the term only reads ``.training``."""

    def __init__(self, training: bool = True) -> None:
        self.training = training


class _Dec:
    """Minimal decoder stub exposing ``backbone.get_input_embeddings().weight``."""

    def __init__(self, weight: torch.Tensor) -> None:
        embeddings = type("E", (), {"weight": weight})()
        self.backbone = type(
            "B", (), {"get_input_embeddings": lambda self: embeddings},
        )()


def _make_stub(
    *,
    embed_weight: torch.Tensor,
    enabled: bool = True,
    weight: float = 0.1,
    tolerance: float = 2.0,
    tolerances: dict[str, float] | None = None,
    target_ratio: float = 4.2,
    family: str = "qwen35",
    training: bool = True,
) -> KRKBTrainer:
    stub = KRKBTrainer.__new__(KRKBTrainer)
    stub.device = torch.device("cpu")
    stub.encoder = _Enc(training=training)
    stub._decoder_family = family
    stub._decoders_by_family = {family: _Dec(embed_weight)}
    stub._proj_norm_reg_enabled = enabled
    stub._proj_norm_reg_weight = weight
    # Mirrors the trainer attributes: ``tolerance`` is the scalar fallback,
    # ``tolerances`` the optional per-family dict (empty = scalar-only mode).
    stub._proj_norm_reg_tolerance = tolerance
    stub._proj_norm_reg_tolerances = dict(tolerances or {})
    stub._proj_norm_reg_target_ratios = {family: target_ratio}
    stub._proj_norm_reg_embed_ref_cache = {}
    stub._proj_norm_ratio_accum = {}
    return stub


def _unit_norm_embed(vocab: int = 32, dim: int = 8) -> torch.Tensor:
    """Embedding table whose rows all have L2-norm 1 -> mean row-norm == 1, so
    ``embed_ref == 1`` and ``target == target_ratio`` exactly."""
    torch.manual_seed(0)
    return F.normalize(torch.randn(vocab, dim), dim=-1)


def _reps_at_norm(n: int, dim: int, norm_val: float) -> nn.Parameter:
    torch.manual_seed(1)
    return nn.Parameter(F.normalize(torch.randn(n, dim), dim=-1) * norm_val)


def test_embed_ref_norm_computed_and_cached():
    w = _unit_norm_embed()
    stub = _make_stub(embed_weight=w)
    ref = KRKBTrainer._proj_norm_reg_embed_ref_norm(stub, "qwen35")
    assert ref == pytest.approx(1.0, abs=1e-4)
    # cached: mutating the weight must not change the cached value.
    w.data.mul_(10.0)
    ref2 = KRKBTrainer._proj_norm_reg_embed_ref_norm(stub, "qwen35")
    assert ref2 == pytest.approx(1.0, abs=1e-4)


def test_term_zero_when_in_band():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim))
    # target = 4.2 * 1.0 = 4.2; reps exactly at target => zero penalty.
    reps = _reps_at_norm(6, dim, norm_val=4.2)
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    assert float(term.detach()) == pytest.approx(0.0, abs=1e-6)


def test_term_zero_at_band_edge():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), tolerance=2.0)
    # target*tolerance = 8.4 is the outer edge of the permissive band; still ~0.
    reps = _reps_at_norm(6, dim, norm_val=8.39)
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    assert float(term.detach()) == pytest.approx(0.0, abs=1e-4)


def test_term_large_when_inflated_30x():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), weight=0.1)
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    # log(30) - log(2) = 2.708; penalty = 0.1 * 2.708^2 ~= 0.733.
    assert float(term.detach()) > 0.5
    # and the logged norm-ratio reflects the 30x inflation (~126 = 30*4.2).
    metrics = KRKBTrainer._proj_norm_reg_step_metrics(stub)
    assert metrics["proj_norm_ratio_qwen35"] == pytest.approx(126.0, rel=0.02)


def test_penalty_monotonic_in_inflation():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim))
    vals = []
    for mult in (3.0, 10.0, 30.0, 100.0):
        stub._proj_norm_ratio_accum = {}
        reps = _reps_at_norm(6, dim, norm_val=4.2 * mult)
        vals.append(float(KRKBTrainer._projection_norm_reg_term(stub, [reps]).detach()))
    import itertools

    assert vals == sorted(vals)
    assert all(v2 > v1 for v1, v2 in itertools.pairwise(vals))


def test_grad_reaches_projection_and_backbone():
    """The whole point: gradient from the norm-reg term must flow through the
    projection layer INTO the producing backbone, so it constrains what the
    encoder produces. Uses a backbone->projection stand-in chain."""
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), weight=0.1)
    torch.manual_seed(3)
    backbone = nn.Linear(dim, dim)
    projection = nn.Linear(dim, dim)
    x = torch.randn(6, dim)
    # Scale into the runaway regime so the penalty (and its gradient) is active.
    reps = 20.0 * projection(backbone(x))
    assert reps.requires_grad
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    assert term.requires_grad
    term.backward()
    assert projection.weight.grad is not None
    assert backbone.weight.grad is not None
    assert projection.weight.grad.abs().sum() > 0
    assert backbone.weight.grad.abs().sum() > 0


def test_disabled_returns_zero_no_grad():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), enabled=False)
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    assert not term.requires_grad
    assert float(term) == pytest.approx(0.0, abs=1e-9)


def test_eval_mode_returns_zero():
    """During eval the reps do not require grad and must not be penalized."""
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), training=False)
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    term = KRKBTrainer._projection_norm_reg_term(stub, [reps])
    assert float(term) == pytest.approx(0.0, abs=1e-9)


def test_non_grad_reps_are_skipped():
    """Zero-fallback splices (no grad) must not contribute (log(0) trap)."""
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim))
    zero_fallback = torch.zeros(1, dim)  # requires_grad False
    inflated = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    term_mixed = KRKBTrainer._projection_norm_reg_term(stub, [zero_fallback, inflated])
    stub._proj_norm_ratio_accum = {}
    term_only = KRKBTrainer._projection_norm_reg_term(stub, [inflated])
    # The no-grad zero fallback is skipped, so mixed == only-inflated (finite).
    assert torch.isfinite(term_mixed)
    assert float(term_mixed.detach()) == pytest.approx(float(term_only.detach()), rel=1e-5)


def test_weight_scales_penalty():
    dim = 8
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    s_lo = _make_stub(embed_weight=_unit_norm_embed(dim=dim), weight=0.1)
    s_hi = _make_stub(embed_weight=_unit_norm_embed(dim=dim), weight=0.4)
    t_lo = float(KRKBTrainer._projection_norm_reg_term(s_lo, [reps]).detach())
    t_hi = float(KRKBTrainer._projection_norm_reg_term(s_hi, [reps]).detach())
    assert t_hi == pytest.approx(4.0 * t_lo, rel=1e-4)


def test_scalar_tolerance_backcompat_matches_dict_equivalent():
    """Back-compat: a scalar tolerance behaves exactly as before, and a dict
    entry with the same value for the active family is numerically identical."""
    dim = 8
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    s_scalar = _make_stub(embed_weight=_unit_norm_embed(dim=dim), tolerance=2.0)
    s_dict = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim),
        tolerance=2.0,
        tolerances={"qwen35": 2.0},
    )
    t_scalar = float(KRKBTrainer._projection_norm_reg_term(s_scalar, [reps]).detach())
    t_dict = float(KRKBTrainer._projection_norm_reg_term(s_dict, [reps]).detach())
    assert t_scalar > 0.0
    assert t_dict == pytest.approx(t_scalar, rel=1e-6)


def test_dict_tolerance_per_family_qwen_tightened_falcon_not():
    """The drift scenario the per-family tolerance exists for: qwen35 at
    norm-ratio 8.5 (target 4.2 -> 8.5/4.2 ~ 2.02x target) is firmly penalized
    under its tightened tol 1.4, while the SAME rep/target ratio for falcon_h1
    is judged against falcon's tol 2.0 (~zero at the band edge)."""
    import math

    dim = 8
    tols = {"qwen35": 1.4, "falcon_h1": 2.0}
    log_over = math.log(8.5 / 4.2)  # ~0.705, the shared rep/target log-ratio

    # qwen: embed_ref == 1 -> rep norm 8.5 IS the 8.5x norm-ratio.
    s_q = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim), tolerances=tols,
        target_ratio=4.2, family="qwen35",
    )
    reps_q = _reps_at_norm(6, dim, norm_val=8.5)
    t_q = float(KRKBTrainer._projection_norm_reg_term(s_q, [reps_q]).detach())
    expected_q = 0.1 * (log_over - math.log(1.4)) ** 2  # ~0.0136
    assert t_q > 0.0
    assert t_q == pytest.approx(expected_q, rel=1e-3)

    # falcon at the same rep/target ratio (0.9 * 8.5/4.2 ~ 1.82x embed-norm):
    # judged against ITS tol 2.0, not qwen's 1.4.
    s_f = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim), tolerances=tols,
        target_ratio=0.9, family="falcon_h1",
    )
    reps_f = _reps_at_norm(6, dim, norm_val=0.9 * 8.5 / 4.2)
    t_f = float(KRKBTrainer._projection_norm_reg_term(s_f, [reps_f]).detach())
    expected_f = 0.1 * max(0.0, log_over - math.log(2.0)) ** 2  # ~1.4e-5
    assert t_f == pytest.approx(expected_f, abs=1e-5)
    assert t_q > 100.0 * t_f


def test_dict_tolerance_falcon_stable_in_band_stays_zero():
    """Tightening qwen must not touch the stable falcon: at the observed
    ~0.62x embed-norm (target 0.9, band [0.45, 1.8] at tol 2.0) the penalty
    is exactly zero even with the per-family dict active."""
    dim = 8
    s_f = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim),
        tolerances={"qwen35": 1.4, "falcon_h1": 2.0},
        target_ratio=0.9, family="falcon_h1",
    )
    reps_f = _reps_at_norm(6, dim, norm_val=0.62)
    t_f = float(KRKBTrainer._projection_norm_reg_term(s_f, [reps_f]).detach())
    assert t_f == pytest.approx(0.0, abs=1e-9)


def test_dict_tolerance_same_ratio_in_band_loose_out_of_band_tight():
    """A rep at 1.9x its family target is OUT of band under tol 1.4 but IN
    band under tol 2.0 — the exact asymmetry the per-family dict provides."""
    dim = 8
    tols = {"qwen35": 1.4, "falcon_h1": 2.0}
    s_q = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim), tolerances=tols,
        target_ratio=4.2, family="qwen35",
    )
    t_q = float(KRKBTrainer._projection_norm_reg_term(
        s_q, [_reps_at_norm(6, dim, norm_val=4.2 * 1.9)],
    ).detach())
    assert t_q > 1e-4  # penalized under tol 1.4

    s_f = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim), tolerances=tols,
        target_ratio=0.9, family="falcon_h1",
    )
    t_f = float(KRKBTrainer._projection_norm_reg_term(
        s_f, [_reps_at_norm(6, dim, norm_val=0.9 * 1.9)],
    ).detach())
    assert t_f == pytest.approx(0.0, abs=1e-9)  # in band under tol 2.0


def test_missing_family_falls_back_to_scalar_default():
    """A family absent from the tolerance dict uses the scalar default —
    numerically identical to the scalar-only (pre-change) behavior."""
    dim = 8
    reps = _reps_at_norm(6, dim, norm_val=4.2 * 30.0)
    s_missing = _make_stub(
        embed_weight=_unit_norm_embed(dim=dim),
        tolerance=2.0,
        tolerances={"falcon_h1": 1.1},  # qwen35 absent -> scalar 2.0 applies
        family="qwen35",
    )
    s_scalar = _make_stub(embed_weight=_unit_norm_embed(dim=dim), tolerance=2.0)
    t_missing = float(KRKBTrainer._projection_norm_reg_term(s_missing, [reps]).detach())
    t_scalar = float(KRKBTrainer._projection_norm_reg_term(s_scalar, [reps]).detach())
    assert t_missing == pytest.approx(t_scalar, rel=1e-6)


def test_live_weight_handler():
    dim = 8
    stub = _make_stub(embed_weight=_unit_norm_embed(dim=dim), weight=0.1)
    KRKBTrainer._handle_projection_norm_reg_weight(stub, 0.25)
    assert stub._proj_norm_reg_weight == pytest.approx(0.25)
    # invalid values are ignored (kept at last good value).
    KRKBTrainer._handle_projection_norm_reg_weight(stub, -1.0)
    assert stub._proj_norm_reg_weight == pytest.approx(0.25)
    KRKBTrainer._handle_projection_norm_reg_weight(stub, "bogus")
    assert stub._proj_norm_reg_weight == pytest.approx(0.25)
