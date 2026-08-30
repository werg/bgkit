"""The rescaled arm must change magnitude and NOTHING else.

WHY IT EXISTS. Measured on the summarization base 2026-08-29: normalising
spliced reps to the decoder's mean token-embedding norm — directions untouched
— flips the per-token rep gap from -0.064 to +0.046 at l0=0.63, and from -0.007
to +0.095 at l0=0.10, the harsher budget where Phase 2 runs. Reps sit at ~500x
embed norm, so scale appears to be mechanically suppressing information the
reps already carry. This arm asks the same question on Phase-2's own path.

The whole inference rests on the transform being magnitude-only. If it perturbs
direction at all, a "scale effect" is really a content effect and the
conclusion inverts. That is what these tests pin.
"""

from __future__ import annotations

import inspect
import types
from pathlib import Path

import torch

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
from bgkit.models.decoder import ReconstructionDecoder


def _trainer(embed_norm: float = 0.5, dim: int = 16) -> KRKBTrainer:
    t = object.__new__(KRKBTrainer)
    w = torch.randn(64, dim)
    w = w / w.norm(dim=-1, keepdim=True) * embed_norm      # every row == embed_norm
    backbone = types.SimpleNamespace(get_input_embeddings=lambda: types.SimpleNamespace(weight=w))
    t.decoder = types.SimpleNamespace(backbone=backbone)
    t._ablation_mode = KRKBTrainer.ABLATION_RESCALED
    return t


def test_directions_are_exactly_preserved() -> None:
    """The load-bearing property: cosine similarity 1.0 for every vector."""
    t = _trainer()
    reps = torch.randn(32, 16) * 500.0                     # ~the real operating point
    out = t._apply_context_ablation(reps, skip=False)
    cos = torch.nn.functional.cosine_similarity(reps.float(), out.float(), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_magnitude_becomes_the_embedding_norm() -> None:
    t = _trainer(embed_norm=0.5)
    reps = torch.randn(32, 16) * 500.0
    out = t._apply_context_ablation(reps, skip=False)
    assert torch.allclose(out.float().norm(dim=-1), torch.full((32,), 0.5), atol=1e-4)


def test_shape_and_count_unchanged() -> None:
    """Same number of spliced vectors, so sequence length and position ids are
    identical between arms — otherwise the comparison is confounded."""
    t = _trainer()
    reps = torch.randn(7, 16) * 123.0
    assert t._apply_context_ablation(reps, skip=False).shape == reps.shape


def test_other_ablation_modes_are_untouched() -> None:
    """Additive change: zeroed/noise/none must behave exactly as before."""
    t = _trainer()
    reps = torch.randn(8, 16) * 500.0

    t._ablation_mode = KRKBTrainer.ABLATION_ZEROED
    assert torch.count_nonzero(t._apply_context_ablation(reps, skip=False)) == 0

    t._ablation_mode = KRKBTrainer.ABLATION_NONE
    assert torch.equal(t._apply_context_ablation(reps, skip=False), reps)


def test_skip_still_wins_over_rescale() -> None:
    """TOPICS_ONLY / NEITHER collapse to one zero vector; rescaling must not
    resurrect content in an arm meant to have none."""
    t = _trainer()
    out = t._apply_context_ablation(torch.randn(9, 16) * 500.0, skip=True)
    assert out.shape == (1, 16)
    assert torch.count_nonzero(out) == 0


def test_already_scaled_reps_are_a_near_noop() -> None:
    """If reps ever DO sit at embed norm, the arm must not move them —
    otherwise it would manufacture a difference where none exists."""
    t = _trainer(embed_norm=0.5)
    reps = torch.randn(16, 16)
    reps = reps / reps.norm(dim=-1, keepdim=True) * 0.5
    out = t._apply_context_ablation(reps, skip=False)
    assert torch.allclose(out.float(), reps.float(), atol=1e-5)


def test_zero_vectors_do_not_produce_nan() -> None:
    """A dead rep must not divide by zero and poison the whole eval."""
    t = _trainer()
    reps = torch.zeros(4, 16)
    assert torch.isfinite(t._apply_context_ablation(reps, skip=False)).all()


def test_metrics_are_derived_not_left_to_hand_comparison() -> None:
    """A two-number comparison done by eye is how the summ_summary autoencoding
    row got read as a summarization result for a day. The deltas are computed."""
    import inspect
    src = inspect.getsource(KRKBTrainer.evaluate)
    assert 'metrics["eval/rescale_gain/nats"] = b_l - r_l' in src
    assert 'metrics["eval/rep_gain_rescaled/nats"] = z_l - r_l' in src


def test_rescale_arm_declares_its_expected_norm_ratio() -> None:
    """The rescale arm targets ratio 1.0 BY CONSTRUCTION, so the splice guard
    must band around 1.0 during it, not around the learned raw reference.

    Observed 2026-08-30: the base eval emitted ``spliced_rep_norm_out_of_band
    degenerate=True`` on every sampled check of the rescaled arm (ratio 1.0 vs
    reference 37.6). A guard that warns continuously through a healthy run is
    an instrument failure — it trains the reader to ignore the one signal that
    catches real rep-inflation.
    """
    assert KRKBTrainer._REP_ABLATION_EXPECTED_RATIO[KRKBTrainer.ABLATION_RESCALED] == 1.0


def test_rescaled_is_not_classified_degenerate() -> None:
    """Suppressing the guard (the ``zeroed``/``noise`` treatment) was the cheap
    fix and the wrong one: rescaled reps are VALID, and during that arm the
    guard is the only automatic check that the rescale actually applied."""
    assert KRKBTrainer.ABLATION_RESCALED not in KRKBTrainer._DEGENERATE_REP_ABLATIONS


def test_expectation_reaches_the_decoder_through_the_ablation_setter() -> None:
    """One funnel. A second assignment site is how the guard flag and the
    trainer's ablation state drift apart."""
    src = inspect.getsource(KRKBTrainer._ablation_mode.fset)
    assert "_rep_norm_guard_expected_ratio" in src
    assert "_REP_ABLATION_EXPECTED_RATIO" in src
    whole = Path(inspect.getfile(KRKBTrainer)).read_text()
    assert whole.count("dec._rep_norm_guard_expected_ratio") == 1


def test_guard_bands_around_the_expectation_when_declared() -> None:
    """Decoder side: the drift comparison must use the declared expectation and
    fall back to the learned reference when there is none."""
    src = inspect.getsource(ReconstructionDecoder._maybe_guard_spliced_rep_norm)
    assert "_rep_norm_guard_expected_ratio" in src
    assert "band_ref = expected if expected is not None else ref" in src
    # the absolute collapse floor must STILL apply — an expectation of 1.0 must
    # not make "no reps at all" look healthy
    assert "ratio < lo or (" in src
