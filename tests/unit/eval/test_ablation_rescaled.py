"""One rescale implementation, shared by every consumer.

`bgkit.eval.ablations` is the module whose own docstring calls this the
"mandatory ablation suite ... every training stage". The rescale transform and
the RESCALED condition belong here rather than in any one trainer: a per-caller
copy is how a representation-interface contract drifts apart between call
sites, which is the failure class this project keeps re-encountering.

MEASURED BASIS (summarization base, source->summary, per-token gap, n=6318):
    l0=0.63  raw -0.0640 -> rescaled +0.0455
    l0=0.32  raw -0.0618 -> rescaled +0.0496
    l0=0.10  raw -0.0068 -> rescaled +0.0945
Directions untouched in all cases, so the swing is attributable to magnitude.
"""

from __future__ import annotations

import pytest
import torch

from bgkit.eval.ablations import (
    AblationCondition,
    AblationResult,
    _modify_survivors,
    compute_ablation_gap,
    rescale_to_embed_norm,
)


def _embed(n: int = 64, d: int = 16, norm: float = 0.5) -> torch.Tensor:
    w = torch.randn(n, d)
    return w / w.norm(dim=-1, keepdim=True) * norm


def test_directions_exactly_preserved() -> None:
    """The property the whole inference rests on. If rescaling perturbed
    direction, a 'scale effect' would really be a content effect and the
    conclusion would invert."""
    reps = torch.randn(32, 16) * 500.0
    out = rescale_to_embed_norm(reps, _embed())
    cos = torch.nn.functional.cosine_similarity(reps.float(), out.float(), dim=-1)
    assert torch.allclose(cos, torch.ones_like(cos), atol=1e-5)


def test_magnitude_matches_the_embedding_norm() -> None:
    out = rescale_to_embed_norm(torch.randn(32, 16) * 500.0, _embed(norm=0.5))
    assert torch.allclose(out.float().norm(dim=-1), torch.full((32,), 0.5), atol=1e-4)


def test_empty_and_zero_inputs_are_safe() -> None:
    """A turn with no survivors, or a dead rep, must not NaN an entire eval."""
    assert rescale_to_embed_norm(torch.zeros(0, 16), _embed()).numel() == 0
    assert torch.isfinite(rescale_to_embed_norm(torch.zeros(4, 16), _embed())).all()


def test_condition_dispatch_matches_the_direct_call() -> None:
    reps = torch.randn(8, 16) * 500.0
    emb = _embed()
    via = _modify_survivors(reps, AblationCondition.SURVIVORS_RESCALED, emb)
    assert torch.equal(via, rescale_to_embed_norm(reps, emb))


def test_rescaled_without_embed_weight_fails_loudly() -> None:
    """It cannot silently fall back to 'present' — that would report a
    no-op arm as a real measurement, the exact silent-success failure shape."""
    with pytest.raises(ValueError, match="embed_weight"):
        _modify_survivors(torch.randn(4, 16), AblationCondition.SURVIVORS_RESCALED)


def test_existing_conditions_unchanged() -> None:
    """Additive: present/zeroed/noise must behave exactly as before."""
    reps = torch.randn(8, 16)
    assert torch.equal(
        _modify_survivors(reps, AblationCondition.SURVIVORS_PRESENT), reps,
    )
    assert torch.count_nonzero(
        _modify_survivors(reps, AblationCondition.SURVIVORS_ZEROED),
    ) == 0


def test_gap_signs_are_what_the_names_claim() -> None:
    """present_vs_rescaled POSITIVE means rescaling helps; rescaled_vs_zeroed
    is the value-of-reps signal with scale fixed."""
    res = [
        AblationResult(AblationCondition.SURVIVORS_PRESENT, {"loss": 3.43}),
        AblationResult(AblationCondition.SURVIVORS_ZEROED, {"loss": 3.36}),
        AblationResult(AblationCondition.SURVIVORS_RESCALED, {"loss": 3.32}),
    ]
    gaps = compute_ablation_gap(res)
    # raw reps HURT relative to zeroed (the measured base behaviour)
    assert gaps["present_vs_zeroed_loss_gap"] < 0
    # rescaling helps ...
    assert gaps["present_vs_rescaled_loss_gap"] > 0
    # ... and flips the value-of-reps signal positive
    assert gaps["rescaled_vs_zeroed_loss_gap"] > 0


def test_there_is_exactly_one_rescale_implementation() -> None:
    """Guards against the copy re-appearing in a trainer or probe."""
    import inspect

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer
    src = inspect.getsource(KRKBTrainer._rescale_to_embed_norm)
    assert "rescale_to_embed_norm" in src
    # it must DELEGATE, not recompute the norm ratio itself
    assert "norm(dim=-1, keepdim=True)" not in src
