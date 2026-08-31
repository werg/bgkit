"""rep_gain must not read zero for a reason unrelated to rep dependence.

Measured on v10 at step 250: the `reconstruct` family scored exact_match
0.000 in BOTH the reps arm and the zeroed arm -- reproducing a 266-character
span exactly is brutal early in training -- so its EM gain was 0.000 by
construction while token_f1 sat at 0.382. A metric that reads zero for a
reason unrelated to the phenomenon is how this project has repeatedly lost
months, so the gain is computed on both metrics.
"""

from __future__ import annotations

import pytest


def _gains(metrics: dict, datasets: list[str]) -> dict:
    """The computation under test, extracted from KRKBTrainer.evaluate."""
    out = {}
    for ds in datasets:
        for metric in ("exact_match", "token_f1"):
            base = metrics.get(f"eval/{ds}/{metric}")
            zeroed = metrics.get(f"eval/ablation/zeroed/{ds}/{metric}")
            if base is not None and zeroed is not None:
                out[f"eval/rep_gain/{ds}/{metric}"] = base - zeroed
    return out


def test_a_family_at_zero_exact_match_still_reports_a_token_f1_gain():
    """The v10 case exactly."""
    m = {
        "eval/reconstruct/exact_match": 0.0,
        "eval/ablation/zeroed/reconstruct/exact_match": 0.0,
        "eval/reconstruct/token_f1": 0.382,
        "eval/ablation/zeroed/reconstruct/token_f1": 0.190,
    }
    g = _gains(m, ["reconstruct"])
    assert g["eval/rep_gain/reconstruct/exact_match"] == 0.0
    assert g["eval/rep_gain/reconstruct/token_f1"] == pytest.approx(0.192)


def test_both_metrics_are_emitted_for_every_dataset():
    m = {}
    for ds in ("lognav", "grepset"):
        for metric, base, zero in (("exact_match", 0.3, 0.2), ("token_f1", 0.5, 0.4)):
            m[f"eval/{ds}/{metric}"] = base
            m[f"eval/ablation/zeroed/{ds}/{metric}"] = zero
    g = _gains(m, ["lognav", "grepset"])
    assert set(g) == {
        "eval/rep_gain/lognav/exact_match", "eval/rep_gain/lognav/token_f1",
        "eval/rep_gain/grepset/exact_match", "eval/rep_gain/grepset/token_f1",
    }


def test_a_missing_ablation_arm_emits_nothing_rather_than_a_bogus_zero():
    """Without the zeroed arm there IS no gain, and reporting 0.0 would read
    as 'the reps do not help' -- which is how the capability decayed
    unnoticed for months with eval_ablation_modes empty."""
    m = {"eval/lognav/exact_match": 0.3, "eval/lognav/token_f1": 0.5}
    assert _gains(m, ["lognav"]) == {}


def test_a_negative_gain_is_preserved():
    """Reps actively hurting is real information and must not be clipped."""
    m = {
        "eval/fileneedle/exact_match": 0.10,
        "eval/ablation/zeroed/fileneedle/exact_match": 0.13,
        "eval/fileneedle/token_f1": 0.30,
        "eval/ablation/zeroed/fileneedle/token_f1": 0.31,
    }
    g = _gains(m, ["fileneedle"])
    assert g["eval/rep_gain/fileneedle/exact_match"] == pytest.approx(-0.03)
    assert g["eval/rep_gain/fileneedle/token_f1"] == pytest.approx(-0.01)


def test_a_dataset_absent_from_the_eval_split_is_skipped():
    assert _gains({"eval/lognav/exact_match": 0.3}, ["reconstruct"]) == {}


def test_the_trainer_computes_both_metrics():
    """Pins the loop to the source, so dropping token_f1 fails here."""
    import inspect

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    src = inspect.getsource(KRKBTrainer.evaluate)
    assert 'for metric in ("exact_match", "token_f1")' in src
    assert '"eval/rep_gain/token_f1"' in src
