"""A trainer that splices reps must report whether the reps do anything.

THE FAILURE THIS EXISTS TO PREVENT. `summarization_round_robin` ran 52,000
steps splicing compressed reps into a decoder while containing ZERO
occurrences of ablation/zeroed/rep_gain, and its checkpoint records
``metrics: null``. Nobody knew whether the reps contributed. When a downstream
Phase-2 regression appeared, it was diagnosed for a day against a baseline
that had never actually been measured — and the eventual measurement showed
the baseline was ~0 too, so the "regression" was not one.

A silent gap is indistinguishable from a healthy pipeline. So the absence has
to be NOISY. The check lives in ``_normalize_eval_metrics`` because that is the
single chokepoint both eval call sites pass through: one check covers every
trainer, instead of five bespoke edits that would drift apart.
"""

from __future__ import annotations

import structlog

from bgkit.training.base_trainer import BaseTrainer


def _norm_and_check(trainer, metrics):
    """Both steps, as the eval path does them: pure normalise, then check."""
    out = trainer._normalize_eval_metrics(metrics)
    trainer._check_rep_dependence_reported(out)
    return out


class _Quiet(BaseTrainer):
    """Splices reps, reports nothing — the summarization_round_robin shape."""

    SPLICES_REPS = True

    def __init__(self) -> None:  # bypass BaseTrainer.__init__ entirely
        pass

    # BaseTrainer is abstract; these exist only to make it instantiable.
    def setup(self) -> None: ...
    def _forward_backward(self, batch):
        ...
    def evaluate(self) -> dict[str, float]:
        return {}


class _Reports(_Quiet):
    SPLICES_REPS = True


class _NoReps(_Quiet):
    SPLICES_REPS = False


def test_splicing_trainer_with_no_rep_metric_warns() -> None:
    with structlog.testing.capture_logs() as logs:
        _norm_and_check(_Quiet(), {"loss": 1.0, "exact_match": 0.5})
    assert [x for x in logs if x["event"] == "rep_dependence_unmeasured"]


def test_rep_gain_metric_satisfies_the_check() -> None:
    with structlog.testing.capture_logs() as logs:
        _norm_and_check(_Reports(), {"loss": 1.0, "rep_gain/nats": 0.01})
    assert not [x for x in logs if x["event"] == "rep_dependence_unmeasured"]


def test_ablation_metric_also_satisfies_it() -> None:
    """Either form counts — some trainers report an ablation arm rather than a
    derived gain, and requiring one exact key name would just be brittle."""
    with structlog.testing.capture_logs() as logs:
        _Reports()._normalize_eval_metrics(
            {"loss": 1.0, "ablation/zeroed/loss": 1.2},
        )
    assert not [x for x in logs if x["event"] == "rep_dependence_unmeasured"]


def test_trainers_that_do_not_splice_are_not_nagged() -> None:
    """Projection warmups, pruning distillation etc. have no reps to measure;
    warning there would train people to ignore the warning."""
    with structlog.testing.capture_logs() as logs:
        _norm_and_check(_NoReps(), {"loss": 1.0})
    assert not [x for x in logs if x["event"] == "rep_dependence_unmeasured"]


def test_prefixing_behaviour_is_unchanged() -> None:
    """The check is additive — the method's original job must still work."""
    out = _norm_and_check(_NoReps(), {"loss": 1.0, "eval/already": 2.0})
    assert out == {"eval/loss": 1.0, "eval/already": 2.0}


def test_every_rep_splicing_trainer_declares_the_flag() -> None:
    """The claim and the check must stay in sync: any trainer that hands reps
    to a decoder has to opt in, or the check silently covers nothing."""
    import importlib
    import inspect

    expected = {
        "bgkit.training.phase1.summarization_round_robin": "SummarizationRoundRobinTrainer",
        "bgkit.training.phase1.decoder_init": "DecoderInitTrainer",
        "bgkit.training.phase1.projection_repair": "ProjectionRepairTrainer",
        "bgkit.training.phase1.commit_encoding": "CommitEncodingTrainer",
        "bgkit.training.phase2.kr_kb_trainer": "KRKBTrainer",
        "bgkit.training.blob_sft_trainer": "BlobSFTTrainer",
        "bgkit.training.phase3.distillation_trainer": "DistillationTrainer",
    }
    for mod_name, cls_name in expected.items():
        mod = importlib.import_module(mod_name)
        cls = getattr(mod, cls_name)
        assert inspect.isclass(cls)
        assert cls.SPLICES_REPS is True, f"{cls_name} splices reps but did not declare it"
