"""Teacher-forced eval runs the decoders in train mode (Falcon-H1's eval-mode
Mixer path is numerically different), restores their previous mode, and the
eval pass reports per-decoder-family metrics (2026-08-22)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_teacher_forced_decoders_toggles_and_restores():
    t = KRKBTrainer.__new__(KRKBTrainer)
    qwen, falcon = torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)
    qwen.eval()
    falcon.eval()
    t.decoder = qwen
    t._decoders_by_family = {"qwen35": qwen, "falcon_h1": falcon}
    with t._teacher_forced_decoders():
        assert qwen.training and falcon.training
    assert not qwen.training and not falcon.training
    qwen.train()
    with t._teacher_forced_decoders():
        assert qwen.training and falcon.training
    assert qwen.training and not falcon.training  # each restored to its own state


def test_eval_pass_reports_per_family_metrics():
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.step_cfg = {}
    t._round_robin = True
    t._decoder_family = None
    t._set_active_decoder = lambda fam: setattr(t, "_decoder_family", fam)
    t._ensure_eval_shared_tree = lambda sample: None
    s = [SimpleNamespace(dataset_name="toy", gold_answer="a") for _ in range(4)]
    t.eval_dataloader = [s[:2], s[2:]]
    results = {
        "qwen35": {"loss": 1.0, "tokens": 10, "correct": 5, "answer_correct": 2,
                   "answer_tokens": 4, "em": 1.0, "f1": 0.8, "tool_call_id": None},
        "falcon_h1": {"loss": 9.0, "tokens": 10, "correct": 1, "answer_correct": 0,
                      "answer_tokens": 4, "em": 0.0, "f1": 0.1, "tool_call_id": None},
    }
    t._eval_one_sample = lambda sample, accum: dict(results[t._decoder_family])
    m = t._eval_pass()
    assert m["eval/loss"] == pytest.approx(5.0)
    assert m["eval/family/qwen35/loss"] == pytest.approx(1.0)
    assert m["eval/family/falcon_h1/loss"] == pytest.approx(9.0)
    assert m["eval/family/qwen35/token_f1"] == pytest.approx(0.8)
    assert m["eval/family/falcon_h1/n_samples"] == 2.0
    # Explicit loader override (the train-subset probe): same pass, other source.
    m2 = t._eval_pass(loader=[s[:1]])
    assert m2["eval/n_samples"] == 1.0


def test_train_subset_batches_are_fixed_and_evenly_spaced():
    t = KRKBTrainer.__new__(KRKBTrainer)
    t.train_dataset = list(range(100))
    b = t._train_subset_batches(4)
    assert b == [[0], [25], [50], [75]]
    assert t._train_subset_batches(4) == b  # deterministic
    t.train_dataset = list(range(3))
    assert t._train_subset_batches(10) == [[0], [1], [2]]
    assert t._train_subset_batches(0) == []


def test_free_running_pass_reports_per_dataset_and_invalid_reasons(monkeypatch):
    import bgkit.eval.kb_trajectory_eval as kte

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.step_cfg = {}
    t._round_robin = False
    t._decoder_family = "qwen35"
    samples = [
        SimpleNamespace(dataset_name="fileneedle", gold_answer="a"),
        SimpleNamespace(dataset_name="lognav", gold_answer="b"),
        SimpleNamespace(dataset_name="fileneedle", gold_answer="c"),
    ]
    t.eval_dataloader = [[s] for s in samples]
    outcomes = {
        "a": {"answer_exact_match": 1.0, "answer_token_f1": 1.0, "invalid_reason": ""},
        "b": {"answer_exact_match": 0.0, "answer_token_f1": 0.0,
              "invalid_reason": "malformed_tool_call", "raw_text": "<tool_call>oops"},
        "c": {"answer_exact_match": 0.0, "answer_token_f1": 0.5, "invalid_reason": ""},
    }
    monkeypatch.setattr(
        kte, "evaluate_free_running_sample",
        lambda trainer, sample, **kw: dict(outcomes[sample.gold_answer]),
    )
    m = t._eval_free_running_pass(8)
    p = "eval/kb/free_running"
    assert m[f"{p}/n_samples"] == 3.0
    assert m[f"{p}/answer_exact_match"] == pytest.approx(1 / 3)
    assert m[f"{p}/fileneedle/answer_exact_match"] == pytest.approx(0.5)
    assert m[f"{p}/fileneedle/n_samples"] == 2.0
    assert m[f"{p}/lognav/answer_exact_match"] == 0.0
    assert m[f"{p}/qwen35/answer_token_f1"] == pytest.approx(0.5)
    assert m[f"{p}/invalid_rate"] == pytest.approx(1 / 3)
    assert m[f"{p}/invalid/malformed_tool_call"] == pytest.approx(1 / 3)


def test_free_running_probe_failure_does_not_propagate():
    """A defect in the diagnostic probe must not kill training."""
    t = KRKBTrainer.__new__(KRKBTrainer)

    def boom(n):
        raise RuntimeError("probe defect")

    t._eval_free_running_pass = boom
    assert t._free_running_metrics_guarded(4) == {"eval/kb/free_running/failed": 1.0}
    t._eval_free_running_pass = lambda n: {"eval/kb/free_running/n_samples": float(n)}
    m = t._free_running_metrics_guarded(4)
    assert m["eval/kb/free_running/failed"] == 0.0 and m["eval/kb/free_running/n_samples"] == 4.0


def test_free_running_pass_slices_evenly_across_the_eval_set(monkeypatch):
    """The eval set is index-sorted by dataset; a bounded pass must stride
    across it, not take the first N (2026-08-23: 64/64 lognav)."""
    import bgkit.eval.kb_trajectory_eval as kte

    t = KRKBTrainer.__new__(KRKBTrainer)
    t.step_cfg = {}
    t._round_robin = False
    t._decoder_family = "qwen35"
    names = ["lognav"] * 40 + ["fileneedle"] * 40 + ["grepset"] * 20 + ["swerecall"] * 60
    t.eval_dataloader = [[SimpleNamespace(dataset_name=n, gold_answer="")] for n in names]
    monkeypatch.setattr(
        kte, "evaluate_free_running_sample",
        lambda trainer, sample, **kw: {"answer_exact_match": 0.0, "invalid_reason": ""},
    )
    m = t._eval_free_running_pass(16)
    p = "eval/kb/free_running"
    assert m[f"{p}/n_samples"] == 16.0
    assert m[f"{p}/lognav/n_samples"] == 4.0
    assert m[f"{p}/fileneedle/n_samples"] == 4.0
    assert m[f"{p}/grepset/n_samples"] == 2.0
    assert m[f"{p}/swerecall/n_samples"] == 6.0


def test_eval_family_follows_training_mix():
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._qwen_decoder_prob = 1.0
    assert [t._eval_family_for_index(i) for i in range(4)] == ["qwen35"] * 4
    t._qwen_decoder_prob = 0.0
    assert [t._eval_family_for_index(i) for i in range(2)] == ["falcon_h1"] * 2
    t._qwen_decoder_prob = 0.8
    assert [t._eval_family_for_index(i) for i in range(4)] == [
        "qwen35", "falcon_h1", "qwen35", "falcon_h1",
    ]


def test_oracle_span_requires_live_l0():
    """oracle_span forces span survival through the LIVE L0 selection; the
    cached path (pre-baked rows) must fail loudly, never silently no-op."""
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._live_l0 = False
    t._ablation_mode = KRKBTrainer.ABLATION_ORACLE_SPAN
    with pytest.raises(RuntimeError, match="oracle_span"):
        t._l0_for_articles("toy", ["a1"])


def test_oracle_span_leaves_rep_vectors_untouched():
    """oracle_span changes WHICH positions survive, not the vectors — the
    context-ablation transform must pass reps through unchanged."""
    t = KRKBTrainer.__new__(KRKBTrainer)
    t._ablation_mode = KRKBTrainer.ABLATION_ORACLE_SPAN
    reps = torch.randn(3, 4)
    assert t._apply_context_ablation(reps, skip=False) is reps
