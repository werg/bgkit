"""A reps score is uninterpretable without a floor and a ceiling beside it.

v8's pooled teacher-forced token-F1 was 0.386, which reads as a mediocre but
working model. Its zeroed floor was 0.386 and its full-text ceiling 0.752:
fraction captured ~0. That comparison had to be assembled by hand from three
separate invocations, which is also how it went un-run for weeks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("torch")

_SPEC = importlib.util.spec_from_file_location(
    "eval_phase2_kb",
    Path(__file__).resolve().parents[3] / "scripts" / "eval_phase2_kb.py",
)
ev = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(ev)


def _rows(dataset: str, f1s: list[float], ems: list[float] | None = None):
    ems = ems if ems is not None else [0.0] * len(f1s)
    return [
        {
            "dataset": dataset,
            "gold_answer": "g",
            "answer_token_f1": f1,
            "answer_exact_match": em,
        }
        for f1, em in zip(f1s, ems, strict=True)
    ]


def test_fraction_captured_is_the_share_of_the_headroom():
    arms = {
        "zeroed": _rows("grepset", [0.2, 0.2]),
        "none": _rows("grepset", [0.4, 0.4]),
        "full_text": _rows("grepset", [0.6, 0.6]),
    }
    cell = ev._ceiling_table(arms)["grepset"]["kb/answer_token_f1"]
    assert cell["headroom"] == pytest.approx(0.4)
    assert cell["fraction_captured"] == pytest.approx(0.5)


def test_no_headroom_reports_none_not_a_ratio():
    """The v8 case: the ceiling arm scores no better than the floor, so there
    is nothing to capture a fraction OF. Dividing by ~0 there produces a huge
    number of arbitrary sign and would read as a triumph or a disaster."""
    arms = {
        "zeroed": _rows("fileneedle", [0.5]),
        "none": _rows("fileneedle", [0.52]),
        "full_text": _rows("fileneedle", [0.5]),
    }
    cell = ev._ceiling_table(arms)["fileneedle"]["kb/answer_token_f1"]
    assert cell["fraction_captured"] is None


def test_a_reps_arm_that_beats_neither_floor_reports_a_negative_fraction():
    """Below the floor is real information and must not be clipped to 0."""
    arms = {
        "zeroed": _rows("lognav", [0.4]),
        "none": _rows("lognav", [0.3]),
        "full_text": _rows("lognav", [0.8]),
    }
    cell = ev._ceiling_table(arms)["lognav"]["kb/answer_token_f1"]
    assert cell["fraction_captured"] < 0


def test_table_is_split_per_dataset_and_overall():
    arms = {
        name: _rows("a", [lo, lo]) + _rows("b", [hi, hi])
        for name, lo, hi in (("zeroed", 0.0, 0.0), ("none", 0.1, 0.9), ("full_text", 1.0, 1.0))
    }
    table = ev._ceiling_table(arms)
    assert set(table) == {"overall", "a", "b"}
    assert table["a"]["kb/answer_token_f1"]["fraction_captured"] == pytest.approx(0.1)
    assert table["b"]["kb/answer_token_f1"]["fraction_captured"] == pytest.approx(0.9)
    # The pooled number sits between them and hides both -- the reason the
    # per-dataset split is not optional.
    assert table["overall"]["kb/answer_token_f1"]["fraction_captured"] == pytest.approx(0.5)


def test_table_is_empty_unless_all_three_arms_ran():
    assert ev._ceiling_table({"none": _rows("a", [0.5])}) == {}
    assert ev._ceiling_table(
        {"none": _rows("a", [0.5]), "zeroed": _rows("a", [0.1])}
    ) == {}


def test_exact_match_is_reported_alongside_f1():
    """F1 gives partial credit for naming one member of a set-valued answer,
    so a family can look like it is reading while getting the answer wrong."""
    arms = {
        "zeroed": _rows("grepset", [0.3], [0.0]),
        "none": _rows("grepset", [0.6], [0.0]),
        "full_text": _rows("grepset", [0.9], [1.0]),
    }
    entry = ev._ceiling_table(arms)["grepset"]
    assert entry["kb/answer_token_f1"]["fraction_captured"] == pytest.approx(0.5)
    assert entry["kb/answer_exact_match"]["fraction_captured"] == pytest.approx(0.0)


def test_samples_without_a_gold_answer_do_not_dilute_the_average():
    no_gold = {"dataset": "a", "gold_answer": ""}
    arms = {
        "zeroed": [*_rows("a", [0.0]), no_gold],
        "none": [*_rows("a", [0.5]), no_gold],
        "full_text": [*_rows("a", [1.0]), no_gold],
    }
    cell = ev._ceiling_table(arms)["a"]["kb/answer_token_f1"]
    assert cell["none"] == pytest.approx(0.5)
    assert cell["fraction_captured"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The scoring loop the sweep runs once per arm
# ---------------------------------------------------------------------------


class _StubDecoders:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


class _StubModel:
    def __init__(self):
        self.modes = []

    def eval(self):
        self.modes.append("eval")

    def train(self):
        self.modes.append("train")


class _StubSample:
    def __init__(self, name, question):
        self.dataset_name = name
        self.question = question


class _StubTrainer:
    _round_robin = False

    def __init__(self, samples):
        self.model = _StubModel()
        self.eval_dataloader = [samples[:2], samples[2:]]
        self._decoder_family = "qwen35"
        self.ablations = []
        self.cleared = 0

    def _teacher_forced_decoders(self):
        return _StubDecoders()

    def set_ablation_mode(self, mode):
        self.ablations.append(mode)

    def _clear_eval_shared_tree(self):
        self.cleared += 1


def _stub_scoring(monkeypatch, marker="x"):
    monkeypatch.setattr(
        ev, "evaluate_sample",
        lambda _t, s: {"gold_answer": "g", "answer_token_f1": 0.5,
                       "answer_exact_match": 0.0, "pred_answer": marker},
    )
    monkeypatch.setattr(
        ev, "evaluate_free_running_sample",
        lambda _t, s, **_kw: {"answer_token_f1": 0.1, "answer_exact_match": 0.0},
    )


def test_score_samples_scores_every_sample_it_is_given(monkeypatch):
    _stub_scoring(monkeypatch)
    samples = [_StubSample("a", f"q{i}") for i in range(4)]
    rows = ev._score_samples(
        _StubTrainer(samples), samples, free_running_limit=0,
        max_tool_calls=1, max_new_tokens=8, force_first_call=False,
    )
    assert len(rows) == 4
    assert [r["question"] for r in rows] == ["q0", "q1", "q2", "q3"]
    assert all("free_running" not in r for r in rows)


def test_generation_is_capped_at_the_limit(monkeypatch):
    _stub_scoring(monkeypatch)
    samples = [_StubSample("a", f"q{i}") for i in range(5)]
    rows = ev._score_samples(
        _StubTrainer(samples), samples, free_running_limit=2,
        max_tool_calls=1, max_new_tokens=8, force_first_call=False,
    )
    assert sum("free_running" in r for r in rows) == 2


def test_collect_eval_samples_caps_and_flattens_batches():
    samples = [_StubSample("a", f"q{i}") for i in range(4)]
    t = _StubTrainer(samples)
    assert len(ev._collect_eval_samples(t, 3)) == 3
    assert len(ev._collect_eval_samples(t, 99)) == 4


def test_every_arm_scores_the_same_sample_list(monkeypatch):
    """The point of materialising the samples once. If each arm re-drew from
    the dataloader, a ceiling would be partly a difference in the draw."""
    _stub_scoring(monkeypatch)
    samples = [_StubSample("a", f"q{i}") for i in range(4)]
    trainer = _StubTrainer(samples)
    arms = {}
    for mode in ("", "zeroed", "full_text"):
        trainer.set_ablation_mode(mode or None)
        arms[mode or "none"] = ev._score_samples(
            trainer, samples, free_running_limit=0,
            max_tool_calls=1, max_new_tokens=8, force_first_call=False,
        )
    assert trainer.ablations == [None, "zeroed", "full_text"]
    questions = {tuple(r["question"] for r in rows) for rows in arms.values()}
    assert len(questions) == 1
    assert ev._ceiling_table(arms)["a"]["kb/answer_token_f1"][
        "fraction_captured"
    ] is None  # identical arms => no headroom, and no invented ratio


def test_decoder_family_follows_the_sample_index_not_a_running_counter(monkeypatch):
    """An arm that scores fewer samples must not shift which family each
    sample gets, or the arms stop being comparable."""
    _stub_scoring(monkeypatch)
    seen = []

    class _RR(_StubTrainer):
        _round_robin = True

        def _set_active_decoder(self, family):
            seen.append(family)
            self._decoder_family = family

        def _eval_family_for_index(self, i):
            return "qwen35" if i % 2 == 0 else "falcon_h1"

    samples = [_StubSample("a", f"q{i}") for i in range(4)]
    ev._score_samples(
        _RR(samples), samples[:2], free_running_limit=0,
        max_tool_calls=1, max_new_tokens=8, force_first_call=False,
    )
    ev._score_samples(
        _RR(samples), samples, free_running_limit=0,
        max_tool_calls=1, max_new_tokens=8, force_first_call=False,
    )
    assert seen == ["qwen35", "falcon_h1"] + ["qwen35", "falcon_h1"] * 2


def test_the_shared_eval_tree_is_cleared_after_each_arm(monkeypatch):
    _stub_scoring(monkeypatch)
    samples = [_StubSample("a", "q")]
    t = _StubTrainer(samples)
    ev._score_samples(t, samples, free_running_limit=0, max_tool_calls=1,
                      max_new_tokens=8, force_first_call=False)
    ev._score_samples(t, samples, free_running_limit=0, max_tool_calls=1,
                      max_new_tokens=8, force_first_call=False)
    assert t.cleared == 2
