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
