"""The verdict has to be fixed BEFORE the numbers arrive, or it is a story.

This session has twice over-read a single number in a favourable direction
(a shared_frac of 0.26 called "the constant is gone" when the healthy base
sits at 0.927; a rank alarm on one short document). A threshold in code,
committed before the run finishes, is the cheap defence.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "verdict_repdist",
    Path(__file__).resolve().parents[3] / "scripts" / "verdict_repdist.py",
)
verdict = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(verdict)


def _report(tmp_path, **checkpoints) -> str:
    data = {
        name: {
            "reps": {
                "top1": top1, "chance_top1": 0.008,
                "eff_rank_within": rank, "shared_frac_within_doc": 0.9,
            },
        }
        for name, (top1, rank) in checkpoints.items()
    }
    p = tmp_path / "r.json"
    p.write_text(json.dumps(data))
    return str(p)


def _run(monkeypatch, path, treatment="v9"):
    monkeypatch.setattr(
        "sys.argv", ["verdict_repdist.py", path, "--treatment", treatment],
    )
    return verdict.main()


def test_a_checkpoint_near_the_base_passes(tmp_path, monkeypatch, capsys):
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(0.84, 11.0))
    assert _run(monkeypatch, r) == 0
    assert "VERDICT: PASS" in capsys.readouterr().out


def test_a_checkpoint_at_the_collapsed_level_fails(tmp_path, monkeypatch, capsys):
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(0.031, 1.01))
    assert _run(monkeypatch, r) == 1
    out = capsys.readouterr().out
    assert "VERDICT: FAIL" in out
    # And it must say how to tell the two muting routes apart.
    assert "mean_cos_to_corpus" in out


def test_the_middle_is_reported_as_partial_not_as_a_win(tmp_path, monkeypatch, capsys):
    """The failure mode this file exists for: a 0.3 is neither the base nor
    the collapse, and calling it a win is the tempting reading."""
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(0.30, 5.0))
    assert _run(monkeypatch, r) == 1
    out = capsys.readouterr().out
    assert "VERDICT: PARTIAL" in out
    assert "Do not report this as a win" in out


def test_the_threshold_is_half_the_base_and_is_stated(tmp_path, monkeypatch, capsys):
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(0.449, 6.0))
    assert _run(monkeypatch, r) == 0
    assert "pass threshold 0.449" in capsys.readouterr().out


def test_a_pass_does_not_claim_the_task_is_solved(tmp_path, monkeypatch, capsys):
    """Rep-identifiability is necessary, not sufficient -- the whole reason
    the ceiling table exists."""
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(0.9, 12.0))
    _run(monkeypatch, r)
    assert "necessary, not sufficient" in capsys.readouterr().out


def test_a_missing_treatment_is_indeterminate_not_a_pass(tmp_path, monkeypatch, capsys):
    r = _report(tmp_path, base=(0.898, 12.97))
    assert _run(monkeypatch, r) == 2
    assert "INDETERMINATE" in capsys.readouterr().out


def test_an_empty_report_is_indeterminate(tmp_path, monkeypatch, capsys):
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"a": {"error": "boom"}}))
    assert _run(monkeypatch, str(p)) == 2
    assert "INDETERMINATE" in capsys.readouterr().out


def test_every_measured_checkpoint_is_printed_beside_the_verdict(
    tmp_path, monkeypatch, capsys,
):
    r = _report(
        tmp_path, base=(0.898, 12.97), v8_old=(0.031, 1.01), v9_run=(0.8, 10.0),
    )
    _run(monkeypatch, r)
    out = capsys.readouterr().out
    for name in ("base", "v8_old", "v9_run"):
        assert name in out


@pytest.mark.parametrize("top1,expected", [(0.9, 0), (0.449, 0), (0.4, 1), (0.0, 1)])
def test_the_boundary_is_exact(tmp_path, monkeypatch, top1, expected):
    r = _report(tmp_path, base=(0.898, 12.97), v9_run=(top1, 5.0))
    assert _run(monkeypatch, r) == expected
