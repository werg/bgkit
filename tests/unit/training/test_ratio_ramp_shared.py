"""The compression ramp arithmetic must live in exactly one place.

Five trainers define ``_current_target_ratio`` and all five bodies differ, so
this is NOT five copies of one function — ``pruning_distill`` samples the ratio
randomly, ``commit_encoding`` delegates to a multi-level curriculum. Collapsing
the POLICIES would be wrong. What was genuinely duplicated is the ARITHMETIC:
``decoder_init`` and ``compression`` hand-rolled ``linear_ratio``'s
interpolation, and in doing so dropped its ``max(end, ...)`` clamp — the thing
that enforces the monotone-descent rule (compression may only ever decrease).
"""

from __future__ import annotations

import ast
import inspect
import itertools
import re
from pathlib import Path

import pytest

from bgkit.training.compression_curriculum import linear_ratio
from bgkit.training.phase1.compression import CompressionTrainer
from bgkit.training.phase1.decoder_init import DecoderInitTrainer

START, END, RAMP = 0.95, 0.10, 100


def _compression(step: int) -> float:
    t = object.__new__(CompressionTrainer)
    t._head_warmup_steps = 0
    t._target_ratio_override = None
    t.global_step = step
    t._target_ratio_start, t._target_ratio_end = START, END
    t._target_ratio_ramp_steps = RAMP
    return t._current_target_ratio()


def _decoder_init(step: int, intro: int = 20) -> float:
    t = object.__new__(DecoderInitTrainer)
    t._target_ratio_override = None
    t._compression_active = True
    t._compression_introduction_step = intro
    t.global_step = step
    t._target_ratio_start, t._target_ratio_end = START, END
    t._target_ratio_ramp_steps = RAMP
    return t._current_target_ratio()


@pytest.mark.parametrize("step", [0, 1, 25, 50, 99, 100, 999])
def test_compression_ramp_matches_the_shared_helper(step: int) -> None:
    assert _compression(step) == pytest.approx(
        linear_ratio(step, START, END, RAMP)
    )


@pytest.mark.parametrize("step", [0, 5, 20, 45, 70, 120, 999])
def test_decoder_init_ramp_matches_the_shared_helper(step: int) -> None:
    """Same helper, ramp measured from the introduction step."""
    assert _decoder_init(step) == pytest.approx(
        linear_ratio(step, START, END, RAMP, introduction_step=20)
    )


def test_the_ramp_never_ascends() -> None:
    """The invariant the hand-rolled copies silently lacked. A misconfigured
    start < end must clamp, not invert the curriculum."""
    assert linear_ratio(50, 0.10, 0.95, 100) == pytest.approx(0.95)
    seq = [_compression(s) for s in range(0, 200, 7)]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(seq))


def test_no_hand_rolled_interpolation_survives() -> None:
    """The shape that was duplicated: ``start + t * (end - start)``. If it
    reappears, someone re-forked the ramp instead of extending the helper."""
    pat = re.compile(r"_target_ratio_start\s*\+\s*t\s*\*")
    for f in Path("src/bgkit/training").rglob("*.py"):
        assert not pat.search(f.read_text()), f"hand-rolled ramp back in {f}"


def test_genuinely_different_policies_were_left_alone() -> None:
    """Guards against over-unification: these two are not ramps and must not be
    folded into one. Recorded so a future pass does not 'finish the job'."""
    from bgkit.training.distillation.pruning_distill import PruningDistillTrainer
    from bgkit.training.phase1.commit_encoding import CommitEncodingTrainer

    rnd = inspect.getsource(PruningDistillTrainer._current_target_ratio)
    assert "random.uniform" in rnd, "stochastic ratio policy, not a linear ramp"
    shim = inspect.getsource(CommitEncodingTrainer._current_target_ratio)
    assert "_curriculum_state" in shim, "multi-level curriculum, not a linear ramp"


def test_both_call_sites_import_the_helper() -> None:
    """``linear_ratio`` is referenced inside a method body, so a missing import
    is a NameError at the first TRAINING STEP, not at import — it would survive
    every smoke check. (This exact miss happened while writing this refactor.)"""
    for mod in ("compression", "decoder_init"):
        src = Path(f"src/bgkit/training/phase1/{mod}.py").read_text()
        names = {
            a.name if a.asname is None else a.asname
            for n in ast.walk(ast.parse(src))
            if isinstance(n, ast.ImportFrom) for a in n.names
        }
        assert "linear_ratio" in names, f"{mod}.py uses linear_ratio without importing it"
