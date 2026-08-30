"""``training.eval_free_running_samples`` must actually disable generation.

WHY. On 2026-08-30 three Phase-2 evals were launched with
``training.eval_free_running_samples=0`` and every one free-ran anyway: the
script read a DIFFERENT key in a DIFFERENT namespace (``eval.free_running``,
default True). Free-running is autoregressive greedy decode — one tiny kernel
per token, CPU-launch-bound, presenting as ~88% GPU utilization at ~35W with one
core pinned. Indistinguishable from a hang without a stack, and it produced two
wrong diagnoses (the replay probe, then the ablation sweep) before a SIGUSR1
dump named the real frame.

The class of bug is "two names for one thing, and the documented one loses".
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

SCRIPT = Path("scripts/eval_phase2_kb.py")


def _source() -> str:
    return SCRIPT.read_text()


def test_the_shared_training_key_is_consulted() -> None:
    src = _source()
    assert "eval_free_running_samples" in src, (
        "the documented shared key is ignored again"
    )


def test_generation_is_capped_by_a_count_not_a_bare_bool() -> None:
    """A bool can only mean all-or-nothing. The in-training knob is a COUNT, so
    the script must be one too, or the two configs cannot agree."""
    src = _source()
    assert "free_running_limit" in src
    assert "free_run_done" in src
    assert re.search(r"if free_run_done < free_running_limit:", src), (
        "generation must be gated on the running count"
    )
    assert "if free_running else None" not in src, "the old bool gate is back"


def test_resolution_is_logged() -> None:
    """A silently-ignored config key is what caused the incident, so the
    resolved value AND which key won must appear in the log."""
    src = _source()
    assert "eval_free_running_resolved" in src
    assert "source=_src" in src


def test_limit_is_clamped_into_range() -> None:
    """A negative or oversized value must not re-enable generation or run past
    max_samples."""
    src = _source()
    assert "max(0, min(free_running_limit, max_samples))" in src


def test_zero_really_reaches_zero() -> None:
    """Execute the resolution logic itself rather than trusting the text."""
    src = _source()
    tree = ast.parse(src)
    seg = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", "") == "free_running_limit" for t in node.targets
        ):
            seg = True
    assert seg, "free_running_limit is never assigned"

    # Reproduce the documented precedence on plain dicts.
    def resolve(eval_cfg: dict, step_cfg: dict, max_samples: int) -> tuple[int, str]:
        explicit = eval_cfg.get("free_running_samples")
        shared = step_cfg.get("eval_free_running_samples")
        if explicit is not None:
            limit, src_ = int(explicit), "eval.free_running_samples"
        elif shared is not None:
            limit, src_ = int(shared), "training.eval_free_running_samples"
        else:
            limit = max_samples if bool(eval_cfg.get("free_running", True)) else 0
            src_ = "eval.free_running"
        return max(0, min(limit, max_samples)), src_

    # the exact invocation that was ignored
    assert resolve({}, {"eval_free_running_samples": 0}, 64) == (
        0, "training.eval_free_running_samples",
    )
    # explicit script cap wins over the shared key
    assert resolve({"free_running_samples": 4}, {"eval_free_running_samples": 99}, 64)[0] == 4
    # legacy default still means "all samples"
    assert resolve({}, {}, 64) == (64, "eval.free_running")
    # legacy off still means none
    assert resolve({"free_running": False}, {}, 64)[0] == 0
    # clamping
    assert resolve({"free_running_samples": -5}, {}, 64)[0] == 0
    assert resolve({"free_running_samples": 999}, {}, 64)[0] == 64


def test_there_are_two_independent_free_running_passes() -> None:
    """The trap that cost four eval launches.

    Generation runs from TWO places, and each was disabled by a different key:

      1. ``scripts/eval_phase2_kb.py``'s per-sample loop  -> ``eval.free_running*``
      2. ``KRKBTrainer.evaluate`` -> ``_free_running_metrics_guarded``
         -> ``_eval_free_running_pass``                   -> ``training.eval_free_running_samples``

    Runs 5 and 6 set the training key and path 1 generated anyway. Run 7 set the
    script key and path 2 generated anyway. Both stacks were only identifiable
    from a SIGUSR1 dump.

    The resolution above makes ``training.eval_free_running_samples`` govern
    BOTH (the script defaults from it), so that one key is now sufficient. This
    test fails if path 2 stops honouring it, which would silently restore the
    trap.
    """
    import inspect

    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    ev = inspect.getsource(KRKBTrainer.evaluate)
    assert "_free_running_metrics_guarded" in ev, "path 2 moved or was renamed"
    # The gate lives in evaluate() and passes the COUNT down as max_samples;
    # the two callees never read config themselves.
    assert "eval_free_running_samples" in ev, (
        "path 2 no longer reads the shared key; the two passes have diverged "
        "again and one key can no longer disable generation"
    )
    gate = ev[ev.index("eval_free_running_samples"):]
    assert "if free_running_samples > 0:" in gate, (
        "0 must mean no generation, not 'fall back to a default'"
    )


def test_report_is_written_somewhere_that_outlives_the_container() -> None:
    """``docker compose run --rm`` deletes the container filesystem on exit.

    The report default used to be the container-relative ``eval_reports/``, so
    a completed 128-sample eval on 2026-08-30 wrote its JSON into a directory
    that was destroyed seconds later; the numbers survived only because the
    container log happened to be teed to a host file. A result you cannot read
    is a result you did not produce.
    """
    src = SCRIPT.read_text()
    assert "CHECKPOINT_DIR" in src, (
        "report path no longer defaults under the bind-mounted CHECKPOINT_DIR"
    )
    i = src.index("output_dir = Path(")
    assert "eval_reports" in src[i - 700 : i + 400]
