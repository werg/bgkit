"""Phase 2 must be able to run L0-only, with no L1 intermediary.

WHY. Every positive rep measurement on the base was L0-ONLY — the probe goes
through ``encoder.forward``, whose ``target_ratio_l1 is None`` branch skips the
bridge and L1 entirely. Rescaled L0 survivors are worth +0.0455 / +0.0496 /
+0.0945 nats at l0 = 0.63 / 0.32 / 0.10. With L1 in the path on wide-net the
same rescale is worth +0.0012 and rep_gain sits at 0.007. L1's own diagnostics
agree it is not contributing: backbone gradient ~35,000x below its head's,
selection at chance (span survival 0.471 against 0.499 retention), weights
moving 0.0228 across an entire phase.

AND IT WAS NOT EXPRESSIBLE. ``_run_l1_batch`` treats ``target_ratio=None`` as
"sample one", not "skip", so L1 was hard-wired into every Phase-2 run — the
configuration that produced every positive result could not be reached by the
Phase-2 trainer at all.

Budget note: two-stage r0 x r1 collapses to single-stage r0, so l0_retention
must be set to the EFFECTIVE budget (0.10 x 0.50 -> 0.05).
"""

from __future__ import annotations

import inspect

from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer


def test_default_is_unchanged_l1_stays_on() -> None:
    """Additive: every existing run must behave exactly as before."""
    src = inspect.getsource(KRKBTrainer.setup)
    assert 'self.step_cfg.get("use_l1", True)' in src


def _l0_only_branch(src: str) -> tuple[int, str]:
    """Locate the L0-only branch and return (offset, its comment-free code).

    Found by SEMANTICS (the guard reads ``_use_l1``) rather than by matching a
    literal condition string. The first version of these tests pinned the exact
    text ``if not getattr(self, "_use_l1", True):`` and broke the moment the
    branch legitimately merged with the SKIP_LEVEL sentinel — a passing-to-
    failing flip on an improvement, which trains you to edit the assert instead
    of reading the diff.
    """
    off = src.index('"_use_l1"')
    off = src.rindex("if ", 0, off)
    body = src[off : src.index("must_keep_l1 = None", off)]
    # The branch comment NAMES what it bypasses; a raw substring check would
    # read that prose as a call.
    return off, "\n".join(
        ln for ln in body.split("\n") if not ln.lstrip().startswith("#")
    )


def test_l0_only_branch_skips_bridge_and_l1() -> None:
    """The branch must bypass auto_reproduce AND run_l1_and_project — going
    through either would defeat the point."""
    _, code = _l0_only_branch(inspect.getsource(KRKBTrainer._run_l1_batch))
    assert "auto_reproduce(" not in code
    assert "run_l1_and_project(" not in code
    assert "get_active_projection_block()" in code


def test_skip_sentinel_and_config_share_one_branch() -> None:
    """``SKIP_LEVEL`` (this call skips L1) and ``use_l1: false`` (the run has no
    L1) must reach the SAME code, not two branches that drift apart. Two ways to
    express one state is how the L0-only path came to be unreachable from the
    Phase-2 trainer in the first place."""
    src = inspect.getsource(KRKBTrainer._run_l1_batch)
    off, _ = _l0_only_branch(src)
    cond = src[off : src.index(":\n", off)]
    assert "is_skip(" in cond and "_use_l1" in cond
    assert src.count('"_use_l1"') == 1, "a second _use_l1 guard means it forked"


def test_l0_only_returns_before_the_l1_work() -> None:
    """It must return from inside the branch, so none of the L1 machinery
    (theta accumulation, _pending_l1_outputs, utility-grad capture) runs."""
    src = inspect.getsource(KRKBTrainer._run_l1_batch)
    branch, _ = _l0_only_branch(src)
    ret = src.index("_per_turn_from_projected", branch)
    l1_work = src.index("must_keep_l1 = None")
    assert branch < ret < l1_work


def test_both_paths_share_one_per_turn_split() -> None:
    """The re-interleaving of None fallbacks and mode-tagged drill survivors
    must exist once. Duplicating it is how two code paths silently diverge —
    the stated reason run_l1_and_project is shared with encoder.forward."""
    src = inspect.getsource(KRKBTrainer._run_l1_batch)
    assert src.count("_per_turn_from_projected(") == 2      # L0-only + L1 path
    helper = inspect.getsource(KRKBTrainer._per_turn_from_projected)
    assert "_resolve_special_survivor" in helper
    assert "zero_fallback" in helper


def test_l0_only_mirrors_the_encoder_reference_branch() -> None:
    """encoder.forward already had the correct L0-only shape; the trainer's
    branch must call the same projection block the same way rather than
    inventing a second convention."""
    from bgkit.models.encoder import BgKITEncoder
    ref = inspect.getsource(BgKITEncoder.forward)
    trainer_src = inspect.getsource(KRKBTrainer._run_l1_batch)
    for token in ("get_active_projection_block", "cu_seqlens=", "position_ids=",
                  "survivor_mask=None"):
        assert token in ref
        assert token in trainer_src


def test_l1_survivor_stash_is_cleared_not_left_stale() -> None:
    """Probes read _last_l1_survivor_mask. In L0-only mode there is no L1
    selection, and leaving a stale mask from a previous batch would let a
    diagnostic report last-batch selection as this batch's."""
    src = inspect.getsource(KRKBTrainer._run_l1_batch)
    head = src[: src.index("must_keep_l1 = None")]
    assert "self._last_l1_survivor_mask = None" in head
