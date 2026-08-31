"""The L0->L1 bridge output scale must not DRIFT from its operating point.

WHY THIS CONTRACT IS LOAD-BEARING. L1's backbone is a deepcopy of L0's, so it
expects INPUT-EMBEDDING-distributed inputs; ``l0.auto_reproduce`` exists solely
to convert L0's last-block hidden states into that space. In the Phase-2 path
L1 ALSO receives pinned article-ID embeddings taken straight from
``embed_tokens``, which are correctly left un-bridged because they are already
in the bridge's codomain. That makes the two streams share one sequence — so if
the bridge MOVES, L1 starts reading one sequence written in two scales.

CORRECTED 2026-08-29. The first version of this guard banded the ratio near
1.0, reading "maps back to input-embedding space" literally. Measured, the
bridge runs at ~502x embed norm on the summarization base and ~636x in the live
Phase-2 run — and the base was TRAINED there, so that IS the operating point
and an absolute band flags the design rather than a fault. The decoder's own
_maybe_guard_spliced_rep_norm already banded around a REFERENCE; this now does
the same, keeping only "collapsed toward zero" as an absolute check.

Nothing checked this. The decoder has ``_maybe_guard_spliced_rep_norm`` at the
SPLICE site, but the L1 INPUT was unguarded, while the lineage weight-delta
shows ``l0.auto_repro_head`` moved 0.0170 across Phase 2 — the third-largest
mover after the two survivorship heads. A drifting adapter with no contract
check is the shape-correct/scale-wrong handoff that keeps recurring here.
"""

from __future__ import annotations

import structlog
import torch

from bgkit.models import encoder as encoder_mod
from bgkit.models.encoder import guard_bridge_output_scale


def _embed(n: int = 512, d: int = 64) -> torch.Tensor:
    return torch.randn(n, d) * 0.5


def _reset() -> None:
    encoder_mod._bridge_guard_calls.clear()
    encoder_mod._bridge_guard_reference.clear()


def test_first_call_establishes_the_reference() -> None:
    _reset()
    emb = _embed()
    with structlog.testing.capture_logs() as logs:
        guard_bridge_output_scale(emb[:50].clone(), emb, site="ok", every=1)
    assert len(logs) == 1
    assert logs[0]["event"] == "bridge_output_scale"
    assert logs[0]["log_level"] == "info"
    assert abs(logs[0]["ratio"] - logs[0]["reference_ratio"]) < 1e-6


def test_large_ratio_is_not_a_defect_when_it_is_the_operating_point() -> None:
    """The correction that produced this design.

    The first version asserted the ratio should sit near 1.0, reading
    CLAUDE.md's "maps back to input-embedding space" literally. Measured
    2026-08-29 the bridge runs at ~502x on the summarization base and ~636x in
    the live Phase-2 run — and the base was TRAINED there, so L1's weights are
    adapted to it. An absolute band flags the design, not a fault.
    """
    _reset()
    emb = _embed()
    with structlog.testing.capture_logs() as logs:
        guard_bridge_output_scale(emb[:50] * 502.0, emb, site="op", every=1)
        guard_bridge_output_scale(emb[:50] * 502.0, emb, site="op", every=1)
    assert all(x["event"] == "bridge_output_scale" for x in logs)


def test_drift_from_the_reference_warns() -> None:
    """What the guard is actually for: the bridge MOVING relative to where
    this lineage operates."""
    _reset()
    emb = _embed()
    with structlog.testing.capture_logs() as logs:
        guard_bridge_output_scale(emb[:50] * 500.0, emb, site="d", every=1)
        guard_bridge_output_scale(emb[:50] * 5000.0, emb, site="d", every=1)
    assert logs[0]["event"] == "bridge_output_scale"
    assert logs[1]["event"] == "bridge_output_scale_out_of_band"
    assert logs[1]["drift"] > 2.0


def test_collapsed_bridge_warns_absolutely() -> None:
    """A collapse toward zero is broken at ANY operating point, so it is
    checked absolutely rather than against the reference — otherwise a run
    that starts collapsed would establish "collapsed" as normal."""
    _reset()
    emb = _embed()
    with structlog.testing.capture_logs() as logs:
        guard_bridge_output_scale(emb[:50] * 1e-6, emb, site="tiny", every=1)
    assert logs[0]["event"] == "bridge_output_scale_out_of_band"
    assert logs[0]["degenerate"] is True


def test_sampled_not_every_call() -> None:
    """It runs inside the training hot path; it must cost ~nothing."""
    _reset()
    emb = _embed()
    with structlog.testing.capture_logs() as logs:
        for _ in range(200):
            guard_bridge_output_scale(emb[:50].clone(), emb, site="s", every=100)
    assert len(logs) == 2


def test_empty_bridge_is_a_noop() -> None:
    """A turn with no L0 survivors must not emit a divide-by-zero ratio."""
    _reset()
    with structlog.testing.capture_logs() as logs:
        guard_bridge_output_scale(torch.zeros(0, 64), _embed(), site="e", every=1)
    assert logs == []


def test_env_disable_is_honoured() -> None:
    """Must be switchable off without editing code, like the decoder's guard."""
    _reset()
    emb = _embed()
    prev = encoder_mod._BRIDGE_GUARD_ENABLED
    try:
        encoder_mod._BRIDGE_GUARD_ENABLED = False
        with structlog.testing.capture_logs() as logs:
            guard_bridge_output_scale(emb[:50] * 200.0, emb, site="off", every=1)
        assert logs == []
    finally:
        encoder_mod._BRIDGE_GUARD_ENABLED = prev


def test_both_bridge_sites_are_guarded() -> None:
    """There are TWO bridge call sites — encoder.forward and KRKBTrainer's
    _run_l1_batch — and they must not be able to drift apart, which is the
    stated reason run_l1_and_project is shared between them.

    Checked through the entry point each site actually calls, not by matching
    the guard function's name in the source: the sites go through
    ``observe_bridge_scale`` so the band is anchored to the checkpoint's
    persisted reference, and a test that pinned the old name would have failed
    on that improvement while a site that quietly stopped guarding would still
    pass. So: each site must call an entry point, and the entry point must
    reach the guard.
    """
    import inspect

    from bgkit.models.encoder import BgKITEncoder
    from bgkit.training.phase2.kr_kb_trainer import KRKBTrainer

    entry_points = ("observe_bridge_scale", "guard_bridge_output_scale")
    for fn in (BgKITEncoder.forward, KRKBTrainer._run_l1_batch):
        src = inspect.getsource(fn)
        assert any(name in src for name in entry_points), fn.__qualname__
    assert "guard_bridge_output_scale" in inspect.getsource(
        BgKITEncoder.observe_bridge_scale
    )


def test_the_wrapper_anchors_on_the_checkpointed_reference() -> None:
    """``observe_bridge_scale`` exists to pass the LINEAGE's reference in. If
    it stopped doing that, the guard would silently return to re-anchoring on
    every process start -- the reason a 484 -> 70985 runaway never fired."""
    import inspect

    from bgkit.models.encoder import BgKITEncoder

    src = inspect.getsource(BgKITEncoder.observe_bridge_scale)
    assert "reference=" in src
    assert "bridge_reference" in src
