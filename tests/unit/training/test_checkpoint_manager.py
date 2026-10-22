"""Tests for CheckpointManager."""

import json

from bgkit.training.checkpoint_manager import CheckpointManager


def _make_ckpt(tmp_path, name, step, metrics=None, phase="test", run_name=None):
    """Create a mock checkpoint dir with metadata.json."""
    ckpt = tmp_path / name
    ckpt.mkdir()
    meta = {"phase": phase, "step": step, "epoch": 0, "parent_checkpoint": None}
    if run_name is not None:
        meta["run_name"] = run_name
    if metrics is not None:
        meta["metrics"] = metrics
    (ckpt / "metadata.json").write_text(json.dumps(meta))
    # Add a dummy file to verify deletion
    (ckpt / "model.pt").write_text("data")
    return ckpt


def test_prune_keeps_latest():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgr = CheckpointManager(keep_best=0, keep_latest=2, metric="eval/loss")

        paths = []
        for i in range(5):
            p = _make_ckpt(tmp_path, f"ckpt_{i}", step=i)
            mgr.record(p, i, None)
            paths.append(p)

        deleted = mgr.prune()
        assert len(deleted) == 3
        # Latest 2 should survive
        assert paths[3].exists()
        assert paths[4].exists()
        assert not paths[0].exists()
        assert not paths[1].exists()
        assert not paths[2].exists()


def test_prune_keeps_best():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgr = CheckpointManager(
            keep_best=2, keep_latest=1, metric="eval/loss", lower_is_better=True
        )

        p0 = _make_ckpt(tmp_path, "ckpt_0", 0, {"eval/loss": 0.5})
        mgr.record(p0, 0, {"eval/loss": 0.5})

        p1 = _make_ckpt(tmp_path, "ckpt_1", 1, {"eval/loss": 0.3})
        mgr.record(p1, 1, {"eval/loss": 0.3})

        p2 = _make_ckpt(tmp_path, "ckpt_2", 2, {"eval/loss": 0.8})
        mgr.record(p2, 2, {"eval/loss": 0.8})

        p3 = _make_ckpt(tmp_path, "ckpt_3", 3, {"eval/loss": 0.1})
        mgr.record(p3, 3, {"eval/loss": 0.1})

        deleted = mgr.prune()
        # Best 2: p3 (0.1) and p1 (0.3). Latest 1: p3.
        # So p0 and p2 should be deleted.
        assert p0 not in [d for d in deleted if d == p0] or not p0.exists()
        assert p1.exists()  # best #2
        assert not p2.exists()  # worst
        assert p3.exists()  # best #1 + latest


def test_none_metrics_excluded_from_best():
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        mgr = CheckpointManager(
            keep_best=1, keep_latest=1, metric="eval/loss", lower_is_better=True
        )

        p0 = _make_ckpt(tmp_path, "ckpt_0", 0, {"eval/loss": 0.5})
        mgr.record(p0, 0, {"eval/loss": 0.5})

        p1 = _make_ckpt(tmp_path, "ckpt_1", 1)  # emergency save, no metrics
        mgr.record(p1, 1, None)

        p2 = _make_ckpt(tmp_path, "ckpt_2", 2, {"eval/loss": 0.3})
        mgr.record(p2, 2, {"eval/loss": 0.3})

        mgr.prune()
        # Best 1: p2 (0.3). Latest 1: p2. So p0 and p1 deleted.
        assert not p0.exists()
        assert not p1.exists()
        assert p2.exists()


def test_load_existing(tmp_path):
    _make_ckpt(tmp_path, "ckpt_a", 10, {"eval/loss": 0.2})
    _make_ckpt(tmp_path, "ckpt_b", 20, {"eval/loss": 0.4})
    _make_ckpt(tmp_path, "ckpt_c", 30)

    mgr = CheckpointManager(keep_best=1, keep_latest=1, metric="eval/loss")
    mgr.load_existing(tmp_path)

    assert len(mgr._records) == 3

    mgr.prune()
    # Best 1: ckpt_a (0.2). Latest 1: ckpt_c (last by step order).
    assert (tmp_path / "ckpt_a").exists()
    assert (tmp_path / "ckpt_c").exists()
    assert not (tmp_path / "ckpt_b").exists()


def test_empty_records():
    mgr = CheckpointManager()
    assert mgr.prune() == []


def test_load_existing_nonexistent_dir(tmp_path):
    mgr = CheckpointManager()
    mgr.load_existing(tmp_path / "nonexistent")
    assert len(mgr._records) == 0


def test_load_existing_filters_by_phase(tmp_path):
    """Checkpoints from other phases are ignored, never pruned."""
    _make_ckpt(tmp_path, "ice_step10", 10, {"eval/mse": 0.2}, phase="ice")
    _make_ckpt(tmp_path, "joint_step20", 20, {"eval/mse": 0.4}, phase="joint_block_pretrain")
    _make_ckpt(tmp_path, "ice_step30", 30, {"eval/mse": 0.1}, phase="ice")

    mgr = CheckpointManager(
        keep_best=1, keep_latest=1, metric="eval/mse", phase="ice"
    )
    mgr.load_existing(tmp_path)

    # Only ICE checkpoints loaded
    assert len(mgr._records) == 2

    mgr.prune()
    # Best 1 (ice_step30, 0.1) + latest 1 (ice_step30) → ice_step10 pruned
    assert not (tmp_path / "ice_step10").exists()
    assert (tmp_path / "ice_step30").exists()
    # Joint checkpoint untouched
    assert (tmp_path / "joint_step20").exists()


def test_load_existing_filters_by_run_name_within_phase(tmp_path):
    own_old = _make_ckpt(
        tmp_path, "own_old", 10, phase="phase2_kb", run_name="stage_b",
    )
    own_new = _make_ckpt(
        tmp_path, "own_new", 20, phase="phase2_kb", run_name="stage_b",
    )
    other = _make_ckpt(
        tmp_path, "smoke_newer", 999, phase="phase2_kb", run_name="smoke",
    )

    mgr = CheckpointManager(
        keep_best=0,
        keep_latest=1,
        phase="phase2_kb",
        run_name="stage_b",
    )
    mgr.load_existing(tmp_path)
    mgr.prune()

    assert not own_old.exists()
    assert own_new.exists()
    assert other.exists()


def test_load_existing_sorts_by_step_not_lexical(tmp_path):
    """keep_latest must use step order, not lexicographic path order.

    Checkpoint names contain unpadded step numbers, so lexical sort
    puts step 100 before step 20 (because '1' < '2').
    """
    _make_ckpt(tmp_path, "test_step5", 5)
    _make_ckpt(tmp_path, "test_step100", 100)
    _make_ckpt(tmp_path, "test_step20", 20)

    mgr = CheckpointManager(keep_best=0, keep_latest=1, phase="test")
    mgr.load_existing(tmp_path)

    # Records should be sorted by step: 5, 20, 100
    assert [r.step for r in mgr._records] == [5, 20, 100]

    mgr.prune()
    # Latest 1 by step = step 100
    assert (tmp_path / "test_step100").exists()
    assert not (tmp_path / "test_step5").exists()
    assert not (tmp_path / "test_step20").exists()


def test_keep_latest_zero_prunes_all(tmp_path):
    """keep_latest=0, keep_best=0 should prune every checkpoint."""
    mgr = CheckpointManager(keep_best=0, keep_latest=0, metric="eval/loss")

    paths = []
    for i in range(3):
        p = _make_ckpt(tmp_path, f"ckpt_{i}", step=i)
        mgr.record(p, i, None)
        paths.append(p)

    deleted = mgr.prune()
    assert len(deleted) == 3
    for p in paths:
        assert not p.exists()


def test_no_phase_filter_loads_all(tmp_path):
    """When phase is None, all checkpoints are loaded (backward compat)."""
    _make_ckpt(tmp_path, "ice_step10", 10, phase="ice")
    _make_ckpt(tmp_path, "joint_step20", 20, phase="joint_block_pretrain")

    mgr = CheckpointManager(phase=None)
    mgr.load_existing(tmp_path)

    assert len(mgr._records) == 2
