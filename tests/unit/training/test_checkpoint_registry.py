"""Tests for CheckpointRegistry."""

import json
from unittest.mock import MagicMock

import pytest

from bgkit.training.checkpoint_registry import (
    CheckpointRegistry,
    RegistryEntry,
    normalize_checkpoint_name,
    resolve_checkpoint,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(name="ckpt_1", phase="ice", step=100, **kwargs) -> RegistryEntry:
    defaults = dict(
        epoch=0,
        timestamp="2026-02-20T22:05:22+00:00",
        status="completed",
        on_disk=True,
    )
    defaults.update(kwargs)
    return RegistryEntry(name=name, phase=phase, step=step, **defaults)


def _make_ckpt_on_disk(tmp_path, name, step, phase="ice", metrics=None, parent=None):
    """Create a mock checkpoint dir with metadata.json on disk."""
    ckpt = tmp_path / name
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = {
        "phase": phase,
        "step": step,
        "epoch": 0,
        "parent_checkpoint": parent,
    }
    if metrics is not None:
        meta["metrics"] = metrics
    (ckpt / "metadata.json").write_text(json.dumps(meta))
    (ckpt / "model.pt").write_text("fake weights")
    return ckpt


# ---------------------------------------------------------------------------
# normalize_checkpoint_name
# ---------------------------------------------------------------------------


def test_normalize_absolute_path():
    assert (
        normalize_checkpoint_name("/workspace/checkpoints/ice_step29999_20260220_220522")
        == "ice_step29999_20260220_220522"
    )


def test_normalize_bare_name():
    assert normalize_checkpoint_name("ice_step29999_20260220_220522") == (
        "ice_step29999_20260220_220522"
    )


# ---------------------------------------------------------------------------
# Register and retrieve
# ---------------------------------------------------------------------------


def test_register_and_get(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    entry = _make_entry()
    reg.register(entry)
    assert reg.get("ckpt_1") is entry
    assert reg.get("nonexistent") is None


def test_register_persists_to_disk(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1))
    reg.register(_make_entry(name="b", step=2))

    # Reload from disk
    reg2 = CheckpointRegistry(tmp_path)
    assert reg2.get("a") is not None
    assert reg2.get("b") is not None
    assert reg2.get("a").step == 1


def test_atomic_save_creates_registry_json(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry())
    data = json.loads((tmp_path / "registry.json").read_text())
    assert data["version"] == 1
    assert "updated_at" in data
    assert len(data["entries"]) == 1


# ---------------------------------------------------------------------------
# Mark pruned
# ---------------------------------------------------------------------------


def test_mark_pruned(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="x", status="completed", on_disk=True))
    reg.mark_pruned("x")
    e = reg.get("x")
    assert e.status == "pruned"
    assert e.on_disk is False


def test_mark_pruned_nonexistent_noop(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.mark_pruned("nonexistent")  # should not raise


# ---------------------------------------------------------------------------
# List with filters
# ---------------------------------------------------------------------------


def test_list_all(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=2, phase="ice"))
    reg.register(_make_entry(name="b", step=1, phase="phase1_step6"))
    entries = reg.list_entries()
    assert len(entries) == 2
    # Sorted by step
    assert entries[0].name == "b"
    assert entries[1].name == "a"


def test_list_filter_phase(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", phase="ice", step=1))
    reg.register(_make_entry(name="b", phase="phase1_step6", step=2))
    assert len(reg.list_entries(phase="ice")) == 1


def test_list_filter_status(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", status="completed", step=1))
    reg.register(_make_entry(name="b", status="pruned", step=2))
    assert len(reg.list_entries(status="pruned")) == 1


def test_list_filter_tags(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", tags=["baseline", "v1"], step=1))
    reg.register(_make_entry(name="b", tags=["v1"], step=2))
    assert len(reg.list_entries(tags=["baseline"])) == 1
    assert len(reg.list_entries(tags=["v1"])) == 2


# ---------------------------------------------------------------------------
# Best by metric
# ---------------------------------------------------------------------------


def test_best_lower_is_better(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics={"eval/mse": 5.0}))
    reg.register(_make_entry(name="b", step=2, metrics={"eval/mse": 3.0}))
    reg.register(_make_entry(name="c", step=3, metrics={"eval/mse": 7.0}))
    best = reg.best(phase="ice", metric="eval/mse", lower_is_better=True)
    assert best.name == "b"


def test_best_higher_is_better(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics={"eval/pearson": 0.3}))
    reg.register(_make_entry(name="b", step=2, metrics={"eval/pearson": 0.8}))
    best = reg.best(phase="ice", metric="eval/pearson", lower_is_better=False)
    assert best.name == "b"


def test_best_on_disk_only_excludes_pruned(tmp_path):
    """Default on_disk_only=True should exclude pruned entries."""
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics={"eval/mse": 1.0}))
    reg.register(_make_entry(name="b", step=2, metrics={"eval/mse": 0.5}))
    reg.mark_pruned("b")

    best = reg.best(phase="ice", metric="eval/mse")
    assert best.name == "a"  # b is pruned, so a wins


def test_best_on_disk_only_false_includes_pruned(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics={"eval/mse": 1.0}))
    reg.register(_make_entry(name="b", step=2, metrics={"eval/mse": 0.5}))
    reg.mark_pruned("b")

    best = reg.best(phase="ice", metric="eval/mse", on_disk_only=False)
    assert best.name == "b"  # pruned but included


def test_best_no_matching_entries(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", phase="other", step=1, metrics={"eval/mse": 1.0}))
    assert reg.best(phase="ice", metric="eval/mse") is None


def test_best_no_metrics(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics=None))
    assert reg.best(phase="ice", metric="eval/mse") is None


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------


def test_annotate_notes_and_tags(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1))
    assert reg.annotate("a", notes="good run", tags=["baseline"])
    e = reg.get("a")
    assert e.notes == "good run"
    assert "baseline" in e.tags


def test_annotate_merges_tags(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, tags=["v1"]))
    reg.annotate("a", tags=["v2"])
    e = reg.get("a")
    assert set(e.tags) == {"v1", "v2"}


def test_annotate_nonexistent(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    assert reg.annotate("nonexistent", notes="x") is False


# ---------------------------------------------------------------------------
# Backfill
# ---------------------------------------------------------------------------


def test_backfill_from_disk(tmp_path):
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 3.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200)

    reg = CheckpointRegistry(tmp_path)
    count = reg.backfill(tmp_path)
    assert count == 2
    assert reg.get("ice_step100_20260220_220522") is not None
    assert reg.get("ice_step100_20260220_220522").metrics == {"eval/mse": 3.0}


def test_backfill_skips_existing(tmp_path):
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100)
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="ice_step100_20260220_220522", step=100))
    count = reg.backfill(tmp_path)
    assert count == 0


def test_backfill_normalizes_parent_checkpoint(tmp_path):
    _make_ckpt_on_disk(
        tmp_path,
        "ice_step200_20260221_100000",
        200,
        parent="/workspace/checkpoints/ice_step100_20260220_220522",
    )
    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    e = reg.get("ice_step200_20260221_100000")
    assert e.parent_checkpoint == "ice_step100_20260220_220522"


def test_backfill_nonexistent_dir(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    assert reg.backfill(tmp_path / "nonexistent") == 0


def test_backfill_and_best(tmp_path):
    """backfill should make on-disk checkpoints queryable via best()."""
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 3.0})

    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    best = reg.best(phase="ice", metric="eval/mse")
    assert best is not None
    assert best.name == "ice_step200_20260221_100000"


def test_backfill_reconciles_deleted_dirs(tmp_path):
    """backfill marks entries as pruned when their dirs no longer exist on disk."""
    # Register an entry for a checkpoint that's "on disk"
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="ice_step100", step=100, metrics={"eval/mse": 2.0}))
    assert reg.get("ice_step100").on_disk is True

    # The directory doesn't actually exist — backfill should reconcile
    reg.backfill(tmp_path)

    e = reg.get("ice_step100")
    assert e.on_disk is False
    assert e.status == "pruned"


def test_backfill_reconcile_excludes_from_best(tmp_path):
    """After reconciliation, deleted checkpoints are excluded from best()."""
    # Register two entries — one with better metric but no dir on disk
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="gone", step=100, metrics={"eval/mse": 1.0}))
    _make_ckpt_on_disk(tmp_path, "present_20260220_220522", 200,
                       phase="ice", metrics={"eval/mse": 5.0})
    reg.backfill(tmp_path)

    # "gone" had better MSE but was reconciled to pruned
    best = reg.best(phase="ice", metric="eval/mse")
    assert best is not None
    assert best.name == "present_20260220_220522"


def test_backfill_reconcile_persists(tmp_path):
    """Reconciliation changes survive reload."""
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="vanished", step=50, metrics={"eval/mse": 1.0}))
    reg.backfill(tmp_path)

    # Reload from disk
    reg2 = CheckpointRegistry(tmp_path)
    e = reg2.get("vanished")
    assert e.on_disk is False
    assert e.status == "pruned"


def test_backfill_reconcile_leaves_existing_dirs_alone(tmp_path):
    """Entries whose dirs still exist keep on_disk=True after backfill."""
    _make_ckpt_on_disk(tmp_path, "still_here_20260220_220522", 100, phase="ice")
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="still_here_20260220_220522", step=100))
    reg.backfill(tmp_path)

    e = reg.get("still_here_20260220_220522")
    assert e.on_disk is True
    assert e.status == "completed"


# ---------------------------------------------------------------------------
# Empty / missing registry
# ---------------------------------------------------------------------------


def test_empty_registry(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    assert reg.list_entries() == []
    assert reg.get("x") is None
    assert reg.best(phase="ice", metric="eval/mse") is None


def test_missing_registry_file(tmp_path):
    # No registry.json exists — should work fine
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1))
    assert reg.get("a") is not None


# ---------------------------------------------------------------------------
# Forward compatibility
# ---------------------------------------------------------------------------


def test_extra_fields_in_json_ignored(tmp_path):
    """Extra fields in registry.json don't crash loading."""
    data = {
        "version": 1,
        "updated_at": "2026-02-23T10:00:00+00:00",
        "entries": [
            {
                "name": "ckpt_1",
                "phase": "ice",
                "step": 100,
                "epoch": 0,
                "timestamp": "2026-02-20T22:05:22+00:00",
                "status": "completed",
                "on_disk": True,
                "future_field": "should be ignored",
                "another_field": 42,
            }
        ],
    }
    (tmp_path / "registry.json").write_text(json.dumps(data))
    reg = CheckpointRegistry(tmp_path)
    e = reg.get("ckpt_1")
    assert e is not None
    assert e.step == 100


# ---------------------------------------------------------------------------
# CheckpointManager + Registry integration
# ---------------------------------------------------------------------------


def test_checkpoint_manager_prune_calls_mark_pruned(tmp_path):
    """CheckpointManager.prune() with registry marks pruned entries."""
    from bgkit.training.checkpoint_manager import CheckpointManager

    reg = CheckpointRegistry(tmp_path)
    entry = _make_entry(name="ice_step100", step=100, metrics={"eval/mse": 5.0})
    reg.register(entry)

    mgr = CheckpointManager(
        keep_best=0, keep_latest=0, metric="eval/mse", registry=reg
    )
    ckpt_path = tmp_path / "ice_step100"
    ckpt_path.mkdir(exist_ok=True)
    (ckpt_path / "metadata.json").write_text("{}")
    mgr.record(ckpt_path, 100, {"eval/mse": 5.0})
    mgr.prune()

    assert reg.get("ice_step100").status == "pruned"
    assert reg.get("ice_step100").on_disk is False


def test_checkpoint_manager_prune_without_registry(tmp_path):
    """CheckpointManager without registry still works unchanged."""
    from bgkit.training.checkpoint_manager import CheckpointManager

    mgr = CheckpointManager(keep_best=0, keep_latest=0, metric="eval/mse")
    ckpt_path = tmp_path / "ckpt_0"
    ckpt_path.mkdir()
    (ckpt_path / "model.pt").write_text("data")
    mgr.record(ckpt_path, 0, None)
    deleted = mgr.prune()
    assert len(deleted) == 1
    assert not ckpt_path.exists()


# ---------------------------------------------------------------------------
# Auto-resolution: best() returns on-disk, not pruned
# ---------------------------------------------------------------------------


def test_auto_resolution_prefers_on_disk(tmp_path):
    """When both on-disk and pruned exist, best() with on_disk_only returns on-disk."""
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", step=1, metrics={"eval/mse": 1.0}, on_disk=False,
                             status="pruned"))
    reg.register(_make_entry(name="b", step=2, metrics={"eval/mse": 2.0}, on_disk=True))

    # a has better metric but is pruned; b is on disk
    best = reg.best(phase="ice", metric="eval/mse", on_disk_only=True)
    assert best.name == "b"


# ---------------------------------------------------------------------------
# _build_registry_entry and _registry_parent (BaseTrainer helpers)
# ---------------------------------------------------------------------------


class _FakeTrainer:
    """Minimal stand-in for BaseTrainer to test registry helpers without GPU."""

    def __init__(self, tmp_path, run_name=None):
        from types import SimpleNamespace

        # Mirror the production OmegaConf cfg, which supports BOTH attribute
        # access (cfg.training) and dict-style .get("run_name", default).
        class _Cfg(SimpleNamespace):
            def get(self, key, default=None):
                return getattr(self, key, default)

        self.cfg = _Cfg(training=SimpleNamespace(phase="ice"), run_name=run_name)
        self.global_step = 500
        self.epoch = 1
        self._last_checkpoint_path = None
        self._input_sources = None

    # Bind the real methods
    from bgkit.training.base_trainer import BaseTrainer

    _registry_parent = BaseTrainer._registry_parent
    _build_registry_entry = BaseTrainer._build_registry_entry


def test_build_registry_entry_fields(tmp_path):
    """_build_registry_entry populates all expected fields."""
    trainer = _FakeTrainer(tmp_path)
    trainer._input_sources = {"ice": "ice_step100"}

    # Create a fake checkpoint dir with a file so disk_size is computed
    ckpt = tmp_path / "ice_step500_20260224_120000"
    ckpt.mkdir()
    (ckpt / "model.pt").write_text("fake")

    entry = trainer._build_registry_entry(
        ckpt, {"eval/mse": 2.5}, wandb_run=None, parent_checkpoint="ice_step400",
    )

    assert entry.name == "ice_step500_20260224_120000"
    assert entry.phase == "ice"
    assert entry.step == 500
    assert entry.epoch == 1
    assert entry.status == "completed"
    assert entry.on_disk is True
    assert entry.metrics == {"eval/mse": 2.5}
    assert entry.parent_checkpoint == "ice_step400"
    assert entry.input_sources == {"ice": "ice_step100"}
    assert entry.wandb_run_id is None
    assert entry.disk_size_bytes > 0
    assert entry.timestamp  # non-empty ISO string


def test_build_registry_entry_interrupted_status(tmp_path):
    trainer = _FakeTrainer(tmp_path)
    ckpt = tmp_path / "ice_step500_20260224_120000"
    ckpt.mkdir()

    entry = trainer._build_registry_entry(
        ckpt, None, wandb_run=None, status="interrupted",
    )
    assert entry.status == "interrupted"
    assert entry.metrics is None


def test_build_registry_entry_with_wandb_run(tmp_path):
    trainer = _FakeTrainer(tmp_path)
    ckpt = tmp_path / "ice_step500_20260224_120000"
    ckpt.mkdir()

    wandb_run = MagicMock()
    wandb_run.id = "abc123"

    entry = trainer._build_registry_entry(
        ckpt, None, wandb_run=wandb_run,
    )
    assert entry.wandb_run_id == "abc123"


def test_registry_parent_normalizes_path():
    from bgkit.training.base_trainer import BaseTrainer

    class T:
        _last_checkpoint_path = "/workspace/checkpoints/ice_step400_20260224_110000"
    T._registry_parent = BaseTrainer._registry_parent

    assert T()._registry_parent() == "ice_step400_20260224_110000"


def test_registry_parent_none_when_no_prior():
    from bgkit.training.base_trainer import BaseTrainer

    class T:
        _last_checkpoint_path = None
    T._registry_parent = BaseTrainer._registry_parent

    assert T()._registry_parent() is None


def test_parent_captured_before_save_not_self(tmp_path):
    """Verify the intended call pattern: parent is captured before save_checkpoint
    mutates _last_checkpoint_path, so parent != self."""
    trainer = _FakeTrainer(tmp_path)
    trainer._last_checkpoint_path = "/workspace/checkpoints/ice_step400_20260224"

    # Capture parent BEFORE save (as the train loop does)
    parent = trainer._registry_parent()
    assert parent == "ice_step400_20260224"

    # Simulate what save_checkpoint does
    new_ckpt = tmp_path / "ice_step500_20260224_120000"
    new_ckpt.mkdir()
    trainer._last_checkpoint_path = str(new_ckpt)

    # Build entry with the pre-captured parent
    entry = trainer._build_registry_entry(
        new_ckpt, None, wandb_run=None, parent_checkpoint=parent,
    )
    assert entry.parent_checkpoint == "ice_step400_20260224"
    assert entry.parent_checkpoint != entry.name  # not self-referential


def test_input_sources_in_registry_entry():
    """CompressionTrainer sets input_sources for cross-phase lineage."""
    entry = RegistryEntry(
        name="phase1_step5_step5000_20260220_220522",
        phase="phase1_step6",
        step=5000,
        epoch=0,
        timestamp="2026-02-20T22:05:22+00:00",
        input_sources={"ice": "ice_step29999_20260220_220522", "step1": "step1_ckpt"},
    )
    assert entry.input_sources["ice"] == "ice_step29999_20260220_220522"
    assert entry.input_sources["step1"] == "step1_ckpt"


# ---------------------------------------------------------------------------
# resolve_checkpoint()
# ---------------------------------------------------------------------------


def test_resolve_checkpoint_finds_best(tmp_path):
    """resolve_checkpoint should backfill and return the best on-disk checkpoint."""
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 3.0})

    result = resolve_checkpoint(tmp_path, phase="ice", metric="eval/mse")
    assert result == tmp_path / "ice_step200_20260221_100000"


def test_resolve_checkpoint_no_match_raises(tmp_path):
    """resolve_checkpoint should raise ValueError when no matching checkpoint exists."""
    with pytest.raises(ValueError, match="no ice checkpoint found"):
        resolve_checkpoint(tmp_path, phase="ice", metric="eval/mse")


def test_resolve_checkpoint_skips_pruned(tmp_path):
    """resolve_checkpoint should skip pruned checkpoints and return the on-disk one."""
    # Create two checkpoints, delete the better one's dir
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 1.0})

    # Register the better one, then mark pruned (simulate deletion)
    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    reg.mark_pruned("ice_step200_20260221_100000")

    # resolve_checkpoint creates a fresh registry, so re-backfill will reconcile
    import shutil
    shutil.rmtree(tmp_path / "ice_step200_20260221_100000")

    result = resolve_checkpoint(tmp_path, phase="ice", metric="eval/mse")
    assert result == tmp_path / "ice_step100_20260220_220522"


def test_resolve_checkpoint_with_label(tmp_path):
    """Label appears in the error message."""
    with pytest.raises(ValueError, match="bgkit_checkpoint"):
        resolve_checkpoint(
            tmp_path, phase="joint_block_pretrain", metric="eval/mse_repro",
            label="bgkit_checkpoint",
        )


# ---------------------------------------------------------------------------
# Backfill: reappearing checkpoint recovery
# ---------------------------------------------------------------------------


def test_backfill_recovers_reappearing_checkpoint(tmp_path):
    """Backfill should un-prune entries whose dirs have reappeared."""
    # Create checkpoint and register it
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 2.0})
    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    assert reg.get("ice_step100_20260220_220522").on_disk is True

    # Mark as pruned (simulate deletion)
    reg.mark_pruned("ice_step100_20260220_220522")
    assert reg.get("ice_step100_20260220_220522").status == "pruned"
    assert reg.get("ice_step100_20260220_220522").on_disk is False

    # Dir still exists on disk (or was restored) -- backfill should recover it
    reg2 = CheckpointRegistry(tmp_path)
    reg2.backfill(tmp_path)
    e = reg2.get("ice_step100_20260220_220522")
    assert e.on_disk is True
    assert e.status == "completed"


# ---------------------------------------------------------------------------
# latest() and resolve_latest_checkpoint()
# ---------------------------------------------------------------------------

from bgkit.training.checkpoint_registry import resolve_latest_checkpoint


def test_latest_returns_highest_step(tmp_path):
    """latest() should return the checkpoint with the highest step."""
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 3.0})
    _make_ckpt_on_disk(tmp_path, "ice_step150_20260221_050000", 150, metrics={"eval/mse": 1.0})

    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    result = reg.latest(phase="ice")
    assert result is not None
    assert result.name == "ice_step200_20260221_100000"


def test_latest_skips_pruned(tmp_path):
    """latest() should skip pruned checkpoints."""
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 3.0})

    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    reg.mark_pruned("ice_step200_20260221_100000")

    result = reg.latest(phase="ice")
    assert result is not None
    assert result.name == "ice_step100_20260220_220522"


def test_latest_returns_none_for_empty_phase(tmp_path):
    reg = CheckpointRegistry(tmp_path)
    assert reg.latest(phase="ice") is None


def test_resolve_latest_checkpoint_finds_latest(tmp_path):
    _make_ckpt_on_disk(tmp_path, "ice_step100_20260220_220522", 100, metrics={"eval/mse": 5.0})
    _make_ckpt_on_disk(tmp_path, "ice_step200_20260221_100000", 200, metrics={"eval/mse": 3.0})

    result = resolve_latest_checkpoint(tmp_path, phase="ice")
    assert result == tmp_path / "ice_step200_20260221_100000"


def test_resolve_latest_checkpoint_returns_none_when_empty(tmp_path):
    result = resolve_latest_checkpoint(tmp_path, phase="ice")
    assert result is None


# ---------------------------------------------------------------------------
# Run-scoped auto-resume (FIX 1): a run must only auto-resume its OWN
# checkpoints, never another run sharing the same phase.
# ---------------------------------------------------------------------------


def _make_ckpt_on_disk_with_run(tmp_path, name, step, run_name, phase="phase2_kb"):
    """Create a mock checkpoint dir whose metadata.json records a run_name."""
    ckpt = tmp_path / name
    ckpt.mkdir(parents=True, exist_ok=True)
    meta = {
        "phase": phase,
        "step": step,
        "epoch": 0,
        "parent_checkpoint": None,
        "run_name": run_name,
    }
    (ckpt / "metadata.json").write_text(json.dumps(meta))
    (ckpt / "model.pt").write_text("fake weights")
    return ckpt


def test_run_scoped_does_not_resume_other_run(tmp_path):
    """A run_name-B run must NOT auto-resume run_name-A's checkpoint, even
    though both share phase=phase2_kb (the latent-crash scenario)."""
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step100_20260220_220522", 100, "run_a")
    # run_b asks for its own latest; there is none → cold start (None).
    result = resolve_latest_checkpoint(tmp_path, phase="phase2_kb", run_name="run_b")
    assert result is None


def test_run_scoped_resumes_same_run(tmp_path):
    """Same-run resume still works: run_a resolves its own latest checkpoint."""
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step100_20260220_220522", 100, "run_a")
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step200_20260221_100000", 200, "run_a")
    result = resolve_latest_checkpoint(tmp_path, phase="phase2_kb", run_name="run_a")
    assert result == tmp_path / "phase2_kb_step200_20260221_100000"


def test_run_scoped_picks_own_among_mixed_runs(tmp_path):
    """With interleaved runs sharing a phase, each resolves only its own
    latest — the higher-step foreign checkpoint is ignored."""
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step100_20260220_220522", 100, "run_a")
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step300_20260222_100000", 300, "run_b")
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step200_20260221_100000", 200, "run_a")
    result = resolve_latest_checkpoint(tmp_path, phase="phase2_kb", run_name="run_a")
    assert result == tmp_path / "phase2_kb_step200_20260221_100000"


def test_run_scoped_ignores_checkpoints_without_run_name(tmp_path):
    """Old checkpoints lacking a recorded run_name are never cross-matched to
    a named run — the run cold-starts instead of grabbing them."""
    _make_ckpt_on_disk(tmp_path, "phase2_kb_step100_20260220_220522", 100, phase="phase2_kb")
    result = resolve_latest_checkpoint(tmp_path, phase="phase2_kb", run_name="run_a")
    assert result is None


def test_no_run_name_preserves_phase_only_behavior(tmp_path):
    """When run_name is None (unnamed run), behaviour is unchanged: phase-only,
    latest step wins — including over run-tagged checkpoints."""
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step100_20260220_220522", 100, "run_a")
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step200_20260221_100000", 200, "run_b")
    result = resolve_latest_checkpoint(tmp_path, phase="phase2_kb", run_name=None)
    assert result == tmp_path / "phase2_kb_step200_20260221_100000"


def test_latest_run_name_filter(tmp_path):
    """CheckpointRegistry.latest(run_name=...) filters in memory too."""
    reg = CheckpointRegistry(tmp_path)
    reg.register(_make_entry(name="a", phase="phase2_kb", step=100, run_name="run_a"))
    reg.register(_make_entry(name="b", phase="phase2_kb", step=200, run_name="run_b"))
    reg.register(_make_entry(name="c", phase="phase2_kb", step=150, run_name="run_a"))
    latest_a = reg.latest(phase="phase2_kb", run_name="run_a")
    assert latest_a.name == "c"
    assert reg.latest(phase="phase2_kb", run_name="run_missing") is None
    # No filter → phase-only, highest step.
    assert reg.latest(phase="phase2_kb").name == "b"


def test_backfill_records_run_name_from_metadata(tmp_path):
    """backfill() reads run_name out of metadata.json into the registry entry."""
    _make_ckpt_on_disk_with_run(tmp_path, "phase2_kb_step100_20260220_220522", 100, "run_a")
    reg = CheckpointRegistry(tmp_path)
    reg.backfill(tmp_path)
    entry = reg.get("phase2_kb_step100_20260220_220522")
    assert entry is not None
    assert entry.run_name == "run_a"


def test_build_registry_entry_records_run_name(tmp_path):
    """_build_registry_entry captures cfg.run_name."""
    trainer = _FakeTrainer(tmp_path, run_name="run_a")
    ckpt = tmp_path / "ice_step500_20260224_120000"
    ckpt.mkdir()
    entry = trainer._build_registry_entry(ckpt, None, wandb_run=None)
    assert entry.run_name == "run_a"
