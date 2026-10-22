"""NVMe residue from FINISHED runs must be reclaimed, and never at the cost of
the only complete copy.

2026-09-13 amendment: un-archived residue is no longer "left in place forever" —
the NVMe is staging and the HDD holds the history, so the sweep ARCHIVES it and
then reclaims the NVMe copy. Leaving it in place is how the root filesystem
reached 100% (255 dirs / 184 GB): the HDD pruner (``archive_keep_last_n``)
deletes old archives by policy, after which ``_archive_is_complete`` reported
every older NVMe copy as un-archived. ``_archive_dir_throttled`` now leaves an
``.archived`` marker on the NVMe copy so a policy-pruned archive is not mistaken
for a never-archived one.

Two defects measured on 2026-08-30, on a filesystem that had reached 94% full
while the retention policy worked exactly as written:

1. ``_prune_fast_dir`` is scoped to the CURRENT phase+run_name, so a completed
   run's last ``keep_last_n`` checkpoints become permanent — no later run with a
   different name ever considers them. Seven finished runs had stranded 17 fully
   archived checkpoints, ~140 GB.
2. It pruned the NVMe copy when the archive merely ``.exists()``. An archive
   directory is created before the copy completes, so an interrupted archival
   leaves a real directory holding a subset. Observed: fast 9082 MB / 2 *.pt vs
   archive 3028 MB / 1 *.pt. Pruning on existence would have destroyed the only
   complete copy.
"""

from __future__ import annotations

import json

from bgkit.training.checkpoint_archiver import (
    _archive_is_complete,
    sweep_finished_run_residue,
)


def _ckpt(root, name, *, phase="phase2_kb", run="old_run", files=("model.pt",),
          size=32):
    d = root / name
    d.mkdir(parents=True)
    (d / "metadata.json").write_text(json.dumps({"phase": phase, "run_name": run}))
    for f in files:
        (d / f).write_bytes(b"x" * size)
    return d


def test_finished_run_residue_is_reclaimed(tmp_path) -> None:
    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    _ckpt(fast, "phase2_kb_step100_run-dead", run="dead_run")
    _ckpt(arch, "phase2_kb_step100_run-dead", run="dead_run")

    removed, _ = sweep_finished_run_residue(fast, arch, "phase2_kb", "live_run")
    assert removed == 1
    assert not (fast / "phase2_kb_step100_run-dead").exists()
    assert (arch / "phase2_kb_step100_run-dead").exists(), "archive must survive"


def test_active_run_is_never_touched(tmp_path) -> None:
    """Retention for the live run belongs to _prune_fast_dir; this sweep must
    not delete a checkpoint the running job may still resume from."""
    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    _ckpt(fast, "phase2_kb_step200_run-live", run="live_run")
    _ckpt(arch, "phase2_kb_step200_run-live", run="live_run")

    removed, _ = sweep_finished_run_residue(fast, arch, "phase2_kb", "live_run")
    assert removed == 0
    assert (fast / "phase2_kb_step200_run-live").exists()


def test_truncated_archive_never_costs_the_only_complete_copy(tmp_path) -> None:
    """The measured near-miss: archive dir EXISTS but holds 1 of 2 files."""
    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    _ckpt(fast, "c", run="dead", files=("model.pt", "optimizer.pt"))
    _ckpt(arch, "c", run="dead", files=("model.pt",))  # interrupted copy

    removed, _ = sweep_finished_run_residue(fast, arch, "phase2_kb", "live")
    # The truncated archive is replaced by a complete one BEFORE the NVMe copy goes.
    assert removed == 1
    assert not (fast / "c").exists()
    assert (arch / "c" / "optimizer.pt").exists()
    assert (arch / "c" / "model.pt").stat().st_size == 32


def test_truncated_file_is_caught_not_just_missing_one(tmp_path) -> None:
    """Same name, smaller size — a partial write, not a missing file."""
    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    _ckpt(fast, "c", run="dead", size=64)
    _ckpt(arch, "c", run="dead", size=8)
    assert _archive_is_complete(fast / "c", arch / "c") is False

    removed, _ = sweep_finished_run_residue(fast, arch, "phase2_kb", "live")
    assert removed == 1
    assert (arch / "c" / "model.pt").stat().st_size == 64, "re-archived in full"


def test_unarchived_checkpoint_is_archived_then_reclaimed(tmp_path) -> None:
    """No archive at all means the NVMe copy is the ONLY copy — so it is
    landed on the HDD first, and only then reclaimed."""
    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    _ckpt(fast, "orphan", run="dead")

    removed, _ = sweep_finished_run_residue(fast, arch, "phase2_kb", "live")
    assert removed == 1
    assert not (fast / "orphan").exists()
    assert (arch / "orphan" / "model.pt").stat().st_size == 32
    assert (arch / "orphan" / "metadata.json").exists()


def test_archive_leaves_marker_and_marker_allows_fast_prune(tmp_path) -> None:
    """The 2026-09-13 leak: archived, then the HDD copy pruned by policy, then
    the NVMe copy kept forever because the archive "was incomplete"."""
    import shutil

    from bgkit.training.checkpoint_archiver import (
        ARCHIVED_MARKER,
        _archive_dir_throttled,
        _prune_fast_dir,
    )

    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    for step in (100, 200, 300, 400):
        _ckpt(fast, f"phase2_kb_step{step}_run-live", run="live")
    old = fast / "phase2_kb_step100_run-live"
    _archive_dir_throttled(old, arch / old.name)
    assert (old / ARCHIVED_MARKER).exists()
    shutil.rmtree(arch / old.name)  # archive_keep_last_n policy removed it

    never = fast / "phase2_kb_step200_run-live"  # never archived: only copy
    _prune_fast_dir(fast, arch, "phase2_kb", keep_last_n=2, run_name="live")
    assert not old.exists(), "archived-then-policy-pruned copy must be reclaimed"
    assert never.exists(), "the only copy must never be reclaimed"


def test_stale_tmp_dirs_are_aged_out_but_fresh_ones_are_not(tmp_path) -> None:
    """``._tmp_*`` are interrupted writes (three held 19 GB here), but a FRESH
    one may be an in-flight write from a concurrent process."""
    import os
    import time

    fast, arch = tmp_path / "fast", tmp_path / "arch"
    fast.mkdir()
    arch.mkdir()
    old = fast / "._tmp_ancient"
    old.mkdir()
    new = fast / "._tmp_inflight"
    new.mkdir()
    long_ago = time.time() - 48 * 3600
    os.utime(old, (long_ago, long_ago))

    _, tmp_removed = sweep_finished_run_residue(
        fast, arch, "phase2_kb", "live", stale_tmp_hours=24.0,
    )
    assert tmp_removed == 1
    assert not old.exists()
    assert new.exists(), "an in-flight write must not be deleted"


def test_prune_fast_dir_checks_completeness_not_existence() -> None:
    """The original bug, pinned at the source."""
    import inspect

    from bgkit.training.checkpoint_archiver import _prune_fast_dir
    src = inspect.getsource(_prune_fast_dir)
    assert "_safe_to_drop_fast_copy(" in src
    assert ".exists()" not in src, "existence check is back; truncation slips through"
