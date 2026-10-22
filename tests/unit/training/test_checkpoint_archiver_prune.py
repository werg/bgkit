"""Tests for the HDD archive retention cap (``_prune_archive_dir``).

The permanent HDD archive historically grew unbounded (348 dirs / 2.4 TB filled
a 3.6 TB disk and crashed training). ``_prune_archive_dir`` caps it to the last
``keep_last_n`` checkpoints for the *current phase only* — never touching other
phases / runs, never removing a mid-write dir, always keeping the newest.
"""

from __future__ import annotations

import json
from pathlib import Path

from bgkit.training.checkpoint_archiver import _prune_archive_dir

PHASE = "phase2_kb"
OTHER = "phase1_other"


def _make_ckpt(parent: Path, phase: str, step: int, *, with_metadata: bool = True) -> Path:
    name = f"{phase}_step{step}_20260610_000000"
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    if with_metadata:
        (d / "metadata.json").write_text(json.dumps({"phase": phase, "step": step}))
    (d / "model.pt").write_bytes(b"\x00" * 1024)
    return d


def _set_run(path: Path, run_name: str) -> None:
    meta = json.loads((path / "metadata.json").read_text())
    meta["run_name"] = run_name
    (path / "metadata.json").write_text(json.dumps(meta))


def _steps(parent: Path, phase: str) -> list[int]:
    return sorted(
        int(p.name.split("step")[1].split("_")[0]) for p in parent.glob(f"{phase}_step*")
    )


def test_prune_keeps_newest_n_of_phase(tmp_path: Path) -> None:
    for step in (5, 10, 15, 20, 25):
        _make_ckpt(tmp_path, PHASE, step)
    # Other phase / other run — must survive untouched.
    for step in (100, 200, 300):
        _make_ckpt(tmp_path, OTHER, step)

    _prune_archive_dir(tmp_path, PHASE, keep_last_n=3)

    assert _steps(tmp_path, PHASE) == [15, 20, 25]  # newest 3
    assert _steps(tmp_path, OTHER) == [100, 200, 300]  # untouched


def test_prune_skips_midwrite_dir_without_metadata(tmp_path: Path) -> None:
    _make_ckpt(tmp_path, PHASE, 5)
    _make_ckpt(tmp_path, PHASE, 10)
    # An old dir that is mid-write (no metadata.json yet) must not be deleted,
    # even though it falls outside keep_last_n.
    _make_ckpt(tmp_path, PHASE, 1, with_metadata=False)
    _make_ckpt(tmp_path, PHASE, 15)
    _make_ckpt(tmp_path, PHASE, 20)

    _prune_archive_dir(tmp_path, PHASE, keep_last_n=2)

    # newest 2 (15, 20) kept; step 1 skipped (mid-write); 5/10 pruned.
    assert _steps(tmp_path, PHASE) == [1, 15, 20]


def test_prune_always_keeps_newest(tmp_path: Path) -> None:
    _make_ckpt(tmp_path, PHASE, 5)
    newest = _make_ckpt(tmp_path, PHASE, 10)
    _prune_archive_dir(tmp_path, PHASE, keep_last_n=0)
    assert newest.exists()
    assert _steps(tmp_path, PHASE) == [10]


def test_prune_noop_when_under_cap(tmp_path: Path) -> None:
    for step in (5, 10):
        _make_ckpt(tmp_path, PHASE, step)
    _prune_archive_dir(tmp_path, PHASE, keep_last_n=3)
    assert _steps(tmp_path, PHASE) == [5, 10]


def test_prune_is_scoped_to_run_name(tmp_path: Path) -> None:
    own_old = _make_ckpt(tmp_path, PHASE, 5)
    own_new = _make_ckpt(tmp_path, PHASE, 10)
    other = _make_ckpt(tmp_path, PHASE, 999)
    _set_run(own_old, "stage_b")
    _set_run(own_new, "stage_b")
    _set_run(other, "smoke")

    _prune_archive_dir(tmp_path, PHASE, keep_last_n=1, run_name="stage_b")

    assert not own_old.exists()
    assert own_new.exists()
    assert other.exists()
