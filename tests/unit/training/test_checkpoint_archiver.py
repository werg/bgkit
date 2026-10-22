"""Tests for NVMe fast-write + async HDD archive checkpoint path.

Covers the archiver (throttled copy, atomic rename, NVMe retention, recovery)
and the registry's multi-dir backfill + NVMe-first resume resolution. No torch
dependency — checkpoints are faked as a metadata.json + a payload file.
"""

from __future__ import annotations

import json
from pathlib import Path

from bgkit.training.checkpoint_archiver import (
    CheckpointArchiver,
    archive_pending_into,
)
from bgkit.training.checkpoint_registry import (
    CheckpointRegistry,
    resolve_latest_checkpoint,
)

PHASE = "phase1_test"


def _make_ckpt(parent: Path, step: int, payload_mb: int = 1) -> Path:
    """Create a fake checkpoint dir with metadata.json + a payload file."""
    name = f"{PHASE}_step{step}_20260610_000000"
    d = parent / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"phase": PHASE, "step": step, "epoch": 0}))
    (d / "model.pt").write_bytes(b"\xab" * (payload_mb * 1024 * 1024))
    return d


def test_archiver_copies_and_is_byte_exact(tmp_path: Path) -> None:
    fast = tmp_path / "nvme"
    hdd = tmp_path / "hdd"
    fast.mkdir()
    hdd.mkdir()
    src = _make_ckpt(fast, 100, payload_mb=5)  # >fsync chunk to exercise throttling

    arch = CheckpointArchiver(archive_dir=hdd, phase=PHASE, keep_last_n=3)
    arch.enqueue(src, fast)
    assert arch.wait_idle(timeout=30.0)

    dst = hdd / src.name
    assert dst.exists()
    assert (dst / "metadata.json").read_text() == (src / "metadata.json").read_text()
    assert (dst / "model.pt").read_bytes() == (src / "model.pt").read_bytes()
    # No staging dir left behind
    assert not list(hdd.glob("._archiving_*"))


def test_archiver_prunes_fast_dir_to_keep_last_n(tmp_path: Path) -> None:
    fast = tmp_path / "nvme"
    hdd = tmp_path / "hdd"
    fast.mkdir()
    hdd.mkdir()
    arch = CheckpointArchiver(archive_dir=hdd, phase=PHASE, keep_last_n=2)
    for step in (100, 200, 300):
        src = _make_ckpt(fast, step)
        arch.enqueue(src, fast)
    assert arch.wait_idle(timeout=30.0)

    # All three archived to HDD
    assert {p.name for p in hdd.glob(f"{PHASE}_step*")} == {
        f"{PHASE}_step{s}_20260610_000000" for s in (100, 200, 300)
    }
    # NVMe keeps only the last 2 (by step)
    remaining = sorted(
        int(p.name.split("step")[1].split("_")[0]) for p in fast.glob(f"{PHASE}_step*")
    )
    assert remaining == [200, 300]


def test_archiver_never_prunes_unarchived(tmp_path: Path) -> None:
    fast = tmp_path / "nvme"
    hdd = tmp_path / "hdd"
    fast.mkdir()
    hdd.mkdir()
    # Pre-existing NVMe checkpoints with NO HDD copy must not be deleted by prune.
    for step in (100, 200, 300, 400):
        _make_ckpt(fast, step)
    arch = CheckpointArchiver(archive_dir=hdd, phase=PHASE, keep_last_n=1)
    newest = _make_ckpt(fast, 500)
    arch.enqueue(newest, fast)
    assert arch.wait_idle(timeout=30.0)

    # 500 archived; older ones lack HDD copies so prune leaves them in place.
    surviving = sorted(
        int(p.name.split("step")[1].split("_")[0]) for p in fast.glob(f"{PHASE}_step*")
    )
    assert surviving == [100, 200, 300, 400, 500]


def test_registry_backfill_counts_nvme_only_as_on_disk(tmp_path: Path) -> None:
    hdd = tmp_path / "hdd"
    fast = tmp_path / "nvme"
    hdd.mkdir()
    fast.mkdir()
    # Checkpoint exists ONLY on NVMe (archive still "in flight")
    _make_ckpt(fast, 100)

    reg = CheckpointRegistry(hdd)
    reg.backfill(hdd, extra_dirs=[fast])
    latest = reg.latest(phase=PHASE)
    assert latest is not None
    assert latest.step == 100
    assert latest.on_disk is True  # not falsely pruned despite absent from HDD


def test_resolve_prefers_nvme_then_falls_back_to_hdd(tmp_path: Path) -> None:
    hdd = tmp_path / "hdd"
    fast = tmp_path / "nvme"
    hdd.mkdir()
    fast.mkdir()

    # Older checkpoint archived to HDD only (NVMe pruned); newer still on NVMe.
    _make_ckpt(hdd, 100)
    newer_fast = _make_ckpt(fast, 200)

    resolved = resolve_latest_checkpoint(hdd, PHASE, fast_dir=fast)
    assert resolved == newer_fast  # prefers NVMe copy of the latest

    # When the latest is only on HDD, resolve returns the HDD path.
    _make_ckpt(hdd, 300)
    resolved2 = resolve_latest_checkpoint(hdd, PHASE, fast_dir=fast)
    assert resolved2 == hdd / f"{PHASE}_step300_20260610_000000"


def test_archive_pending_into_recovers_orphans(tmp_path: Path) -> None:
    hdd = tmp_path / "hdd"
    fast = tmp_path / "nvme"
    hdd.mkdir()
    fast.mkdir()
    _make_ckpt(fast, 100)
    _make_ckpt(fast, 200)
    _make_ckpt(hdd, 100)  # 100 already archived; 200 is the orphan

    copied = archive_pending_into(hdd, fast, PHASE)
    assert copied == [f"{PHASE}_step200_20260610_000000"]
    assert (hdd / f"{PHASE}_step200_20260610_000000" / "model.pt").exists()
