"""Background checkpoint archiver: NVMe live-write + async copy to the HDD archive.

Motivation (DGX Spark, 2026-06-10)
----------------------------------
``CHECKPOINT_DIR`` lives on a ~27 MB/s spinning USB HDD (``/mnt/external``). A
single checkpoint for the summarization phase is ~9 GB, so a synchronous
``torch.save`` + ``fsync`` blocks the train loop for ~5.5 min AND leaves a
multi-GB dirty page-cache backlog on the 128 GB *unified* memory pool — the
source of the historical post-save spike / "first step after checkpoint_saved"
hang.

Throttling saves to limit exposure is the wrong fix: checkpointing exists to
*bound* loss from crashes, so spacing it out maximizes loss-per-crash while
still hitting the dangerous op. The right fix is to make saving cheap + safe,
then save often. This module does that:

1. The trainer writes the checkpoint to a **fast NVMe live dir** (fsync there is
   ~15 s and the dirty pages drain near-instantly, so no backlog, no spike).
   Training resumes immediately.
2. This archiver copies the checkpoint NVMe -> HDD **in a background daemon
   thread**, in bounded chunks with a periodic ``fsync`` so dirty pages never
   exceed ``fsync_every_bytes`` — the dirty-page backlog that stalled CUDA
   allocations can never accumulate. Full file history is preserved on the HDD.
3. After a checkpoint is safely on the HDD, the archiver prunes the NVMe copy if
   it is older than the ``keep_last_n`` most recent (by step) — keeping recent
   checkpoints on NVMe for fast resume while the HDD keeps the configured
   run-scoped archive history.

Crash robustness: resume prefers the NVMe copy if present, else the HDD copy
(see ``checkpoint_registry.resolve_latest_checkpoint``). A crash mid-archive
therefore loses nothing — the just-written checkpoint is intact on NVMe.
"""

from __future__ import annotations

import contextlib
import json
import os
import queue
import shutil
import threading
from collections.abc import Iterable
from pathlib import Path

import structlog

logger = structlog.get_logger()

# Copy in 64 MB reads; fsync after every 256 MB written so the kernel dirty-page
# set for the slow HDD stays bounded well below the multi-GB backlog that caused
# the post-save reclaim stall.
_COPY_CHUNK_BYTES = 64 * 1024 * 1024
_FSYNC_EVERY_BYTES = 256 * 1024 * 1024


def _drop_cache(fd: int) -> None:
    """Best-effort POSIX_FADV_DONTNEED — Linux-only; ignore where unsupported."""
    with contextlib.suppress(AttributeError, OSError):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)


def _copy_file_throttled(src: Path, dst: Path) -> None:
    """Copy one file with periodic fsync so dirty pages stay bounded.

    Also drops the written pages from cache (POSIX_FADV_DONTNEED) — the archive
    copy is write-once cold storage and competes for the unified pool otherwise.
    """
    written_since_sync = 0
    fd = os.open(dst, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        with open(src, "rb", buffering=0) as fsrc:
            while True:
                buf = fsrc.read(_COPY_CHUNK_BYTES)
                if not buf:
                    break
                off = 0
                while off < len(buf):
                    off += os.write(fd, buf[off:])
                written_since_sync += len(buf)
                if written_since_sync >= _FSYNC_EVERY_BYTES:
                    os.fsync(fd)
                    _drop_cache(fd)
                    written_since_sync = 0
        os.fsync(fd)
        _drop_cache(fd)
    finally:
        os.close(fd)
    shutil.copymode(src, dst)


def _archive_dir_throttled(src_dir: Path, dst_dir: Path) -> None:
    """Copy a checkpoint directory NVMe -> HDD atomically and throttled.

    Writes into ``dst_dir.parent/._archiving_<name>`` then renames to
    ``dst_dir`` so a crash mid-copy never leaves a half-written checkpoint under
    the final name (backfill ignores ``._`` prefixes).
    """
    if dst_dir.exists():
        return  # already archived (idempotent re-enqueue)
    staging = dst_dir.parent / f"._archiving_{dst_dir.name}"
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)
    for f in sorted(src_dir.iterdir()):
        if f.is_file():
            _copy_file_throttled(f, staging / f.name)
    # Durable rename of the directory entry on the HDD.
    staging.rename(dst_dir)
    dfd = os.open(dst_dir.parent, os.O_RDONLY)
    try:
        os.fsync(dfd)
    finally:
        os.close(dfd)


def _matches_run(path: Path, phase: str, run_name: str | None) -> bool:
    """Return whether checkpoint metadata belongs to the requested run."""
    try:
        meta = json.loads((path / "metadata.json").read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if meta.get("phase") != phase:
        return False
    return run_name is None or meta.get("run_name") == run_name


def _prune_fast_dir(
    fast_dir: Path,
    archive_dir: Path,
    phase: str,
    keep_last_n: int,
    run_name: str | None = None,
) -> None:
    """Remove NVMe checkpoint copies older than the ``keep_last_n`` most recent.

    Only removes a NVMe copy whose HDD archive exists — never deletes the only
    copy of a checkpoint.
    """
    def _step(p: Path) -> int:
        import re
        m = re.search(r"step(\d+)", p.name)
        return int(m.group(1)) if m else -1

    live = sorted(
        (p for p in fast_dir.glob(f"{phase}_step*") if p.is_dir()
         and not p.name.startswith("._") and _matches_run(p, phase, run_name)),
        key=_step,
        reverse=True,
    )
    for stale in live[keep_last_n:]:
        if (archive_dir / stale.name).exists():
            shutil.rmtree(stale, ignore_errors=True)
            logger.info("checkpoint_fast_pruned", name=stale.name)


def _prune_archive_dir(
    archive_dir: Path,
    phase: str,
    keep_last_n: int,
    run_name: str | None = None,
) -> None:
    """Remove HDD archive checkpoints older than the ``keep_last_n`` most recent.

    Mirrors :func:`_prune_fast_dir` but operates on the permanent HDD archive so
    it cannot grow unbounded (the 2026-06 3.6 TB-full crash). Safety rules:

    * Only ``{phase}_step*`` dirs are considered — other phases / other runs are
      never touched.
    * A dir mid-write (no ``metadata.json`` yet, or an ``._`` staging prefix) is
      skipped, so a checkpoint still being archived is never removed.
    * The single newest dir is always kept regardless of ``keep_last_n``.
    """
    def _step(p: Path) -> int:
        import re
        m = re.search(r"step(\d+)", p.name)
        return int(m.group(1)) if m else -1

    archived = sorted(
        (p for p in archive_dir.glob(f"{phase}_step*") if p.is_dir()
         and not p.name.startswith("._") and _matches_run(p, phase, run_name)),
        key=_step,
        reverse=True,
    )
    if not archived:
        return
    newest = archived[0]
    for stale in archived[max(keep_last_n, 1):]:
        if stale == newest:
            continue  # never delete the single newest
        if not (stale / "metadata.json").exists():
            continue  # mid-write / incomplete — do not touch
        shutil.rmtree(stale, ignore_errors=True)
        logger.info("checkpoint_archive_pruned", name=stale.name)


class CheckpointArchiver:
    """Single-process background archiver (one daemon worker, FIFO queue).

    Use one instance per training run. ``enqueue`` is non-blocking; the worker
    drains the queue, copying each checkpoint to the HDD and pruning the NVMe
    dir. ``wait_idle`` blocks until all queued archives finish (call before
    process exit so the HDD has every checkpoint).
    """

    def __init__(
        self,
        archive_dir: Path,
        phase: str,
        run_name: str | None = None,
        keep_last_n: int = 3,
        archive_keep_last_n: int | None = None,
    ) -> None:
        self._archive_dir = Path(archive_dir)
        self._phase = phase
        self._run_name = run_name
        self._keep_last_n = keep_last_n
        self._archive_keep_last_n = archive_keep_last_n
        self._q: queue.Queue[tuple[Path, Path] | None] = queue.Queue()
        self._thread = threading.Thread(
            target=self._run, name="ckpt-archiver", daemon=True
        )
        self._started = False
        self._lock = threading.Lock()

    def _ensure_started(self) -> None:
        with self._lock:
            if not self._started:
                self._thread.start()
                self._started = True

    def enqueue(self, fast_path: Path, fast_dir: Path) -> None:
        """Queue a freshly-written NVMe checkpoint for archival to the HDD."""
        self._ensure_started()
        self._q.put((Path(fast_path), Path(fast_dir)))

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                self._q.task_done()
                return
            fast_path, fast_dir = item
            try:
                dst = self._archive_dir / fast_path.name
                _archive_dir_throttled(fast_path, dst)
                logger.info(
                    "checkpoint_archived",
                    name=fast_path.name,
                    dst=str(dst),
                )
                _prune_fast_dir(
                    fast_dir,
                    self._archive_dir,
                    self._phase,
                    self._keep_last_n,
                    self._run_name,
                )
                if self._archive_keep_last_n is not None:
                    try:
                        _prune_archive_dir(
                            self._archive_dir,
                            self._phase,
                            self._archive_keep_last_n,
                            self._run_name,
                        )
                    except Exception as exc:  # never let a prune kill the archiver
                        logger.error(
                            "checkpoint_archive_prune_failed",
                            error=str(exc),
                        )
            except Exception as exc:
                logger.error(
                    "checkpoint_archive_failed",
                    name=fast_path.name,
                    error=str(exc),
                )
            finally:
                self._q.task_done()

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until all queued archives complete. Returns True if drained."""
        if not self._started:
            return True
        if timeout is None:
            self._q.join()
            return True
        # join() has no timeout; emulate with the unfinished-tasks counter.
        import time
        deadline = time.monotonic() + timeout
        while self._q.unfinished_tasks > 0:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.2)
        return True


def archive_pending_into(
    archive_dir: Path,
    fast_dir: Path,
    phase: str,
    run_name: str | None = None,
) -> list[str]:
    """Synchronously archive any NVMe checkpoints missing from the HDD.

    Recovery helper for startup: if the process died mid-archive, this lands any
    NVMe-only checkpoints on the HDD before training begins. Returns names copied.
    """
    archive_dir = Path(archive_dir)
    fast_dir = Path(fast_dir)
    if not fast_dir.exists():
        return []
    copied: list[str] = []
    cands: Iterable[Path] = sorted(
        p for p in fast_dir.glob(f"{phase}_step*")
        if p.is_dir()
        and not p.name.startswith("._")
        and _matches_run(p, phase, run_name)
    )
    for p in cands:
        if not (archive_dir / p.name).exists():
            try:
                _archive_dir_throttled(p, archive_dir / p.name)
                copied.append(p.name)
            except Exception as exc:
                logger.error("checkpoint_archive_recover_failed", name=p.name, error=str(exc))
    if copied:
        logger.info("checkpoint_archive_recovered", names=copied)
    return copied
