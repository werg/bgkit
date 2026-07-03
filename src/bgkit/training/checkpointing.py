"""Save/load checkpoints with phase metadata and lineage tracking."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
import torch

logger = structlog.get_logger()


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""

    phase: str  # e.g. "phase1_step5", "phase2"
    step: int
    epoch: int
    parent_checkpoint: str | None  # Lineage: which checkpoint this was initialized from
    metrics: dict[str, float] | None = None
    # LR schedule params at save time — restored on resume for schedule continuity
    schedule_params: dict[str, float] | None = None
    # Training loop state for seamless resume (early stopping, wandb run ID, etc.)
    training_state: dict | None = None
    # Optimizer type used when saving ("adamw", "adamw8bit", "muon").
    # None for checkpoints saved before this field was added.
    optimizer_type: str | None = None
    # Originating run name (cfg.run_name). Used to scope auto-resume so a run
    # only resumes its OWN checkpoints, never another run sharing the same phase
    # (e.g. all Phase 2 KB runs share phase="phase2_kb"). None for checkpoints
    # saved before this field was added — never cross-matched to a named run.
    run_name: str | None = None


def save_checkpoint(
    checkpoint_dir: Path,
    metadata: CheckpointMetadata,
    **state_dicts,
) -> Path:
    """Save a checkpoint with metadata.

    Writes to a temporary directory first, then atomically renames to the
    final name.  If the process is killed mid-save the temp directory
    (``._tmp_*``) is left behind and ignored by auto-resolve / backfill.

    Args:
        checkpoint_dir: Base directory for checkpoints.
        metadata: Phase and training metadata.
        **state_dicts: Named state dicts to save (model, optimizer, etc).

    Returns:
        Path to the saved checkpoint directory.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    ckpt_name = f"{metadata.phase}_step{metadata.step}_{timestamp}"
    tmp_path = checkpoint_dir / f"._tmp_{ckpt_name}"
    final_path = checkpoint_dir / ckpt_name
    tmp_path.mkdir(parents=True, exist_ok=True)

    # Save metadata
    meta_path = tmp_path / "metadata.json"
    meta_path.write_text(json.dumps(asdict(metadata), indent=2))

    # Save each state dict
    written: list[Path] = []
    for name, state_dict in state_dicts.items():
        fp = tmp_path / f"{name}.pt"
        torch.save(state_dict, fp)
        written.append(fp)

    # Flush the just-written files to stable storage BEFORE returning.
    #
    # ``torch.save`` only writes into the OS page cache; the actual write to
    # disk completes asynchronously. On a slow / rotational ``CHECKPOINT_DIR``
    # (DGX Spark's ``/mnt/external`` is a spinning HDD) a large checkpoint —
    # this phase saves ~15 GB, incl. a 12 GB Muon optimizer state for 1.586 B
    # params — leaves a multi-GB *dirty* page-cache backlog draining slowly to
    # the disk. On the 128 GB *unified* memory pool those dirty pages cannot be
    # reclaimed until writeback finishes, so the first post-save CUDA
    # allocation (``torch.empty_like`` inside the next forward) stalls trying to
    # reclaim them and the allocator retry-thrashes into a global OOM — the
    # recurring "first train step after checkpoint_saved" hang. ``fsync`` makes
    # the pages clean (instantly reclaimable) before training resumes, and also
    # makes the checkpoint durable across a crash. Cheap for the small
    # checkpoints other phases write; decisive for this one.
    for fp in written:
        fd = os.open(fp, os.O_RDONLY)
        try:
            os.fsync(fd)
            # Unified-memory: drop the just-written checkpoint pages from the
            # page cache. They are write-once and only re-read on the NEXT
            # process restart, so the ~6-9 GB of clean cache has no value here
            # and competes with the post-save training step's allocations in the
            # single shared LPDDR5X pool. DONTNEED returns those bytes to the
            # pool immediately instead of waiting for LRU eviction under
            # pressure.
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            except (AttributeError, OSError):
                pass  # posix_fadvise is Linux-only; best-effort
        finally:
            os.close(fd)

    # Atomic rename — if killed before this line, the temp dir is ignored
    tmp_path.rename(final_path)

    # Durably persist the rename (directory entry) too, so a crash immediately
    # after this function can't leave the checkpoint un-findable.
    dir_fd = os.open(checkpoint_dir, os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)

    logger.info(
        "checkpoint_saved",
        path=str(final_path),
        step=metadata.step,
        keys=list(state_dicts.keys()),
    )
    return final_path


def load_checkpoint(checkpoint_path: Path) -> tuple[CheckpointMetadata, dict]:
    """Load a checkpoint and its metadata.

    Args:
        checkpoint_path: Path to checkpoint directory.

    Returns:
        (metadata, state_dicts) tuple where state_dicts maps name -> state_dict.
    """
    meta_path = checkpoint_path / "metadata.json"
    meta_dict = json.loads(meta_path.read_text())
    # Filter to known fields for forward-compatibility: a checkpoint saved by
    # newer code may contain extra metadata keys unknown to this version.
    known_fields = {f.name for f in CheckpointMetadata.__dataclass_fields__.values()}
    metadata = CheckpointMetadata(**{k: v for k, v in meta_dict.items() if k in known_fields})

    state_dicts = {}
    for pt_file in sorted(checkpoint_path.glob("*.pt")):
        name = pt_file.stem
        state_dicts[name] = torch.load(pt_file, map_location="cpu", weights_only=True)

    logger.info(
        "checkpoint_loaded",
        path=str(checkpoint_path),
        step=metadata.step,
        keys=list(state_dicts.keys()),
    )
    return metadata, state_dicts
