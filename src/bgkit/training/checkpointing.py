"""Save/load checkpoints with phase metadata and lineage tracking."""

from __future__ import annotations

import contextlib
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

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


def write_checkpoint_files(
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
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    run_suffix = ""
    if metadata.run_name:
        safe_run = re.sub(r"[^A-Za-z0-9_.-]+", "-", metadata.run_name).strip("-.")
        if safe_run:
            run_suffix = f"_run-{safe_run[:80]}"
    ckpt_name = f"{metadata.phase}_step{metadata.step}_{timestamp}{run_suffix}"
    tmp_path = checkpoint_dir / f"._tmp_{ckpt_name}"
    final_path = checkpoint_dir / ckpt_name
    if tmp_path.exists() or final_path.exists():
        # Extremely unlikely with microsecond timestamps, but two workers can
        # checkpoint the same run concurrently. Never merge payloads.
        ckpt_name = f"{ckpt_name}_{uuid4().hex[:8]}"
        tmp_path = checkpoint_dir / f"._tmp_{ckpt_name}"
        final_path = checkpoint_dir / ckpt_name
    tmp_path.mkdir(parents=True, exist_ok=False)

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
            with contextlib.suppress(AttributeError, OSError):
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
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


# Sub-state name -> the prefix it carries in the flat, single-``model`` layout.
# Phase-1 trainers save each managed model as its own ``<name>.pt`` with
# unprefixed keys; Phase-2 saves one flat ``model.pt``. Anything that reads
# ``state["model"]`` therefore raises KeyError on every Phase-1 checkpoint.
_PHASE1_SUBSTATE_PREFIXES = {
    "encoder": "encoder.",
    "decoder_qwen": "decoders.qwen35.",
    "decoder_falcon": "decoders.falcon_h1.",
    "decoder": "decoder.",
}


def normalize_model_state(state_dicts: dict) -> dict:
    """Return ``state_dicts`` with a single flat ``["model"]`` sub-state.

    TWO ON-DISK LAYOUTS EXIST and consumers kept assuming the newer one:

    * Phase-2 / KRKBTrainer     ``{"model": {...}, "optimizer_state_by_name": ...}``
    * Phase-1 (summarization,
      commit-encoding, ...)     ``{"encoder": {...}, "decoder_qwen": {...},
                                   "decoder_falcon": {...}, ...}`` with
                                  UNPREFIXED keys inside each sub-state.

    ``state_dicts["model"]`` on a Phase-1 checkpoint raises ``KeyError``. On
    2026-08-28 that turned every lineage comparison against the Phase-1 base
    into a silent ``SKIPPED (KeyError)`` row, so the one measurement that could
    distinguish "Phase-2 training broke the projection" from "it was never
    aligned" was never actually made — the question stayed open for a day on a
    missing three-line adapter.

    This is deliberately shared rather than fixed per-script: the same
    assumption is in ``analyze_embedding_deviation``, ``eval_phase1`` and
    ``probe_decoder_damage_locus``, and any future tool comparing across the
    Phase-1 -> Phase-2 boundary will hit it too.

    Idempotent: a checkpoint already in the flat layout is returned unchanged.
    """
    if "model" in state_dicts:
        return state_dicts
    merged: dict = {}
    for name, prefix in _PHASE1_SUBSTATE_PREFIXES.items():
        sub = state_dicts.get(name)
        if not isinstance(sub, dict):
            continue
        for k, v in sub.items():
            merged[f"{prefix}{k}"] = v
    if not merged:
        raise KeyError(
            "unrecognised checkpoint layout: expected a flat 'model' sub-state "
            f"or one of {sorted(_PHASE1_SUBSTATE_PREFIXES)}; got {sorted(state_dicts)}"
        )
    out = dict(state_dicts)
    out["model"] = merged
    return out


def save_checkpoint(*_args, **_kwargs):
    """REMOVED — use ``BaseTrainer._write_checkpoint`` instead.

    This was the copy-paste path. Because a module-level ``save_checkpoint``
    was importable, the cheapest way to give a trainer a different on-disk
    layout was to copy BaseTrainer.save_checkpoint and call this directly —
    which skips NVMe fast-dir routing and async HDD archival. Twelve trainers
    copied it; ELEVEN carried the resulting bug, and the fix made in one of
    them in June 2026 could not propagate to the others.

    Declare the layout with ``checkpoint_models()`` / ``checkpoint_extra_state()``
    and let BaseTrainer do the write. The low-level writer is still available as
    ``write_checkpoint_files`` for infrastructure that genuinely needs it.
    """
    raise RuntimeError(
        "save_checkpoint() was removed: it bypasses NVMe routing and archival. "
        "Declare layout via checkpoint_models()/checkpoint_extra_state(); "
        "BaseTrainer._write_checkpoint performs the write."
    )
