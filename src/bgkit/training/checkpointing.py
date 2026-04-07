"""Save/load checkpoints with phase metadata and lineage tracking."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import structlog
import torch

logger = structlog.get_logger()


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""

    phase: str  # e.g. "phase1_step4", "phase2"
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
    for name, state_dict in state_dicts.items():
        torch.save(state_dict, tmp_path / f"{name}.pt")

    # Atomic rename — if killed before this line, the temp dir is ignored
    tmp_path.rename(final_path)

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
