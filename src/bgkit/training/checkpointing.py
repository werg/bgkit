"""Save/load checkpoints with phase metadata and lineage tracking."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class CheckpointMetadata:
    """Metadata stored with each checkpoint."""

    phase: str  # e.g. "phase1_step2", "phase2"
    step: int
    epoch: int
    parent_checkpoint: str | None  # Lineage: which checkpoint this was initialized from
    metrics: dict[str, float] | None = None


def save_checkpoint(
    checkpoint_dir: Path,
    metadata: CheckpointMetadata,
    **state_dicts,
) -> Path:
    """Save a checkpoint with metadata.

    Args:
        checkpoint_dir: Base directory for checkpoints.
        metadata: Phase and training metadata.
        **state_dicts: Named state dicts to save (model, optimizer, etc).

    Returns:
        Path to the saved checkpoint directory.
    """
    # TODO: Implement with torch.save / safetensors
    raise NotImplementedError


def load_checkpoint(checkpoint_path: Path) -> tuple[CheckpointMetadata, dict]:
    """Load a checkpoint and its metadata.

    Args:
        checkpoint_path: Path to checkpoint directory.

    Returns:
        (metadata, state_dicts) tuple.
    """
    # TODO: Implement
    raise NotImplementedError
