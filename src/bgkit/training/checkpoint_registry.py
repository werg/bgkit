"""Checkpoint registry: persistent record of all checkpoints across training runs.

Survives checkpoint pruning. No torch dependency (runs on host for CLI).
"""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import structlog

logger = structlog.get_logger()

_SCHEMA_VERSION = 1


def resolve_checkpoint(
    checkpoint_dir: Path,
    phase: str,
    metric: str,
    lower_is_better: bool = True,
    label: str = "",
) -> Path:
    """Backfill registry, find best on-disk checkpoint for a phase.

    Raises ValueError if no checkpoint found.
    """
    registry = CheckpointRegistry(checkpoint_dir)
    registry.backfill(checkpoint_dir)
    best = registry.best(phase=phase, metric=metric, lower_is_better=lower_is_better)
    if best is None:
        raise ValueError(
            f"{label or phase} checkpoint_path=auto but no {phase} checkpoint found "
            f"in registry or on disk. Run {phase} training first, or set an explicit path."
        )
    return checkpoint_dir / best.name


def resolve_latest_checkpoint(
    checkpoint_dir: Path,
    phase: str,
    fast_dir: Path | None = None,
) -> Path | None:
    """Find the latest on-disk checkpoint for a phase, or None if no match.

    Used for auto-resume: silently returns None when no checkpoint exists
    (first run), unlike resolve_checkpoint which raises.

    When ``fast_dir`` (the NVMe live dir) is given, both it and
    ``checkpoint_dir`` (the HDD archive) are scanned, and the physical path
    prefers the NVMe copy when present (faster resume; also the only intact copy
    if the process died mid-archive). Recent checkpoints live on NVMe; older
    ones only on the HDD — both resolve correctly.
    """
    registry = CheckpointRegistry(checkpoint_dir)
    extra = [Path(fast_dir)] if fast_dir is not None else None
    registry.backfill(checkpoint_dir, extra_dirs=extra)
    latest = registry.latest(phase=phase)
    if latest is None:
        return None
    if fast_dir is not None:
        fast_path = Path(fast_dir) / latest.name
        if fast_path.exists():
            return fast_path
    return checkpoint_dir / latest.name


def normalize_checkpoint_name(path_or_name: str) -> str:
    """Extract checkpoint directory name from an absolute path or bare name.

    Examples:
        "/workspace/checkpoints/ice_step29999_..." -> "ice_step29999_..."
        "ice_step29999_..." -> "ice_step29999_..."
    """
    return Path(path_or_name).name


@dataclass
class RegistryEntry:
    """A single checkpoint record in the registry."""

    # Identity
    name: str  # checkpoint dir name
    phase: str
    step: int
    epoch: int
    timestamp: str  # ISO-8601

    # Status
    status: str = "completed"  # completed | interrupted | pruned
    on_disk: bool = True

    # Metrics and config
    metrics: dict[str, float] | None = None
    config_snapshot: dict | None = None
    wandb_run_id: str | None = None
    disk_size_bytes: int | None = None

    # Lineage
    parent_checkpoint: str | None = None  # within-phase, dir name
    input_sources: dict[str, str] | None = None  # cross-phase, e.g. {"ice": "ice_step29999_..."}

    # Annotations
    notes: str | None = None
    tags: list[str] = field(default_factory=list)


class CheckpointRegistry:
    """Persistent registry of all checkpoints ever saved.

    Stores a ``registry.json`` file in the checkpoint directory.
    Entries survive pruning (status changes to ``pruned``, ``on_disk`` to False).

    Args:
        checkpoint_dir: Directory containing checkpoints and registry.json.
    """

    def __init__(self, checkpoint_dir: Path) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._registry_path = self._checkpoint_dir / "registry.json"
        self._entries: dict[str, RegistryEntry] = {}
        self._load()

    def _load(self) -> None:
        """Load registry from disk if it exists."""
        if not self._registry_path.exists():
            return
        try:
            data = json.loads(self._registry_path.read_text())
            for entry_dict in data.get("entries", []):
                # Forward-compat: ignore unknown fields
                known = {f.name for f in RegistryEntry.__dataclass_fields__.values()}
                filtered = {k: v for k, v in entry_dict.items() if k in known}
                entry = RegistryEntry(**filtered)
                self._entries[entry.name] = entry
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("registry_load_error", path=str(self._registry_path), error=str(exc))

    def _save(self) -> None:
        """Atomic write: write to temp file then rename."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "version": _SCHEMA_VERSION,
            "updated_at": datetime.now(UTC).isoformat(),
            "entries": [asdict(e) for e in self._entries.values()],
        }
        content = json.dumps(data, indent=2)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._checkpoint_dir), suffix=".tmp", prefix=".registry_"
        )
        closed = False
        try:
            os.write(fd, content.encode())
            os.close(fd)
            closed = True
            os.replace(tmp_path, str(self._registry_path))
        except BaseException:
            if not closed:
                os.close(fd)
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
            raise

    def register(self, entry: RegistryEntry) -> None:
        """Add or update a registry entry and persist."""
        self._entries[entry.name] = entry
        self._save()

    def mark_pruned(self, name: str) -> None:
        """Mark a checkpoint as pruned (no longer on disk)."""
        entry = self._entries.get(name)
        if entry is not None:
            entry.status = "pruned"
            entry.on_disk = False
            self._save()

    def get(self, name: str) -> RegistryEntry | None:
        """Look up a single entry by checkpoint name."""
        return self._entries.get(name)

    def list_entries(
        self,
        phase: str | None = None,
        status: str | None = None,
        tags: list[str] | None = None,
    ) -> list[RegistryEntry]:
        """List entries with optional filters, sorted by step."""
        results = list(self._entries.values())
        if phase is not None:
            results = [e for e in results if e.phase == phase]
        if status is not None:
            results = [e for e in results if e.status == status]
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.issubset(set(e.tags))]
        results.sort(key=lambda e: e.step)
        return results

    def best(
        self,
        phase: str,
        metric: str,
        lower_is_better: bool = True,
        on_disk_only: bool = True,
    ) -> RegistryEntry | None:
        """Find the best checkpoint by a metric.

        Args:
            phase: Filter to this training phase.
            metric: Metric key (e.g. "eval/mse").
            lower_is_better: If True, lower metric values are better.
            on_disk_only: If True (default), exclude pruned checkpoints.

        Returns:
            Best RegistryEntry or None if no matching entries.
        """
        candidates = [
            e for e in self._entries.values()
            if e.phase == phase
            and e.metrics is not None
            and metric in e.metrics
        ]
        if on_disk_only:
            candidates = [e for e in candidates if e.on_disk]
        if not candidates:
            return None
        return min(candidates, key=lambda e: e.metrics[metric] * (1 if lower_is_better else -1))

    def latest(
        self,
        phase: str,
        on_disk_only: bool = True,
    ) -> RegistryEntry | None:
        """Find the latest (highest step) checkpoint for a phase.

        Args:
            phase: Filter to this training phase.
            on_disk_only: If True (default), exclude pruned checkpoints.

        Returns:
            Latest RegistryEntry or None if no matching entries.
        """
        candidates = [e for e in self._entries.values() if e.phase == phase]
        if on_disk_only:
            candidates = [e for e in candidates if e.on_disk]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.step)

    def annotate(
        self,
        name: str,
        notes: str | None = None,
        tags: list[str] | None = None,
    ) -> bool:
        """Add notes and/or tags to an existing entry.

        Returns:
            True if the entry was found and updated, False otherwise.
        """
        entry = self._entries.get(name)
        if entry is None:
            return False
        if notes is not None:
            entry.notes = notes
        if tags is not None:
            existing = set(entry.tags)
            existing.update(tags)
            entry.tags = sorted(existing)
        self._save()
        return True

    def backfill(
        self,
        checkpoint_dir: Path | None = None,
        extra_dirs: list[Path] | None = None,
    ) -> int:
        """Scan on-disk checkpoints and reconcile with registry.

        - Registers on-disk checkpoints not yet in the registry.
        - Marks registry entries whose directories no longer exist on disk
          as pruned (``on_disk=False``), so ``best()`` won't return them.

        Args:
            checkpoint_dir: Directory to scan. Defaults to the registry's checkpoint_dir.
            extra_dirs: Additional directories to scan (e.g. the NVMe live dir).
                A checkpoint present in ANY scanned dir counts as on-disk, so a
                checkpoint that exists only on NVMe (HDD archive still in flight)
                is never falsely marked pruned.

        Returns:
            Number of newly registered entries.
        """
        primary = Path(checkpoint_dir) if checkpoint_dir is not None else self._checkpoint_dir
        scan_dirs = [primary] + [Path(d) for d in (extra_dirs or [])]
        scan_dirs = [d for d in scan_dirs if d.exists()]
        if not scan_dirs:
            return 0

        def _present(name: str) -> bool:
            return any((d / name).exists() for d in scan_dirs)

        # Reconcile: mark entries whose dirs are gone (from EVERY scanned dir) as pruned
        dirty = False
        for entry in self._entries.values():
            if entry.on_disk and not _present(entry.name):
                entry.on_disk = False
                entry.status = "pruned"
                dirty = True
                logger.info("registry_reconcile_pruned", name=entry.name)
        # Recovery: un-prune entries whose dirs have reappeared (e.g. restored from backup).
        # Note: if status was "interrupted" before pruning, mark_pruned() overwrote it to
        # "pruned". We can't distinguish that case, so we recover to "completed" as the
        # pragmatic default.
        for entry in self._entries.values():
            if not entry.on_disk and _present(entry.name):
                entry.on_disk = True
                if entry.status == "pruned":
                    entry.status = "completed"
                dirty = True
                logger.info("registry_reconcile_recovered", name=entry.name)

        if dirty:
            self._save()

        count = 0
        seen_this_scan: set[str] = set()
        meta_files = [mf for d in scan_dirs for mf in d.glob("*/metadata.json")]
        for meta_file in meta_files:
            ckpt_dir = meta_file.parent
            name = ckpt_dir.name
            # Skip incomplete checkpoints (atomic save / archive not yet renamed)
            if name.startswith("._"):
                continue
            if name in self._entries or name in seen_this_scan:
                continue
            seen_this_scan.add(name)

            try:
                meta = json.loads(meta_file.read_text())
            except (json.JSONDecodeError, OSError):
                continue

            # Compute disk size
            disk_size = sum(f.stat().st_size for f in ckpt_dir.rglob("*") if f.is_file())

            # Normalize parent_checkpoint from absolute path to dir name
            parent = meta.get("parent_checkpoint")
            if parent is not None:
                parent = normalize_checkpoint_name(parent)

            # Extract wandb_run_id from training_state if present
            training_state = meta.get("training_state") or {}
            wandb_run_id = training_state.get("wandb_run_id")

            # Build config snapshot from schedule_params if available
            config_snapshot = None
            schedule = meta.get("schedule_params")
            if schedule:
                config_snapshot = {
                    "phase": meta.get("phase", "unknown"),
                    **schedule,
                }

            entry = RegistryEntry(
                name=name,
                phase=meta.get("phase", "unknown"),
                step=meta.get("step", 0),
                epoch=meta.get("epoch", 0),
                timestamp=_infer_timestamp(name),
                status="completed",
                on_disk=True,
                metrics=meta.get("metrics"),
                config_snapshot=config_snapshot,
                wandb_run_id=wandb_run_id,
                disk_size_bytes=disk_size,
                parent_checkpoint=parent,
            )
            self._entries[name] = entry
            count += 1

        if count > 0:
            self._save()
            logger.info("registry_backfill", new_entries=count)
        return count


def _infer_timestamp(name: str) -> str:
    """Try to parse timestamp from checkpoint dir name (e.g. ice_step29999_20260220_220522).

    Falls back to empty string if parsing fails.
    """
    parts = name.split("_")
    if len(parts) >= 4:
        # Last two parts should be date and time
        date_part = parts[-2]
        time_part = parts[-1]
        if len(date_part) == 8 and len(time_part) == 6:
            try:
                dt = datetime.strptime(f"{date_part}_{time_part}", "%Y%m%d_%H%M%S")
                return dt.replace(tzinfo=UTC).isoformat()
            except ValueError:
                pass
    return ""
