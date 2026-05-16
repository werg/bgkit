"""Checkpoint pruning: keep best-K by metric + latest-N, delete the rest."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import structlog

logger = structlog.get_logger()


@dataclass
class _CheckpointRecord:
    path: Path
    step: int
    metrics: dict[str, float] | None = None


class CheckpointManager:
    """Manages checkpoint lifecycle: recording, ranking, and pruning.

    Args:
        keep_best: Number of best checkpoints to keep (by eval metric).
        keep_latest: Number of most-recent checkpoints to keep.
        metric: Eval metric key for best-K ranking (e.g. ``"eval/mse"``).
        lower_is_better: If True, lower metric values are better.
        registry: Optional CheckpointRegistry to notify on prune.
    """

    def __init__(
        self,
        keep_best: int = 3,
        keep_latest: int = 2,
        metric: str = "eval/loss",
        lower_is_better: bool = True,
        phase: str | None = None,
        registry=None,
    ) -> None:
        self.keep_best = keep_best
        self.keep_latest = keep_latest
        self.metric = metric
        self.lower_is_better = lower_is_better
        self.phase = phase
        self._registry = registry
        self._records: list[_CheckpointRecord] = []

    def record(self, path: Path, step: int, metrics: dict[str, float] | None = None) -> None:
        """Register a saved checkpoint."""
        self._records.append(_CheckpointRecord(path=Path(path), step=step, metrics=metrics))

    def load_existing(self, checkpoint_dir: Path) -> None:
        """Scan existing checkpoints and populate records from metadata.json.

        Only loads checkpoints matching ``self.phase`` (if set), so checkpoints
        from other training phases in the same directory are never pruned.
        Records are sorted by step so that ``keep_latest`` works correctly
        regardless of filesystem or lexicographic ordering.
        """
        if not checkpoint_dir.exists():
            return

        loaded: list[_CheckpointRecord] = []
        for meta_file in checkpoint_dir.glob("*/metadata.json"):
            ckpt_path = meta_file.parent
            try:
                meta = json.loads(meta_file.read_text())
                # Skip checkpoints from other phases
                if self.phase is not None and meta.get("phase") != self.phase:
                    continue
                step = meta.get("step", 0)
                metrics = meta.get("metrics", None)
                loaded.append(
                    _CheckpointRecord(path=ckpt_path, step=step, metrics=metrics)
                )
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "checkpoint_metadata_error", path=str(ckpt_path), error=str(exc)
                )

        # Sort by step so keep_latest selects truly recent checkpoints
        loaded.sort(key=lambda r: r.step)
        self._records.extend(loaded)

        logger.info("checkpoint_manager_loaded", count=len(loaded))

    def prune(self) -> list[Path]:
        """Delete checkpoints beyond keep_best + keep_latest.

        Returns:
            List of deleted checkpoint paths.
        """
        if not self._records:
            return []

        keep: set[Path] = set()

        # Keep latest N (by insertion order, which is chronological)
        if self.keep_latest > 0:
            for rec in self._records[-self.keep_latest :]:
                keep.add(rec.path)

        # Keep best K (only among those with metrics containing the target key)
        scored = [
            rec for rec in self._records
            if rec.metrics is not None and self.metric in rec.metrics
        ]
        scored.sort(key=lambda r: r.metrics[self.metric], reverse=not self.lower_is_better)
        for rec in scored[: self.keep_best]:
            keep.add(rec.path)

        # Loss-reweighting sanity check: if the metric values across the
        # CURRENT candidate set span >5x in magnitude, the metric scale
        # likely changed mid-run (e.g. loss reweighting). Ranking older
        # pre-reweight checkpoints against newer post-reweight ones by
        # raw value is misleading and was the 2026-05-12 / 2026-05-14
        # Phase C pitfall. Warn loudly.
        if len(scored) >= 2:
            vals = [r.metrics[self.metric] for r in scored]
            non_zero_vals = [abs(v) for v in vals if v != 0]
            if non_zero_vals:
                scale = max(non_zero_vals) / max(min(non_zero_vals), 1e-9)
                if scale > 5.0:
                    logger.warning(
                        "checkpoint_metric_scale_shift_detected",
                        metric=self.metric,
                        scale_ratio=scale,
                        min_value=min(vals),
                        max_value=max(vals),
                        hint=(
                            "Magnitude of the pruning metric varies >5x across "
                            "registered checkpoints — likely a loss reweighting "
                            "happened mid-run. Pruning by this metric will favor "
                            "the regime with the smaller absolute value, which "
                            "may NOT be the better-trained checkpoint. Consider "
                            "switching the metric to a sign/scale-stable signal "
                            "like eval/cos_sim or eval/perplexity, or manually "
                            "delete pre-reweighting registry entries."
                        ),
                    )

        # Delete the rest
        deleted: list[Path] = []
        surviving: list[_CheckpointRecord] = []
        for rec in self._records:
            if rec.path in keep:
                surviving.append(rec)
            else:
                if rec.path.exists():
                    shutil.rmtree(rec.path)
                    logger.info("checkpoint_pruned", path=str(rec.path), step=rec.step)
                if self._registry is not None:
                    self._registry.mark_pruned(rec.path.name)
                deleted.append(rec.path)

        self._records = surviving
        return deleted
