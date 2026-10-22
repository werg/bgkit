"""KILT Phase 2 dataset wrapper.

Adds KILT-specific functionality on top of Phase2QADataset:
- Per-sample ``task_name`` from metadata
- Parsed ``provenance`` (list of Wikipedia page IDs) from metadata
- Multi-task sampling weights via ``task_weights`` parameter
- Enriched ``metadata`` property with parsed provenance

The metadata parquet is expected to contain ``task_name`` and
``provenance_json`` columns (written by ``convert_hf_to_mmap.py``).
"""

from __future__ import annotations

import json
import random
from collections import defaultdict

import numpy as np

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample


class KILTDataset(Phase2QADataset):
    """KILT knowledge-intensive language tasks dataset.

    Extends the generic loader with:

    * ``task_name(idx)`` -- returns the KILT task name for a given sample.
    * ``provenance(idx)`` -- returns parsed list of Wikipedia page IDs that
      ground the answer.
    * Multi-task sampling support: ``task_weights`` maps task names to
      relative weights for curriculum-based task mixing.
    * ``task_indices`` -- mapping from task name to list of dataset indices
      for that task, useful for stratified evaluation.
    """

    # Columns we need from the parquet beyond the base set.
    _EXTRA_COLUMNS = ("task_name", "provenance_json")

    def __init__(
        self,
        data_dir: str,
        *,
        task_weights: dict[str, float] | None = None,
        **kwargs,
    ):
        # Ensure extra columns are loaded (if they exist in the parquet).
        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name="kilt",
            metadata_columns=columns,
            **kwargs,
        )

        self._task_weights = task_weights or {}

        # Build task -> valid-idx mapping.
        self._task_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            task = str(meta.get("task_name", "unknown"))
            self._task_to_indices[task].append(idx)

        # Pre-compute per-index sampling weights when task_weights are given.
        if self._task_weights:
            self._sample_weights = self._compute_sample_weights()
        else:
            self._sample_weights = None

    # ------------------------------------------------------------------
    # Per-sample accessors
    # ------------------------------------------------------------------

    def task_name(self, idx: int) -> str:
        """Return the KILT task name for sample *idx*."""
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        return str(meta.get("task_name", "unknown"))

    def provenance(self, idx: int) -> list[str]:
        """Return parsed Wikipedia page IDs grounding sample *idx*'s answer.

        Returns an empty list when the ``provenance_json`` column is absent or
        the value is not valid JSON.
        """
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        raw = meta.get("provenance_json")
        if not raw:
            return []
        try:
            parsed = json.loads(str(raw))
            if isinstance(parsed, list):
                return [str(pid) for pid in parsed]
        except (json.JSONDecodeError, ValueError):
            pass
        return []

    # ------------------------------------------------------------------
    # Task-level helpers
    # ------------------------------------------------------------------

    @property
    def task_indices(self) -> dict[str, list[int]]:
        """Mapping from task name to list of dataset indices for that task."""
        return dict(self._task_to_indices)

    @property
    def task_names(self) -> list[str]:
        """Sorted list of unique task names present in this dataset."""
        return sorted(self._task_to_indices.keys())

    @property
    def sample_weights(self) -> np.ndarray | None:
        """Per-sample weights for weighted random sampling.

        Returns ``None`` when no ``task_weights`` were provided.  Otherwise
        returns a float64 array of length ``len(self)`` where each element is
        proportional to the weight assigned to its task, normalized so that
        within each task the weight is ``task_weight / task_count``.
        """
        return self._sample_weights

    def _compute_sample_weights(self) -> np.ndarray:
        """Build per-sample weight array from task_weights."""
        weights = np.ones(len(self), dtype=np.float64)
        for task, indices in self._task_to_indices.items():
            task_weight = self._task_weights.get(task, 1.0)
            if indices:
                per_sample = task_weight / len(indices)
                for idx in indices:
                    weights[idx] = per_sample
        # Normalize so weights sum to len(self).
        total = weights.sum()
        if total > 0:
            weights *= len(self) / total
        return weights

    def sample_task_balanced(
        self,
        n: int,
        *,
        seed: int = 42,
    ) -> list[int]:
        """Sample *n* indices with task-balanced weighting.

        When ``task_weights`` are configured, samples proportionally.
        Otherwise, samples uniformly from each task in round-robin fashion.
        """
        rng = random.Random(seed)

        if self._sample_weights is not None:
            # Weighted sampling.
            population = list(range(len(self)))
            w = self._sample_weights.tolist()
            return rng.choices(population, weights=w, k=n)

        # Round-robin across tasks.
        task_pools: dict[str, list[int]] = {
            task: list(indices) for task, indices in self._task_to_indices.items()
        }
        for pool in task_pools.values():
            rng.shuffle(pool)

        result: list[int] = []
        task_order = sorted(task_pools.keys())
        pointers = {task: 0 for task in task_order}
        while len(result) < n:
            for task in task_order:
                if len(result) >= n:
                    break
                pool = task_pools[task]
                if not pool:
                    continue
                ptr = pointers[task] % len(pool)
                result.append(pool[ptr])
                pointers[task] = ptr + 1
        return result

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]

        # Inject task_name.
        task = str(meta.get("task_name", "unknown"))
        sample.metadata["task_name"] = task

        # Parse and inject provenance.
        raw = meta.get("provenance_json")
        if raw:
            try:
                parsed = json.loads(str(raw))
                if isinstance(parsed, list):
                    sample.metadata["provenance"] = [str(pid) for pid in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # Add task as a tag if not already present.
        if task != "unknown" and task not in sample.tags:
            sample.tags.append(task)

        return sample

    # ------------------------------------------------------------------
    # Enriched metadata property
    # ------------------------------------------------------------------

    @property
    def metadata(self) -> list[dict[str, object]]:
        """Metadata list enriched with parsed provenance.

        Each dict in the returned list has a ``provenance`` key containing a
        list of Wikipedia page-ID strings (possibly empty) in addition to all
        columns loaded from the parquet.
        """
        enriched: list[dict[str, object]] = []
        for row in self._metadata_rows:
            entry = dict(row)
            raw = entry.get("provenance_json")
            if raw:
                try:
                    parsed = json.loads(str(raw))
                    if isinstance(parsed, list):
                        entry["provenance"] = [str(pid) for pid in parsed]
                    else:
                        entry["provenance"] = []
                except (json.JSONDecodeError, ValueError):
                    entry["provenance"] = []
            else:
                entry["provenance"] = []
            enriched.append(entry)
        return enriched
