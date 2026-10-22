"""Multi-session user-memory dataset wrapper.

Adds memory-dataset-specific functionality on top of Phase2QADataset:
- Per-sample ``memory_type`` from metadata
- Episode-level grouping for session assembly
- ``get_episode_sessions(episode_id)`` for retrieving all sessions in an episode
- Answer-containing session selection and distractor session sampling

The metadata parquet is expected to contain ``memory_type`` and either
``document_id`` or ``id`` as episode identifiers (written by
``convert_memory_datasets.py``).
"""

from __future__ import annotations

import random
from collections import defaultdict

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample


class MemoryDataset(Phase2QADataset):
    """Multi-session conversational memory dataset.

    Covers MSC, SHARE, Chronicles, PerLTQA, and LAPS memory benchmarks.

    Extends the generic loader with:

    * ``memory_type(idx)`` -- returns the memory annotation type for a
      sample (e.g. ``persona``, ``persona_entries``, ``preference``,
      ``knowledge_update``, ``temporal``, ``profile``, etc.).
    * ``episode_indices`` -- mapping from episode ID to list of dataset
      indices belonging to that episode.
    * ``get_episode_sessions(episode_id)`` -- returns all sample indices
      for a given episode, enabling session-level operations.
    * ``get_answer_sessions(episode_id)`` -- returns indices likely to
      contain the answer (useful for answer-containing session selection).
    * ``get_distractor_sessions(episode_id, exclude, n)`` -- samples
      distractor session indices from the same episode.
    """

    # Columns we need from the parquet beyond the base set.
    _EXTRA_COLUMNS = ("memory_type",)

    def __init__(self, data_dir: str, dataset_name: str = "memory", **kwargs):
        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name=dataset_name,
            metadata_columns=columns,
            **kwargs,
        )

        # Build episode -> [valid_idx, ...] mapping.
        # Use document_id as episode identifier (convert_memory_datasets sets
        # document_id = episode_id).
        self._episode_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            episode_id = str(
                meta.get("document_id", meta.get("id", orig_idx)),
            )
            self._episode_to_indices[episode_id].append(idx)

    # ------------------------------------------------------------------
    # Per-sample accessors
    # ------------------------------------------------------------------

    def memory_type(self, idx: int) -> str:
        """Return the memory annotation type for sample *idx*.

        Common types (from ``convert_memory_datasets.py``):
        ``persona``, ``persona_entries``, ``personal_events``,
        ``mutual_events``, ``shared_memories``, ``knowledge_update``,
        ``temporal``, ``preference``, ``profile``.
        """
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        return str(meta.get("memory_type", "unknown"))

    # ------------------------------------------------------------------
    # Episode grouping
    # ------------------------------------------------------------------

    @property
    def episode_indices(self) -> dict[str, list[int]]:
        """Mapping from episode ID to list of dataset indices."""
        return dict(self._episode_to_indices)

    @property
    def episode_ids(self) -> list[str]:
        """Sorted list of unique episode identifiers in this dataset."""
        return sorted(self._episode_to_indices.keys())

    def get_episode_sessions(self, episode_id: str) -> list[int]:
        """Return all dataset indices belonging to *episode_id*.

        Indices are in insertion order (matching the order sessions were
        added during conversion).

        Args:
            episode_id: The episode identifier (matches ``document_id``
                in metadata).

        Returns:
            List of dataset indices, possibly empty if *episode_id* is
            not found.
        """
        return list(self._episode_to_indices.get(episode_id, []))

    # ------------------------------------------------------------------
    # Answer-containing session selection
    # ------------------------------------------------------------------

    def get_answer_sessions(self, episode_id: str) -> list[int]:
        """Return indices within *episode_id* that are likely answer sources.

        Heuristic: sessions whose ``memory_type`` indicates direct factual
        content (``persona``, ``persona_entries``, ``profile``,
        ``knowledge_update``) are preferred as answer-containing sessions.
        When no such sessions exist, all sessions for the episode are returned.

        This supports the Phase 2 Track C training strategy where the model
        must identify which past session(s) contain the information needed
        to answer the question.
        """
        answer_types = frozenset({
            "persona",
            "persona_entries",
            "profile",
            "knowledge_update",
            "preference",
        })

        indices = self._episode_to_indices.get(episode_id, [])
        if not indices:
            return []

        answer_indices = []
        for idx in indices:
            mt = self.memory_type(idx)
            if mt in answer_types:
                answer_indices.append(idx)

        # Fall back to all sessions when no type-based filtering matched.
        return answer_indices if answer_indices else list(indices)

    # ------------------------------------------------------------------
    # Distractor session sampling
    # ------------------------------------------------------------------

    def get_distractor_sessions(
        self,
        episode_id: str,
        exclude: set[int] | frozenset[int] | None = None,
        n: int = 3,
        *,
        seed: int | None = None,
    ) -> list[int]:
        """Sample distractor session indices from the same episode.

        Distractors are sessions from the same episode that are NOT in
        *exclude* (typically the answer-containing sessions).  This
        enables a training curriculum where the model must distinguish
        relevant from irrelevant context within the same conversation
        history.

        Args:
            episode_id: The episode to sample distractors from.
            exclude: Set of dataset indices to exclude (e.g. the
                answer-containing sessions).
            n: Maximum number of distractors to return.
            seed: Optional RNG seed for reproducibility.

        Returns:
            List of dataset indices (up to *n*).
        """
        exclude = exclude or set()
        indices = self._episode_to_indices.get(episode_id, [])
        candidates = [idx for idx in indices if idx not in exclude]

        if not candidates:
            return []

        rng = random.Random(seed)
        return rng.sample(candidates, min(n, len(candidates)))

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]

        # Inject memory_type and episode_id into sample metadata.
        mt = str(meta.get("memory_type", "unknown"))
        sample.metadata["memory_type"] = mt
        sample.metadata["episode_id"] = str(
            meta.get("document_id", meta.get("id", orig_idx)),
        )

        # Add memory_type as a tag.
        if mt != "unknown" and mt not in sample.tags:
            sample.tags.append(mt)

        return sample
