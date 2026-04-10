"""SearchQA Phase 2 dataset wrapper.

Adds SearchQA-specific functionality on top of Phase2QADataset:
- Multi-snippet handling: metadata about how many snippets are packed
  together in the document field
- ``query_id`` exposure for QueryAwareBatchSampler

SearchQA stores multiple web-search snippets per query concatenated into
the document token stream (separated by double newlines in the source text).
The ``snippet_count`` metadata column (when present) records how many
snippets are packed.
"""

from __future__ import annotations

from collections import defaultdict

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample


class SearchQADataset(Phase2QADataset):
    """SearchQA web-snippet retrieval dataset.

    Extends the generic loader with:

    * ``snippet_count(idx)`` -- returns the number of web snippets packed
      into the document for sample *idx*.
    * ``query_id`` injected into each sample's ``metadata`` dict for use
      by ``QueryAwareBatchSampler``.
    * ``query_indices`` -- mapping from query_id to list of dataset indices.
    """

    _EXTRA_COLUMNS = ("query_id", "snippet_count")

    def __init__(self, data_dir: str, **kwargs):
        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name="searchqa",
            metadata_columns=columns,
            **kwargs,
        )

        # Build query_id -> [valid_idx, ...] mapping.
        self._query_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            qid = str(meta.get("query_id", meta.get("id", orig_idx)))
            self._query_to_indices[qid].append(idx)

    # ------------------------------------------------------------------
    # Per-sample accessors
    # ------------------------------------------------------------------

    def snippet_count(self, idx: int) -> int:
        """Return the number of snippets packed into sample *idx*'s document.

        Falls back to ``1`` when the ``snippet_count`` column is absent.
        """
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        raw = meta.get("snippet_count")
        if raw is not None:
            try:
                return int(raw)
            except (ValueError, TypeError):
                pass
        return 1

    # ------------------------------------------------------------------
    # Query grouping
    # ------------------------------------------------------------------

    @property
    def query_indices(self) -> dict[str, list[int]]:
        """Mapping from query_id to list of dataset indices."""
        return dict(self._query_to_indices)

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]

        # Inject query_id so QueryAwareBatchSampler can read it.
        sample.metadata["query_id"] = str(
            meta.get("query_id", meta.get("id", orig_idx)),
        )

        # Inject snippet_count for downstream multi-snippet handling.
        sample.metadata["snippet_count"] = self.snippet_count(idx)

        return sample
