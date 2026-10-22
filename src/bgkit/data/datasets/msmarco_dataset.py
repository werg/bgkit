"""MS MARCO Phase 2 dataset wrapper.

Adds MS MARCO-specific functionality on top of Phase2QADataset:
- Relevance judgments (``is_selected``) from metadata
- ``query_id`` exposure for QueryAwareBatchSampler
- Distractor sampling via ``get_relevant_doc_ids``

The metadata parquet is expected to contain ``is_selected_json`` (JSON-encoded
list of 0/1 per passage) and ``query_id`` columns.  When these columns are
absent the wrapper degrades gracefully -- ``relevance_judgments`` returns an
empty list and ``query_id`` falls back to the sample id.
"""

from __future__ import annotations

import json
from collections import defaultdict

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample


class MSMARCODataset(Phase2QADataset):
    """MS MARCO passage retrieval dataset.

    Extends the generic loader with:

    * ``relevance_judgments`` -- per-sample list of 0/1 flags indicating which
      of the original passages were marked relevant by annotators.
    * ``query_id`` injected into each sample's ``metadata`` dict so that
      ``QueryAwareBatchSampler`` can group by query.
    * ``get_relevant_doc_ids(query_idx)`` -- returns document indices that
      share the same ``query_id`` as the given sample *and* have at least one
      selected passage, enabling curriculum-aware distractor selection.
    """

    # Additional metadata columns we need beyond the base set.
    _EXTRA_COLUMNS = ("is_selected_json", "query_id", "query_type")

    def __init__(self, data_dir: str, **kwargs):
        # Ensure our extra columns are loaded from the parquet (if they exist).
        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name="msmarco_passage",
            metadata_columns=columns,
            **kwargs,
        )

        # Build query-id -> [valid_idx, ...] mapping for distractor sampling.
        self._query_to_indices: dict[str, list[int]] = defaultdict(list)
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            qid = str(meta.get("query_id", meta.get("id", orig_idx)))
            self._query_to_indices[qid].append(idx)

    # ------------------------------------------------------------------
    # Relevance judgments
    # ------------------------------------------------------------------

    @property
    def relevance_judgments(self) -> list[list[int]]:
        """Per-sample relevance flags from the ``is_selected_json`` metadata column.

        Returns a list aligned with ``__len__`` (i.e. valid indices only).
        Each entry is a list of ints (0 or 1) corresponding to the original
        passage ordering.  Returns an empty inner list when the column is
        missing or unparseable.
        """
        results: list[list[int]] = []
        for idx in range(len(self)):
            orig_idx = int(self._valid_indices[idx])
            meta = self._metadata_rows[orig_idx]
            raw = meta.get("is_selected_json")
            if raw:
                try:
                    parsed = json.loads(str(raw))
                    if isinstance(parsed, list):
                        results.append([int(v) for v in parsed])
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            results.append([])
        return results

    # ------------------------------------------------------------------
    # Distractor / relevant-doc helpers
    # ------------------------------------------------------------------

    def get_relevant_doc_ids(self, query_idx: int) -> list[int]:
        """Return dataset indices sharing the same query_id as *query_idx*.

        Only includes indices whose ``is_selected_json`` contains at least one
        positive (``1``) flag.  Excludes *query_idx* itself.
        """
        orig_idx = int(self._valid_indices[query_idx])
        meta = self._metadata_rows[orig_idx]
        qid = str(meta.get("query_id", meta.get("id", orig_idx)))

        relevant: list[int] = []
        for sibling_idx in self._query_to_indices.get(qid, []):
            if sibling_idx == query_idx:
                continue
            sibling_orig = int(self._valid_indices[sibling_idx])
            sibling_meta = self._metadata_rows[sibling_orig]
            raw = sibling_meta.get("is_selected_json")
            if raw:
                try:
                    selected = json.loads(str(raw))
                    if isinstance(selected, list) and any(int(v) for v in selected):
                        relevant.append(sibling_idx)
                        continue
                except (json.JSONDecodeError, ValueError):
                    pass
            # When no is_selected data, include by default
            relevant.append(sibling_idx)
        return relevant

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]

        # Inject query_id so QueryAwareBatchSampler can read it from metadata.
        sample.metadata["query_id"] = str(
            meta.get("query_id", meta.get("id", orig_idx)),
        )

        # Attach relevance flags directly when available.
        raw = meta.get("is_selected_json")
        if raw:
            try:
                parsed = json.loads(str(raw))
                if isinstance(parsed, list):
                    sample.metadata["is_selected"] = [int(v) for v in parsed]
            except (json.JSONDecodeError, ValueError):
                pass

        # Surface query_type tag if present.
        qt = meta.get("query_type")
        if qt and str(qt) not in sample.tags:
            sample.tags.append(str(qt))

        return sample
