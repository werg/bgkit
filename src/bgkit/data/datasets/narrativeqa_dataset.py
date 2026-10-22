"""NarrativeQA Phase 2 dataset wrapper.

Adds NarrativeQA-specific functionality on top of Phase2QADataset:
- ``max_document_len`` override to 65536 (long documents -- books and
  movie scripts)
- Per-sample ``document_type`` from metadata (book vs movie script)

The metadata parquet may contain a ``document_type`` column.  When absent,
the wrapper infers the type heuristically from the ``document_id`` or
defaults to ``"unknown"``.
"""

from __future__ import annotations

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset, merge_metadata_columns
from bgkit.data.datasets.qa_sample import QASample

# NarrativeQA documents are full books or movie scripts -- much longer than
# the default 8192 token cap used by other Phase 2 datasets.
_DEFAULT_NARRATIVEQA_MAX_DOC_LEN = 65536


class NarrativeQADataset(Phase2QADataset):
    """NarrativeQA long-document QA dataset.

    Extends the generic loader with:

    * ``max_document_len`` defaulting to 65536 (vs the generic 8192) to
      accommodate full book / movie-script documents.
    * ``document_type(idx)`` -- returns ``"book"`` or ``"movie_script"``
      for sample *idx*, derived from metadata.
    """

    _EXTRA_COLUMNS = ("document_type",)

    def __init__(self, data_dir: str, **kwargs):
        # Override max_document_len default for long documents.
        if "max_document_len" not in kwargs:
            kwargs["max_document_len"] = _DEFAULT_NARRATIVEQA_MAX_DOC_LEN

        explicit = kwargs.pop("metadata_columns", None)
        columns = merge_metadata_columns(data_dir, self._EXTRA_COLUMNS, explicit)
        super().__init__(
            data_dir,
            dataset_name="narrativeqa",
            metadata_columns=columns,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Per-sample accessors
    # ------------------------------------------------------------------

    def document_type(self, idx: int) -> str:
        """Return the document type for sample *idx*.

        Returns ``"book"`` or ``"movie_script"`` when the ``document_type``
        metadata column is present.  Falls back to ``"unknown"`` otherwise.
        """
        orig_idx = int(self._valid_indices[idx])
        meta = self._metadata_rows[orig_idx]
        dt = meta.get("document_type")
        if dt:
            return str(dt)
        return "unknown"

    # ------------------------------------------------------------------
    # __getitem__ override
    # ------------------------------------------------------------------

    def __getitem__(self, idx: int) -> QASample:
        sample = super().__getitem__(idx)

        # Inject document_type into sample metadata.
        dt = self.document_type(idx)
        sample.metadata["document_type"] = dt

        # Add document_type as a tag for topic embedding training.
        if dt != "unknown" and dt not in sample.tags:
            sample.tags.append(dt)

        return sample
