"""PubMedQA Phase 2 dataset wrapper."""

from __future__ import annotations

from bgkit.data.datasets.phase2_qa_dataset import Phase2QADataset


class PubMedQADataset(Phase2QADataset):
    """Thin wrapper around the generic Phase 2 QA mmap loader."""

    def __init__(self, data_dir: str, **kwargs):
        super().__init__(data_dir, dataset_name="pubmedqa", **kwargs)
