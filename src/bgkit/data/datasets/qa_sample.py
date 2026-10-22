"""Phase 2 QA sample types."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from bgkit.data.datasets.compression_dataset import FileCompressionSample


@dataclass
class QASample(FileCompressionSample):
    """Single-document QA sample for Phase 2 retrieval training.

    Extends ``FileCompressionSample`` with explicit question/answer fields and
    lightweight metadata used by topic embeddings and per-benchmark evaluation.
    """

    question_token_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long),
    )
    answer_token_ids: torch.Tensor = field(
        default_factory=lambda: torch.empty(0, dtype=torch.long),
    )
    sample_id: str | None = None
    dataset_name: str | None = None
    document_id: str | None = None
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, object] = field(default_factory=dict)
