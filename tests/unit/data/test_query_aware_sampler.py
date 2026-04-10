"""Tests for the query-aware batch sampler."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.qa_sample import QASample
from bgkit.data.samplers import QueryAwareBatchSampler


@dataclass
class _Dataset:
    items: list[QASample]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _sample(query_id: str, sample_id: str) -> QASample:
    tensor = torch.tensor([1, 2], dtype=torch.long)
    return QASample(
        objective="phase2_qa",
        content_token_ids=tensor,
        content_attention_mask=torch.ones_like(tensor, dtype=torch.bool),
        compression_ratio=0.0,
        compression_level=0,
        target_token_ids=tensor,
        target_attention_mask=torch.ones_like(tensor, dtype=torch.bool),
        target_loss_mask=torch.tensor([False, True]),
        prefix_ids=tensor,
        compression_prompt_ids=tensor,
        question_token_ids=tensor,
        answer_token_ids=tensor,
        sample_id=sample_id,
        metadata={"query_id": query_id},
    )


def test_query_aware_sampler_groups_by_query():
    dataset = _Dataset([
        _sample("q1", "a"),
        _sample("q1", "b"),
        _sample("q2", "c"),
    ])
    sampler = QueryAwareBatchSampler(dataset, batch_size=2, shuffle=False)
    batches = list(iter(sampler))
    assert batches == [[0, 1], [2]]
