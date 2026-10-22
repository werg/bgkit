"""Tests for the enhanced QueryAwareBatchSampler with distractor curriculum."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.qa_sample import QASample
from bgkit.data.samplers import QueryAwareBatchSampler


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class _DummyDataset:
    items: list[QASample]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def _sample(query_id: str, sample_id: str) -> QASample:
    t = torch.tensor([1, 2], dtype=torch.long)
    return QASample(
        objective="phase2_qa",
        content_token_ids=t,
        content_attention_mask=torch.ones_like(t, dtype=torch.bool),
        compression_ratio=0.0,
        compression_level=0,
        target_token_ids=t,
        target_attention_mask=torch.ones_like(t, dtype=torch.bool),
        target_loss_mask=torch.tensor([False, True]),
        prefix_ids=t,
        compression_prompt_ids=t,
        question_token_ids=t,
        answer_token_ids=t,
        sample_id=sample_id,
        metadata={"query_id": query_id},
    )


def _build_dataset(n_queries: int = 3, samples_per_query: int = 2) -> _DummyDataset:
    items = []
    for qi in range(n_queries):
        for si in range(samples_per_query):
            items.append(_sample(f"q{qi}", f"q{qi}_s{si}"))
    return _DummyDataset(items)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDistractorRampCurriculum:
    def test_distractor_count_at_step_zero(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=0, n_distractors_end=4, distractor_ramp_steps=100,
        )
        sampler.set_step(0)
        assert sampler._n_distractors == 0

    def test_distractor_count_ramps_linearly(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=0, n_distractors_end=10, distractor_ramp_steps=100,
        )
        sampler.set_step(50)
        assert sampler._n_distractors == 5

        sampler.set_step(100)
        assert sampler._n_distractors == 10

    def test_distractor_count_clamps_at_end(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=0, n_distractors_end=10, distractor_ramp_steps=100,
        )
        sampler.set_step(200)  # Past the ramp
        assert sampler._n_distractors == 10

    def test_distractor_start_nonzero(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=2, n_distractors_end=8, distractor_ramp_steps=100,
        )
        sampler.set_step(0)
        assert sampler._n_distractors == 2

        sampler.set_step(50)
        assert sampler._n_distractors == 5

        sampler.set_step(100)
        assert sampler._n_distractors == 8


class TestDistractorSampling:
    def test_with_zero_distractors_matches_basic_behavior(self):
        ds = _build_dataset(n_queries=2, samples_per_query=2)
        sampler = QueryAwareBatchSampler(
            ds, batch_size=10, shuffle=False,
            n_distractors_start=0, n_distractors_end=0,
        )
        batches = list(iter(sampler))
        # All 4 samples should appear exactly once
        all_indices = [idx for batch in batches for idx in batch]
        assert sorted(all_indices) == [0, 1, 2, 3]

    def test_with_distractors_includes_extra_indices(self):
        ds = _build_dataset(n_queries=2, samples_per_query=2)
        sampler = QueryAwareBatchSampler(
            ds, batch_size=100, shuffle=False,
            n_distractors_start=3, n_distractors_end=3, distractor_ramp_steps=1,
        )
        sampler.set_step(1)
        batches = list(iter(sampler))
        all_indices = [idx for batch in batches for idx in batch]
        # With distractors, we should have more than the original 4 indices
        # Each of 2 queries adds 3 distractors -> 4 + 6 = 10 total
        assert len(all_indices) > 4

    def test_set_step_updates_distractor_count(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=0, n_distractors_end=5, distractor_ramp_steps=10,
        )
        assert sampler._n_distractors == 0
        sampler.set_step(10)
        assert sampler._n_distractors == 5


class TestLenWithDistractors:
    def test_len_without_distractors(self):
        ds = _build_dataset(n_queries=3, samples_per_query=2)
        sampler = QueryAwareBatchSampler(
            ds, batch_size=2, shuffle=False,
            n_distractors_start=0, n_distractors_end=0,
        )
        # 6 samples, batch_size=2 -> ceil(6/2) = 3 batches
        assert len(sampler) == 3

    def test_len_with_distractors_accounts_for_extras(self):
        ds = _build_dataset(n_queries=3, samples_per_query=2)
        sampler = QueryAwareBatchSampler(
            ds, batch_size=4, shuffle=False,
            n_distractors_start=2, n_distractors_end=2, distractor_ramp_steps=1,
        )
        sampler.set_step(1)
        # 6 original + 3 queries * 2 distractors = 12 total, batch_size=4 -> 3 batches
        assert len(sampler) == 3

    def test_default_zero_distractors(self):
        ds = _build_dataset()
        sampler = QueryAwareBatchSampler(ds, batch_size=10, shuffle=False)
        assert sampler._n_distractors == 0
