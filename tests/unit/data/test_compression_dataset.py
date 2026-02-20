"""Tests for CompressionDataset, sub-datasets, samplers, and collators."""
from __future__ import annotations

import json
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from torch.utils.data import Dataset

from bgkit.data.collators import collate_compression
from bgkit.data.datasets.compression_dataset import (
    CompressionDataset,
    DescriptionSubset,
    FileCompressionSample,
    RepoCompressionSample,
)
from bgkit.data.samplers import LengthSortedBatchSampler

# ---------------------------------------------------------------------------
# Mock sub-datasets
# ---------------------------------------------------------------------------


class MockFileSubset(Dataset):
    """Mock sub-dataset returning FileCompressionSample."""

    def __init__(self, size: int, base_length: int = 100, objective: str = "test"):
        self._size = size
        self._base_length = base_length
        self._objective = objective
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def token_length(self, idx: int) -> int:
        return self._base_length + idx % 50

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> FileCompressionSample:
        length = self._base_length + idx % 50
        return FileCompressionSample(
            objective=self._objective,
            content_token_ids=torch.ones(length, dtype=torch.long),
            content_attention_mask=torch.ones(length, dtype=torch.bool),
                compression_ratio=0.2,
            compression_level=0,
            target_token_ids=torch.ones(50, dtype=torch.long),
            target_attention_mask=torch.ones(50, dtype=torch.bool),
            target_loss_mask=torch.ones(50, dtype=torch.long),
            prefix_ids=torch.ones(20, dtype=torch.long),
            compression_prompt_ids=torch.ones(10, dtype=torch.long),
        )


class MockRepoSubset(Dataset):
    """Mock sub-dataset returning RepoCompressionSample."""

    def __init__(self, size: int, base_length: int = 200, objective: str = "test_repo"):
        self._size = size
        self._base_length = base_length
        self._objective = objective
        self._epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self._epoch = epoch

    def token_length(self, idx: int) -> int:
        return self._base_length + idx % 30

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> RepoCompressionSample:
        n_files = 2 + idx % 3
        file_len = self._base_length // n_files
        return RepoCompressionSample(
            objective=self._objective,
            file_token_ids=[
                torch.ones(file_len + f, dtype=torch.long) for f in range(n_files)
            ],
            file_attention_masks=[
                torch.ones(file_len + f, dtype=torch.bool) for f in range(n_files)
            ],
            compression_ratio=0.15,
            compression_level=1,
            target_token_ids=torch.ones(60, dtype=torch.long),
            target_attention_mask=torch.ones(60, dtype=torch.bool),
            target_loss_mask=torch.ones(60, dtype=torch.long),
            prefix_ids=torch.ones(20, dtype=torch.long),
            compression_prompt_ids=torch.ones(10, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# CompressionDataset tests
# ---------------------------------------------------------------------------


class TestCompressionDataset:
    def test_proportional_sizes(self):
        """Sub-dataset sizes should be proportional to weights within +-1 sample."""
        subsets = {
            "data_reconstruction": MockFileSubset(400, objective="data_reconstruction"),
            "description_generation": MockFileSubset(200, objective="description_generation"),
            "structural_relational": MockFileSubset(150, objective="structural_relational"),
            "commit_reproduction": MockRepoSubset(250, objective="commit_reproduction"),
        }
        weights = {
            "data_reconstruction": 0.40,
            "description_generation": 0.20,
            "structural_relational": 0.15,
            "commit_reproduction": 0.25,
        }
        ds = CompressionDataset(subsets=subsets, weights=weights)

        total = len(ds)
        assert total == 1000  # sum of natural sizes

        # Check proportions
        for i, key in enumerate(ds._objective_order):
            effective_size = ds._effective_sizes[i]
            expected_frac = weights[key]
            actual_frac = effective_size / total
            assert abs(actual_frac - expected_frac) < 0.01, (
                f"{key}: expected {expected_frac:.2f}, got {actual_frac:.2f}"
            )

    def test_index_mapping_covers_all_objectives(self):
        """objective_for_idx should return correct objective for each range."""
        subsets = {
            "data_reconstruction": MockFileSubset(40, objective="data_reconstruction"),
            "description_generation": MockFileSubset(20, objective="description_generation"),
            "commit_reproduction": MockRepoSubset(25, objective="commit_reproduction"),
        }
        weights = {
            "data_reconstruction": 0.40,
            "description_generation": 0.20,
            "commit_reproduction": 0.25,
        }
        ds = CompressionDataset(subsets=subsets, weights=weights)

        seen_objectives = set()
        for i in range(len(ds)):
            obj = ds.objective_for_idx(i)
            seen_objectives.add(obj)

        assert seen_objectives == {"data_reconstruction", "description_generation",
                                    "commit_reproduction"}

    def test_index_mapping_contiguous_ranges(self):
        """Each objective should occupy a contiguous range of indices."""
        subsets = {
            "data_reconstruction": MockFileSubset(100, objective="data_reconstruction"),
            "description_generation": MockFileSubset(50, objective="description_generation"),
        }
        ds = CompressionDataset(subsets=subsets)

        # Collect objectives for all indices
        objectives = [ds.objective_for_idx(i) for i in range(len(ds))]

        # Verify contiguous: once an objective changes, it should not come back
        seen = set()
        prev = None
        for obj in objectives:
            if obj != prev:
                assert obj not in seen, f"{obj} appeared in non-contiguous range"
                seen.add(obj)
                prev = obj

    def test_token_length_stability(self):
        """token_length should return same value across multiple calls."""
        subsets = {
            "data_reconstruction": MockFileSubset(50, objective="data_reconstruction"),
        }
        ds = CompressionDataset(subsets=subsets)

        for i in range(len(ds)):
            l1 = ds.token_length(i)
            l2 = ds.token_length(i)
            assert l1 == l2, f"token_length({i}) returned {l1} then {l2}"

    def test_getitem_returns_correct_type(self):
        """File subsets should return FileCompressionSample."""
        subsets = {
            "data_reconstruction": MockFileSubset(10, objective="data_reconstruction"),
            "commit_reproduction": MockRepoSubset(10, objective="commit_reproduction"),
        }
        ds = CompressionDataset(subsets=subsets)

        # First range should be file samples
        for i in range(ds._offsets[0], ds._offsets[1]):
            sample = ds[i]
            if ds.objective_for_idx(i) == "data_reconstruction":
                assert isinstance(sample, FileCompressionSample)
            else:
                assert isinstance(sample, RepoCompressionSample)

    def test_set_epoch_propagation(self):
        """set_epoch should propagate to all sub-datasets."""
        s1 = MockFileSubset(10, objective="data_reconstruction")
        s2 = MockRepoSubset(10, objective="commit_reproduction")
        subsets = {"data_reconstruction": s1, "commit_reproduction": s2}
        ds = CompressionDataset(subsets=subsets)

        ds.set_epoch(5)
        assert s1._epoch == 5
        assert s2._epoch == 5

    def test_empty_subsets(self):
        """Empty subsets should result in length 0."""
        ds = CompressionDataset(subsets={})
        assert len(ds) == 0

    def test_index_out_of_range(self):
        """Accessing out of range should raise IndexError."""
        subsets = {"data_reconstruction": MockFileSubset(10)}
        ds = CompressionDataset(subsets=subsets)

        with pytest.raises(IndexError):
            ds[len(ds)]
        with pytest.raises(IndexError):
            ds[-1]

    def test_upsampling_wraps_around(self):
        """When effective size > natural size, indices should wrap around."""
        subsets = {
            "data_reconstruction": MockFileSubset(5, objective="data_reconstruction"),
            "commit_reproduction": MockRepoSubset(100, objective="commit_reproduction"),
        }
        weights = {
            "data_reconstruction": 0.50,
            "commit_reproduction": 0.50,
        }
        ds = CompressionDataset(subsets=subsets, weights=weights)

        # data_reconstruction has natural size 5 but should get ~50% of total
        # All samples should still be accessible (wrap around)
        for i in range(len(ds)):
            sample = ds[i]
            assert sample is not None

    def test_file_compression_sample_fields(self):
        """FileCompressionSample should have all required fields with correct types."""
        subsets = {"data_reconstruction": MockFileSubset(5)}
        ds = CompressionDataset(subsets=subsets)
        sample = ds[0]
        assert isinstance(sample, FileCompressionSample)
        assert isinstance(sample.content_token_ids, torch.Tensor)
        assert isinstance(sample.content_attention_mask, torch.Tensor)
        assert isinstance(sample.compression_ratio, float)
        assert isinstance(sample.compression_level, int)
        assert sample.compression_level == 0
        assert isinstance(sample.target_token_ids, torch.Tensor)
        assert isinstance(sample.target_attention_mask, torch.Tensor)
        assert isinstance(sample.target_loss_mask, torch.Tensor)
        assert isinstance(sample.prefix_ids, torch.Tensor)
        assert isinstance(sample.compression_prompt_ids, torch.Tensor)

    def test_repo_compression_sample_fields(self):
        """RepoCompressionSample should have all required fields with correct types."""
        subsets = {"commit_reproduction": MockRepoSubset(5)}
        ds = CompressionDataset(subsets=subsets)
        sample = ds[0]
        assert isinstance(sample, RepoCompressionSample)
        assert isinstance(sample.file_token_ids, list)
        assert all(isinstance(t, torch.Tensor) for t in sample.file_token_ids)
        assert isinstance(sample.file_attention_masks, list)
        assert isinstance(sample.compression_ratio, float)
        assert isinstance(sample.compression_level, int)
        assert sample.compression_level == 1
        assert isinstance(sample.target_token_ids, torch.Tensor)

    def test_curriculum_state_l1_enable(self):
        """set_curriculum_state should return True when L1 is newly enabled."""

        class MockL1Subset(MockFileSubset):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.l1_called = False

            def enable_l1(self):
                self.l1_called = True

        sub = MockL1Subset(10, objective="description_generation")
        subsets = {"description_generation": sub}
        ds = CompressionDataset(subsets=subsets)

        # First enable
        result = ds.set_curriculum_state(step=100, l1_enabled=True)
        assert result is True
        assert sub.l1_called

        # Second call should return False
        result = ds.set_curriculum_state(step=200, l1_enabled=True)
        assert result is False

    def test_single_subset(self):
        """Should work with just one subset."""
        subsets = {"data_reconstruction": MockFileSubset(100)}
        ds = CompressionDataset(subsets=subsets)
        assert len(ds) == 100
        assert ds.objective_for_idx(0) == "data_reconstruction"
        assert ds.objective_for_idx(99) == "data_reconstruction"


# ---------------------------------------------------------------------------
# Collator tests
# ---------------------------------------------------------------------------


class TestCollateCompression:
    def test_collate_file_samples(self):
        """Collating FileCompressionSample should produce correct batch dict."""
        samples = [
            MockFileSubset(5, base_length=100, objective="data_reconstruction")[i]
            for i in range(3)
        ]
        batch = collate_compression(samples)

        assert batch["sample_type"] == "file"
        assert len(batch["objectives"]) == 3
        assert batch["content_token_ids"].shape[0] == 3
        assert batch["content_attention_mask"].shape == batch["content_token_ids"].shape
        assert batch["target_token_ids"].shape[0] == 3
        assert batch["target_loss_mask"].shape == batch["target_token_ids"].shape
        assert batch["prefix_ids"].shape[0] == 3
        assert batch["compression_prompt_ids"].shape[0] == 3
        assert batch["compression_ratios"].shape == (3,)
        assert batch["compression_levels"].shape == (3,)

    def test_collate_repo_samples(self):
        """Collating RepoCompressionSample should produce 3D file tensor."""
        samples = [
            MockRepoSubset(5, base_length=200, objective="commit_reproduction")[i]
            for i in range(3)
        ]
        batch = collate_compression(samples)

        assert batch["sample_type"] == "repo"
        assert len(batch["objectives"]) == 3
        assert batch["file_token_ids"].dim() == 3  # (batch, max_files, max_len)
        assert batch["file_attention_masks"].shape == batch["file_token_ids"].shape
        assert batch["file_count"].shape == (3,)
        assert batch["target_token_ids"].shape[0] == 3

    def test_collate_mixed_batch(self):
        """Mixed batch should produce both file_batch and repo_batch."""
        file_samples = [MockFileSubset(5)[0]]
        repo_samples = [MockRepoSubset(5)[0]]
        batch = collate_compression(file_samples + repo_samples)

        assert batch["mixed"] is True
        assert "file_batch" in batch
        assert "repo_batch" in batch
        assert batch["file_batch"]["sample_type"] == "file"
        assert batch["repo_batch"]["sample_type"] == "repo"

    def test_collate_variable_lengths(self):
        """Collator should pad shorter sequences to max length in batch."""
        samples = [
            FileCompressionSample(
                objective="test",
                content_token_ids=torch.ones(length, dtype=torch.long),
                content_attention_mask=torch.ones(length, dtype=torch.bool),
                        compression_ratio=0.2,
                compression_level=0,
                target_token_ids=torch.ones(30, dtype=torch.long),
                target_attention_mask=torch.ones(30, dtype=torch.bool),
                target_loss_mask=torch.ones(30, dtype=torch.long),
                prefix_ids=torch.ones(10, dtype=torch.long),
                compression_prompt_ids=torch.ones(5, dtype=torch.long),
            )
            for length in [50, 100, 75]
        ]
        batch = collate_compression(samples)

        # Content should be padded to max length (100)
        assert batch["content_token_ids"].shape == (3, 100)
        # Attention mask should reflect actual lengths
        assert batch["content_attention_mask"][0].sum() == 50
        assert batch["content_attention_mask"][1].sum() == 100
        assert batch["content_attention_mask"][2].sum() == 75

    def test_collate_variable_file_counts(self):
        """Repo samples with different file counts should be padded correctly."""
        samples = [
            RepoCompressionSample(
                objective="test",
                file_token_ids=[torch.ones(50, dtype=torch.long)] * n_files,
                file_attention_masks=[torch.ones(50, dtype=torch.bool)] * n_files,
                    compression_ratio=0.15,
                compression_level=1,
                target_token_ids=torch.ones(30, dtype=torch.long),
                target_attention_mask=torch.ones(30, dtype=torch.bool),
                target_loss_mask=torch.ones(30, dtype=torch.long),
                prefix_ids=torch.ones(10, dtype=torch.long),
                compression_prompt_ids=torch.ones(5, dtype=torch.long),
            )
            for n_files in [2, 4, 3]
        ]
        batch = collate_compression(samples)

        # File dim should be padded to max (4)
        assert batch["file_token_ids"].shape[1] == 4
        assert batch["file_count"].tolist() == [2, 4, 3]
        # Extra file slots should be zero-masked
        assert batch["file_attention_masks"][0, 3].sum() == 0  # sample 0 has 2 files


# ---------------------------------------------------------------------------
# LengthSortedBatchSampler tests
# ---------------------------------------------------------------------------


class TestLengthSortedBatchSampler:
    def test_all_indices_yielded_once(self):
        """Every index should appear exactly once."""
        ds = MockFileSubset(50, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=False)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(50))

    def test_all_indices_yielded_once_with_shuffle(self):
        """Shuffled sampler should still yield all indices exactly once."""
        ds = MockFileSubset(50, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=True)

        all_indices = []
        for batch in sampler:
            all_indices.extend(batch)

        assert sorted(all_indices) == list(range(50))

    def test_respects_token_budget(self):
        """No batch should exceed max_batch_tokens (except singleton overflow)."""
        ds = MockFileSubset(50, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=False)

        for batch in sampler:
            max_len = max(ds.token_length(i) for i in batch)
            # Either within budget, or singleton overflow
            assert len(batch) * max_len <= 500 or len(batch) == 1

    def test_sorted_within_batches(self):
        """Samples within a batch should be sorted by length (no shuffle)."""
        ds = MockFileSubset(50, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=1000, shuffle=False)

        for batch in sampler:
            lengths = [ds.token_length(i) for i in batch]
            assert lengths == sorted(lengths)

    def test_shuffle_changes_batch_order(self):
        """Different epochs should produce different batch orderings."""
        ds = MockFileSubset(100, base_length=50)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=True)

        sampler.set_epoch(0)
        batches_0 = [b[:] for b in sampler]

        sampler.set_epoch(1)
        batches_1 = [b[:] for b in sampler]

        # Batch contents should be the same, but order may differ
        assert sorted(tuple(b) for b in batches_0) == sorted(tuple(b) for b in batches_1)

        # With enough samples, order should differ
        flat_0 = [idx for b in batches_0 for idx in b]
        flat_1 = [idx for b in batches_1 for idx in b]
        if len(batches_0) > 1:
            assert flat_0 != flat_1, "Epoch shuffle should change batch order"

    def test_rebuild(self):
        """Rebuild should recompute batches."""
        ds = MockFileSubset(20, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=False)
        original_batches = len(sampler)

        # Rebuild should produce same result for unchanged dataset
        sampler.rebuild()
        assert len(sampler) == original_batches

    def test_len_matches_actual(self):
        """len() should match actual number of batches yielded."""
        ds = MockFileSubset(100, base_length=50)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=False)
        assert len(sampler) == len(list(sampler))

    def test_empty_dataset(self):
        """Empty dataset should yield no batches."""
        ds = MockFileSubset(0)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500)
        assert len(sampler) == 0
        assert list(sampler) == []

    def test_single_sample(self):
        """Single sample should yield one batch."""
        ds = MockFileSubset(1, base_length=100)
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500)
        batches = list(sampler)
        assert len(batches) == 1
        assert batches[0] == [0]

    def test_singleton_overflow(self):
        """A sample exceeding budget should be emitted as a batch of 1."""

        class BigSmallDataset(Dataset):
            def token_length(self, idx):
                return 10000 if idx == 0 else 100

            def __len__(self):
                return 5

        ds = BigSmallDataset()
        sampler = LengthSortedBatchSampler(ds, max_batch_tokens=500, shuffle=False)

        found_overflow = False
        for batch in sampler:
            if 0 in batch:
                assert len(batch) == 1, "Overflow sample should be singleton"
                found_overflow = True

        assert found_overflow


# ---------------------------------------------------------------------------
# RepoCompressionSample.direct_l1 tests
# ---------------------------------------------------------------------------


class TestDirectL1Flag:
    def test_repo_compression_sample_direct_l1_default(self):
        """direct_l1 field should default to False."""
        sample = RepoCompressionSample(
            objective="test",
            file_token_ids=[torch.ones(50, dtype=torch.long)],
            file_attention_masks=[torch.ones(50, dtype=torch.bool)],
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.ones(30, dtype=torch.long),
            target_attention_mask=torch.ones(30, dtype=torch.bool),
            target_loss_mask=torch.ones(30, dtype=torch.long),
            prefix_ids=torch.ones(10, dtype=torch.long),
            compression_prompt_ids=torch.ones(5, dtype=torch.long),
        )
        assert sample.direct_l1 is False

    def test_collate_repo_direct_l1_flag(self):
        """direct_l1=True should propagate as per-sample bool tensor."""
        samples = [
            RepoCompressionSample(
                objective="commit_reproduction",
                file_token_ids=[torch.ones(50, dtype=torch.long)],
                file_attention_masks=[torch.ones(50, dtype=torch.bool)],
                    compression_ratio=0.2,
                compression_level=1,
                target_token_ids=torch.ones(30, dtype=torch.long),
                target_attention_mask=torch.ones(30, dtype=torch.bool),
                target_loss_mask=torch.ones(30, dtype=torch.long),
                prefix_ids=torch.ones(10, dtype=torch.long),
                compression_prompt_ids=torch.ones(5, dtype=torch.long),
                direct_l1=True,
            )
            for _ in range(3)
        ]
        batch = collate_compression(samples)
        assert batch["direct_l1"].dtype == torch.bool
        assert batch["direct_l1"].all()

    def test_collate_repo_direct_l1_mixed(self):
        """Mixed direct_l1 batch preserves per-sample flags."""
        s1 = RepoCompressionSample(
            objective="commit_reproduction",
            file_token_ids=[torch.ones(50, dtype=torch.long)],
            file_attention_masks=[torch.ones(50, dtype=torch.bool)],
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.ones(30, dtype=torch.long),
            target_attention_mask=torch.ones(30, dtype=torch.bool),
            target_loss_mask=torch.ones(30, dtype=torch.long),
            prefix_ids=torch.ones(10, dtype=torch.long),
            compression_prompt_ids=torch.ones(5, dtype=torch.long),
            direct_l1=True,
        )
        s2 = RepoCompressionSample(
            objective="description_generation",
            file_token_ids=[torch.ones(40, dtype=torch.long)] * 2,
            file_attention_masks=[torch.ones(40, dtype=torch.bool)] * 2,
            compression_ratio=0.2,
            compression_level=1,
            target_token_ids=torch.ones(30, dtype=torch.long),
            target_attention_mask=torch.ones(30, dtype=torch.bool),
            target_loss_mask=torch.ones(30, dtype=torch.long),
            prefix_ids=torch.ones(10, dtype=torch.long),
            compression_prompt_ids=torch.ones(5, dtype=torch.long),
            direct_l1=False,
        )
        batch = collate_compression([s1, s2])
        assert batch["direct_l1"][0] is True or batch["direct_l1"][0].item() is True
        assert batch["direct_l1"][1] is False or batch["direct_l1"][1].item() is False


# ---------------------------------------------------------------------------
# DescriptionSubset L1 target selection tests
# ---------------------------------------------------------------------------


class _MockTokenizer:
    """Minimal tokenizer for DescriptionSubset tests."""

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._next_id = 100

    def _get_id(self, char: str) -> int:
        if char not in self._vocab:
            self._vocab[char] = self._next_id
            self._next_id += 1
        return self._vocab[char]

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        return [self._get_id(c) for c in text]

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False):
        parts = []
        for msg in messages:
            parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>\n")
        result = "".join(parts)
        if tokenize:
            return self.encode(result, add_special_tokens=False)
        return result


class _MockTokenDataset(Dataset):
    """Token dataset with metadata for DescriptionSubset join tests."""

    def __init__(self, entries: list[dict]):
        """entries: list of {repo_path, file_path, commit_sha, token_ids}."""
        self._entries = entries
        self._lengths = np.array([len(e["token_ids"]) for e in entries], dtype=np.int32)
        self._chunk_file_idx = np.arange(len(entries), dtype=np.int32)

    @property
    def lengths(self):
        return self._lengths

    @property
    def chunk_file_indices(self):
        return self._chunk_file_idx

    def get_metadata_table(self):
        return pa.table({
            "repo_path": [e["repo_path"] for e in self._entries],
            "file_path": [e["file_path"] for e in self._entries],
            "commit_sha": [e["commit_sha"] for e in self._entries],
        })

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        e = self._entries[idx]
        return {
            "token_ids": torch.tensor(e["token_ids"], dtype=torch.long),
            "file_path": e["file_path"],
            "language": e.get("language", "python"),
        }


class _MockDescDataset(Dataset):
    """Mock file-level description dataset."""

    def __init__(self, entries: dict[tuple[str, str, str], list[int]]):
        """entries: {(rp, fp, cs): token_ids_list}."""
        self._entries = entries
        self._ordered_keys = list(entries.keys())
        self._key_to_vi = {k: i for i, k in enumerate(self._ordered_keys)}

    def lookup_index(self, rp, fp, cs):
        return self._key_to_vi.get((rp, fp, cs))

    def has_key(self, rp, fp, cs):
        return (rp, fp, cs) in self._key_to_vi

    @property
    def lengths(self):
        return np.array(
            [len(v) for v in self._entries.values()], dtype=np.int32,
        )

    def __len__(self):
        return len(self._entries)

    def __getitem__(self, idx):
        key = self._ordered_keys[idx]
        return {"token_ids": torch.tensor(self._entries[key], dtype=torch.long)}


class _MockRepoDescDataset:
    """Mock repo-level description dataset."""

    def __init__(self, entries: dict[tuple[str, str], list[int]]):
        """entries: {(rp, cs): token_ids_list}."""
        self._entries = entries

    def has_key(self, rp, cs):
        return (rp, cs) in self._entries

    def get_by_key(self, rp, cs):
        ids = self._entries.get((rp, cs))
        if ids is None:
            return None
        return {"token_ids": torch.tensor(ids, dtype=torch.long)}


def _make_desc_subset(
    token_entries, file_desc_entries, repo_desc_entries=None,
):
    """Helper to build a DescriptionSubset with mocks."""
    from bgkit.data.chat_template import TOOL_CONFIGS

    token_ds = _MockTokenDataset(token_entries)
    desc_ds = _MockDescDataset(file_desc_entries)
    repo_desc_ds = (
        _MockRepoDescDataset(repo_desc_entries) if repo_desc_entries else None
    )
    tokenizer = _MockTokenizer()
    variant_bank = [{
        "system_prompt": "You are a helpful assistant.",
        "user_prompt": "Describe {file_path}",
        "compression_prompt": "Generate a description.",
        "response_prefix": "Description of {file_path}:",
    }]
    config = TOOL_CONFIGS["description_gen"]

    subset = DescriptionSubset(
        token_ds, desc_ds, tokenizer, variant_bank, config,
        seed=42, repo_description_dataset=repo_desc_ds,
    )
    return subset


class TestDescriptionSubsetL1Targets:
    def test_description_l1_uses_repo_target(self):
        """With repo_desc_ds provided, L1 should use repo-level description."""
        token_entries = [
            {"repo_path": "owner/repo", "file_path": "a.py",
             "commit_sha": "abc", "token_ids": [1, 2, 3]},
            {"repo_path": "owner/repo", "file_path": "b.py",
             "commit_sha": "abc", "token_ids": [4, 5, 6]},
        ]
        file_desc = {
            ("owner/repo", "a.py", "abc"): [10, 11],
            ("owner/repo", "b.py", "abc"): [12, 13],
        }
        repo_desc = {
            ("owner/repo", "abc"): [99, 98, 97],  # distinct from file descs
        }

        subset = _make_desc_subset(token_entries, file_desc, repo_desc)
        subset.enable_l1()

        # Get L1 sample
        sample = subset[0]
        assert isinstance(sample, RepoCompressionSample)

        # The target should contain the repo description tokens (99, 98, 97)
        # embedded within the chat template. Verify by checking the loss-masked
        # region contains the repo desc tokens.
        target_ids = sample.target_token_ids
        loss_mask = sample.target_loss_mask
        content_region = target_ids[loss_mask.bool()].tolist()

        # The mock tokenizer maps each char to an ID, so the repo desc tokens
        # [99, 98, 97] will be spliced directly into the template via
        # tokenize_with_sentinel. The content_token_ids passed to
        # tokenize_with_sentinel are the repo desc tokens themselves.
        # They appear in the full token sequence at the loss-masked positions.
        assert content_region == [99, 98, 97]

    def test_description_l1_falls_back_to_file_target(self):
        """When repo_desc_ds has no entry, L1 should fall back to file-level."""
        token_entries = [
            {"repo_path": "owner/repo", "file_path": "a.py",
             "commit_sha": "abc", "token_ids": [1, 2, 3]},
        ]
        file_desc = {
            ("owner/repo", "a.py", "abc"): [10, 11, 12],
        }
        # Repo desc dataset exists but has no entry for this repo/commit
        repo_desc = {
            ("other/repo", "xyz"): [99, 98],
        }

        subset = _make_desc_subset(token_entries, file_desc, repo_desc)
        subset.enable_l1()

        sample = subset[0]
        assert isinstance(sample, RepoCompressionSample)

        # Should use file-level description [10, 11, 12]
        target_ids = sample.target_token_ids
        loss_mask = sample.target_loss_mask
        content_region = target_ids[loss_mask.bool()].tolist()
        assert content_region == [10, 11, 12]

    def test_description_l1_template_context_is_repo_scoped(self):
        """When using repo-level target, template should use repo_path not file_path."""
        token_entries = [
            {"repo_path": "owner/repo", "file_path": "src/main.py",
             "commit_sha": "abc", "token_ids": [1, 2, 3], "language": "python"},
        ]
        file_desc = {
            ("owner/repo", "src/main.py", "abc"): [10, 11],
        }
        repo_desc = {
            ("owner/repo", "abc"): [99, 98],
        }

        subset = _make_desc_subset(token_entries, file_desc, repo_desc)
        subset.enable_l1()

        # Patch tokenize_with_sentinel to capture args
        captured_calls = []
        original_tws = None

        from bgkit.data import chat_template as ct_module

        original_tws = ct_module.tokenize_with_sentinel

        def tracking_tws(tokenizer, variant, config, file_path, language, content_ids):
            captured_calls.append((file_path, language))
            return original_tws(tokenizer, variant, config, file_path, language, content_ids)

        with patch.object(ct_module, "tokenize_with_sentinel", tracking_tws):
            # Also need to patch the import in compression_dataset module
            from bgkit.data.datasets import compression_dataset as cd_module

            with patch.object(cd_module, "tokenize_with_sentinel", tracking_tws):
                sample = subset[0]

        assert isinstance(sample, RepoCompressionSample)
        # The L1 call should use repo_path and "" (not file_path and language)
        assert len(captured_calls) >= 1
        last_call = captured_calls[-1]
        assert last_call[0] == "owner/repo", f"Expected repo_path, got {last_call[0]}"
        assert last_call[1] == "", f"Expected empty language, got {last_call[1]}"

    def test_description_l0_still_uses_file_target(self):
        """L0 path should still use file-level description even with repo_desc_ds."""
        token_entries = [
            {"repo_path": "owner/repo", "file_path": "a.py",
             "commit_sha": "abc", "token_ids": [1, 2, 3]},
        ]
        file_desc = {
            ("owner/repo", "a.py", "abc"): [10, 11],
        }
        repo_desc = {
            ("owner/repo", "abc"): [99, 98],
        }

        subset = _make_desc_subset(token_entries, file_desc, repo_desc)
        # Don't enable L1 — should remain L0
        sample = subset[0]
        assert isinstance(sample, FileCompressionSample)

        target_ids = sample.target_token_ids
        loss_mask = sample.target_loss_mask
        content_region = target_ids[loss_mask.bool()].tolist()
        assert content_region == [10, 11]


# ---------------------------------------------------------------------------
# from_config repo description discovery test
# ---------------------------------------------------------------------------


def _create_mmap_artifacts(path, repo_paths, file_paths, commit_shas, token_lengths):
    """Create minimal mmap artifacts (tokens, offsets, manifest, metadata)."""
    path.mkdir(parents=True, exist_ok=True)

    offsets = [0]
    all_tokens = []
    for length in token_lengths:
        all_tokens.extend(range(1, length + 1))
        offsets.append(offsets[-1] + length)

    np.save(path / "tokens.npy", np.array(all_tokens, dtype=np.int32))
    np.save(path / "offsets.npy", np.array(offsets, dtype=np.int64))

    manifest = {"schema_version": 1, "row_count": len(token_lengths),
                "total_tokens": len(all_tokens)}
    (path / "manifest.json").write_text(json.dumps(manifest))

    columns = {"repo_path": repo_paths, "commit_sha": commit_shas}
    if file_paths is not None:
        columns["file_path"] = file_paths
    pq.write_table(pa.table(columns), path / "metadata.parquet")


class TestFromConfigRepoDiscovery:
    def test_from_config_discovers_repo_sibling(self, tmp_path):
        """from_config should auto-discover repo/ sibling of file/ desc dir."""
        # Create file-scope description artifacts
        file_dir = tmp_path / "descriptions" / "file"
        _create_mmap_artifacts(
            file_dir,
            repo_paths=["owner/repo"],
            file_paths=["a.py"],
            commit_shas=["abc"],
            token_lengths=[10],
        )

        # Create repo-scope description artifacts
        repo_dir = tmp_path / "descriptions" / "repo"
        _create_mmap_artifacts(
            repo_dir,
            repo_paths=["owner/repo"],
            file_paths=None,  # repo scope has no file_path
            commit_shas=["abc"],
            token_lengths=[15],
        )

        # Create token dataset artifacts
        token_dir = tmp_path / "tokens"
        token_dir.mkdir(parents=True)
        # tokens.npy, offsets.npy, manifest.json, metadata.parquet, chunk_map.npy
        np.save(token_dir / "tokens.npy", np.array([1, 2, 3, 4, 5], dtype=np.int32))
        np.save(token_dir / "offsets.npy", np.array([0, 5], dtype=np.int64))
        np.save(token_dir / "chunk_map.npy", np.array([0], dtype=np.int32))
        (token_dir / "manifest.json").write_text(json.dumps({
            "schema_version": 1, "row_count": 1, "total_tokens": 5,
        }))
        pq.write_table(
            pa.table({
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
                "language": ["python"],
            }),
            token_dir / "metadata.parquet",
        )

        # Build a minimal config object
        class _Cfg:
            file_tokens_path = str(token_dir)
            description_tokens_path = str(tmp_path / "descriptions")
            structural_tokens_path = None
            commit_tokens_path = None
            prompt_variants_dir = None
            max_seq_len = 512

        tokenizer = _MockTokenizer()

        ds = CompressionDataset.from_config(_Cfg(), tokenizer, seed=42)

        # Verify description_generation subset was loaded
        assert "description_generation" in ds._subsets
        desc_subset = ds._subsets["description_generation"]

        # Verify repo_desc_ds was loaded
        assert desc_subset._repo_desc_ds is not None
        assert desc_subset._repo_desc_ds.has_key("owner/repo", "abc")
