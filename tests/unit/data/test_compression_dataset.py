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
from bgkit.data.samplers import PackedTokenBudgetSampler

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
    """Packed-schema collator tests.

    Wave 0.4 rewrote ``collate_compression`` to emit flat ``(N,)`` tensors
    with ``cu_seqlens`` segmentation, replacing the old padded
    ``(B, L_max)`` outputs. These tests assert the packed contract.
    """

    def test_collate_file_samples(self):
        """Collating FileCompressionSample yields flat content with cu_seqlens."""
        samples = [
            MockFileSubset(5, base_length=100, objective="data_reconstruction")[i]
            for i in range(3)
        ]
        batch = collate_compression(samples)

        assert batch["sample_type"] == "file"
        assert len(batch["objectives"]) == 3
        # Packed: content is flat over all 3 samples.
        n_content = batch["content_token_ids"].shape[0]
        assert batch["content_cu_seqlens"].shape == (4,)
        assert int(batch["content_cu_seqlens"][-1].item()) == n_content
        assert batch["content_position_ids"].shape == (n_content,)
        # Target / prefix / prompt are also flat with their own cu_seqlens.
        assert batch["target_cu_seqlens"].shape == (4,)
        assert batch["target_loss_mask"].shape == batch["target_token_ids"].shape
        assert batch["prefix_cu_seqlens"].shape == (4,)
        assert batch["prompt_cu_seqlens"].shape == (4,)
        assert batch["compression_ratios"].shape == (3,)
        assert batch["compression_levels"].shape == (3,)

    def test_collate_repo_samples(self):
        """Repo samples yield two-level packing: cu_file_seqlens + cu_repo_seqlens."""
        samples = [
            MockRepoSubset(5, base_length=200, objective="commit_reproduction")[i]
            for i in range(3)
        ]
        batch = collate_compression(samples)

        assert batch["sample_type"] == "repo"
        assert len(batch["objectives"]) == 3
        # Packed repo: content is flat over all files in all repos.
        total_files = int(batch["cu_file_seqlens"].shape[0]) - 1
        # cu_repo_seqlens has B+1 entries (3 repos → 4).
        assert batch["cu_repo_seqlens"].shape == (4,)
        assert int(batch["cu_repo_seqlens"][-1].item()) == total_files
        # Prompt is tiled per file: prompt_cu_seqlens aligned 1:1 with cu_file_seqlens.
        assert batch["prompt_cu_seqlens"].shape[0] == total_files + 1
        # Per-repo target / prefix (B+1).
        assert batch["target_cu_seqlens"].shape == (4,)
        assert batch["prefix_cu_seqlens"].shape == (4,)

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
        """Variable per-sample content lengths preserved in flat buffer + cu_seqlens."""
        lengths = [50, 100, 75]
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
            for length in lengths
        ]
        batch = collate_compression(samples)

        # Content flat length == sum(lengths).
        assert batch["content_token_ids"].shape == (sum(lengths),)
        # cu_seqlens cumulative matches lengths.
        cu = batch["content_cu_seqlens"].tolist()
        assert cu == [0, 50, 150, 225]
        # Per-sample length recoverable as cu[i+1] - cu[i].
        per_sample_lengths = [cu[i + 1] - cu[i] for i in range(len(lengths))]
        assert per_sample_lengths == lengths
        # content_max_seqlen is a scalar int (largest per-sample length).
        assert batch["content_max_seqlen"] == 100

    def test_collate_variable_file_counts(self):
        """Repo samples with different file counts: file segments preserved in cu_file_seqlens."""
        file_counts = [2, 4, 3]
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
            for n_files in file_counts
        ]
        batch = collate_compression(samples)

        total_files = sum(file_counts)
        # cu_file_seqlens covers every file segment.
        assert batch["cu_file_seqlens"].shape[0] == total_files + 1
        assert int(batch["cu_file_seqlens"][-1].item()) == total_files * 50
        # cu_repo_seqlens indexes into cu_file_seqlens at repo boundaries.
        assert batch["cu_repo_seqlens"].tolist() == [0, 2, 6, 9]
        # Each repo's file count recoverable as cu_repo[i+1] - cu_repo[i].
        recovered = [
            int(batch["cu_repo_seqlens"][i + 1].item())
            - int(batch["cu_repo_seqlens"][i].item())
            for i in range(len(file_counts))
        ]
        assert recovered == file_counts


# ---------------------------------------------------------------------------
# LengthSortedBatchSampler tests — REMOVED.
#
# ``LengthSortedBatchSampler`` was deleted in Wave 0.2 of the FA4 packed-
# attention migration. Its replacement, ``PackedTokenBudgetSampler``, is
# covered by ``tests/unit/data/test_samplers.py``. The former class-level
# behaviours (length-bucketing, batch-sort-within) no longer exist: packed
# attention has a flat quadratic budget per microbatch so neither sorting
# nor bucketing is needed.
# ---------------------------------------------------------------------------


def test_packed_token_budget_sampler_importable():
    """Smoke test: the replacement sampler should import and construct."""
    lengths = [100, 200, 50, 400]
    sampler = PackedTokenBudgetSampler(
        dataset=None, lengths=lengths, max_batch_tokens=500, shuffle=False,
    )
    batches = list(sampler)
    # All indices must be emitted exactly once.
    flat = [i for b in batches for i in b]
    assert sorted(flat) == list(range(len(lengths)))


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

    def apply_chat_template(self, messages, tokenize=True, add_generation_prompt=False, **kwargs):
        parts = []
        prev_role = None
        for msg in messages:
            role = msg["role"]
            content = msg.get("content", "")
            if role == "assistant" and "tool_calls" in msg:
                tc_parts = []
                for tc in msg["tool_calls"]:
                    fn = tc["function"]
                    args = fn["arguments"]
                    param_strs = "".join(
                        f"<parameter={k}>\n{v}\n</parameter>"
                        for k, v in args.items()
                    )
                    tc_parts.append(
                        f"<tool_call>\n<function={fn['name']}>"
                        f"{param_strs}</function>\n</tool_call>"
                    )
                content = "".join(tc_parts)
            if role == "tool":
                role = "user"
                content = f"<tool_response>\n{content}\n</tool_response>"
            if msg["role"] == "assistant" and prev_role in ("user", "tool"):
                content = f"<think>\n\n</think>\n\n{content}"
            parts.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
            prev_role = msg["role"]
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


# ---------------------------------------------------------------------------
# Variant bank loading tests
# ---------------------------------------------------------------------------


def _create_mmap_dir(path, tokens, offsets, metadata_columns=None):
    """Create a minimal mmap dataset directory (tokens, offsets, manifest, metadata)."""
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "tokens.npy", np.array(tokens, dtype=np.int32))
    np.save(path / "offsets.npy", np.array(offsets, dtype=np.int64))
    (path / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "row_count": len(offsets) - 1,
        "total_tokens": len(tokens),
    }))
    if metadata_columns is not None:
        pq.write_table(pa.table(metadata_columns), path / "metadata.parquet")


class TestVariantBankLoading:
    """Tests for per-objective variant bank loading in from_config."""

    def _make_full_config(self, tmp_path, variants_dir=None):
        """Create config with all four objective datasets populated."""
        # Token dataset (shared by data_reconstruction, description, structural)
        token_dir = tmp_path / "tokens"
        _create_mmap_dir(
            token_dir,
            tokens=[1, 2, 3, 4, 5],
            offsets=[0, 5],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
                "language": ["python"],
            },
        )
        np.save(token_dir / "chunk_map.npy", np.array([0], dtype=np.int32))

        # Description dataset (file scope)
        desc_dir = tmp_path / "descriptions" / "file"
        _create_mmap_dir(
            desc_dir,
            tokens=[10, 11, 12],
            offsets=[0, 3],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
            },
        )

        # Structural dataset
        struct_dir = tmp_path / "structural"
        _create_mmap_dir(
            struct_dir,
            tokens=[20, 21, 22],
            offsets=[0, 3],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
                "structural_type": ["skeleton"],
            },
        )

        # Commit dataset (no metadata.parquet needed)
        commit_dir = tmp_path / "commits"
        _create_mmap_dir(
            commit_dir,
            tokens=[30, 31, 32, 33],
            offsets=[0, 4],
        )

        class _Cfg:
            file_tokens_path = str(token_dir)
            description_tokens_path = str(tmp_path / "descriptions")
            structural_tokens_path = str(struct_dir)
            commit_tokens_path = str(commit_dir)
            prompt_variants_dir = str(variants_dir) if variants_dir else None
            max_seq_len = 512

        return _Cfg()

    def _make_minimal_config(self, tmp_path, variants_dir=None):
        """Create config with only data_reconstruction (token dataset)."""
        token_dir = tmp_path / "tokens"
        _create_mmap_dir(
            token_dir,
            tokens=[1, 2, 3, 4, 5],
            offsets=[0, 5],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["abc"],
                "language": ["python"],
            },
        )
        np.save(token_dir / "chunk_map.npy", np.array([0], dtype=np.int32))

        class _Cfg:
            file_tokens_path = str(token_dir)
            description_tokens_path = None
            structural_tokens_path = None
            commit_tokens_path = None
            prompt_variants_dir = str(variants_dir) if variants_dir else None
            max_seq_len = 512

        return _Cfg()

    def test_per_objective_banks_routed_to_all_subsets(self, tmp_path):
        """Each objective gets its own variant bank when per-objective files exist."""
        variants_dir = tmp_path / "variants"
        variants_dir.mkdir()

        def _make_variant(name):
            return [{"system_prompt": name, "user_prompt": f"{name} {{file_path}}",
                     "compression_prompt": f"{name}.", "response_prefix": "{file_path}:"}]

        (variants_dir / "file_read_repro.json").write_text(json.dumps(_make_variant("fr")))
        (variants_dir / "description_gen.json").write_text(json.dumps(_make_variant("dg")))
        (variants_dir / "structural_repro.json").write_text(json.dumps(_make_variant("sr")))
        (variants_dir / "commit_repro.json").write_text(json.dumps(_make_variant("cr")))

        cfg = self._make_full_config(tmp_path, variants_dir)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        # All four objectives should be present
        assert set(ds._subsets.keys()) == {
            "data_reconstruction", "description_generation",
            "structural_relational", "commit_reproduction",
        }

        # Each subset should have its own distinct variant bank
        assert ds._subsets["data_reconstruction"]._variants[0]["system_prompt"] == "fr"
        assert ds._subsets["description_generation"]._variants[0]["system_prompt"] == "dg"
        assert ds._subsets["structural_relational"]._variants[0]["system_prompt"] == "sr"
        assert ds._subsets["commit_reproduction"]._variants[0]["system_prompt"] == "cr"

    def test_missing_bank_uses_objective_specific_defaults(self, tmp_path):
        """Missing per-objective files should not reuse file-read prompts."""
        variants_dir = tmp_path / "variants"
        variants_dir.mkdir()

        shared_variant = [{"system_prompt": "shared", "user_prompt": "Read {file_path}",
                           "compression_prompt": "Reproduce.", "response_prefix": "{file_path}:"}]
        (variants_dir / "file_read_repro.json").write_text(json.dumps(shared_variant))

        cfg = self._make_full_config(tmp_path, variants_dir)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        assert ds._subsets["data_reconstruction"]._variants[0]["system_prompt"] == "shared"
        assert (
            "bgkit_describe"
            in ds._subsets["description_generation"]._variants[0]["system_prompt"]
        )
        assert (
            "bgkit_extract_structure"
            in ds._subsets["structural_relational"]._variants[0]["system_prompt"]
        )
        assert (
            "bgkit_reproduce_commit"
            in ds._subsets["commit_reproduction"]._variants[0]["system_prompt"]
        )

    def test_empty_bank_falls_back_to_objective_default(self, tmp_path):
        """Empty per-objective files should use the matching built-in default."""
        variants_dir = tmp_path / "variants"
        variants_dir.mkdir()

        shared_variant = [{"system_prompt": "shared", "user_prompt": "Read {file_path}",
                           "compression_prompt": "Reproduce.", "response_prefix": "{file_path}:"}]
        (variants_dir / "file_read_repro.json").write_text(json.dumps(shared_variant))
        (variants_dir / "description_gen.json").write_text("[]")  # empty

        cfg = self._make_full_config(tmp_path, variants_dir)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        assert (
            "bgkit_describe"
            in ds._subsets["description_generation"]._variants[0]["system_prompt"]
        )
        assert ds._subsets["data_reconstruction"]._variants[0]["system_prompt"] == "shared"

    def test_no_variants_dir_uses_fallback(self, tmp_path):
        """Without a variants dir, each objective gets a matching built-in prompt bank."""
        cfg = self._make_full_config(tmp_path, variants_dir=None)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        assert (
            ds._subsets["data_reconstruction"]._variants[0]["user_prompt"]
            == "Read the file `{file_path}`"
        )
        assert (
            ds._subsets["description_generation"]._variants[0]["user_prompt"]
            == "Describe `{file_path}`"
        )
        assert (
            ds._subsets["structural_relational"]._variants[0]["user_prompt"]
            == "Extract the structure of `{file_path}`"
        )
        assert (
            ds._subsets["commit_reproduction"]._variants[0]["user_prompt"]
            == "Reproduce the commit from repository {file_path}"
        )

    def test_commit_repro_accepts_commit_encoding_alias(self, tmp_path):
        """Step-4 prompt banks should be accepted for step-5 commit reproduction."""
        variants_dir = tmp_path / "variants"
        variants_dir.mkdir()
        (variants_dir / "file_read_repro.json").write_text(json.dumps([{
            "system_prompt": "shared",
            "user_prompt": "Read {file_path}",
            "compression_prompt": "Reproduce.",
            "response_prefix": "{file_path}:",
        }]))
        (variants_dir / "commit_encoding.json").write_text(json.dumps([{
            "system_prompt": "commit-alias",
            "user_prompt": "Commit {file_path}",
            "compression_prompt": "Commit alias.",
            "response_prefix": "Commit:",
        }]))

        cfg = self._make_full_config(tmp_path, variants_dir)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        assert ds._subsets["commit_reproduction"]._variants[0]["system_prompt"] == "commit-alias"

    def test_single_file_variant_path(self, tmp_path):
        """A single JSON file as prompt_variants_dir loads for all objectives."""
        variant_file = tmp_path / "variants.json"
        variant = [{"system_prompt": "single", "user_prompt": "Read {file_path}",
                     "compression_prompt": "Reproduce.", "response_prefix": "{file_path}:"}]
        variant_file.write_text(json.dumps(variant))

        cfg = self._make_full_config(tmp_path, variants_dir=variant_file)
        tokenizer = _MockTokenizer()
        ds = CompressionDataset.from_config(cfg, tokenizer, seed=42)

        for key in ds._subsets:
            assert ds._subsets[key]._variants[0]["system_prompt"] == "single", (
                f"{key} should use single-file bank"
            )
