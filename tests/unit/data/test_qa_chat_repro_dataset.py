"""Tests for QAChatReproDataset: chat-formatted QA for decoder init."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.qa_chat_repro_dataset import QAChatReproDataset
from tests.unit.data.test_chat_repro_dataset import MockTokenizer
from tests.unit.data.test_qa_conditioned_dataset import (
    MockTokenDataset,
    _create_qa_artifacts,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mock_token_ds():
    return MockTokenDataset([
        {
            "token_ids": [100, 200, 300],
            "file_path": "src/main.py",
            "language": "python",
            "repo_path": "owner/repo",
            "commit_sha": "sha1",
        },
        {
            "token_ids": [400, 500],
            "file_path": "lib/utils.js",
            "language": "javascript",
            "repo_path": "owner/repo",
            "commit_sha": "sha1",
        },
    ])


@pytest.fixture()
def mock_qa_ds(tmp_path):
    d = tmp_path / "qa"
    d.mkdir()
    _create_qa_artifacts(
        d,
        answer_lists=[[10, 20, 30], [40, 50]],
        question_lists=[[1, 2], [3, 4, 5]],
        metadata_columns={
            "repo_path": ["owner/repo", "owner/repo"],
            "file_path": ["src/main.py", "src/main.py"],
            "commit_sha": ["sha1", "sha1"],
        },
    )
    from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset

    return MmapQAConditionedDataset(str(d))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestQAChatReproDataset:
    def test_basic_construction(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        # 2 QA rows both join to src/main.py (single chunk)
        assert len(ds) == 2

    def test_no_join_for_missing_key(self, tmp_path):
        token_ds = MockTokenDataset([
            {
                "token_ids": [1, 2],
                "file_path": "a.py",
                "language": "python",
                "repo_path": "owner/repo",
                "commit_sha": "sha1",
            },
        ])
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[10, 20]],
            question_lists=[[1]],
            metadata_columns={
                "repo_path": ["owner/OTHER"],
                "file_path": ["a.py"],
                "commit_sha": ["sha1"],
            },
        )
        from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset

        qa_ds = MmapQAConditionedDataset(str(d))
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(qa_ds, token_ds, tokenizer)
        assert len(ds) == 0

    def test_multi_row_takes_first(self, tmp_path):
        """When the same (owner/repo, file_path) appears multiple times in
        the token_dataset (e.g. scanned at different commits), the join
        takes the FIRST occurrence rather than dropping the QA sample.
        Previously this test asserted exclusion (single-chunk-only); we
        now keep the QA sample paired against the first token row to
        recover ~22% more training data, since chunking-into-multiple-
        rows isn't actually happening in our token mmap."""
        token_ds = MockTokenDataset([
            {
                "token_ids": [1, 2],
                "file_path": "big.py",
                "language": "python",
                "repo_path": "owner/repo",
                "commit_sha": "sha1",
            },
            {
                "token_ids": [3, 4],
                "file_path": "big.py",
                "language": "python",
                "repo_path": "owner/repo",
                "commit_sha": "sha2",  # different commit, same file path
            },
        ])
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[10, 20]],
            question_lists=[[1]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["big.py"],
                "commit_sha": ["sha1"],
            },
        )
        from bgkit.data.datasets.qa_conditioned_dataset import MmapQAConditionedDataset

        qa_ds = MmapQAConditionedDataset(str(d))
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(qa_ds, token_ds, tokenizer)
        assert len(ds) == 1
        # First occurrence (sha1, token_ids [1,2]) wins.
        assert ds._joined_indices[0] == (0, 0)

    def test_getitem_returns_token_ids(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        sample = ds[0]
        assert "token_ids" in sample
        assert isinstance(sample["token_ids"], torch.Tensor)
        assert len(sample["token_ids"]) > 0

    def test_getitem_returns_loss_mask(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        sample = ds[0]
        assert "loss_mask" in sample
        assert sample["loss_mask"].shape == sample["token_ids"].shape

    def test_suffix_ids_property(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        assert isinstance(ds.suffix_ids, torch.Tensor)
        assert len(ds.suffix_ids) > 0

    def test_lengths_property(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        assert len(ds.lengths) == len(ds)
        assert all(ds.lengths > 0)

    def test_set_epoch(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer, seed=42)
        ds.set_epoch(5)
        assert ds._epoch_seed == 42 + 5

    def test_language_in_sample(self, mock_token_ds, mock_qa_ds):
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        sample = ds[0]
        assert "language" in sample

    def test_content_token_ids_are_source_file_not_answer(
        self, mock_token_ds, mock_qa_ds,
    ):
        """content_token_ids must be source file tokens for the BgKIT encoder,
        NOT answer tokens (which only appear in the decoder sequence)."""
        tokenizer = MockTokenizer()
        ds = QAChatReproDataset(mock_qa_ds, mock_token_ds, tokenizer)
        sample = ds[0]
        # Source file for src/main.py has tokens [100, 200, 300]
        expected_source = torch.tensor([100, 200, 300], dtype=torch.long)
        assert torch.equal(sample["content_token_ids"], expected_source)
