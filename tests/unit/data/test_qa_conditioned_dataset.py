"""Tests for MmapQAConditionedDataset and QAConditionedSubset."""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.qa_conditioned_dataset import (
    MmapQAConditionedDataset,
    QAConditionedSubset,
)


# ---------------------------------------------------------------------------
# Fixture: create QA mmap artifacts (answer + question tokens + metadata)
# ---------------------------------------------------------------------------


def _create_qa_artifacts(
    data_dir: Path,
    answer_lists: list[list[int]],
    question_lists: list[list[int]],
    metadata_columns: dict[str, list],
) -> Path:
    """Write QA mmap artifacts (answer tokens, question tokens, metadata)."""
    # Answer tokens
    all_answer = []
    answer_offsets = [0]
    for tids in answer_lists:
        all_answer.extend(tids)
        answer_offsets.append(len(all_answer))

    np.save(data_dir / "tokens.npy", np.array(all_answer, dtype=np.int32))
    np.save(data_dir / "offsets.npy", np.array(answer_offsets, dtype=np.int64))

    # Question tokens
    all_question = []
    question_offsets = [0]
    for tids in question_lists:
        all_question.extend(tids)
        question_offsets.append(len(all_question))

    np.save(data_dir / "question_tokens.npy", np.array(all_question, dtype=np.int32))
    np.save(data_dir / "question_offsets.npy", np.array(question_offsets, dtype=np.int64))

    # Manifest
    manifest = {
        "schema_version": 1,
        "row_count": len(answer_lists),
        "total_tokens": len(all_answer),
    }
    (data_dir / "manifest.json").write_text(json.dumps(manifest))

    # Metadata
    arrow_cols: dict[str, pa.Array] = {}
    for col_name, values in metadata_columns.items():
        arrow_cols[col_name] = pa.array(values, type=pa.string())
    pq.write_table(pa.table(arrow_cols), data_dir / "metadata.parquet")

    return data_dir


class MockTokenDataset:
    """Minimal mock of MmapTokenDataset with file_key() and metadata."""

    def __init__(self, samples: list[dict]):
        self._samples = samples
        self._lengths = np.array(
            [len(s["token_ids"]) for s in samples], dtype=np.int32
        )

    @property
    def lengths(self) -> np.ndarray:
        return self._lengths

    def file_key(self, idx: int):
        s = self._samples[idx]
        rp = s.get("repo_path", "")
        fp = s.get("file_path", "")
        cs = s.get("commit_sha", "")
        if not rp or not fp:
            return None
        return (rp, fp, cs)

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        s = self._samples[idx]
        return {
            "token_ids": torch.tensor(s["token_ids"], dtype=torch.long),
            "file_path": s["file_path"],
            "language": s.get("language", "python"),
            "repo_path": s.get("repo_path", ""),
            "commit_sha": s.get("commit_sha", ""),
        }


# ---------------------------------------------------------------------------
# Tests: MmapQAConditionedDataset
# ---------------------------------------------------------------------------


class TestMmapQAConditionedDataset:
    def test_basic_load(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[1, 2, 3], [10, 20, 30, 40]],
            question_lists=[[5, 6], [7, 8, 9]],
            metadata_columns={
                "repo_path": ["owner/repo1", "owner/repo1"],
                "file_path": ["a.py", "b.py"],
                "commit_sha": ["abc123", "abc123"],
            },
        )
        ds = MmapQAConditionedDataset(str(d))
        assert len(ds) == 2

    def test_getitem_returns_answer_and_question(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[1, 2, 3]],
            question_lists=[[5, 6, 7]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["main.py"],
                "commit_sha": ["sha1"],
            },
        )
        ds = MmapQAConditionedDataset(str(d))
        sample = ds[0]
        assert "answer_token_ids" in sample
        assert "question_token_ids" in sample
        assert "token_ids" not in sample  # renamed to answer_token_ids
        assert sample["answer_token_ids"].tolist() == [1, 2, 3]
        assert sample["question_token_ids"].tolist() == [5, 6, 7]

    def test_file_key(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[1, 2], [3, 4]],
            question_lists=[[5], [6]],
            metadata_columns={
                "repo_path": ["owner/repo1", "owner/repo2"],
                "file_path": ["a.py", "b.py"],
                "commit_sha": ["sha1", "sha2"],
            },
        )
        ds = MmapQAConditionedDataset(str(d))
        assert ds.file_key(0) == ("owner/repo1", "a.py", "sha1")
        assert ds.file_key(1) == ("owner/repo2", "b.py", "sha2")

    def test_truncation(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[list(range(100))],
            question_lists=[[1, 2]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["sha"],
            },
        )
        ds = MmapQAConditionedDataset(str(d), max_seq_len=10)
        sample = ds[0]
        assert sample["answer_token_ids"].shape == (10,)

    def test_pickle_roundtrip(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[1, 2, 3]],
            question_lists=[[4, 5]],
            metadata_columns={
                "repo_path": ["owner/repo"],
                "file_path": ["a.py"],
                "commit_sha": ["sha"],
            },
        )
        ds = MmapQAConditionedDataset(str(d))
        data = pickle.dumps(ds)
        ds2 = pickle.loads(data)
        assert len(ds2) == 1
        sample = ds2[0]
        assert sample["answer_token_ids"].tolist() == [1, 2, 3]

    def test_zero_length_filtered(self, tmp_path):
        d = tmp_path / "qa"
        d.mkdir()
        _create_qa_artifacts(
            d,
            answer_lists=[[], [1, 2]],  # first entry is empty
            question_lists=[[1], [2]],
            metadata_columns={
                "repo_path": ["owner/r1", "owner/r2"],
                "file_path": ["a.py", "b.py"],
                "commit_sha": ["s1", "s2"],
            },
        )
        ds = MmapQAConditionedDataset(str(d))
        assert len(ds) == 1  # zero-length filtered


# ---------------------------------------------------------------------------
# Tests: QAConditionedSubset
# ---------------------------------------------------------------------------


class TestQAConditionedSubset:
    @pytest.fixture()
    def mock_token_ds(self):
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
    def mock_qa_ds(self, tmp_path):
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
        return MmapQAConditionedDataset(str(d))

    def test_join_count(self, mock_token_ds, mock_qa_ds):
        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(
            mock_token_ds, mock_qa_ds, MockTokenizer(), seed=42,
        )
        # 2 QA rows both join to "src/main.py" (single chunk)
        assert len(subset) == 2

    def test_no_join_for_missing_file(self, tmp_path):
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
                "repo_path": ["owner/OTHER"],  # no match
                "file_path": ["a.py"],
                "commit_sha": ["sha1"],
            },
        )
        qa_ds = MmapQAConditionedDataset(str(d))

        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(token_ds, qa_ds, MockTokenizer())
        assert len(subset) == 0

    def test_multi_chunk_files_excluded(self, tmp_path):
        """Files chunked into >1 piece should not join."""
        # Two chunks mapping to same file key
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
                "repo_path": ["owner/repo"],
                "file_path": ["big.py"],
                "commit_sha": ["sha1"],
            },
        )
        qa_ds = MmapQAConditionedDataset(str(d))

        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(token_ds, qa_ds, MockTokenizer())
        assert len(subset) == 0  # multi-chunk excluded

    def test_getitem_returns_file_compression_sample(self, mock_token_ds, mock_qa_ds):
        from bgkit.data.datasets.compression_dataset import FileCompressionSample
        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(
            mock_token_ds, mock_qa_ds, MockTokenizer(), seed=42,
        )
        sample = subset[0]
        assert isinstance(sample, FileCompressionSample)
        assert sample.objective == "query_conditioned"
        assert sample.compression_level == 0
        # Content should be the source file tokens
        assert sample.content_token_ids.tolist() == [100, 200, 300]

    def test_set_epoch(self, mock_token_ds, mock_qa_ds):
        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(
            mock_token_ds, mock_qa_ds, MockTokenizer(), seed=42,
        )
        subset.set_epoch(5)
        assert subset._epoch_seed == 42 + 5

    def test_lengths_property(self, mock_token_ds, mock_qa_ds):
        from tests.unit.data.test_chat_repro_dataset import MockTokenizer

        subset = QAConditionedSubset(
            mock_token_ds, mock_qa_ds, MockTokenizer(), seed=42,
        )
        assert len(subset.lengths) == len(subset)
        # Each length should be > 0
        assert all(subset.lengths > 0)
