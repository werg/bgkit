"""Tests for CommitReproDataset."""

from __future__ import annotations

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.commit_repro_dataset import CommitReproDataset


def _write_test_shard(path, num_rows=5):
    """Write a minimal test Parquet shard."""
    table = pa.table({
        "repo_path": pa.array([f"/repo/{i}" for i in range(num_rows)], type=pa.string()),
        "sha": pa.array([f"{'a' * 39}{i}" for i in range(num_rows)], type=pa.string()),
        "message": pa.array([f"commit {i}" for i in range(num_rows)], type=pa.string()),
        "num_files": pa.array([1] * num_rows, type=pa.int32()),
        "is_cross_file": pa.array([False] * num_rows, type=pa.bool_()),
        "token_ids": pa.array(
            [np.array([100 + i, 200 + i, 300 + i], dtype=np.int32) for i in range(num_rows)],
            type=pa.list_(pa.int32()),
        ),
    })
    pq.write_table(table, path, compression="zstd")


class TestCommitReproDataset:
    def test_len(self, tmp_path):
        _write_test_shard(tmp_path / "shard_00000.parquet", num_rows=5)
        _write_test_shard(tmp_path / "shard_00001.parquet", num_rows=3)

        ds = CommitReproDataset(str(tmp_path))
        assert len(ds) == 8

    def test_getitem(self, tmp_path):
        _write_test_shard(tmp_path / "shard_00000.parquet", num_rows=3)

        ds = CommitReproDataset(str(tmp_path))
        sample = ds[0]

        assert "token_ids" in sample
        assert isinstance(sample["token_ids"], torch.Tensor)
        assert sample["token_ids"].dtype == torch.int64
        assert len(sample["token_ids"]) == 3

    def test_empty_dir(self, tmp_path):
        ds = CommitReproDataset(str(tmp_path))
        assert len(ds) == 0

    def test_ignores_non_shard_files(self, tmp_path):
        _write_test_shard(tmp_path / "shard_00000.parquet", num_rows=2)
        # Write a non-shard file
        (tmp_path / "manifest.jsonl").write_text("{}\n")

        ds = CommitReproDataset(str(tmp_path))
        assert len(ds) == 2

    def test_all_items_accessible(self, tmp_path):
        _write_test_shard(tmp_path / "shard_00000.parquet", num_rows=4)

        ds = CommitReproDataset(str(tmp_path))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["token_ids"].shape[0] > 0

    def test_max_seq_len_truncates(self, tmp_path):
        """Tokens longer than max_seq_len should be truncated."""
        # Write a shard with longer token sequences
        table = pa.table({
            "repo_path": pa.array(["/repo/0"], type=pa.string()),
            "sha": pa.array(["a" * 40], type=pa.string()),
            "message": pa.array(["msg"], type=pa.string()),
            "num_files": pa.array([1], type=pa.int32()),
            "is_cross_file": pa.array([False], type=pa.bool_()),
            "token_ids": pa.array(
                [np.arange(100, dtype=np.int32)],
                type=pa.list_(pa.int32()),
            ),
        })
        pq.write_table(table, tmp_path / "shard_00000.parquet", compression="zstd")

        ds = CommitReproDataset(str(tmp_path), max_seq_len=10)
        sample = ds[0]
        assert sample["token_ids"].shape[0] == 10
