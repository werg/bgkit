"""Tests for ICEDataset."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.ice_dataset import ICEDataset


@pytest.fixture
def ice_shard_dir(tmp_path: Path) -> Path:
    """Create a test fixture with a small ICE label shard."""
    shard_dir = tmp_path / "ice_labels"
    shard_dir.mkdir()

    # Create a small test shard
    n_rows = 5
    table = pa.table({
        "repo_path": pa.array([f"repo/{i}" for i in range(n_rows)], type=pa.string()),
        "file_path": pa.array([f"file_{i}.py" for i in range(n_rows)], type=pa.string()),
        "language": pa.array(["Python"] * n_rows, type=pa.string()),
        "chunk_idx": pa.array(list(range(n_rows)), type=pa.int32()),
        "token_ids": pa.array(
            [np.array(list(range(10 + i * 5)), dtype=np.int32) for i in range(n_rows)],
            type=pa.list_(pa.int32()),
        ),
        "ce_values": pa.array(
            [
                np.array([0.5 * j for j in range(9 + i * 5)], dtype=np.float16)
                for i in range(n_rows)
            ],
            type=pa.list_(pa.float16()),
        ),
    })

    pq.write_table(table, shard_dir / "shard_00000.parquet")
    return shard_dir


class TestICEDataset:
    def test_len(self, ice_shard_dir: Path):
        ds = ICEDataset(str(ice_shard_dir))
        assert len(ds) == 5

    def test_getitem_returns_dict(self, ice_shard_dir: Path):
        ds = ICEDataset(str(ice_shard_dir))
        sample = ds[0]
        assert isinstance(sample, dict)
        assert "token_ids" in sample
        assert "ce_values" in sample

    def test_getitem_tensor_types(self, ice_shard_dir: Path):
        ds = ICEDataset(str(ice_shard_dir))
        sample = ds[0]
        assert isinstance(sample["token_ids"], torch.Tensor)
        assert isinstance(sample["ce_values"], torch.Tensor)
        assert sample["token_ids"].dtype == torch.int64
        assert sample["ce_values"].dtype == torch.float32

    def test_getitem_shapes(self, ice_shard_dir: Path):
        ds = ICEDataset(str(ice_shard_dir))
        sample = ds[0]
        # First row: 10 token_ids, 9 ce_values
        assert sample["token_ids"].shape == (10,)
        assert sample["ce_values"].shape == (9,)

    def test_all_items_accessible(self, ice_shard_dir: Path):
        ds = ICEDataset(str(ice_shard_dir))
        for i in range(len(ds)):
            sample = ds[i]
            assert sample["token_ids"].ndim == 1
            assert sample["ce_values"].ndim == 1

    def test_empty_dir(self, tmp_path: Path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        ds = ICEDataset(str(empty_dir))
        assert len(ds) == 0

    def test_multiple_shards(self, tmp_path: Path):
        shard_dir = tmp_path / "multi"
        shard_dir.mkdir()

        for shard_idx in range(3):
            table = pa.table({
                "repo_path": pa.array(["repo/a"], type=pa.string()),
                "file_path": pa.array([f"file_{shard_idx}.py"], type=pa.string()),
                "language": pa.array(["Python"], type=pa.string()),
                "chunk_idx": pa.array([0], type=pa.int32()),
                "token_ids": pa.array(
                    [np.array([1, 2, 3], dtype=np.int32)], type=pa.list_(pa.int32())
                ),
                "ce_values": pa.array(
                    [np.array([0.5, 1.0], dtype=np.float16)], type=pa.list_(pa.float16())
                ),
            })
            pq.write_table(table, shard_dir / f"shard_{shard_idx:05d}.parquet")

        ds = ICEDataset(str(shard_dir))
        assert len(ds) == 3
