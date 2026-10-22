"""Tests for the precomputed L0 cache."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.datasets.precomputed_l0_cache import PrecomputedL0Cache


def _write_cache(base: Path) -> Path:
    shard = base / "shard_000"
    shard.mkdir(parents=True)
    np.save(shard / "survivors.npy", np.array([
        [1.0, 0.0],
        [2.0, 0.0],
        [3.0, 0.0],
        [4.0, 0.0],
    ], dtype=np.float32))
    np.save(shard / "offsets.npy", np.array([0, 3, 4], dtype=np.int64))
    np.save(shard / "ice_scores.npy", np.array([0.3, 0.9, 0.1, 0.8], dtype=np.float32))
    pq.write_table(
        pa.table({
            "document_id": pa.array(["doc-a", "doc-b"]),
            "shard_id": pa.array(["shard_000", "shard_000"]),
            "row_index": pa.array([0, 1], type=pa.int32()),
        }),
        base / "index.parquet",
    )
    return base


def test_precomputed_l0_cache_subselects_by_retention(tmp_path):
    cache = PrecomputedL0Cache(str(_write_cache(tmp_path / "cache")))
    survivors = cache.get_survivors("doc-a", retention_ratio=0.34)
    assert survivors.shape == (1, 2)
    assert survivors.tolist() == [[2.0, 0.0]]


def test_precomputed_l0_cache_batches_and_pads(tmp_path):
    cache = PrecomputedL0Cache(str(_write_cache(tmp_path / "cache")))
    padded, mask = cache.get_survivors_batch(["doc-a", "doc-b"], retention_ratio=1.0)
    assert padded.shape == (2, 3, 2)
    assert mask.tolist() == [[True, True, True], [True, False, False]]
