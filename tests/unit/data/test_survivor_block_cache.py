"""Tests for the generic node-keyed :class:`SurvivorBlockCache`."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.l0_cache import (
    L0Cache,
    SurvivorBlockCache,
    SurvivorBlockCacheWriter,
    update_dataset_index,
)


def _populate(cache_dir, dataset, blocks, id_column="node_id"):
    writer = SurvivorBlockCacheWriter(cache_dir, dataset, "shard_0000")
    for nid, arr in blocks.items():
        writer.add(nid, arr)
    _, index_rows = writer.finalize()
    update_dataset_index(
        cache_dir, dataset, "shard_0000", index_rows, id_column=id_column,
    )


def test_node_keyed_roundtrip(tmp_path):
    blocks = {
        "root": np.ones((2, 4), dtype=np.float16),
        "topic/a": np.full((3, 4), 5, dtype=np.float16),
        "topic/b": np.full((1, 4), 9, dtype=np.float16),
    }
    _populate(tmp_path, "toy", blocks)

    cache = SurvivorBlockCache(tmp_path)
    assert cache.has("toy", "root")
    assert cache.has("toy", "topic/a")
    assert not cache.has("toy", "missing")
    assert sorted(cache.node_ids("toy")) == ["root", "topic/a", "topic/b"]

    r = cache.get("toy", "topic/a")
    assert r.shape == (3, 4)
    assert torch.allclose(r.float(), torch.full((3, 4), 5.0))


def test_node_keyed_miss_raises(tmp_path):
    _populate(tmp_path, "toy", {"n0": np.ones((1, 4), dtype=np.float16)})
    cache = SurvivorBlockCache(tmp_path)
    with pytest.raises(KeyError):
        cache.get("toy", "absent")


def test_l0cache_reads_node_keyed_index(tmp_path):
    """L0Cache (article_id column) and SurvivorBlockCache (node_id) read the
    same on-disk layout — the reader is id-column tolerant."""
    _populate(tmp_path, "toy", {"n0": np.ones((2, 4), dtype=np.float16)},
              id_column="node_id")
    # L0Cache pins article_id but must still read a node_id index.
    cache = L0Cache(tmp_path)
    assert cache.has("toy", "n0")
    assert cache.get("toy", "n0").shape == (2, 4)


def test_shard_rollover_idempotent_index(tmp_path):
    _populate(tmp_path, "toy", {"a": np.ones((1, 4), dtype=np.float16)})
    # Add a second shard, refresh index — first shard rows survive.
    w = SurvivorBlockCacheWriter(tmp_path, "toy", "shard_0001")
    w.add("b", np.full((2, 4), 3, dtype=np.float16))
    _, rows = w.finalize()
    update_dataset_index(tmp_path, "toy", "shard_0001", rows, id_column="node_id")

    cache = SurvivorBlockCache(tmp_path)
    assert cache.has("toy", "a") and cache.has("toy", "b")
    assert cache.get("toy", "b").shape == (2, 4)
