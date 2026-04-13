"""Tests for the namespaced L0 cache."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.l0_cache import L0Cache, L0CacheWriter, update_dataset_index


def _populate_dataset(cache_dir, dataset: str, articles: dict[str, np.ndarray]) -> None:
    writer = L0CacheWriter(cache_dir, dataset, "shard_0000")
    for aid, arr in articles.items():
        writer.add(aid, arr)
    _, index_rows = writer.finalize()
    update_dataset_index(cache_dir, dataset, "shard_0000", index_rows)


def test_l0_cache_key_namespacing(tmp_path):
    # Two datasets with a colliding article id.
    a = {"alpha": np.ones((3, 4), dtype=np.float16)}
    b = {"alpha": np.full((2, 4), 7, dtype=np.float16)}
    _populate_dataset(tmp_path, "dataset_a", a)
    _populate_dataset(tmp_path, "dataset_b", b)

    cache = L0Cache(tmp_path)
    rows_a = cache.get("dataset_a", "alpha")
    rows_b = cache.get("dataset_b", "alpha")
    assert rows_a.shape == (3, 4)
    assert rows_b.shape == (2, 4)
    assert torch.allclose(rows_a.float(), torch.ones(3, 4))
    assert torch.allclose(rows_b.float(), torch.full((2, 4), 7.0))


def test_l0_cache_batch(tmp_path):
    articles = {
        "art_a": np.ones((3, 4), dtype=np.float16),
        "art_b": np.full((5, 4), 2, dtype=np.float16),
        "art_c": np.full((2, 4), 3, dtype=np.float16),
    }
    _populate_dataset(tmp_path, "ds", articles)
    cache = L0Cache(tmp_path)
    batch, mask = cache.get_batch("ds", ["art_a", "art_b", "art_c"])
    assert batch.shape == (3, 5, 4)
    assert mask.shape == (3, 5)
    # Padding rows should be zero; real rows match input
    assert mask[0, :3].all() and not mask[0, 3:].any()
    assert mask[1].all()
    assert mask[2, :2].all() and not mask[2, 2:].any()


def test_l0_cache_miss_raises(tmp_path):
    _populate_dataset(
        tmp_path, "ds", {"present": np.ones((1, 4), dtype=np.float16)},
    )
    cache = L0Cache(tmp_path)
    with pytest.raises(KeyError):
        cache.get("ds", "absent")
