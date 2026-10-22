"""Tests for the ``(dataset, document_id)`` article token store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

torch = pytest.importorskip("torch")

from bgkit.data.article_token_store import ArticleTokenStore


def _write_mmap_dataset(
    root: Path,
    dataset: str,
    docs: list[tuple[str, list[int]]],
    tag_list: list[str] | None = None,
) -> None:
    """Write a minimal Phase 2 mmap layout matching convert_hf_to_mmap.py."""
    d = root / dataset
    d.mkdir(parents=True, exist_ok=True)

    tokens: list[int] = []
    offsets: list[int] = [0]
    for _doc_id, token_ids in docs:
        tokens.extend(token_ids)
        offsets.append(len(tokens))
    np.save(d / "tokens.npy", np.asarray(tokens, dtype=np.int32))
    offsets_array = np.asarray(offsets, dtype=np.int64)
    np.save(d / "offsets.npy", offsets_array)

    tl = tag_list or []
    table = pa.Table.from_pylist(
        [
            {
                "id": doc_id,
                "document_id": doc_id,
                "dataset_name": dataset,
                "tag_list_json": json.dumps(tl),
            }
            for doc_id, _ in docs
        ],
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
            ("tag_list_json", pa.string()),
        ]),
    )
    pq.write_table(table, d / "metadata.parquet")
    (d / "manifest.json").write_text(json.dumps({
        "schema_version": 1,
        "row_count": len(docs),
        "total_tokens": len(tokens),
        "offsets_sha256": hashlib.sha256(offsets_array.tobytes()).hexdigest(),
    }))


def test_get_by_document_id(tmp_path):
    _write_mmap_dataset(
        tmp_path,
        "ds",
        [("doc_a", [1, 2, 3]), ("doc_b", [4, 5]), ("doc_c", [6, 7, 8, 9])],
    )
    store = ArticleTokenStore(tmp_path)
    a = store.get("ds", "doc_a")
    b = store.get("ds", "doc_b")
    c = store.get("ds", "doc_c")
    assert a.tolist() == [1, 2, 3]
    assert b.tolist() == [4, 5]
    assert c.tolist() == [6, 7, 8, 9]


def test_missing_document_id_raises(tmp_path):
    _write_mmap_dataset(tmp_path, "ds", [("doc_a", [1, 2])])
    store = ArticleTokenStore(tmp_path)
    assert store.has("ds", "doc_a")
    assert not store.has("ds", "missing")
    with pytest.raises(KeyError):
        store.get("ds", "missing")


def test_multi_dataset_isolation(tmp_path):
    """Two datasets with colliding document_ids stay separated."""
    _write_mmap_dataset(tmp_path, "ds_a", [("alpha", [1, 1, 1])])
    _write_mmap_dataset(tmp_path, "ds_b", [("alpha", [9, 9, 9, 9])])
    store = ArticleTokenStore(tmp_path)
    a = store.get("ds_a", "alpha")
    b = store.get("ds_b", "alpha")
    assert a.tolist() == [1, 1, 1]
    assert b.tolist() == [9, 9, 9, 9]


def test_get_batch_pads_and_masks(tmp_path):
    _write_mmap_dataset(
        tmp_path,
        "ds",
        [
            ("doc_a", [1, 2, 3]),
            ("doc_b", [4, 5, 6, 7, 8]),
            ("doc_c", [9]),
        ],
    )
    store = ArticleTokenStore(tmp_path)
    batch, mask = store.get_batch("ds", ["doc_a", "doc_b", "doc_c"])
    assert batch.shape == (3, 5)
    assert mask.shape == (3, 5)
    assert batch[0, :3].tolist() == [1, 2, 3]
    assert not mask[0, 3:].any()
    assert mask[1].all()
    assert batch[2, 0].item() == 9
    assert not mask[2, 1:].any()


def test_missing_dataset_raises(tmp_path):
    store = ArticleTokenStore(tmp_path)
    with pytest.raises(FileNotFoundError):
        store.get("ghost", "doc_a")


def test_document_ids_list(tmp_path):
    _write_mmap_dataset(
        tmp_path,
        "ds",
        [("doc_a", [1]), ("doc_b", [2]), ("doc_c", [3])],
    )
    store = ArticleTokenStore(tmp_path)
    assert set(store.document_ids("ds")) == {"doc_a", "doc_b", "doc_c"}


def test_duplicate_document_ids_are_rejected(tmp_path):
    _write_mmap_dataset(tmp_path, "ds", [("same", [1]), ("same", [2])])
    store = ArticleTokenStore(tmp_path)
    with pytest.raises(ValueError, match="duplicate document_id"):
        store.document_ids("ds")


def test_mixed_generation_counts_are_rejected(tmp_path):
    _write_mmap_dataset(tmp_path, "ds", [("a", [1, 2])])
    manifest_path = tmp_path / "ds" / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["total_tokens"] = 99
    manifest_path.write_text(json.dumps(manifest))
    store = ArticleTokenStore(tmp_path)
    with pytest.raises(ValueError, match="counts do not match"):
        store.get("ds", "a")
