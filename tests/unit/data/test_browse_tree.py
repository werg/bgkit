"""Tests for browse tree construction, serialization, and rendering."""

from __future__ import annotations

import pytest

from bgkit.data.browse_tree import BrowseNode, BrowseTree
from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig


def _small_tree() -> BrowseTree:
    builder = BrowseTreeBuilder(TaggingConfig(dataset="toy", leaf_cap=10, fanout_cap=10))
    # 5 top-tags * 5 sub-tags * 5 articles
    for top in ["Physics", "Biology", "Chemistry", "Math", "History"]:
        for sub in [f"{top}_sub{j}" for j in range(3)]:
            for art in [f"{sub}_article{k}" for k in range(3)]:
                builder.add_article(art, [top, sub])
    return builder.build()


def test_browse_tree_roundtrip(tmp_path):
    tree = _small_tree()
    path = tmp_path / "toy.parquet"
    tree.save(path)
    loaded = BrowseTree.load(path, dataset="toy")
    assert len(loaded) == len(tree)
    assert "root" in loaded
    assert "Physics" in loaded
    children = loaded.children("root")
    assert {c.id for c in children} >= {"Physics", "Biology"}


def test_browse_tree_topic_list():
    tree = _small_tree()
    tops = tree.top_level_topic_list()
    assert set(tops) == {"Physics", "Biology", "Chemistry", "Math", "History"}


def test_browse_tree_path_to_leaf():
    tree = _small_tree()
    leaves = [n for n in [tree.get(c) for c in tree.get("Physics").children]]
    # First descend into a sub-tag
    sub = leaves[0]
    assert sub.is_leaf_tag
    path = tree.path_to(sub.id)
    assert path[0] == "root"
    assert path[-1] == sub.id


def test_browse_tree_articles_under_tag():
    tree = _small_tree()
    articles = tree.articles("Physics")
    assert len(articles) == 9  # 3 sub-tags * 3 articles


def test_fanout_capping_subdivides():
    builder = BrowseTreeBuilder(
        TaggingConfig(dataset="big", leaf_cap=5, fanout_cap=100),
    )
    # One leaf with 20 articles — must be sub-divided into ≤5 each.
    for i in range(20):
        builder.add_article(f"article_{i:03d}", ["OverLarge"])
    tree = builder.build()
    # Walk every leaf under OverLarge and verify size <= 5.
    def walk(nid: str) -> list[str]:
        node = tree.get(nid)
        if node.is_leaf_tag:
            return [nid]
        out: list[str] = []
        for c in node.children:
            if tree.get(c).is_article:
                continue
            out.extend(walk(c))
        return out
    leaves = walk("OverLarge")
    assert leaves, "OverLarge should have been split into sub-leaves"
    for nid in leaves:
        assert len(tree.get(nid).articles) <= 5


def test_leaf_tag_for_article_lookup():
    tree = _small_tree()
    # Pick any article id we created
    leaf = tree.leaf_tag_for_article("Physics_sub0_article1")
    assert leaf == "Physics/Physics_sub0"


def test_builder_ingests_git_history_with_repo_hierarchy(tmp_path, caplog):
    """Backwards-compat: without commit_ts, falls back to flat root → org → repo."""
    import logging
    import runpy
    import sys

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = tmp_path / "git_history"
    ds_dir.mkdir()
    np.save(ds_dir / "tokens.npy", np.zeros(30, dtype=np.int32))
    np.save(
        ds_dir / "offsets.npy",
        np.array([0, 10, 20, 30], dtype=np.int64),
    )
    table = pa.Table.from_pylist(
        [
            {
                "id": "c0",
                "document_id": "c0",
                "dataset_name": "git_history",
                "tag_list_json": "[]",
                "repo_path": "torvalds/linux",
                "commit_sha": "deadbeef",
            },
            {
                "id": "c1",
                "document_id": "c1",
                "dataset_name": "git_history",
                "tag_list_json": "[]",
                "repo_path": "torvalds/linux",
                "commit_sha": "cafebabe",
            },
            {
                "id": "c2",
                "document_id": "c2",
                "dataset_name": "git_history",
                "tag_list_json": "[]",
                "repo_path": "pytorch/pytorch",
                "commit_sha": "f00dface",
            },
        ],
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
            ("tag_list_json", pa.string()),
            ("repo_path", pa.string()),
            ("commit_sha", pa.string()),
        ]),
    )
    pq.write_table(table, ds_dir / "metadata.parquet")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "git_history",
        "--phase2-dir", str(ds_dir),
        "--output-dir", str(output_dir),
    ]
    with caplog.at_level(logging.WARNING, logger="build_browse_tree"):
        runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    from bgkit.data.browse_tree import BrowseTree

    tree = BrowseTree.load(output_dir / "git_history.parquet", dataset="git_history")
    # Top-level: orgs
    tops = tree.top_level_topic_list()
    assert "torvalds" in tops
    assert "pytorch" in tops
    # Each commit lives under org/repo (no year bucket — commit_ts absent)
    linux_leaf = tree.get("torvalds/linux")
    assert linux_leaf.is_leaf_tag
    assert set(linux_leaf.articles) == {"c0", "c1"}
    pt_leaf = tree.get("pytorch/pytorch")
    assert set(pt_leaf.articles) == {"c2"}
    # A warning should have been emitted telling the operator to
    # re-run generate_git_qa.py for the new schema.
    assert any(
        "commit_ts" in rec.getMessage() or "flat commit structure" in rec.getMessage()
        for rec in caplog.records
    )


def test_builder_ingests_git_history_with_year_buckets(tmp_path):
    """With commit_ts present, builds root → org → org/repo → year_YYYY leaves."""
    import runpy
    import sys

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = tmp_path / "git_history"
    ds_dir.mkdir()
    np.save(ds_dir / "tokens.npy", np.zeros(50, dtype=np.int32))
    np.save(
        ds_dir / "offsets.npy",
        np.array([0, 10, 20, 30, 40, 50], dtype=np.int64),
    )
    # Reference Unix timestamps (UTC):
    #   2020-01-01 00:00:00Z -> 1577836800
    #   2020-06-15 12:00:00Z -> 1592222400
    #   2023-01-01 00:00:00Z -> 1672531200
    rows = [
        {  # torvalds/linux, 2020
            "id": "c0", "document_id": "c0", "dataset_name": "git_history",
            "tag_list_json": "[]", "repo_path": "torvalds/linux",
            "commit_sha": "aaa", "commit_ts": 1577836800,
        },
        {  # torvalds/linux, 2020 (same year, different commit)
            "id": "c1", "document_id": "c1", "dataset_name": "git_history",
            "tag_list_json": "[]", "repo_path": "torvalds/linux",
            "commit_sha": "bbb", "commit_ts": 1592222400,
        },
        {  # torvalds/linux, 2023
            "id": "c2", "document_id": "c2", "dataset_name": "git_history",
            "tag_list_json": "[]", "repo_path": "torvalds/linux",
            "commit_sha": "ccc", "commit_ts": 1672531200,
        },
        {  # pytorch/pytorch, 2023
            "id": "c3", "document_id": "c3", "dataset_name": "git_history",
            "tag_list_json": "[]", "repo_path": "pytorch/pytorch",
            "commit_sha": "ddd", "commit_ts": 1672531200,
        },
        {  # legacy/old, unknown timestamp -> falls back to flat repo leaf
            "id": "c4", "document_id": "c4", "dataset_name": "git_history",
            "tag_list_json": "[]", "repo_path": "legacy/old",
            "commit_sha": "eee", "commit_ts": 0,
        },
    ]
    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
            ("tag_list_json", pa.string()),
            ("repo_path", pa.string()),
            ("commit_sha", pa.string()),
            ("commit_ts", pa.int64()),
        ]),
    )
    pq.write_table(table, ds_dir / "metadata.parquet")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "git_history",
        "--phase2-dir", str(ds_dir),
        "--output-dir", str(output_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    from bgkit.data.browse_tree import BrowseTree

    tree = BrowseTree.load(output_dir / "git_history.parquet", dataset="git_history")

    # Top-level: org names
    tops = set(tree.top_level_topic_list())
    assert "torvalds" in tops
    assert "pytorch" in tops
    assert "legacy" in tops

    # torvalds/linux has two year sub-buckets: 2020 and 2023
    linux_2020 = tree.get("torvalds/linux/year_2020")
    linux_2023 = tree.get("torvalds/linux/year_2023")
    assert linux_2020.is_leaf_tag
    assert linux_2023.is_leaf_tag
    assert set(linux_2020.articles) == {"c0", "c1"}
    assert set(linux_2023.articles) == {"c2"}
    # torvalds/linux itself is a sub-tag (not a leaf) — walking articles
    # from it should recurse into both year buckets.
    assert set(tree.articles("torvalds/linux")) == {"c0", "c1", "c2"}

    # pytorch/pytorch has exactly one year_2023 bucket with c3.
    pytorch_2023 = tree.get("pytorch/pytorch/year_2023")
    assert pytorch_2023.is_leaf_tag
    assert set(pytorch_2023.articles) == {"c3"}

    # legacy/old has no usable timestamp, so falls back to the flat
    # repo leaf (the same shape the pre-commit_ts ingester produced).
    legacy_leaf = tree.get("legacy/old")
    assert legacy_leaf.is_leaf_tag
    assert set(legacy_leaf.articles) == {"c4"}


def test_builder_ingests_phase2_mmap_metadata(tmp_path):
    """build_browse_tree.py --phase2-dir reads document_id + tag_list_json."""
    import json
    import runpy
    import sys

    import numpy as np
    import pyarrow as pa
    import pyarrow.parquet as pq

    ds_dir = tmp_path / "memory"
    ds_dir.mkdir()
    np.save(ds_dir / "tokens.npy", np.zeros(10, dtype=np.int32))
    np.save(ds_dir / "offsets.npy", np.array([0, 5, 10], dtype=np.int64))
    table = pa.Table.from_pylist(
        [
            {
                "id": "ep_0",
                "document_id": "ep_0",
                "dataset_name": "memory",
                "tag_list_json": json.dumps(["persona"]),
            },
            {
                "id": "ep_1",
                "document_id": "ep_1",
                "dataset_name": "memory",
                "tag_list_json": json.dumps(["temporal"]),
            },
        ],
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
            ("tag_list_json", pa.string()),
        ]),
    )
    pq.write_table(table, ds_dir / "metadata.parquet")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "memory",
        "--phase2-dir", str(ds_dir),
        "--output-dir", str(output_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    from bgkit.data.browse_tree import BrowseTree

    tree = BrowseTree.load(output_dir / "memory.parquet", dataset="memory")
    # Article IDs should match document_id from the metadata
    assert "ep_0" in tree
    assert "ep_1" in tree
    # Each tag becomes a leaf tag under root
    tops = tree.top_level_topic_list()
    assert "persona" in tops and "temporal" in tops
