"""Tests for title-keyed browse trees and the title → document_id sidecar.

The Phase 2 KB pipeline uses human-readable article titles as browse-tree
node ids for KILT Wikipedia and PubMedQA (when titles are available).
This file covers:

1. ``BrowseTreeBuilder`` accepts multi-character title strings as article
   ids and every title round-trips through parquet (no truncation, no
   hash mangling).
2. ``scripts/build_browse_tree.py --input`` honors the optional
   ``document_id`` column and emits a sidecar JSON whose entries match
   the input rows exactly.
3. The sidecar is suppressed when every row's ``document_id`` equals
   its ``article_id`` (identity map) — that lets legacy numeric-id
   datasets skip the sidecar entirely.
"""

from __future__ import annotations

import json
import runpy
import sys

import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.browse_tree import BrowseTree
from bgkit.data.tagging import BrowseTreeBuilder, TaggingConfig


def test_browse_tree_accepts_title_article_ids(tmp_path):
    """Titles with spaces, punctuation, and Unicode round-trip through
    the builder, parquet write/load, and every query method used by the
    trainer."""
    builder = BrowseTreeBuilder(
        TaggingConfig(dataset="kilt_wikipedia", leaf_cap=10, fanout_cap=10),
    )
    titles = [
        "Black hole",
        "Schrödinger equation",
        "Napoleon I, Emperor of the French",
        "2019 coronavirus pandemic",
        "C++ (programming language)",
    ]
    for title in titles:
        builder.add_article(title, ["Physics", "Concepts"])
    tree = builder.build()

    # The titles must be preserved exactly and resolvable as articles.
    for title in titles:
        assert title in tree
        node = tree.get(title)
        assert node.is_article

    # Round-trip through parquet.
    path = tmp_path / "kilt_wikipedia.parquet"
    tree.save(path)
    loaded = BrowseTree.load(path, dataset="kilt_wikipedia")
    for title in titles:
        assert title in loaded
        assert loaded.get(title).is_article

    # leaf_tag_for_article must resolve titles too — it's the fallback
    # lookup used by _resolve_article_ids when a trajectory references
    # an article directly.
    for title in titles:
        parent = loaded.leaf_tag_for_article(title)
        assert parent is not None
        assert parent.endswith("Concepts")


def test_build_browse_tree_emits_title_sidecar(tmp_path):
    """build_browse_tree.py --input reads the optional document_id column
    and writes a per-dataset title → document_id sidecar JSON."""
    input_path = tmp_path / "kilt_hierarchy.jsonl"
    rows = [
        {
            "article_id": "Black hole",
            "document_id": "4650",
            "tag_path": ["Physics", "Astronomy"],
        },
        {
            "article_id": "Schrödinger equation",
            "document_id": "27223",
            "tag_path": ["Physics", "Quantum mechanics"],
        },
        {
            "article_id": "Napoleon",
            "document_id": "69880",
            "tag_path": ["History", "Figures"],
        },
    ]
    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "kilt_wikipedia",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    parquet_path = output_dir / "kilt_wikipedia.parquet"
    sidecar_path = output_dir / "kilt_wikipedia_title_to_doc_id.json"
    assert parquet_path.exists()
    assert sidecar_path.exists()

    tree = BrowseTree.load(parquet_path, dataset="kilt_wikipedia")
    for row in rows:
        assert row["article_id"] in tree

    with sidecar_path.open() as f:
        sidecar = json.load(f)
    assert sidecar == {
        "Black hole": "4650",
        "Schrödinger equation": "27223",
        "Napoleon": "69880",
    }


def test_build_browse_tree_skips_sidecar_when_identity(tmp_path):
    """When every row's document_id == article_id (or document_id is
    absent), no sidecar file is emitted — the absence signals
    "browse-tree ids are already mmap keys" to downstream consumers."""
    input_path = tmp_path / "numeric_hierarchy.jsonl"
    rows = [
        {"article_id": "doc_001", "tag_path": ["Topic", "Leaf"]},
        {
            "article_id": "doc_002",
            "document_id": "doc_002",
            "tag_path": ["Topic", "Leaf"],
        },
    ]
    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "legacy",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    parquet_path = output_dir / "legacy.parquet"
    sidecar_path = output_dir / "legacy_title_to_doc_id.json"
    assert parquet_path.exists()
    assert not sidecar_path.exists(), (
        "identity-only title → document_id map should not materialize a sidecar"
    )


def test_build_browse_tree_sidecar_has_all_rows(tmp_path):
    """Every article in the input JSONL is present in the emitted sidecar
    (no silent drops even at leaf-cap boundaries).
    """
    input_path = tmp_path / "hierarchy.jsonl"
    rows = []
    for i in range(25):
        rows.append({
            "article_id": f"Article {i}",
            "document_id": f"wiki_{i:04d}",
            "tag_path": ["Topic", "Leaf"],
        })
    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    output_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "toy",
        "--input", str(input_path),
        "--output-dir", str(output_dir),
        # Force sub-division so we exercise the fanout cap path.
        "--leaf-cap", "10",
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    sidecar_path = output_dir / "toy_title_to_doc_id.json"
    assert sidecar_path.exists()
    with sidecar_path.open() as f:
        sidecar = json.load(f)
    assert len(sidecar) == 25
    for row in rows:
        assert sidecar[row["article_id"]] == row["document_id"]

    tree = BrowseTree.load(
        output_dir / "toy.parquet", dataset="toy",
    )
    # Every title must still be in the tree — sub-division is purely a
    # reshaping operation and must not drop articles.
    for row in rows:
        assert row["article_id"] in tree


def test_build_trajectory_set_translates_titles_to_doc_ids(tmp_path):
    """build_trajectory_set.py must emit mmap document_ids, not the
    browse-tree titles, when a sidecar is present. This is the contract
    that keeps scripts/precompute_l0_subset.py feeding ArticleTokenStore
    the canonical document_id it expects."""
    # 1. Build a title-keyed browse tree + sidecar via the CLI.
    input_path = tmp_path / "hierarchy.jsonl"
    rows = [
        {
            "article_id": "Black hole",
            "document_id": "wiki_4650",
            "tag_path": ["Physics", "Astronomy"],
        },
        {
            "article_id": "Schrödinger equation",
            "document_id": "wiki_27223",
            "tag_path": ["Physics", "Quantum mechanics"],
        },
    ]
    with input_path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    browse_tree_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "kilt_wikipedia",
        "--input", str(input_path),
        "--output-dir", str(browse_tree_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")

    # 2. Build a trajectory parquet that references titles via bgkit turns.
    trajectory_rows = [
        {
            "dataset_name": "kilt_wikipedia",
            "scope_template": "topic_list",
            "scope_description": "",
            "topic_list_json": json.dumps(["Physics"]),
            "question": "What is a black hole?",
            "gold_answer": "A region where gravity is so strong nothing escapes.",
            "trajectory_json": json.dumps([
                {
                    "kind": "browse",
                    "args": {"id": "Physics"},
                    "response": "",
                    "loss": True,
                },
                {
                    "kind": "bgkit",
                    "args": {
                        "ids": ["Black hole", "Schrödinger equation"],
                        "query": "What is a black hole?",
                    },
                    "response": "",
                    "loss": True,
                },
                {
                    "kind": "answer",
                    "args": {},
                    "response": "A region ...",
                    "loss": True,
                },
            ]),
        },
    ]
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    trajectory_path = trajectory_dir / "kilt_wikipedia.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            trajectory_rows,
            schema=pa.schema([
                ("dataset_name", pa.string()),
                ("scope_template", pa.string()),
                ("scope_description", pa.string()),
                ("topic_list_json", pa.string()),
                ("question", pa.string()),
                ("gold_answer", pa.string()),
                ("trajectory_json", pa.string()),
            ]),
        ),
        trajectory_path,
    )

    # 3. Run build_trajectory_set.py and verify the emitted rows carry
    #    document_ids, NOT titles.
    output = tmp_path / "trajectory_sets" / "kilt_wikipedia.jsonl"
    sys.argv = [
        "build_trajectory_set.py",
        "--trajectory", str(trajectory_path),
        "--browse-tree", str(browse_tree_dir / "kilt_wikipedia.parquet"),
        "--dataset", "kilt_wikipedia",
        "--output", str(output),
    ]
    runpy.run_path("scripts/build_trajectory_set.py", run_name="__main__")

    assert output.exists()
    emitted = []
    with output.open() as f:
        for line in f:
            line = line.strip()
            if line:
                emitted.append(json.loads(line))
    article_ids = sorted(r["article_id"] for r in emitted)
    assert article_ids == ["wiki_27223", "wiki_4650"]
    for row in emitted:
        assert row["dataset"] == "kilt_wikipedia"


def test_build_trajectory_set_without_sidecar_passthrough(tmp_path):
    """Legacy path: dataset without a sidecar emits browse-tree ids
    unchanged (identity translation)."""
    input_path = tmp_path / "hierarchy.jsonl"
    # Identity ids — the script should still produce a tree and NOT
    # emit a sidecar; build_trajectory_set then passes ids through.
    with input_path.open("w") as f:
        f.write(json.dumps({
            "article_id": "doc_a",
            "tag_path": ["Topic", "Leaf"],
        }) + "\n")
        f.write(json.dumps({
            "article_id": "doc_b",
            "tag_path": ["Topic", "Leaf"],
        }) + "\n")

    browse_tree_dir = tmp_path / "browse_trees"
    sys.argv = [
        "build_browse_tree.py",
        "--dataset", "legacy",
        "--input", str(input_path),
        "--output-dir", str(browse_tree_dir),
    ]
    runpy.run_path("scripts/build_browse_tree.py", run_name="__main__")
    assert not (browse_tree_dir / "legacy_title_to_doc_id.json").exists()

    trajectory_rows = [
        {
            "dataset_name": "legacy",
            "scope_template": "topic_list",
            "scope_description": "",
            "topic_list_json": json.dumps(["Topic"]),
            "question": "q",
            "gold_answer": "a",
            "trajectory_json": json.dumps([
                {
                    "kind": "bgkit",
                    "args": {"ids": ["doc_a", "doc_b"], "query": "q"},
                    "response": "",
                    "loss": True,
                },
                {
                    "kind": "answer",
                    "args": {},
                    "response": "a",
                    "loss": True,
                },
            ]),
        },
    ]
    trajectory_dir = tmp_path / "trajectories"
    trajectory_dir.mkdir()
    trajectory_path = trajectory_dir / "legacy.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            trajectory_rows,
            schema=pa.schema([
                ("dataset_name", pa.string()),
                ("scope_template", pa.string()),
                ("scope_description", pa.string()),
                ("topic_list_json", pa.string()),
                ("question", pa.string()),
                ("gold_answer", pa.string()),
                ("trajectory_json", pa.string()),
            ]),
        ),
        trajectory_path,
    )

    output = tmp_path / "trajectory_sets" / "legacy.jsonl"
    sys.argv = [
        "build_trajectory_set.py",
        "--trajectory", str(trajectory_path),
        "--browse-tree", str(browse_tree_dir / "legacy.parquet"),
        "--dataset", "legacy",
        "--output", str(output),
    ]
    runpy.run_path("scripts/build_trajectory_set.py", run_name="__main__")

    assert output.exists()
    emitted = []
    with output.open() as f:
        for line in f:
            line = line.strip()
            if line:
                emitted.append(json.loads(line))
    ids = sorted(r["article_id"] for r in emitted)
    assert ids == ["doc_a", "doc_b"]
