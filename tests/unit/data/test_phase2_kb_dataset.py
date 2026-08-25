"""Unit tests for the lazy KBTrajectoryDataset.

The dataset switched from eager ``table.to_pylist()`` (which materialized the
whole ~1.87M-row git_commit_repro parquet into a Python list of dicts, ~51GB
resident) to a lazy per-row Arrow slice. These tests pin that the lazy
``__getitem__`` is byte-identical to the old eager path.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bgkit.data.bgkit_tool_template import (
    TrajectoryTurn,
    trajectory_from_json,
    trajectory_to_json,
)
from bgkit.data.datasets.phase2_kb_dataset import KBTrajectoryDataset


def _write_parquet(path):
    traj0 = trajectory_to_json([
        TrajectoryTurn(kind="bgkit", args={"ids": ["repo/w000"], "query": ""}, response="r"),
        TrajectoryTurn(kind="bgkit", args={"ids": ["a"], "query": "q"}),
        TrajectoryTurn(kind="answer", response="the answer"),
    ])
    traj1 = trajectory_to_json([
        TrajectoryTurn(kind="answer", response="just an answer"),
    ])
    table = pa.table({
        "dataset_name": ["git_commit_repro", "git_commit_repro", "git_commit_repro"],
        "scope_template": ["pre_scoped", "topic_list", "pre_scoped"],
        # null scope_description exercises the ``or ""`` branch.
        "scope_description": ["git:foo/bar", None, "git:baz/qux"],
        # null + populated topic_list_json exercise the json-parse branch.
        "topic_list_json": [None, json.dumps(["Physics", "Biology"]), None],
        "question": ["q0", "q1", "q2"],
        "gold_answer": ["g0", "g1", "g2"],
        "trajectory_json": [traj0, traj1, traj0],
        "group_id": ["head-a", "head-b", "head-c"],
        "repo_id": ["foo/bar", "baz/qux", "third/repo"],
        "split": ["train", "eval", "train"],
        "target_sha": ["a", "b", "c"],
        "target_path": ["a.py", "b.py", "c.py"],
        "drill_mode": ["full", "full", "full"],
        "structural_depth": [4, 3, 2],
        "artifact_schema_version": [2, 2, 2],
        "id_scheme_version": [2, 2, 2],
        "source_sha256": ["source", "source", "source"],
    })
    pq.write_table(table, path)
    trajectory_sha = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(".manifest.json").write_text(json.dumps({
        "schema_version": 2,
        "id_scheme_version": 2,
        "id_salt": "test",
        "source_sha256": "source",
        "tree_sha256": "tree",
        "trajectory_sha256": trajectory_sha,
        "row_count": 3,
        "drill_mode_counts": {"full": 3},
        "build_config": {"max_touching": 0},
    }))
    return table


def test_lazy_getitem_matches_eager_to_pylist(tmp_path):
    path = tmp_path / "git_commit_repro.parquet"
    table = _write_parquet(path)

    ds = KBTrajectoryDataset(path)
    assert len(ds) == table.num_rows == 3

    # Reference = the OLD eager path: build from table.to_pylist()[idx].
    eager_rows = pq.read_table(path).to_pylist()

    def build_from_row(row):
        topic_list_raw = row.get("topic_list_json")
        topic_list = list(json.loads(topic_list_raw)) if topic_list_raw else []
        return {
            "dataset_name": str(row["dataset_name"]),
            "scope_template": str(row["scope_template"]),
            "scope_description": str(row.get("scope_description") or ""),
            "topic_list": topic_list,
            "question": str(row["question"]),
            "gold_answer": str(row["gold_answer"]),
            "trajectory_json": str(row["trajectory_json"]),
        }

    for idx in range(len(ds)):
        sample = ds[idx]
        ref = build_from_row(eager_rows[idx])
        assert sample.dataset_name == ref["dataset_name"]
        assert sample.scope_template == ref["scope_template"]
        assert sample.scope_description == ref["scope_description"]
        assert sample.topic_list == ref["topic_list"]
        assert sample.question == ref["question"]
        assert sample.gold_answer == ref["gold_answer"]
        # Trajectory reconstructs identically to deserializing the same json.
        assert sample.trajectory == trajectory_from_json(ref["trajectory_json"])


def test_lazy_slice_row_equals_eager_row(tmp_path):
    """The single-row Arrow slice yields the exact dict the old
    ``to_pylist()[idx]`` produced."""
    path = tmp_path / "git_commit_repro.parquet"
    _write_parquet(path)
    ds = KBTrajectoryDataset(path)
    eager_rows = pq.read_table(path).to_pylist()
    for idx in range(len(ds)):
        lazy_row = ds._table.slice(idx, 1).to_pylist()[0]
        assert lazy_row == eager_rows[idx]


def test_dataset_holds_no_eager_row_list(tmp_path):
    """Regression guard: the dataset must NOT materialize a per-row Python list
    (the ~51GB footprint). Only the compact Arrow table is held."""
    path = tmp_path / "git_commit_repro.parquet"
    _write_parquet(path)
    ds = KBTrajectoryDataset(path)
    assert not hasattr(ds, "_rows")
    assert hasattr(ds, "_table")


def _write_ipc(parquet_path, table, *, chunksize):
    """Write an uncompressed Arrow IPC sibling (multi-batch) for the dataset to
    prefer — mirrors scripts/convert_trajectory_to_feather.py."""
    import pyarrow as pa

    out = parquet_path.with_suffix(".arrow")
    manifest = json.loads(parquet_path.with_suffix(".manifest.json").read_text())
    metadata = dict(table.schema.metadata or {})
    metadata[b"bgkit_source_parquet_sha256"] = manifest[
        "trajectory_sha256"
    ].encode()
    table = table.replace_schema_metadata(metadata)
    with pa.OSFile(str(out), "wb") as sink:
        opts = pa.ipc.IpcWriteOptions(compression=None)
        with pa.ipc.new_file(sink, table.schema, options=opts) as writer:
            writer.write_table(table, max_chunksize=chunksize)
    return out


def test_ipc_mode_preferred_and_identical(tmp_path):
    """When the .arrow sibling exists, the dataset uses PAGED IPC mode and each
    row is byte-identical to the parquet (and to the eager to_pylist path).
    chunksize=2 forces multiple record batches -> exercises the
    idx -> (batch_i, row_in_batch) mapping."""
    path = tmp_path / "git_commit_repro.parquet"
    table = _write_parquet(path)
    _write_ipc(path, table, chunksize=2)  # 3 rows -> batches [2, 1]

    ds = KBTrajectoryDataset(path)
    assert ds._mode == "ipc"
    assert len(ds) == table.num_rows == 3
    # Multiple batches present (offset list has >2 entries: [0, 2, 3]).
    assert ds._batch_starts == [0, 2, 3]

    eager_rows = pq.read_table(path).to_pylist()
    for idx in range(len(ds)):
        assert ds._row_at(idx) == eager_rows[idx]
        sample = ds[idx]
        ref = eager_rows[idx]
        assert sample.dataset_name == str(ref["dataset_name"])
        assert sample.scope_description == str(ref.get("scope_description") or "")
        assert sample.trajectory == trajectory_from_json(str(ref["trajectory_json"]))


def test_parquet_fallback_when_no_ipc(tmp_path):
    """No .arrow sibling -> parquet-lazy fallback (still no eager _rows list)."""
    path = tmp_path / "git_commit_repro.parquet"
    _write_parquet(path)
    ds = KBTrajectoryDataset(path)
    assert ds._mode == "parquet"
    assert not hasattr(ds, "_rows")
    assert ds._table is not None


def test_convert_script_roundtrip(tmp_path):
    """The conversion script produces an .arrow the dataset reads identically."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_conv",
        Path(__file__).resolve().parents[3]
        / "scripts" / "convert_trajectory_to_feather.py",
    )
    conv = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conv)

    path = tmp_path / "git_commit_repro.parquet"
    _write_parquet(path)
    out = conv.convert_one(path, max_chunksize=2)
    assert out == path.with_suffix(".arrow") and out.exists()
    # Idempotent: second call skips (returns None).
    assert conv.convert_one(path, max_chunksize=2) is None

    ds = KBTrajectoryDataset(path)
    assert ds._mode == "ipc"
    eager_rows = pq.read_table(path).to_pylist()
    for idx in range(len(ds)):
        assert ds._row_at(idx) == eager_rows[idx]


def test_git_trajectory_manifest_fingerprint_is_enforced(tmp_path):
    path = tmp_path / "git_commit_repro.parquet"
    _write_parquet(path)
    manifest_path = path.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    manifest["trajectory_sha256"] = "0" * 64
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="does not match its manifest"):
        KBTrajectoryDataset(path)
