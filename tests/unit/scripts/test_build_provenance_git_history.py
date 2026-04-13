"""Tests for ``scripts/build_provenance_git_history.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class _FakeTokenizer:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_module():
    mod_name = "_test_build_provenance_git_history"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_provenance_git_history.py"
    )
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _write_mmap(
    src: Path,
    *,
    rows: list[dict],
    answers: list[list[int]],  # answer tokens live in tokens.npy for git_history
    questions: list[list[int]],
    columns: list[tuple[str, pa.DataType]],
) -> None:
    src.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    assert len(questions) == n_rows and len(answers) == n_rows

    # The git_history converter writes answer tokens into tokens.npy
    # (no separate answer_tokens.npy file).
    flat_a: list[int] = []
    a_offsets = [0]
    for a in answers:
        flat_a.extend(a)
        a_offsets.append(len(flat_a))
    np.save(src / "tokens.npy", np.asarray(flat_a, dtype=np.int32))
    np.save(src / "offsets.npy", np.asarray(a_offsets, dtype=np.int64))

    flat_q: list[int] = []
    q_offsets = [0]
    for q in questions:
        flat_q.extend(q)
        q_offsets.append(len(flat_q))
    np.save(src / "question_tokens.npy", np.asarray(flat_q, dtype=np.int32))
    np.save(src / "question_offsets.npy", np.asarray(q_offsets, dtype=np.int64))

    table = pa.Table.from_pylist(rows, schema=pa.schema(columns))
    pq.write_table(table, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset_name": "git_history",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_gold_article_id_uses_commit_sha_fallback(tmp_path):
    """Current convert_qa_pairs_to_npy.py writes no document_id column;
    the script must fall back to commit_sha."""
    src = tmp_path / "git_history"
    out = tmp_path / "git_history.jsonl"
    _write_mmap(
        src,
        rows=[
            {"repo_path": "torvalds/linux", "commit_sha": "abc123"},
            {"repo_path": "torvalds/linux", "commit_sha": "def456"},
            {"repo_path": "python/cpython", "commit_sha": "xyz789"},
        ],
        questions=[[1], [2], [3]],
        answers=[[10], [20], [30]],
        columns=[
            ("repo_path", pa.string()),
            ("commit_sha", pa.string()),
        ],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 3
    rows = _read_jsonl(out)
    assert [r["gold_article_id"] for r in rows] == ["abc123", "def456", "xyz789"]
    assert [r["scope_description"] for r in rows] == [
        "torvalds/linux", "torvalds/linux", "python/cpython",
    ]
    for r in rows:
        assert r["scope_template"] == "pre_scoped"


def test_document_id_preferred_over_commit_sha(tmp_path):
    src = tmp_path / "git_history"
    out = tmp_path / "git_history.jsonl"
    _write_mmap(
        src,
        rows=[
            {
                "repo_path": "torvalds/linux",
                "commit_sha": "abc123",
                "document_id": "preferred-id",
            },
        ],
        questions=[[1]],
        answers=[[2]],
        columns=[
            ("repo_path", pa.string()),
            ("commit_sha", pa.string()),
            ("document_id", pa.string()),
        ],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "preferred-id"


def test_missing_repo_path_raises(tmp_path):
    import pytest

    src = tmp_path / "git_history"
    out = tmp_path / "git_history.jsonl"
    _write_mmap(
        src,
        rows=[{"commit_sha": "abc"}],
        questions=[[1]],
        answers=[[2]],
        columns=[("commit_sha", pa.string())],
    )

    module = _load_module()
    with pytest.raises(RuntimeError, match="repo_path"):
        module.build_provenance(
            mmap_dir=src,
            output=out,
            tokenizer=_FakeTokenizer(),
        )


def test_empty_repo_path_skipped(tmp_path):
    src = tmp_path / "git_history"
    out = tmp_path / "git_history.jsonl"
    _write_mmap(
        src,
        rows=[
            {"repo_path": "", "commit_sha": "abc"},
            {"repo_path": "valid/repo", "commit_sha": "def"},
        ],
        questions=[[1], [2]],
        answers=[[10], [20]],
        columns=[("repo_path", pa.string()), ("commit_sha", pa.string())],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "def"
