"""Tests for ``scripts/build_provenance_memory.py``."""

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
    mod_name = "_test_build_provenance_memory"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_provenance_memory.py"
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
    questions: list[list[int]],
    answers: list[list[int]],
) -> None:
    src.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    assert len(questions) == n_rows and len(answers) == n_rows

    np.save(src / "tokens.npy", np.zeros(0, dtype=np.int32))
    np.save(src / "offsets.npy", np.zeros(n_rows + 1, dtype=np.int64))

    flat_q: list[int] = []
    q_offsets = [0]
    for q in questions:
        flat_q.extend(q)
        q_offsets.append(len(flat_q))
    np.save(src / "question_tokens.npy", np.asarray(flat_q, dtype=np.int32))
    np.save(src / "question_offsets.npy", np.asarray(q_offsets, dtype=np.int64))

    flat_a: list[int] = []
    a_offsets = [0]
    for a in answers:
        flat_a.extend(a)
        a_offsets.append(len(flat_a))
    np.save(src / "answer_tokens.npy", np.asarray(flat_a, dtype=np.int32))
    np.save(src / "answer_offsets.npy", np.asarray(a_offsets, dtype=np.int64))

    table = pa.Table.from_pylist(
        rows,
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("memory_type", pa.string()),
        ]),
    )
    pq.write_table(table, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset_name": "msc",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_scope_is_episode_id(tmp_path):
    src = tmp_path / "msc"
    out = tmp_path / "memory.jsonl"
    _write_mmap(
        src,
        rows=[
            {
                "id": "conv-17",
                "document_id": "conv-17",
                "memory_type": "persona",
            },
            {
                "id": "conv-42",
                "document_id": "conv-42",
                "memory_type": "preference",
            },
        ],
        questions=[[1, 2], [3, 4]],
        answers=[[10, 11], [20, 21]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 2
    rows = _read_jsonl(out)
    assert [r["scope_description"] for r in rows] == ["conv-17", "conv-42"]
    for r in rows:
        assert r["scope_template"] == "pre_scoped"


def test_rows_from_one_episode_share_the_episode_article(tmp_path):
    """Episode turns intentionally resolve to one browse-tree article."""
    src = tmp_path / "msc"
    out = tmp_path / "memory.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "e0", "document_id": "e0", "memory_type": "persona"},
            {"id": "e0", "document_id": "e0", "memory_type": "persona"},
            {"id": "e0", "document_id": "e0", "memory_type": "persona"},
            {"id": "e1", "document_id": "e1", "memory_type": "persona"},
        ],
        questions=[[1], [2], [3], [4]],
        answers=[[10], [20], [30], [40]],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    rows = _read_jsonl(out)
    gold_ids = [r["gold_article_id"] for r in rows]
    assert gold_ids == ["e0", "e0", "e0", "e1"]


def test_skips_rows_with_empty_tokens(tmp_path):
    src = tmp_path / "msc"
    out = tmp_path / "memory.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "e0", "document_id": "e0", "memory_type": "persona"},
            {"id": "e1", "document_id": "e1", "memory_type": "persona"},
        ],
        questions=[[1], [2]],
        answers=[[], [3]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    rows = _read_jsonl(out)
    assert rows[0]["scope_description"] == "e1"
