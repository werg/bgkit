"""Tests for ``scripts/build_provenance_searchqa.py``."""

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
    mod_name = "_test_build_provenance_searchqa"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_provenance_searchqa.py"
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
        schema=pa.schema([("id", pa.string()), ("document_id", pa.string())]),
    )
    pq.write_table(table, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset_name": "searchqa",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_gold_article_id_equals_document_id(tmp_path):
    src = tmp_path / "searchqa"
    out = tmp_path / "searchqa.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "jeopardy_1", "document_id": "jeopardy_1"},
            {"id": "jeopardy_2", "document_id": "jeopardy_2"},
        ],
        questions=[[1, 2], [3]],
        answers=[[10, 11], [12]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 2
    rows = _read_jsonl(out)
    assert [r["gold_article_id"] for r in rows] == ["jeopardy_1", "jeopardy_2"]
    for r in rows:
        assert r["scope_template"] == "pre_scoped"
        assert r["scope_description"] == "SearchQA web search snippets"


def test_decoded_strings_present(tmp_path):
    src = tmp_path / "searchqa"
    out = tmp_path / "searchqa.jsonl"
    _write_mmap(
        src,
        rows=[{"id": "a", "document_id": "a"}],
        questions=[[9, 8]],
        answers=[[7, 6]],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    rows = _read_jsonl(out)
    assert rows[0]["question"] == "t9 t8"
    assert rows[0]["gold_answer"] == "t7 t6"


def test_empty_tokens_row_skipped(tmp_path):
    src = tmp_path / "searchqa"
    out = tmp_path / "searchqa.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "a", "document_id": "a"},
            {"id": "b", "document_id": "b"},
        ],
        questions=[[1], [2]],
        answers=[[], [3]],  # row 0 has empty answer
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "b"


def test_limit_truncates_output(tmp_path):
    src = tmp_path / "searchqa"
    out = tmp_path / "searchqa.jsonl"
    _write_mmap(
        src,
        rows=[{"id": f"jp_{i}", "document_id": f"jp_{i}"} for i in range(5)],
        questions=[[i + 1] for i in range(5)],
        answers=[[20 + i] for i in range(5)],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
        limit=2,
    )

    assert n == 2
    rows = _read_jsonl(out)
    assert [r["gold_article_id"] for r in rows] == ["jp_0", "jp_1"]


def test_output_path_parent_created(tmp_path):
    src = tmp_path / "searchqa"
    out = tmp_path / "nested" / "dir" / "searchqa.jsonl"
    _write_mmap(
        src,
        rows=[{"id": "jp_0", "document_id": "jp_0"}],
        questions=[[1]],
        answers=[[2]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    assert out.exists()
