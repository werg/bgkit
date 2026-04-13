"""Tests for ``scripts/build_provenance_pubmedqa.py``."""

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
    mod_name = "_test_build_provenance_pubmedqa"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_provenance_pubmedqa.py"
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
            "dataset_name": "pubmedqa",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_gold_article_id_equals_document_id(tmp_path):
    src = tmp_path / "pubmedqa"
    out = tmp_path / "pubmedqa.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "27733282", "document_id": "27733282"},
            {"id": "12345678", "document_id": "12345678"},
        ],
        questions=[[1, 2, 3], [4, 5]],
        answers=[[10, 11, 12], [20, 21]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 2
    rows = _read_jsonl(out)
    assert [r["gold_article_id"] for r in rows] == ["27733282", "12345678"]
    # gold_document_id is always populated with the canonical mmap key.
    assert [r["gold_document_id"] for r in rows] == ["27733282", "12345678"]
    for r in rows:
        assert r["scope_template"] == "topic_list"
        assert r["scope_description"] == ""


def test_pubmedqa_title_map_translation(tmp_path):
    """With --title-map, gold_article_id becomes the mesh hierarchy's
    article_id (human-readable title when available, numeric pubid
    otherwise) while gold_document_id keeps the numeric pubid used by
    the L0 cache."""
    src = tmp_path / "pubmedqa"
    out = tmp_path / "pubmedqa.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "27733282", "document_id": "27733282"},
            {"id": "12345678", "document_id": "12345678"},
            {"id": "99999999", "document_id": "99999999"},
        ],
        questions=[[1], [2], [3]],
        answers=[[10], [11], [12]],
    )
    title_map_path = tmp_path / "pubmedqa_hierarchy.jsonl"
    with title_map_path.open("w") as f:
        f.write(json.dumps({
            "article_id": "Metformin pharmacokinetics in diabetes",
            "document_id": "27733282",
            "tag_path": ["Diseases", "Endocrine"],
        }) + "\n")
        f.write(json.dumps({
            "article_id": "12345678",  # fallback: title == pubid
            "document_id": "12345678",
            "tag_path": ["Diseases", "Uncategorized"],
        }) + "\n")

    module = _load_module()
    title_map = module._load_title_map(title_map_path)
    assert title_map == {
        "27733282": "Metformin pharmacokinetics in diabetes",
        "12345678": "12345678",
    }
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
        title_map=title_map,
    )
    assert n == 3
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "Metformin pharmacokinetics in diabetes"
    assert rows[0]["gold_document_id"] == "27733282"
    # Identity-mapped row keeps the numeric id for both fields.
    assert rows[1]["gold_article_id"] == "12345678"
    assert rows[1]["gold_document_id"] == "12345678"
    # Unknown row (not in title map at all) falls back to numeric id.
    assert rows[2]["gold_article_id"] == "99999999"
    assert rows[2]["gold_document_id"] == "99999999"


def test_decoded_strings_present(tmp_path):
    src = tmp_path / "pubmedqa"
    out = tmp_path / "pubmedqa.jsonl"
    _write_mmap(
        src,
        rows=[{"id": "99", "document_id": "99"}],
        questions=[[7, 8]],
        answers=[[100, 101, 102]],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    rows = _read_jsonl(out)
    assert rows[0]["question"] == "t7 t8"
    assert rows[0]["gold_answer"] == "t100 t101 t102"


def test_empty_tokens_row_skipped(tmp_path):
    src = tmp_path / "pubmedqa"
    out = tmp_path / "pubmedqa.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "1", "document_id": "1"},
            {"id": "2", "document_id": "2"},
        ],
        questions=[[1], [2]],
        answers=[[10], []],   # row 1 has empty answer
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "1"
    assert rows[0]["gold_document_id"] == "1"
