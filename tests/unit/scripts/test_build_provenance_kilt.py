"""Tests for ``scripts/build_provenance_kilt.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class _FakeTokenizer:
    """Renders token IDs as ``t{id}`` tokens — same approach as
    ``tests/unit/scripts/test_reshard_narrativeqa.py``."""

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_module():
    mod_name = "_test_build_provenance_kilt"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3] / "scripts" / "build_provenance_kilt.py"
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
    """Create a minimal KILT-shaped Phase 2 mmap.

    The base tokens.npy / offsets.npy are not read by the provenance
    script, but Phase2QADataset invariants still expect them to exist
    and agree on row count, so we write tiny placeholders.
    """
    src.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    assert len(questions) == n_rows and len(answers) == n_rows

    # Empty document stream, but with a valid offsets array.
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
            ("provenance_json", pa.string()),
        ]),
    )
    pq.write_table(table, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset_name": "kilt_nq",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_provenance_uses_first_wikipedia_id(tmp_path):
    src = tmp_path / "kilt_nq"
    out = tmp_path / "kilt.jsonl"
    _write_mmap(
        src,
        rows=[
            {
                "id": "q0",
                "document_id": "q0",
                "provenance_json": json.dumps(["12345", "67890"]),
            },
            {
                "id": "q1",
                "document_id": "q1",
                "provenance_json": json.dumps(["99"]),
            },
        ],
        questions=[[1, 2], [3, 4]],
        answers=[[10, 11], [12, 13]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 2
    rows = _read_jsonl(out)
    # Without a title map, gold_article_id falls back to the numeric id
    # but gold_document_id is always populated as the canonical mmap key.
    assert [r["gold_article_id"] for r in rows] == ["12345", "99"]
    assert [r["gold_document_id"] for r in rows] == ["12345", "99"]
    for r in rows:
        assert r["scope_template"] == "topic_list"
        assert r["scope_description"] == ""
        assert isinstance(r["question"], str) and r["question"]
        assert isinstance(r["gold_answer"], str) and r["gold_answer"]


def test_provenance_uses_title_map(tmp_path):
    """With --title-map, gold_article_id becomes the human-readable
    wikipedia_title while gold_document_id retains the numeric
    wikipedia_id used by the L0 cache / mmap."""
    src = tmp_path / "kilt_nq"
    out = tmp_path / "kilt.jsonl"
    _write_mmap(
        src,
        rows=[
            {
                "id": "q0",
                "document_id": "q0",
                "provenance_json": json.dumps(["12345"]),
            },
            {
                "id": "q1",
                "document_id": "q1",
                "provenance_json": json.dumps(["99"]),
            },
            {
                "id": "q2",
                "document_id": "q2",
                "provenance_json": json.dumps(["unknown_id"]),
            },
        ],
        questions=[[1], [2], [3]],
        answers=[[10], [11], [12]],
    )
    # Simulate the KILT hierarchy JSONL: article_id = title,
    # document_id = wikipedia_id.
    title_map_path = tmp_path / "kilt_hierarchy.jsonl"
    with title_map_path.open("w") as f:
        f.write(json.dumps({
            "article_id": "Black hole",
            "document_id": "12345",
            "tag_path": ["Physics", "Astronomy"],
        }) + "\n")
        f.write(json.dumps({
            "article_id": "Schrödinger equation",
            "document_id": "99",
            "tag_path": ["Physics", "Quantum mechanics"],
        }) + "\n")

    module = _load_module()
    title_map = module._load_title_map(title_map_path)
    assert title_map == {"12345": "Black hole", "99": "Schrödinger equation"}

    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
        title_map=title_map,
    )
    assert n == 3
    rows = _read_jsonl(out)
    # Known ids translate through the title map.
    assert rows[0]["gold_article_id"] == "Black hole"
    assert rows[0]["gold_document_id"] == "12345"
    assert rows[1]["gold_article_id"] == "Schrödinger equation"
    assert rows[1]["gold_document_id"] == "99"
    # Unknown ids fall back to the numeric id for both fields.
    assert rows[2]["gold_article_id"] == "unknown_id"
    assert rows[2]["gold_document_id"] == "unknown_id"


def test_empty_provenance_falls_back_to_document_id(tmp_path):
    src = tmp_path / "kilt_nq"
    out = tmp_path / "kilt.jsonl"
    _write_mmap(
        src,
        rows=[
            {
                "id": "q0",
                "document_id": "bare-doc-id",
                "provenance_json": json.dumps([]),
            },
        ],
        questions=[[1, 2]],
        answers=[[3, 4]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "bare-doc-id"
    assert rows[0]["gold_document_id"] == "bare-doc-id"


def test_skips_rows_with_empty_tokens(tmp_path):
    src = tmp_path / "kilt_nq"
    out = tmp_path / "kilt.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "q0", "document_id": "q0", "provenance_json": json.dumps(["1"])},
            {"id": "q1", "document_id": "q1", "provenance_json": json.dumps(["2"])},
        ],
        questions=[[1, 2], []],   # row 1 has empty question
        answers=[[3, 4], [5, 6]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1  # row 1 dropped
    rows = _read_jsonl(out)
    assert rows[0]["gold_article_id"] == "1"


def test_output_path_parent_created(tmp_path):
    src = tmp_path / "kilt_nq"
    out = tmp_path / "nested" / "dir" / "kilt.jsonl"
    _write_mmap(
        src,
        rows=[
            {"id": "q0", "document_id": "q0", "provenance_json": json.dumps(["11"])},
        ],
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
