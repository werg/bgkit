"""Tests for ``scripts/build_provenance_kilt_wikipedia.py``."""

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
    mod_name = "_test_build_provenance_kilt_wikipedia"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "build_provenance_kilt_wikipedia.py"
    )
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _write_corpus_mmap(
    src: Path,
    *,
    rows: list[dict],
    docs: list[list[int]],
    schema_fields: list[tuple[str, pa.DataType]] | None = None,
) -> None:
    """Create a minimal KILT Wikipedia corpus-style Phase 2 mmap.

    Corpus-only datasets put article text in ``tokens.npy`` / ``offsets.npy``
    and leave ``question_tokens.npy`` / ``answer_tokens.npy`` as a
    placeholder single-token-per-row stream (matching the behavior of
    ``scripts/convert_hf_to_mmap.py`` for ``kilt_wikipedia``).
    """
    src.mkdir(parents=True, exist_ok=True)
    n_rows = len(rows)
    assert len(docs) == n_rows

    flat_docs: list[int] = []
    d_offsets = [0]
    for d in docs:
        flat_docs.extend(d)
        d_offsets.append(len(flat_docs))
    np.save(src / "tokens.npy", np.asarray(flat_docs, dtype=np.int32))
    np.save(src / "offsets.npy", np.asarray(d_offsets, dtype=np.int64))

    # Placeholder question/answer streams (converter writes [0] per row).
    placeholder_tokens = np.zeros(n_rows, dtype=np.int32)
    placeholder_offsets = np.arange(n_rows + 1, dtype=np.int64)
    np.save(src / "question_tokens.npy", placeholder_tokens)
    np.save(src / "question_offsets.npy", placeholder_offsets)
    np.save(src / "answer_tokens.npy", placeholder_tokens)
    np.save(src / "answer_offsets.npy", placeholder_offsets)

    if schema_fields is None:
        schema_fields = [("id", pa.string()), ("document_id", pa.string())]
    table = pa.Table.from_pylist(rows, schema=pa.schema(schema_fields))
    pq.write_table(table, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({
            "schema_version": 1,
            "dataset_name": "kilt_wikipedia",
            "row_count": n_rows,
        }),
    )


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_question_template_uses_document_id_when_no_title(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[
            {"id": "42", "document_id": "42"},
            {"id": "99", "document_id": "99"},
        ],
        docs=[[1, 2, 3], [4, 5]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 2
    rows = _read_jsonl(out)
    assert rows[0]["question"] == "Tell me about 42"
    assert rows[1]["question"] == "Tell me about 99"
    assert [r["gold_article_id"] for r in rows] == ["42", "99"]
    for r in rows:
        assert r["scope_template"] == "topic_list"
        assert r["scope_description"] == ""


def test_answer_is_prefix_of_document_tokens(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[{"id": "1", "document_id": "1"}],
        docs=[[10, 11, 12, 13, 14, 15, 16, 17]],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
        answer_prefix_tokens=4,
    )

    rows = _read_jsonl(out)
    # Answer truncated to first 4 tokens.
    assert rows[0]["gold_answer"] == "t10 t11 t12 t13"


def test_title_column_used_when_present(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[
            {
                "id": "42",
                "document_id": "42",
                "wikipedia_title": "Douglas Adams",
            },
        ],
        docs=[[1, 2, 3]],
        schema_fields=[
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("wikipedia_title", pa.string()),
        ],
    )

    module = _load_module()
    module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    rows = _read_jsonl(out)
    assert rows[0]["question"] == "Tell me about Douglas Adams"
    # The gold_article_id always stays the document_id, not the title.
    assert rows[0]["gold_article_id"] == "42"


def test_empty_document_tokens_row_skipped(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[
            {"id": "1", "document_id": "1"},
            {"id": "2", "document_id": "2"},
        ],
        docs=[[1, 2], []],  # row 1 has empty document
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


def test_limit_truncates_output(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[{"id": f"w{i}", "document_id": f"w{i}"} for i in range(4)],
        docs=[[i + 1] for i in range(4)],
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
    assert [r["gold_article_id"] for r in rows] == ["w0", "w1"]


def test_output_path_parent_created(tmp_path):
    src = tmp_path / "kilt"
    out = tmp_path / "nested" / "dir" / "kilt_wiki.jsonl"
    _write_corpus_mmap(
        src,
        rows=[{"id": "1", "document_id": "1"}],
        docs=[[10]],
    )

    module = _load_module()
    n = module.build_provenance(
        mmap_dir=src,
        output=out,
        tokenizer=_FakeTokenizer(),
    )

    assert n == 1
    assert out.exists()
