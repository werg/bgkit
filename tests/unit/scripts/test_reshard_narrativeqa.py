"""Tests for ``scripts/reshard_narrativeqa.py``."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


class _FakeTokenizer:
    """Trivial tokenizer stub — renders token IDs as ``t{id}`` tokens.

    The real production tokenizer (``Qwen/Qwen3.5-0.8B``) is gated on HF
    and can't be loaded in CI without network. Tests inject this fake via
    the script's ``reshard()`` entry point, which accepts a tokenizer
    object directly (the argparse ``--tokenizer`` path is only exercised
    by the top-level ``main()``).
    """

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_reshard_module():
    """Import scripts/reshard_narrativeqa.py as a module."""
    mod_name = "_test_reshard_narrativeqa"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    path = Path("scripts/reshard_narrativeqa.py").resolve()
    spec = importlib.util.spec_from_file_location(mod_name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = module
    spec.loader.exec_module(module)
    return module


def _write_source_mmap(
    src: Path,
    *,
    answers: list[list[int]],
) -> None:
    """Build a minimal book-level mmap with two books and three QA rows.

    ``answers`` lets each test override the per-row answer token IDs to
    exercise the short-answer / low-overlap / happy-path code paths.
    """
    assert len(answers) == 3, "this helper assumes 3 QA rows"
    src.mkdir(parents=True, exist_ok=True)
    book_a = list(range(50, 50 + 20))   # 20 tokens: 50..69
    book_b = list(range(100, 100 + 15))  # 15 tokens: 100..114

    all_tokens: list[int] = []
    offsets = [0]
    # Row 0: book_a, qa1
    all_tokens.extend(book_a)
    offsets.append(len(all_tokens))
    # Row 1: book_a, qa2 (same book)
    all_tokens.extend(book_a)
    offsets.append(len(all_tokens))
    # Row 2: book_b, qa1
    all_tokens.extend(book_b)
    offsets.append(len(all_tokens))
    np.save(src / "tokens.npy", np.asarray(all_tokens, dtype=np.int32))
    np.save(src / "offsets.npy", np.asarray(offsets, dtype=np.int64))

    # Questions: one per row, each 2 tokens long so decoding produces a
    # non-empty string regardless of row.
    q_tokens: list[int] = [1, 2, 3, 4, 5, 6]
    q_offsets = [0, 2, 4, 6]
    np.save(src / "question_tokens.npy", np.asarray(q_tokens, dtype=np.int32))
    np.save(src / "question_offsets.npy", np.asarray(q_offsets, dtype=np.int64))

    # Answers are test-specific.
    flat_a: list[int] = []
    a_offsets = [0]
    for a in answers:
        flat_a.extend(a)
        a_offsets.append(len(flat_a))
    np.save(src / "answer_tokens.npy", np.asarray(flat_a, dtype=np.int32))
    np.save(src / "answer_offsets.npy", np.asarray(a_offsets, dtype=np.int64))

    metadata = pa.Table.from_pylist(
        [
            {"id": "ra0", "document_id": "book_A", "dataset_name": "narrativeqa"},
            {"id": "ra1", "document_id": "book_A", "dataset_name": "narrativeqa"},
            {"id": "rb0", "document_id": "book_B", "dataset_name": "narrativeqa"},
        ],
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
        ]),
    )
    pq.write_table(metadata, src / "metadata.parquet")
    (src / "manifest.json").write_text(
        json.dumps({"schema_version": 1, "dataset_name": "narrativeqa", "row_count": 3}),
    )


def _run_reshard(
    src: Path,
    dst: Path,
    *,
    passage_tokens: int = 10,
    short_answer_threshold: int = 5,
    min_overlap: int = 3,
) -> dict:
    module = _load_reshard_module()
    return module.reshard(
        src=src,
        dst=dst,
        passage_tokens=passage_tokens,
        short_answer_threshold=short_answer_threshold,
        min_overlap=min_overlap,
        tokenizer=_FakeTokenizer(),
    )


def _read_provenance(dst: Path) -> list[dict]:
    path = dst / "narrativeqa_provenance.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_reshard_produces_passages_and_provenance(tmp_path):
    """Structural assertions about the output mmap."""
    src = tmp_path / "narrativeqa"
    dst = tmp_path / "narrativeqa_passages"
    # Use longer answers (>= short_answer_threshold=5) with clear overlap
    # so all three rows take the happy path.
    _write_source_mmap(
        src,
        answers=[
            [60, 61, 62, 63, 64, 65],  # overlaps book_A passage 1 (tokens 60..69)
            [50, 51, 52, 53, 54, 55],  # overlaps book_A passage 0 (tokens 50..59)
            [100, 101, 102, 103, 104],  # overlaps book_B passage 0 (tokens 100..109)
        ],
    )

    _run_reshard(src, dst, passage_tokens=10)

    for required in ("tokens.npy", "offsets.npy", "metadata.parquet", "manifest.json"):
        assert (dst / required).exists(), f"missing {required}"

    meta = pq.read_table(dst / "metadata.parquet").to_pylist()
    # book_A: 20 tokens / 10 per passage = 2 passages.
    # book_B: 15 tokens / 10 per passage = 2 passages (last is 5 tokens).
    # Total: 4 passage rows.
    assert len(meta) == 4
    doc_ids = {row["document_id"] for row in meta}
    assert doc_ids == {
        "book_A#p0000", "book_A#p0001", "book_B#p0000", "book_B#p0001",
    }

    # Every passage row's tag_list_json has the parent book as its only tag.
    for row in meta:
        tags = json.loads(row["tag_list_json"])
        assert tags == [row["parent_book"]]

    # Tokens concatenated in order; offsets.npy has length = row_count+1.
    offsets = np.load(dst / "offsets.npy")
    assert offsets.shape == (5,)
    assert offsets[-1] == 35  # 20 + 15

    # Manifest carries the new telemetry fields.
    manifest = json.loads((dst / "manifest.json").read_text())
    assert manifest["schema_version"] == 2
    assert manifest["passage_tokens_target"] == 10
    assert manifest["parent_book_count"] == 2
    assert manifest["row_count"] == 4
    assert manifest["provenance_row_count"] == 3


def test_provenance_uses_decoded_strings(tmp_path):
    """Question / gold_answer should be strings produced by the tokenizer,
    not raw token-ID lists."""
    src = tmp_path / "narrativeqa"
    dst = tmp_path / "narrativeqa_passages"
    _write_source_mmap(
        src,
        answers=[
            [60, 61, 62, 63, 64, 65],
            [50, 51, 52, 53, 54, 55],
            [100, 101, 102, 103, 104],
        ],
    )

    _run_reshard(src, dst, passage_tokens=10)

    rows = _read_provenance(dst)
    assert len(rows) == 3
    for row in rows:
        assert isinstance(row["question"], str)
        assert isinstance(row["gold_answer"], str)
        # The fake tokenizer renders IDs as space-joined "t{id}" tokens.
        assert row["question"].startswith("t")
        assert row["gold_answer"].startswith("t")
        # Raw token ID lists are still present for debugging / alt consumers.
        assert isinstance(row["question_token_ids"], list)
        assert isinstance(row["answer_token_ids"], list)
        assert all(isinstance(i, int) for i in row["answer_token_ids"])

    # Concrete check: row 0's answer is [60, 61, 62, 63, 64, 65] → "t60 t61 t62 t63 t64 t65".
    assert rows[0]["gold_answer"] == "t60 t61 t62 t63 t64 t65"
    # Row 0's question is [1, 2] → "t1 t2".
    assert rows[0]["question"] == "t1 t2"


def test_short_answer_fallback(tmp_path):
    """Short answers (< short_answer_threshold tokens) fall back to the
    book-level gold_article_id."""
    src = tmp_path / "narrativeqa"
    dst = tmp_path / "narrativeqa_passages"
    _write_source_mmap(
        src,
        answers=[
            [42, 43],  # 2 tokens — shorter than default threshold (5)
            [50, 51, 52, 53, 54, 55],  # long, happy path
            [100, 101, 102, 103, 104],  # long, happy path
        ],
    )

    _run_reshard(src, dst, passage_tokens=10)

    rows = _read_provenance(dst)
    assert len(rows) == 3

    # Row 0: short answer → fallback to parent book ID, NOT a passage ID.
    first = rows[0]
    assert first["gold_article_id"] == "book_A"
    assert "#p" not in first["gold_article_id"]
    assert first["gold_resolution"] == "short_answer_fallback"
    assert first["parent_book"] == "book_A"

    # Rows 1 and 2 take the happy path.
    assert rows[1]["gold_resolution"] == "passage"
    assert rows[2]["gold_resolution"] == "passage"

    # Manifest telemetry reflects both outcomes.
    manifest = json.loads((dst / "manifest.json").read_text())
    counts = manifest["gold_resolution_counts"]
    assert counts.get("short_answer_fallback") == 1
    assert counts.get("passage") == 2


def test_low_overlap_fallback(tmp_path):
    """Long answers with no token overlap against any passage fall back to
    the book-level gold_article_id."""
    src = tmp_path / "narrativeqa"
    dst = tmp_path / "narrativeqa_passages"
    # Row 0's answer has 6 tokens (> threshold) but none appear in book_A
    # (tokens 50..69) — the answer tokens are far outside that range.
    _write_source_mmap(
        src,
        answers=[
            [9000, 9001, 9002, 9003, 9004, 9005],  # long, zero overlap
            [50, 51, 52, 53, 54, 55],  # long, happy path
            [100, 101, 102, 103, 104],  # long, happy path
        ],
    )

    _run_reshard(src, dst, passage_tokens=10)

    rows = _read_provenance(dst)
    assert len(rows) == 3

    first = rows[0]
    assert first["gold_article_id"] == "book_A"
    assert first["gold_resolution"] == "low_overlap_fallback"
    assert first["parent_book"] == "book_A"

    assert rows[1]["gold_resolution"] == "passage"
    assert rows[2]["gold_resolution"] == "passage"

    manifest = json.loads((dst / "manifest.json").read_text())
    counts = manifest["gold_resolution_counts"]
    assert counts.get("low_overlap_fallback") == 1
    assert counts.get("passage") == 2


def test_happy_path_resolves_to_correct_passage(tmp_path):
    """A long answer with clear overlap resolves to the right passage."""
    src = tmp_path / "narrativeqa"
    dst = tmp_path / "narrativeqa_passages"
    # Row 0 answer lives entirely in book_A's second passage (tokens 60..69).
    _write_source_mmap(
        src,
        answers=[
            [65, 66, 67, 68, 69],  # 5 tokens, all in passage 1 of book_A
            [50, 51, 52, 53, 54, 55],  # 6 tokens, all in passage 0 of book_A
            [110, 111, 112, 113, 114],  # 5 tokens, all in passage 1 of book_B
        ],
    )

    _run_reshard(src, dst, passage_tokens=10)

    rows = _read_provenance(dst)
    assert rows[0]["gold_article_id"] == "book_A#p0001"
    assert rows[0]["gold_resolution"] == "passage"
    assert rows[0]["parent_book"] == "book_A"
    assert rows[0]["scope_template"] == "pre_scoped"
    assert rows[0]["scope_description"] == "book_A"

    assert rows[1]["gold_article_id"] == "book_A#p0000"
    assert rows[1]["gold_resolution"] == "passage"

    assert rows[2]["gold_article_id"] == "book_B#p0001"
    assert rows[2]["gold_resolution"] == "passage"
