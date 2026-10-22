#!/usr/bin/env python
"""Re-shard NarrativeQA from book-level to passage-level for KB-scale training.

The existing NarrativeQA Phase 2 mmap (at ``$DATA_DIR/mmap/phase2/narrativeqa/``)
stores one row per (book, question) tuple with a ~65k-token ``document``
covering the full book. That's the right granularity for single-doc long-
context QA, but it's useless for KB-scale browse+bgkit training where the
browse tree needs many retrieval-addressable articles per book.

This script produces a parallel mmap at
``$DATA_DIR/mmap/phase2/narrativeqa_passages/`` with:

- **One row per passage** (default ~1000 tokens each, producing ≤~65 passages
  for a typical 65k-token book — safely under the 100 leaf_cap of the
  browse-tree builder). Each row has
  ``document_id = "{book_id}#p{passage_idx:04d}"`` and
  ``tag_list_json = ["{book_id}"]`` so the per-phase2-mmap browse tree
  ingester builds a ``root → book → passage`` two-level tree automatically.
- A sidecar ``narrativeqa_provenance.jsonl`` with one row per original
  QA pair. Each row carries ``(question, gold_answer, gold_article_id,
  scope_template, scope_description)`` where ``question`` and
  ``gold_answer`` are strings decoded from the source token IDs and
  ``gold_article_id`` is either the passage whose token content overlaps
  most with the gold answer (the normal path) or the parent book
  (the fallback path — used when the answer is too short to localize
  by bag-of-tokens overlap, or when no passage shows enough overlap).

Schema compatibility: ``scripts/build_teacher_trajectories.py`` reads
``question``, ``gold_answer``, ``gold_article_id``, ``scope_template``,
``scope_description``. These field names are canonical.

The re-shard uses the same token stream as the original mmap, so no
re-tokenization is needed. A tokenizer is only loaded for decoding the
(short) question/answer token ID streams back to strings for the
provenance sidecar.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class _DecoderLike(Protocol):
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


def _read_existing_mmap(src: Path) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, list[dict],
]:
    tokens = np.load(src / "tokens.npy", mmap_mode="r")
    offsets = np.load(src / "offsets.npy")
    q_tokens = np.load(src / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(src / "question_offsets.npy")
    a_tokens = np.load(src / "answer_tokens.npy", mmap_mode="r")
    a_offsets = np.load(src / "answer_offsets.npy")
    metadata = pq.read_table(src / "metadata.parquet").to_pylist()
    return tokens, offsets, q_tokens, q_offsets, a_tokens, a_offsets, metadata


def _split_into_passages(
    book_tokens: np.ndarray, passage_tokens: int,
) -> list[np.ndarray]:
    """Split a flat token array into fixed-size passages.

    The last passage is allowed to be shorter than ``passage_tokens``.
    Could be extended with chapter detection (look for newline-heavy
    regions or explicit "Chapter N" patterns) — deferred until we have
    evidence it matters.
    """
    n = int(book_tokens.shape[0])
    result: list[np.ndarray] = []
    for start in range(0, n, passage_tokens):
        end = min(start + passage_tokens, n)
        if end > start:
            result.append(np.asarray(book_tokens[start:end]))
    if not result:
        result.append(np.zeros(1, dtype=book_tokens.dtype))
    return result


def _score_passage_overlap(passage: np.ndarray, answer_set: set[int]) -> int:
    """Bag-of-tokens overlap count between a passage and an answer token set."""
    return sum(1 for t in passage.tolist() if int(t) in answer_set)


def _pick_gold_passage(
    passages: list[np.ndarray], answer_ids: np.ndarray,
) -> tuple[int, int]:
    """Heuristic: return (best_passage_idx, overlap_score).

    Ties are broken by the earliest passage (biased toward the start of
    the book, where summaries tend to live).
    """
    if len(passages) == 0:
        return 0, 0
    if answer_ids.size == 0:
        return 0, 0
    answer_set = {int(x) for x in answer_ids.tolist()}
    best_idx = 0
    best_score = -1
    for i, p in enumerate(passages):
        score = _score_passage_overlap(p, answer_set)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx, max(best_score, 0)


class _TrivialDecoder:
    """Test/fallback tokenizer that renders token IDs as ``t{id}`` strings.

    Only used when a real tokenizer isn't available (e.g., CI without network
    access to the gated Qwen3.5 HF repo). Production runs load the real
    tokenizer via ``AutoTokenizer.from_pretrained``.
    """

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_tokenizer(name: str) -> _DecoderLike:
    """Load a HF tokenizer, falling back to a trivial decoder on failure.

    Failures can be: the ``transformers`` package isn't installed (CPU-only
    CI without the ``[gpu]`` extra), the model repo is gated and we have no
    HF token, or there's no network. In all three cases we want to keep
    running — the provenance JSONL will carry placeholder strings, but
    production runs (inside the Docker container with network) will get
    real text.
    """
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:
        logger.warning(
            "Could not load tokenizer %r (%s); falling back to trivial decoder. "
            "Provenance question/answer strings will be placeholders.",
            name,
            exc,
        )
        return _TrivialDecoder()


def reshard(
    src: Path,
    dst: Path,
    passage_tokens: int,
    short_answer_threshold: int,
    min_overlap: int,
    tokenizer: _DecoderLike,
) -> dict:
    """Core re-shard loop. Pure-Python; takes an injected tokenizer so
    tests can bypass the HF load entirely. Returns the manifest dict.
    """
    dst.mkdir(parents=True, exist_ok=True)

    tokens, offsets, q_tokens, q_offsets, a_tokens, a_offsets, metadata = (
        _read_existing_mmap(src)
    )

    # Group QA rows by book (document_id). Each book gets split ONCE;
    # all QA pairs pointing at that book share the resulting passages.
    rows_by_book: dict[str, list[int]] = defaultdict(list)
    for idx, meta in enumerate(metadata):
        book_id = str(meta.get("document_id") or meta.get("id") or idx)
        rows_by_book[book_id].append(idx)

    # Build passage-level arrays.
    new_tokens_list: list[np.ndarray] = []
    new_offsets = [0]
    new_metadata_rows: list[dict] = []
    passage_lookup: dict[str, list[str]] = {}  # book_id -> ordered passage doc_ids
    passages_cache: dict[str, list[np.ndarray]] = {}

    for book_id, row_indices in rows_by_book.items():
        first_row_idx = row_indices[0]
        start = int(offsets[first_row_idx])
        end = int(offsets[first_row_idx + 1])
        book_tokens = np.asarray(tokens[start:end])
        passages = _split_into_passages(book_tokens, passage_tokens)
        passages_cache[book_id] = passages

        passage_ids: list[str] = []
        for p_idx, passage in enumerate(passages):
            passage_doc_id = f"{book_id}#p{p_idx:04d}"
            new_tokens_list.append(passage.astype(np.int32))
            new_offsets.append(new_offsets[-1] + int(passage.shape[0]))
            new_metadata_rows.append({
                "id": passage_doc_id,
                "document_id": passage_doc_id,
                "dataset_name": "narrativeqa_passages",
                "tag_list_json": json.dumps([book_id]),
                "parent_book": book_id,
                "passage_idx": p_idx,
            })
            passage_ids.append(passage_doc_id)
        passage_lookup[book_id] = passage_ids

    flat_tokens = (
        np.concatenate(new_tokens_list)
        if new_tokens_list
        else np.zeros(0, dtype=np.int32)
    )
    np.save(dst / "tokens.npy", flat_tokens)
    np.save(dst / "offsets.npy", np.asarray(new_offsets, dtype=np.int64))

    # Empty question / answer arrays — the KB-scale pipeline reads
    # questions from the provenance JSONL, not from the mmap.
    np.save(dst / "question_tokens.npy", np.zeros(0, dtype=np.int32))
    np.save(
        dst / "question_offsets.npy",
        np.zeros(len(new_metadata_rows) + 1, dtype=np.int64),
    )
    np.save(dst / "answer_tokens.npy", np.zeros(0, dtype=np.int32))
    np.save(
        dst / "answer_offsets.npy",
        np.zeros(len(new_metadata_rows) + 1, dtype=np.int64),
    )

    table = pa.Table.from_pylist(
        new_metadata_rows,
        schema=pa.schema([
            ("id", pa.string()),
            ("document_id", pa.string()),
            ("dataset_name", pa.string()),
            ("tag_list_json", pa.string()),
            ("parent_book", pa.string()),
            ("passage_idx", pa.int32()),
        ]),
    )
    pq.write_table(table, dst / "metadata.parquet")

    # Provenance: one row per ORIGINAL QA pair in the source metadata.
    prov_path = dst / "narrativeqa_provenance.jsonl"
    resolution_counts: dict[str, int] = defaultdict(int)
    n_prov = 0
    with prov_path.open("w") as f:
        for idx, meta in enumerate(metadata):
            book_id = str(meta.get("document_id") or meta.get("id") or idx)
            if book_id not in passage_lookup:
                continue
            passages_for_book = passage_lookup[book_id]
            if not passages_for_book:
                continue
            passages = passages_cache[book_id]

            a_start = int(a_offsets[idx])
            a_end = int(a_offsets[idx + 1])
            answer_ids = np.asarray(a_tokens[a_start:a_end])

            q_start = int(q_offsets[idx])
            q_end = int(q_offsets[idx + 1])
            question_ids = np.asarray(q_tokens[q_start:q_end])

            # Decide gold resolution.
            if int(answer_ids.size) < short_answer_threshold:
                gold_article_id = book_id
                gold_resolution = "short_answer_fallback"
            else:
                gold_p_idx, overlap_score = _pick_gold_passage(
                    passages, answer_ids,
                )
                if overlap_score < min_overlap:
                    gold_article_id = book_id
                    gold_resolution = "low_overlap_fallback"
                elif gold_p_idx >= len(passages_for_book):
                    # Shouldn't happen (passages_for_book is built from passages)
                    # but guard anyway.
                    gold_article_id = book_id
                    gold_resolution = "low_overlap_fallback"
                else:
                    gold_article_id = passages_for_book[gold_p_idx]
                    gold_resolution = "passage"

            question_str = tokenizer.decode(
                question_ids.tolist(), skip_special_tokens=True,
            )
            answer_str = tokenizer.decode(
                answer_ids.tolist(), skip_special_tokens=True,
            )

            f.write(json.dumps({
                "question": question_str,
                "gold_answer": answer_str,
                "gold_article_id": gold_article_id,
                "scope_template": "pre_scoped",
                "scope_description": book_id,
                "parent_book": book_id,
                "gold_resolution": gold_resolution,
                "question_token_ids": question_ids.tolist(),
                "answer_token_ids": answer_ids.tolist(),
            }) + "\n")
            n_prov += 1
            resolution_counts[gold_resolution] += 1

    manifest = {
        "schema_version": 2,
        "dataset_name": "narrativeqa_passages",
        "row_count": len(new_metadata_rows),
        "total_tokens": int(flat_tokens.shape[0]),
        "passage_tokens_target": passage_tokens,
        "parent_book_count": len(rows_by_book),
        "provenance_row_count": n_prov,
        "gold_resolution_counts": dict(resolution_counts),
        "short_answer_threshold": short_answer_threshold,
        "min_overlap": min_overlap,
    }
    (dst / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Path to $DATA_DIR/mmap/phase2/narrativeqa",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to the new passage-level mmap directory.",
    )
    parser.add_argument(
        "--passage-tokens",
        type=int,
        default=1000,
        help=(
            "Target passage length in tokens (default 1000). A 65k-token "
            "book produces ~65 passages, under the 100 leaf_cap of the "
            "browse-tree builder."
        ),
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="Qwen/Qwen3.5-0.8B",
        help=(
            "HF tokenizer name (default: Qwen/Qwen3.5-0.8B — must match "
            "the tokenizer used to build the source mmap). Used only to "
            "decode question/answer token IDs back to strings for the "
            "provenance JSONL."
        ),
    )
    parser.add_argument(
        "--short-answer-threshold",
        type=int,
        default=5,
        help=(
            "If the gold answer has fewer than this many tokens, fall back "
            "to book-level gold_article_id (the bag-of-tokens heuristic is "
            "unreliable for short answers like 'yes' / '42'). Default 5."
        ),
    )
    parser.add_argument(
        "--min-overlap",
        type=int,
        default=3,
        help=(
            "If the best passage's bag-of-tokens overlap with the answer is "
            "below this, fall back to book-level gold_article_id. Default 3."
        ),
    )
    args = parser.parse_args()

    tokenizer = _load_tokenizer(args.tokenizer)
    manifest = reshard(
        src=args.source,
        dst=args.output,
        passage_tokens=args.passage_tokens,
        short_answer_threshold=args.short_answer_threshold,
        min_overlap=args.min_overlap,
        tokenizer=tokenizer,
    )
    print(
        f"wrote {args.output} — {manifest['row_count']} passage rows across "
        f"{manifest['parent_book_count']} books; "
        f"{manifest['provenance_row_count']} provenance rows "
        f"(resolution: {manifest['gold_resolution_counts']})"
    )


if __name__ == "__main__":
    main()
