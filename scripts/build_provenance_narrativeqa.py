#!/usr/bin/env python
"""Build a NarrativeQA provenance JSONL for ``scripts/build_teacher_trajectories.py``.

NarrativeQA is book-scoped: each question is about a specific book, and
browsing the full Wikipedia topic hierarchy isn't useful — the decoder
should start with ``browse(id="root")`` on a per-book browse tree and
drill down to a specific passage.

The authoritative pipeline for NarrativeQA is
``scripts/reshard_narrativeqa.py``, which already writes
``narrativeqa_provenance.jsonl`` directly. This script is a lighter-weight
alternative that works against the original book-level mmap (no passage
resharding) — it produces book-level gold_article_ids and is suitable
when passage resharding isn't desired (e.g. for an initial smoke run on
the browse tree builder before the reshard has completed).

Scope template: ``pre_scoped`` with ``scope_description`` set to the
book title when metadata carries one, otherwise to the book's
``document_id`` (the stable per-book key). Each row's own ``document_id``
is the gold article.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Protocol

import numpy as np
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


class _DecoderLike(Protocol):
    def decode(self, ids: list[int], skip_special_tokens: bool = ...) -> str: ...


class _TrivialDecoder:
    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_tokenizer(name: str) -> _DecoderLike:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Could not load tokenizer %r (%s); falling back to trivial decoder.",
            name, exc,
        )
        return _TrivialDecoder()


def _default_mmap_dir() -> Path | None:
    try:
        from bgkit.env import get_data_dir

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "narrativeqa"
    except Exception:
        return None


def _slice_ids(tokens: np.ndarray, offsets: np.ndarray, idx: int) -> list[int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    if end <= start:
        return []
    return [int(x) for x in np.asarray(tokens[start:end]).tolist()]


# Columns we'll probe for a human-readable book title. None of these are
# written by the current convert_hf_to_mmap.py (NarrativeQA only gets id /
# document_id / dataset_name / tag_list_json), so the fallback path to
# document_id will be the usual one. Listed here so that if the converter
# is ever extended to carry a title column, this script picks it up.
_TITLE_COLUMNS = ("book_title", "title", "parent_book_title")


def build_provenance(
    mmap_dir: Path,
    output: Path,
    tokenizer: _DecoderLike,
) -> int:
    meta_path = mmap_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {mmap_dir}")

    schema = pq.read_schema(meta_path)
    if "document_id" not in schema.names:
        raise RuntimeError(
            f"NarrativeQA metadata at {meta_path} is missing 'document_id'. "
            "Re-run scripts/convert_hf_to_mmap.py to produce a valid mmap.",
        )

    columns = ["id", "document_id"]
    if "book_id" in schema.names:
        columns.append("book_id")
    for candidate in _TITLE_COLUMNS:
        if candidate in schema.names:
            columns.append(candidate)
    # parent_book (written by reshard_narrativeqa) is also accepted as an
    # alternate scope key, but it means the mmap is already passage-sharded
    # and should be handled via the reshard pipeline instead.
    if "parent_book" in schema.names:
        columns.append("parent_book")

    metadata = pq.read_table(meta_path, columns=columns).to_pylist()

    q_tokens = np.load(mmap_dir / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(mmap_dir / "question_offsets.npy")
    a_tokens = np.load(mmap_dir / "answer_tokens.npy", mmap_mode="r")
    a_offsets = np.load(mmap_dir / "answer_offsets.npy")

    n_rows = len(metadata)
    if len(q_offsets) - 1 != n_rows or len(a_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"NarrativeQA mmap at {mmap_dir} has mismatched parallel row counts",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_fallback_scope = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            doc_id = row.get("document_id")
            if not doc_id:
                logger.warning(
                    "narrativeqa row %d missing document_id — skipping", idx,
                )
                n_skipped += 1
                continue

            question_ids = _slice_ids(q_tokens, q_offsets, idx)
            answer_ids = _slice_ids(a_tokens, a_offsets, idx)
            if not question_ids or not answer_ids:
                logger.warning(
                    "narrativeqa row %d has empty question or answer tokens — skipping",
                    idx,
                )
                n_skipped += 1
                continue

            # Prefer a human-readable title if present; else the per-book
            # grouping key (book_id, written by convert_hf_to_mmap.py); else
            # the parent book id (reshard pipeline); else the row's own
            # document_id.
            scope_description: str | None = None
            for key in _TITLE_COLUMNS:
                val = row.get(key)
                if val:
                    scope_description = str(val)
                    break
            if scope_description is None:
                book = row.get("book_id")
                if book:
                    scope_description = str(book)
            if scope_description is None:
                parent = row.get("parent_book")
                if parent:
                    scope_description = str(parent)
            if scope_description is None:
                scope_description = str(doc_id)
                n_fallback_scope += 1

            question_str = tokenizer.decode(question_ids, skip_special_tokens=True)
            answer_str = tokenizer.decode(answer_ids, skip_special_tokens=True)

            f.write(
                json.dumps({
                    "question": question_str,
                    "gold_answer": answer_str,
                    "gold_article_id": str(doc_id),
                    "scope_template": "pre_scoped",
                    "scope_description": scope_description,
                }) + "\n",
            )
            n_written += 1

    logger.info(
        "narrativeqa provenance: wrote=%d fallback_scope=%d skipped=%d",
        n_written, n_fallback_scope, n_skipped,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help="Phase 2 NarrativeQA mmap dir (default: $DATA_DIR/mmap/phase2/narrativeqa)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Destination JSONL path.",
    )
    parser.add_argument(
        "--tokenizer",
        default="Qwen/Qwen3.5-0.8B",
        help="HF tokenizer name (must match the one used to build the mmap).",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if args.mmap_dir is None:
        print(
            "ERROR: --mmap-dir not given and DATA_DIR is not configured.",
            file=sys.stderr,
        )
        sys.exit(2)

    tokenizer = _load_tokenizer(args.tokenizer)
    n = build_provenance(
        mmap_dir=args.mmap_dir,
        output=args.output,
        tokenizer=tokenizer,
    )
    print(f"wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
