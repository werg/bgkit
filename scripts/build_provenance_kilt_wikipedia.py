#!/usr/bin/env python
"""Build a KILT Wikipedia provenance JSONL for ``build_teacher_trajectories.py``.

Unlike the task-side KILT provenance builder
(``scripts/build_provenance_kilt.py``), this one targets the raw
Wikipedia *corpus* mmap at ``$DATA_DIR/mmap/phase2/kilt`` (produced by
``scripts/convert_hf_to_mmap.py kilt_wikipedia``). There are no natural
(question, answer) pairs in that mmap — each row is a single Wikipedia
article, with its text living in the primary ``tokens.npy`` stream and
the per-row ``question_tokens.npy`` / ``answer_tokens.npy`` streams
holding a single placeholder token (the corpus-only branch of
``convert_hf_to_mmap.py``).

For Phase 2 KB training we still want trajectories that *walk the KILT
browse tree* to a specific article. The most direct way to get those
from a pure corpus is to synthesize a trivial ``"Tell me about <title>"``
question per article, with the article's own text as the gold answer.
This keeps the trajectory framework happy without needing a real task
dataset joined in.

We don't have a usable ``wikipedia_title`` column in the current mmap
(the converter drops it — only ``id``, ``document_id``, ``dataset_name``
and ``tag_list_json`` are kept), so the "title" we substitute into the
question template is the row's ``document_id`` (which is the Wikipedia
page id — not a real title, but a stable key that matches the browse
tree).

For the gold answer we decode a prefix of the document text
(``answer_prefix_tokens`` tokens, default 64) from ``tokens.npy``. For
the trivial fallback decoder that yields ``"t<id> t<id> ..."``; for a
real HF tokenizer it yields the article's first ~1-2 sentences.

Scope template: ``topic_list`` — the KILT Wikipedia browse tree built by
``scripts/build_kilt_hierarchy.py`` (or the generic browse-tree builder
driven by ``tag_list_json``) provides the category topics, so the
trajectory builder can surface a topic list to the decoder.
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

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "kilt"
    except Exception:
        return None


def _slice_ids(
    tokens: np.ndarray,
    offsets: np.ndarray,
    idx: int,
    *,
    max_len: int | None = None,
) -> list[int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    if end <= start:
        return []
    if max_len is not None and end - start > max_len:
        end = start + max_len
    return [int(x) for x in np.asarray(tokens[start:end]).tolist()]


_TITLE_COLUMNS = ("wikipedia_title", "title")


def build_provenance(
    mmap_dir: Path,
    output: Path,
    tokenizer: _DecoderLike,
    *,
    answer_prefix_tokens: int = 64,
    limit: int | None = None,
) -> int:
    """Emit provenance JSONL rows for the KILT Wikipedia mmap at ``mmap_dir``.

    One output row per article. Each row's question is the synthetic
    ``"Tell me about <title>"`` template and the gold answer is a short
    prefix of the article's text (first ``answer_prefix_tokens`` tokens).

    Returns the number of rows written.
    """
    meta_path = mmap_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {mmap_dir}")

    schema = pq.read_schema(meta_path)
    if "document_id" not in schema.names:
        raise RuntimeError(
            f"KILT Wikipedia metadata at {meta_path} is missing 'document_id'. "
            "Re-run scripts/convert_hf_to_mmap.py kilt_wikipedia.",
        )

    columns = ["document_id"]
    if "id" in schema.names:
        columns.append("id")
    for key in _TITLE_COLUMNS:
        if key in schema.names:
            columns.append(key)
    metadata = pq.read_table(meta_path, columns=columns).to_pylist()

    # Article text lives in the primary tokens.npy stream (corpus-only
    # datasets put document tokens there, not in answer_tokens.npy).
    doc_tokens = np.load(mmap_dir / "tokens.npy", mmap_mode="r")
    doc_offsets = np.load(mmap_dir / "offsets.npy")

    n_rows = len(metadata)
    if len(doc_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"KILT Wikipedia mmap at {mmap_dir} has mismatched parallel row "
            "counts between metadata.parquet and offsets.npy",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            if limit is not None and n_written >= limit:
                break

            doc_id = row.get("document_id")
            if not doc_id:
                logger.warning(
                    "kilt_wikipedia row %d missing document_id — skipping", idx,
                )
                n_skipped += 1
                continue

            doc_ids = _slice_ids(
                doc_tokens, doc_offsets, idx, max_len=answer_prefix_tokens,
            )
            if not doc_ids:
                logger.warning(
                    "kilt_wikipedia row %d has empty document tokens — skipping",
                    idx,
                )
                n_skipped += 1
                continue

            title: str | None = None
            for key in _TITLE_COLUMNS:
                val = row.get(key)
                if val:
                    title = str(val)
                    break
            if title is None:
                title = str(doc_id)

            question_str = f"Tell me about {title}"
            gold_answer_str = tokenizer.decode(doc_ids, skip_special_tokens=True)

            f.write(
                json.dumps({
                    "question": question_str,
                    "gold_answer": gold_answer_str,
                    "gold_article_id": str(doc_id),
                    "scope_template": "topic_list",
                    "scope_description": "",
                }) + "\n",
            )
            n_written += 1

    logger.info(
        "kilt_wikipedia provenance: wrote=%d skipped=%d", n_written, n_skipped,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help="Phase 2 KILT Wikipedia mmap dir (default: $DATA_DIR/mmap/phase2/kilt)",
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
    parser.add_argument(
        "--answer-prefix-tokens",
        type=int,
        default=64,
        help="Number of document tokens to decode as the gold answer.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Emit at most this many rows (for smoke tests).",
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
        answer_prefix_tokens=args.answer_prefix_tokens,
        limit=args.limit,
    )
    print(f"wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
