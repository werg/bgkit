#!/usr/bin/env python
"""Build a KILT provenance JSONL for ``scripts/build_teacher_trajectories.py``.

KILT tasks (NQ, HotpotQA, FEVER, zsRE, T-REx, WoW, ELI5, AidaYago2, WNED,
CWeb, TriviaQA) store per-query Wikipedia provenance in a ``provenance_json``
column written by ``scripts/convert_hf_to_mmap.py``. This script reads the
Phase 2 mmap for a KILT task, decodes each row's (question, answer) token
streams back to strings, and emits one JSONL row per QA pair with:

  - ``question``:          decoded question text
  - ``gold_answer``:       decoded answer text
  - ``gold_article_id``:   the ``wikipedia_title`` for the first
                           wikipedia_id in the row's provenance. This is
                           the browse-tree node id — the same string the
                           decoder emits during drill-down. Built via
                           ``--title-map`` against the KILT hierarchy
                           JSONL, which carries both ``article_id``
                           (title) and ``document_id`` (wikipedia_id).
                           Falls back to the stringified document_id when
                           provenance is empty (e.g. CWeb rows without a
                           canonical Wikipedia target) or when no title
                           map is provided.
  - ``gold_document_id``:  the canonical mmap key (wikipedia_id) the L0
                           cache and ``ArticleTokenStore`` index. The
                           trainer uses this to fetch survivors after
                           translating ``gold_article_id`` through the
                           title → document_id sidecar, but having it on
                           the provenance row simplifies downstream
                           debugging.
  - ``scope_template``:    always ``"topic_list"`` — KILT answers are
                           drawn from all of Wikipedia; the decoder picks
                           the top-level topic from the browse tree
  - ``scope_description``: empty string (topic_list uses topic_list instead)

The output feeds ``scripts/build_teacher_trajectories.py``, which joins
this against the per-dataset browse tree to assemble teacher trajectories.
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
    """Test/fallback tokenizer that renders IDs as ``t{id}`` tokens.

    Used when a real HF tokenizer is unavailable (e.g. CPU-only CI with no
    network access to gated Qwen3.5). Production runs inside the training
    container load the real tokenizer and get meaningful text back.
    """

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return " ".join(f"t{int(i)}" for i in ids)


def _load_tokenizer(name: str) -> _DecoderLike:
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # pragma: no cover - exercised via fallback path
        logger.warning(
            "Could not load tokenizer %r (%s); falling back to trivial decoder. "
            "Provenance question/answer strings will be placeholders.",
            name,
            exc,
        )
        return _TrivialDecoder()


def _default_mmap_dir() -> Path | None:
    try:
        from bgkit.env import get_data_dir

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "kilt_nq"
    except Exception:
        return None


def _slice_ids(tokens: np.ndarray, offsets: np.ndarray, idx: int) -> list[int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    if end <= start:
        return []
    return [int(x) for x in np.asarray(tokens[start:end]).tolist()]


def _load_title_map(path: Path | None) -> dict[str, str]:
    """Load a ``document_id → article_id`` map from a hierarchy JSONL.

    The KILT hierarchy JSONL rows (emitted by
    ``scripts/build_kilt_hierarchy.py``) carry both ``article_id``
    (the human-readable wikipedia_title, possibly disambiguated) and
    ``document_id`` (the numeric wikipedia_id). We invert the direction
    to resolve a KILT task's provenance list back to titles.

    Returns an empty dict when ``path`` is None or missing — callers
    fall back to the numeric id as gold_article_id in that case.
    """
    if path is None or not path.exists():
        return {}
    doc_to_title: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            stripped = raw.strip()
            if not stripped:
                continue
            try:
                row = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            article_id = row.get("article_id")
            doc_id = row.get("document_id")
            if article_id is None or doc_id is None:
                continue
            doc_to_title[str(doc_id)] = str(article_id)
    return doc_to_title


def build_provenance(
    mmap_dir: Path,
    output: Path,
    tokenizer: _DecoderLike,
    title_map: dict[str, str] | None = None,
) -> int:
    """Emit provenance JSONL rows for the KILT mmap at ``mmap_dir``.

    Returns the number of rows written.
    """
    if title_map is None:
        title_map = {}
    meta_path = mmap_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {mmap_dir}")

    schema = pq.read_schema(meta_path)
    wanted = [
        c for c in ("id", "document_id", "provenance_json") if c in schema.names
    ]
    if "document_id" not in wanted:
        raise RuntimeError(
            f"KILT metadata at {meta_path} is missing 'document_id'. "
            "Re-run scripts/convert_hf_to_mmap.py to produce a valid mmap.",
        )
    metadata = pq.read_table(meta_path, columns=wanted).to_pylist()

    q_tokens = np.load(mmap_dir / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(mmap_dir / "question_offsets.npy")
    a_tokens = np.load(mmap_dir / "answer_tokens.npy", mmap_mode="r")
    a_offsets = np.load(mmap_dir / "answer_offsets.npy")

    n_rows = len(metadata)
    if len(q_offsets) - 1 != n_rows or len(a_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"KILT mmap at {mmap_dir} has mismatched parallel array row counts",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped_no_prov = 0
    n_fallback_docid = 0
    n_title_missing = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            question_ids = _slice_ids(q_tokens, q_offsets, idx)
            answer_ids = _slice_ids(a_tokens, a_offsets, idx)
            if not question_ids or not answer_ids:
                logger.warning(
                    "kilt row %d has empty question or answer tokens — skipping",
                    idx,
                )
                continue

            provenance_raw = row.get("provenance_json") or "[]"
            try:
                provenance = json.loads(str(provenance_raw))
            except json.JSONDecodeError:
                logger.warning(
                    "kilt row %d has malformed provenance_json — skipping", idx,
                )
                continue

            if isinstance(provenance, list) and provenance:
                gold_document_id = str(provenance[0])
            else:
                # Fall back to document_id. For KILT tasks, document_id is
                # the query id (nq/hotpot), not a Wikipedia article — this
                # usually won't match the Wikipedia browse tree, but we
                # emit it anyway so the caller can decide to skip.
                doc_id = row.get("document_id")
                if not doc_id:
                    n_skipped_no_prov += 1
                    logger.warning(
                        "kilt row %d has no provenance and no document_id — skipping",
                        idx,
                    )
                    continue
                gold_document_id = str(doc_id)
                n_fallback_docid += 1

            # Translate numeric wikipedia_id → wikipedia_title via the
            # title map. When absent (no --title-map or the id is not in
            # the loaded hierarchy), fall back to the numeric id string
            # so downstream tools still have *something* to work with,
            # but log the shortfall.
            gold_article_id = title_map.get(gold_document_id)
            if gold_article_id is None:
                gold_article_id = gold_document_id
                if title_map:
                    n_title_missing += 1

            question_str = tokenizer.decode(question_ids, skip_special_tokens=True)
            answer_str = tokenizer.decode(answer_ids, skip_special_tokens=True)

            f.write(
                json.dumps({
                    "question": question_str,
                    "gold_answer": answer_str,
                    "gold_article_id": gold_article_id,
                    "gold_document_id": gold_document_id,
                    "scope_template": "topic_list",
                    "scope_description": "",
                }) + "\n",
            )
            n_written += 1

    logger.info(
        "kilt provenance: wrote=%d fallback_docid=%d skipped=%d title_missing=%d",
        n_written, n_fallback_docid, n_skipped_no_prov, n_title_missing,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help="Phase 2 KILT mmap dir (default: $DATA_DIR/mmap/phase2/kilt_nq)",
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
        "--title-map",
        type=Path,
        default=None,
        help="Path to a KILT hierarchy JSONL emitted by "
        "scripts/build_kilt_hierarchy.py. Rows must carry both "
        "'article_id' (wikipedia_title, already disambiguated) and "
        "'document_id' (wikipedia_id). Used to translate provenance "
        "wikipedia_ids into titles for gold_article_id. Optional: "
        "when omitted, gold_article_id falls back to the numeric id.",
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
    title_map = _load_title_map(args.title_map)
    if args.title_map and not title_map:
        logger.warning(
            "title_map at %s is empty or missing — gold_article_id will "
            "fall back to numeric wikipedia_id",
            args.title_map,
        )
    n = build_provenance(
        mmap_dir=args.mmap_dir,
        output=args.output,
        tokenizer=tokenizer,
        title_map=title_map,
    )
    print(f"wrote {n} rows to {args.output}")


if __name__ == "__main__":
    main()
