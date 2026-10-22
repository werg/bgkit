#!/usr/bin/env python
"""Build a PubMedQA provenance JSONL for ``scripts/build_teacher_trajectories.py``.

PubMedQA stores one QA row per PubMed article; the article's own ``pubid``
(written by ``scripts/convert_hf_to_mmap.py`` as ``document_id``) IS the
gold article. The answer combines ``final_decision`` (yes/no/maybe) with
``long_answer`` when both are present, matching the reader behavior
expected by the decoder at eval time.

Since the conversion step already tokenized and concatenated these into
the ``answer_tokens.npy`` stream, this builder simply decodes them back
to a string. We do NOT need to round-trip through the HF dataset.

Scope template is ``topic_list`` — the decoder picks a PubMed MeSH topic
from the browse tree built by ``scripts/build_mesh_hierarchy.py``.

Each row emits both:

  - ``gold_article_id``:  the browse-tree node id (human-readable title
                          when the mesh hierarchy produced one, falling
                          back to the numeric pubid). Resolved via
                          ``--title-map`` against the mesh hierarchy
                          JSONL.
  - ``gold_document_id``: the numeric pubid (mmap key used by the L0
                          cache / ``ArticleTokenStore``). Always equal to
                          the row's ``document_id``.

When no title map is supplied, ``gold_article_id == gold_document_id``
and the downstream browse tree is expected to use pubid as its node id
— the trainer's title→document_id sidecar will also be empty / absent
in that case, so ``_resolve_article_ids`` is a no-op translation.
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

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "pubmedqa"
    except Exception:
        return None


def _slice_ids(tokens: np.ndarray, offsets: np.ndarray, idx: int) -> list[int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    if end <= start:
        return []
    return [int(x) for x in np.asarray(tokens[start:end]).tolist()]


def _load_title_map(path: Path | None) -> dict[str, str]:
    """Load a ``document_id → article_id`` map from a mesh hierarchy JSONL.

    Each row of ``scripts/build_mesh_hierarchy.py``'s output carries
    ``article_id`` (title-or-pubid, possibly disambiguated) and
    ``document_id`` (the numeric pubid). We invert the pair so this
    script can translate pubid → gold_article_id for the trajectory
    output. An empty / missing file yields an empty dict.
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
    if title_map is None:
        title_map = {}
    meta_path = mmap_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {mmap_dir}")

    schema = pq.read_schema(meta_path)
    if "document_id" not in schema.names:
        raise RuntimeError(
            f"PubMedQA metadata at {meta_path} is missing 'document_id'. "
            "Re-run scripts/convert_hf_to_mmap.py to produce a valid mmap.",
        )

    columns = [c for c in ("id", "document_id") if c in schema.names]
    metadata = pq.read_table(meta_path, columns=columns).to_pylist()

    q_tokens = np.load(mmap_dir / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(mmap_dir / "question_offsets.npy")
    a_tokens = np.load(mmap_dir / "answer_tokens.npy", mmap_mode="r")
    a_offsets = np.load(mmap_dir / "answer_offsets.npy")

    n_rows = len(metadata)
    if len(q_offsets) - 1 != n_rows or len(a_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"PubMedQA mmap at {mmap_dir} has mismatched parallel row counts",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_title_missing = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            doc_id = row.get("document_id")
            if not doc_id:
                logger.warning(
                    "pubmedqa row %d missing document_id — skipping", idx,
                )
                n_skipped += 1
                continue

            question_ids = _slice_ids(q_tokens, q_offsets, idx)
            answer_ids = _slice_ids(a_tokens, a_offsets, idx)
            if not question_ids or not answer_ids:
                logger.warning(
                    "pubmedqa row %d has empty question or answer tokens — skipping",
                    idx,
                )
                n_skipped += 1
                continue

            question_str = tokenizer.decode(question_ids, skip_special_tokens=True)
            # The converter already concatenated final_decision + long_answer
            # into the answer stream, so a single decode call gives us both.
            answer_str = tokenizer.decode(answer_ids, skip_special_tokens=True)

            gold_document_id = str(doc_id)
            gold_article_id = title_map.get(gold_document_id)
            if gold_article_id is None:
                gold_article_id = gold_document_id
                if title_map:
                    n_title_missing += 1

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
        "pubmedqa provenance: wrote=%d skipped=%d title_missing=%d",
        n_written, n_skipped, n_title_missing,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help="Phase 2 PubMedQA mmap dir (default: $DATA_DIR/mmap/phase2/pubmedqa)",
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
        help="Path to a PubMedQA mesh-hierarchy JSONL emitted by "
        "scripts/build_mesh_hierarchy.py. Rows must carry both "
        "'article_id' (possibly a title, possibly the numeric pubid) "
        "and 'document_id' (the numeric pubid). Used to translate "
        "pubid → gold_article_id. Optional: when omitted, "
        "gold_article_id falls back to the numeric pubid.",
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
            "fall back to numeric pubid",
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
