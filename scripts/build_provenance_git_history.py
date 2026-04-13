#!/usr/bin/env python
"""Build a git-history provenance JSONL for ``scripts/build_teacher_trajectories.py``.

Each row in the git_history mmap (written by
``scripts/convert_qa_pairs_to_npy.py`` from
``scripts/generate_git_qa.py`` output) corresponds to one (commit, QA pair).
The gold article is the commit itself — the browse tree built by
``scripts/build_browse_tree.py`` with ``--dataset git_history`` keys leaves
as commit hashes under ``org/repo/year_YYYY`` paths.

Scope template: ``pre_scoped`` with ``scope_description`` set to the row's
``repo_path`` (e.g. ``"torvalds/linux"``). The decoder starts with
``browse(id="root")`` on the per-repo browse tree and drills into years
then commits.

gold_article_id resolution order:
  1. ``document_id`` column (preferred — matches the key used by
     ``build_browse_tree.py``'s git_history ingester)
  2. ``commit_sha`` column (fallback — the current converter's output
     doesn't include a document_id column, so this is usually the path
     that fires; if your browse tree uses a different article ID form
     you'll need to align the two)
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

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "git_history"
    except Exception:
        return None


def _slice_ids(tokens: np.ndarray, offsets: np.ndarray, idx: int) -> list[int]:
    start = int(offsets[idx])
    end = int(offsets[idx + 1])
    if end <= start:
        return []
    return [int(x) for x in np.asarray(tokens[start:end]).tolist()]


def build_provenance(
    mmap_dir: Path,
    output: Path,
    tokenizer: _DecoderLike,
) -> int:
    meta_path = mmap_dir / "metadata.parquet"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.parquet not found in {mmap_dir}")

    schema = pq.read_schema(meta_path)
    if "repo_path" not in schema.names:
        raise RuntimeError(
            f"git_history metadata at {meta_path} is missing 'repo_path'. "
            "Re-run scripts/generate_git_qa.py + "
            "scripts/convert_qa_pairs_to_npy.py to produce a valid mmap.",
        )
    if "commit_sha" not in schema.names and "document_id" not in schema.names:
        raise RuntimeError(
            f"git_history metadata at {meta_path} has neither 'commit_sha' "
            "nor 'document_id' — cannot derive a gold article id.",
        )

    columns: list[str] = ["repo_path"]
    if "document_id" in schema.names:
        columns.append("document_id")
    if "commit_sha" in schema.names:
        columns.append("commit_sha")
    if "question" in schema.names:
        columns.append("question")

    metadata = pq.read_table(meta_path, columns=columns).to_pylist()

    # The question column in git_history metadata is a truncated preview
    # (first 500 chars). The authoritative question tokens live in
    # question_tokens.npy, which we decode back to a string.
    q_tokens = np.load(mmap_dir / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(mmap_dir / "question_offsets.npy")
    # Answer tokens live in tokens.npy for git_history (not answer_tokens.npy) —
    # the QA mmap schema uses the primary tokens stream as the answer stream
    # because there's no separate "document" per row. Fall back to
    # answer_tokens.npy if it exists (future-proof), else use tokens.npy.
    if (mmap_dir / "answer_tokens.npy").exists():
        a_tokens = np.load(mmap_dir / "answer_tokens.npy", mmap_mode="r")
        a_offsets = np.load(mmap_dir / "answer_offsets.npy")
    else:
        a_tokens = np.load(mmap_dir / "tokens.npy", mmap_mode="r")
        a_offsets = np.load(mmap_dir / "offsets.npy")

    n_rows = len(metadata)
    if len(q_offsets) - 1 != n_rows or len(a_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"git_history mmap at {mmap_dir} has mismatched parallel row counts",
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0
    n_fallback_sha = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            repo = row.get("repo_path")
            if not repo:
                logger.warning(
                    "git_history row %d missing repo_path — skipping", idx,
                )
                n_skipped += 1
                continue

            # Prefer document_id when the converter writes one; otherwise
            # fall back to commit_sha (current behavior of
            # convert_qa_pairs_to_npy.py).
            doc_id = row.get("document_id") if "document_id" in columns else None
            if not doc_id:
                doc_id = row.get("commit_sha")
                if doc_id:
                    n_fallback_sha += 1
            if not doc_id:
                logger.warning(
                    "git_history row %d has neither document_id nor commit_sha — "
                    "skipping",
                    idx,
                )
                n_skipped += 1
                continue

            question_ids = _slice_ids(q_tokens, q_offsets, idx)
            answer_ids = _slice_ids(a_tokens, a_offsets, idx)
            if not question_ids or not answer_ids:
                logger.warning(
                    "git_history row %d has empty question or answer tokens — "
                    "skipping",
                    idx,
                )
                n_skipped += 1
                continue

            question_str = tokenizer.decode(question_ids, skip_special_tokens=True)
            answer_str = tokenizer.decode(answer_ids, skip_special_tokens=True)

            f.write(
                json.dumps({
                    "question": question_str,
                    "gold_answer": answer_str,
                    "gold_article_id": str(doc_id),
                    "scope_template": "pre_scoped",
                    "scope_description": str(repo),
                }) + "\n",
            )
            n_written += 1

    logger.info(
        "git_history provenance: wrote=%d fallback_sha=%d skipped=%d",
        n_written, n_fallback_sha, n_skipped,
    )
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help="Phase 2 git_history mmap dir (default: $DATA_DIR/mmap/phase2/git_history)",
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
