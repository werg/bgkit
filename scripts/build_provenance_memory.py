#!/usr/bin/env python
"""Build a user-memory provenance JSONL for ``scripts/build_teacher_trajectories.py``.

Covers the memory datasets produced by ``scripts/convert_memory_datasets.py``:
MSC, SHARE, Chronicles, PerLTQA, LAPS. Each row is one (session/episode,
question, answer) triple where the ``document_id`` is the episode/user
identifier and each sample is scoped to that single user.

Scope template: ``pre_scoped`` with ``scope_description`` set to the
``document_id`` (the episode/user id written by the converter). The
decoder works against a per-dataset browse tree whose leaves are the
individual sessions/turns within an episode; ``gold_article_id`` equals
the row's own ``document_id`` or ``id`` (session/turn id). The browse
tree builder for memory datasets groups these under one node per
episode, so ``pre_scoped`` with the episode id as scope_description is
the right prompt template.
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

        return get_data_dir(must_exist=False) / "mmap" / "phase2" / "msc"
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
    # The converter writes both id and document_id (both set to episode_id).
    # We require at least one.
    if "document_id" not in schema.names and "id" not in schema.names:
        raise RuntimeError(
            f"memory dataset metadata at {meta_path} lacks both 'document_id' "
            "and 'id' columns — cannot derive scope / gold_article_id.",
        )

    columns: list[str] = []
    for candidate in ("id", "document_id", "memory_type", "dataset_name"):
        if candidate in schema.names:
            columns.append(candidate)
    metadata = pq.read_table(meta_path, columns=columns).to_pylist()

    q_tokens = np.load(mmap_dir / "question_tokens.npy", mmap_mode="r")
    q_offsets = np.load(mmap_dir / "question_offsets.npy")
    a_tokens = np.load(mmap_dir / "answer_tokens.npy", mmap_mode="r")
    a_offsets = np.load(mmap_dir / "answer_offsets.npy")

    n_rows = len(metadata)
    if len(q_offsets) - 1 != n_rows or len(a_offsets) - 1 != n_rows:
        raise RuntimeError(
            f"memory mmap at {mmap_dir} has mismatched parallel row counts",
        )

    # For memory datasets, multiple rows share the same episode_id (each row
    # is one Q/A pair within the episode). We synthesize a per-row gold
    # article id by appending the row index when the converter didn't give
    # us a unique-per-row id (which it currently doesn't).
    # Row-level uniqueness matters for the browse tree — otherwise multiple
    # provenance rows collapse onto the same browse leaf.
    episode_row_counters: dict[str, int] = {}

    output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    n_skipped = 0

    with output.open("w") as f:
        for idx, row in enumerate(metadata):
            episode_id = row.get("document_id") or row.get("id")
            if not episode_id:
                logger.warning(
                    "memory row %d lacks document_id/id — skipping", idx,
                )
                n_skipped += 1
                continue

            question_ids = _slice_ids(q_tokens, q_offsets, idx)
            answer_ids = _slice_ids(a_tokens, a_offsets, idx)
            if not question_ids or not answer_ids:
                logger.warning(
                    "memory row %d has empty question or answer tokens — skipping",
                    idx,
                )
                n_skipped += 1
                continue

            # Row-unique gold article id within this episode. Form:
            # ``<episode_id>#r<local_row_idx>``. Memory browse trees built
            # from the same metadata will key leaves the same way (a
            # corresponding builder hook in build_browse_tree.py adds the
            # index suffix per episode). If your browse tree is indexed by
            # the bare episode_id, drop the suffix — but then multiple
            # provenance rows collide onto the same leaf.
            local = episode_row_counters.get(str(episode_id), 0)
            episode_row_counters[str(episode_id)] = local + 1
            gold_article_id = f"{episode_id}#r{local:05d}"

            question_str = tokenizer.decode(question_ids, skip_special_tokens=True)
            answer_str = tokenizer.decode(answer_ids, skip_special_tokens=True)

            f.write(
                json.dumps({
                    "question": question_str,
                    "gold_answer": answer_str,
                    "gold_article_id": gold_article_id,
                    "scope_template": "pre_scoped",
                    "scope_description": str(episode_id),
                }) + "\n",
            )
            n_written += 1

    logger.info("memory provenance: wrote=%d skipped=%d", n_written, n_skipped)
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mmap-dir",
        type=Path,
        default=_default_mmap_dir(),
        help=(
            "Phase 2 memory mmap dir "
            "(default: $DATA_DIR/mmap/phase2/msc; override for share, "
            "chronicles, perltqa, laps)"
        ),
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
