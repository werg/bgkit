"""Shared writer for flat Phase-2 datasets (capability-packaging Family B).

One call turns (windows/articles + flat 2-turn trajectories) into the three
artifacts the ``KRKBTrainer`` flat path consumes: the mmap article store, the
trajectory parquet (with explicit ``split`` column), and — via
``scripts/build_browse_tree.py`` run separately — the flat browse tree.
Used by ``scripts/build_lognav_phase2.py`` and
``scripts/build_fileneedle_phase2.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bgkit.data.bgkit_tool_template import TrajectoryTurn, trajectory_to_json
from bgkit.data.mmap_writer import build_csr_offsets, write_mmap_artifacts

TRAJ_COLUMNS = (
    "dataset_name",
    "scope_template",
    "scope_description",
    "topic_list_json",
    "question",
    "gold_answer",
    "trajectory_json",
    "split",
    "group_id",
    "gold_span_json",
)


def flat_trajectory_row(
    *,
    dataset_name: str,
    doc_id: str,
    question: str,
    answer: str,
    scope_description: str,
    split: str,
    group_id: str,
    gold_span_json: str | None = None,
) -> dict:
    """One flat single-bgkit trajectory row (retrieval call + answer).

    The bgkit turn is a PLAIN LEAF DRILL ``{ids, query}`` — the form
    ``KRKBTrainer._prepare_sample_for_decode`` routes to ``_prepare_l1_turn``
    (live query-conditioned L0 → L1 → projection → REAL survivors). It must NOT
    carry the git-repro drill tags: ``is_head`` routes the turn to the
    shared-tree head drill, which flat datasets have no infrastructure for, so
    every splice silently resolved to a ZERO survivor and the encoder never ran
    (the v1→v4 widenet runs, diagnosed 2026-08-22). The trainer now raises on
    that misrouting; this writer is the data-side half of the contract.

    ``gold_span_json`` = ``"[tok_start, tok_end)"`` of the answer inside the
    tokenized article (see :mod:`bgkit.data.gold_span`) — the v5 span-level
    relevance supervision signal. Optional; consumers ignore it when absent.

    The system prompt (``SYSTEM_PRE_SCOPED``) tells the decoder to start with
    the entrypoint ID NAMED IN THE KNOWLEDGE-BASE DESCRIPTION, so the scope
    names ``doc_id`` explicitly (``"...; entrypoint id: `<doc_id>`"``). Before
    2026-08-23 the description only said e.g. "source file a.py" while the
    trained call carried ``file:o/r:a.py@<blob8>`` — an id the prompt never
    showed, so tool-call id accuracy was structurally 0, ~20 unlearnable
    tokens bore loss per sample, and the free-running evaluator rejected every
    flat call as unsurfaced.
    """
    if "`" in doc_id:
        raise ValueError(f"doc_id may not contain a backtick (scope quoting): {doc_id!r}")
    # Backtick-quoted: ids carry file paths, which may contain spaces/commas.
    scope_description = f"{scope_description}; entrypoint id: `{doc_id}`"
    trajectory = [
        TrajectoryTurn(
            kind="bgkit",
            args={"ids": [doc_id], "query": question},
            response="",
            loss=True,
        ),
        TrajectoryTurn(kind="answer", args={}, response=answer, loss=True),
    ]
    return {
        "dataset_name": dataset_name,
        "scope_template": "pre_scoped",
        "scope_description": scope_description,
        "topic_list_json": None,
        "question": question,
        "gold_answer": answer,
        "trajectory_json": trajectory_to_json(trajectory),
        "split": split,
        "group_id": group_id,
        "gold_span_json": gold_span_json,
    }


def write_flat_phase2_dataset(
    *,
    dataset_name: str,
    doc_ids: list[str],
    doc_tokens: list[np.ndarray],
    traj_rows: list[dict],
    mmap_out: str | Path,
    traj_out: str | Path,
    tokenizer_name: str,
) -> dict:
    """Write mmap article store + trajectory parquet. Returns the manifest."""
    assert doc_ids, "no articles produced"
    assert len(set(doc_ids)) == len(doc_ids), "duplicate article doc_ids"
    assert traj_rows, "no trajectories produced"

    lengths = np.asarray([len(t) for t in doc_tokens], dtype=np.int64)
    tokens = np.concatenate(doc_tokens).astype(np.int32)
    offsets = build_csr_offsets(lengths)
    metadata = pa.table(
        {
            "id": pa.array(range(len(doc_ids)), type=pa.int64()),
            "document_id": pa.array(doc_ids, type=pa.string()),
            "dataset_name": pa.array([dataset_name] * len(doc_ids), type=pa.string()),
            "tag_list_json": pa.array(["[]"] * len(doc_ids), type=pa.string()),
        }
    )
    mmap_dir = Path(mmap_out) / dataset_name
    manifest = write_mmap_artifacts(
        mmap_dir,
        tokens,
        offsets,
        manifest_extra={"dataset": dataset_name, "tokenizer": tokenizer_name},
        metadata_table=metadata,
    )

    traj_table = pa.table({name: pa.array([r[name] for r in traj_rows]) for name in TRAJ_COLUMNS})
    traj_path = Path(traj_out) / f"{dataset_name}.parquet"
    traj_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(traj_table, traj_path)
    return manifest
