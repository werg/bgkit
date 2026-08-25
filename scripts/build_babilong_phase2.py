#!/usr/bin/env python
"""Build a BABILong evaluation set as a flat Phase-2 dataset (the bgkit arm).

The stock-model arms live in ``scripts/baseline_babilong.py``: ``full`` (whole
context, when it fits) and ``truncate`` (keep the LAST N tokens — the naive
budget baseline). This script produces the third arm: the same BABILong
samples, but with the haystack handed to the model as *compressed survivor
reps* through the ordinary flat single-``bgkit`` trajectory the wide-net runs
already train on (``bgkit.data.flat_phase2_writer``), so no new trainer or
eval path is needed —
``scripts/eval_phase2_kb.py +eval.datasets=[babilong_<suffix>]`` runs it and
``scripts/score_babilong_bgkit.py`` scores the resulting per-sample
predictions with BABILong's own ``compare_answers``.

The comparison that matters is **budget matched**: a 32k-token context at
~6x retention lands ~4k reps in the decoder, which is exactly the
``truncate`` arm's 4k token budget. Same decoder, same budget, different
selection mechanism.

Prompt parity with the baseline: the question text carries BABILong's own
``instruction`` / ``examples`` / ``post_prompt`` for the task, so the only
difference between the arms is how the haystack reaches the decoder.

Usage (host venv is fine — CPU only), with
``BL=$NVME/capability_packaging/benchmarks/babilong``:
    .venv/bin/python scripts/build_babilong_phase2.py \
      --data-dir $BL/babilong-1k-samples --repo-dir $BL/babilong-repo \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories \
      --tasks qa1 qa2 --lengths 16k 32k --n-samples 100
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset


def compose_question(cfg: dict, question: str, *, with_examples: bool) -> str:
    """BABILong's own instruction / examples / post-prompt around the question.

    Mirrors babilong's ``DEFAULT_TEMPLATE`` —
    ``{instruction}\\n\\n{examples}\\n\\n{post_prompt}\\n\\n<context>…</context>
    \\n\\nQuestion: {question}`` — with the ``<context>`` block removed,
    because here the context is spliced in as reps by the trainer. Order and
    the ``Question:`` prefix are kept verbatim: they are what makes the bgkit
    arm comparable to the ``full`` / ``truncate`` arms.
    """
    parts = [str(cfg.get("instruction") or "").strip()]
    if with_examples:
        parts.append(str(cfg.get("examples") or "").strip())
    parts.append(str(cfg.get("post_prompt") or "").strip())
    parts.append(f"Question: {question.strip()}")
    return "\n\n".join(p for p in parts if p)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", required=True, help="babilong-1k-samples root")
    ap.add_argument("--repo-dir", required=True, help="babilong repo (for prompts)")
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2"])
    ap.add_argument("--lengths", nargs="+", default=["16k", "32k"])
    ap.add_argument("--n-samples", type=int, default=100)
    ap.add_argument(
        "--dataset-prefix",
        default="babilong",
        help="one dataset per cell, named <prefix>_<task>_<length>",
    )
    ap.add_argument("--no-examples", action="store_true", help="drop the few-shot examples")
    ap.add_argument(
        "--train-filler",
        type=int,
        default=4,
        help="rows past --n-samples marked split=train ONLY to satisfy the "
        "trainer's non-empty-train-partition contract. Never trained on: this "
        "is an eval-only benchmark set, and the eval arm reads the eval split.",
    )
    ap.add_argument(
        "--max-article-tokens",
        type=int,
        default=0,
        help="truncate the haystack to the LAST N tokens (0 = no truncation). "
        "Non-zero makes the arm a compressed-truncation hybrid — report it.",
    )
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    from babilong.prompts import DEFAULT_PROMPTS
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    # One dataset per (task, length) cell: the eval report's own per-dataset
    # breakdown then carries the cell identity, so nothing downstream has to
    # thread a task label through the trainer's per-sample rows.
    created: list[str] = []

    for task in args.tasks:
        cfg = DEFAULT_PROMPTS[task]
        for length in args.lengths:
            path = Path(args.data_dir) / length / f"{task}-00000-of-00001.parquet"
            if not path.exists():
                print(f"skip missing {path}")
                continue
            dataset_name = f"{args.dataset_prefix}_{task}_{length}"
            doc_ids: list[str] = []
            doc_tokens: list[np.ndarray] = []
            traj_rows: list[dict] = []
            token_lengths: list[int] = []
            rows = pq.read_table(path).to_pylist()[: args.n_samples + max(0, args.train_filler)]
            for i, row in enumerate(rows):
                split = "eval" if i < args.n_samples else "train"
                ids = tokenizer.encode(row["input"], add_special_tokens=False)
                if args.max_article_tokens and len(ids) > args.max_article_tokens:
                    ids = ids[-args.max_article_tokens :]
                doc_id = f"babilong:{task}:{length}:{i:04d}"
                doc_ids.append(doc_id)
                doc_tokens.append(np.asarray(ids, dtype=np.int32))
                token_lengths.append(len(ids))
                traj_rows.append(
                    flat_trajectory_row(
                        dataset_name=dataset_name,
                        doc_id=doc_id,
                        question=compose_question(
                            cfg, row["question"], with_examples=not args.no_examples
                        ),
                        # L0 selection is query-conditioned: give it the BARE
                        # question. The composed prompt's few-shot examples
                        # name every answer label, which would pull survivors
                        # toward the wrong locations.
                        query=row["question"].strip(),
                        answer=str(row["target"]).strip(),
                        scope_description=f"a long story transcript ({task}, {length} context)",
                        split=split,
                        # Group = the cell, so the trainer's repo-leakage guard
                        # sees train and eval share a group. They deliberately
                        # do (same task/length, disjoint samples) — group_id is
                        # per-sample here so the guard stays meaningful.
                        group_id=doc_id,
                        gold_span_json=None,
                    )
                )

            manifest = write_flat_phase2_dataset(
                dataset_name=dataset_name,
                doc_ids=doc_ids,
                doc_tokens=doc_tokens,
                traj_rows=traj_rows,
                mmap_out=args.mmap_out,
                traj_out=args.traj_out,
                tokenizer_name=args.tokenizer,
            )
            created.append(dataset_name)
            # Sidecar so the scorer can recover the BARE question from the
            # composed prompt. BABILong's ``compare_answers`` excludes any
            # label already named in the question — and the few-shot EXAMPLES
            # name every label, so scoring against the composed string would
            # mark every sample wrong.
            (Path(args.traj_out) / f"{dataset_name}.babilong_meta.json").write_text(
                json.dumps(
                    {
                        "task": task,
                        "length": length,
                        "instruction": str(cfg.get("instruction") or "").strip(),
                        "examples": (
                            "" if args.no_examples else str(cfg.get("examples") or "").strip()
                        ),
                        "post_prompt": str(cfg.get("post_prompt") or "").strip(),
                        "max_article_tokens": int(args.max_article_tokens),
                    },
                    indent=2,
                )
            )
            lens = np.asarray(token_lengths)
            print(
                f"{dataset_name}: {manifest['row_count']} articles, "
                f"median {int(np.median(lens)):,} tokens (max {int(lens.max()):,}); "
                f"reps at 6x (0.35*0.5) ~{int(np.median(lens) * 0.175):,}, "
                f"at 67x ~{int(np.median(lens) * 0.015):,}",
                flush=True,
            )

    print("datasets:", json.dumps(created))


if __name__ == "__main__":
    main()
