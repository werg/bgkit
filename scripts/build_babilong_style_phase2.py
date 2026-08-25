#!/usr/bin/env python
"""Build ``babilong_style``: long-prose semantic fact recall training data.

The capability BABILong measures — find the handful of fact sentences that
answer a question inside a long stream of unrelated prose — is exactly what
the wide-net selection head is for, and unlike verbatim needle extraction it
is *semantic*, which the oracle-span and v7 verdicts showed is the side of
the line bgkit can actually work on. This builds training data for it.

Contamination boundary, deliberately wide:

- **Signal**: bAbI's ``en-10k`` **train** split. BABILong's benchmark samples
  come from the bAbI **test** split; nothing here overlaps them.
- **Noise**: FineWeb-Edu token shards. BABILong buries its facts in PG19 book
  text, so the model cannot learn the benchmark's noise distribution here —
  it has to learn the task.
- **Prompt**: identical composition to the eval arm
  (``scripts/build_babilong_phase2.py``), and the answer is rendered in the
  sentence BABILong's ``post_prompt`` demands, so there is no train/eval
  prompt mismatch.

Gold spans come out exact and free: we know where we inserted each fact, so
the span-relevance loss can supervise selection directly on the supporting
sentence that names the answer.

Usage (host venv is fine — CPU only), with
``BL=$NVME/capability_packaging/benchmarks/babilong``:
    .venv/bin/python scripts/build_babilong_style_phase2.py \
      --babi-zip $BL/babilong-repo/data/tasks_1-20_v1-2.zip \
      --repo-dir $BL/babilong-repo \
      --noise-dir /home/werg/bgkit-data-nvme/fineweb_edu_v1/tokens \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories \
      --n-train 3000 --n-eval 200
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from bgkit.data.babi_tasks import (
    SUPPORTED_TASKS,
    build_haystack,
    format_answer,
    read_babi_zip,
    word_start_mask,
)
from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import span_to_json

DATASET_NAME = "babilong_style"


class NoiseStream:
    """Endless-ish stream of FineWeb-Edu document token arrays."""

    def __init__(self, noise_dir: str, rng: random.Random) -> None:
        self._shards = sorted(Path(noise_dir).glob("shard_*.parquet"))
        if not self._shards:
            raise FileNotFoundError(f"no shard_*.parquet under {noise_dir}")
        self._rng = rng
        self._shard_i = 0
        self._buf: list[np.ndarray] = []
        self._buf_i = 0

    def _refill(self) -> None:
        if self._shard_i >= len(self._shards):
            # Wrap: noise is a distractor, so reuse is acceptable, but a fresh
            # shuffle keeps the same document from landing in the same place.
            self._shard_i = 0
        table = pq.read_table(self._shards[self._shard_i], columns=["token_ids"])
        self._shard_i += 1
        self._buf = [np.asarray(x, dtype=np.int32) for x in table.column("token_ids").to_pylist()]
        self._rng.shuffle(self._buf)
        self._buf_i = 0

    def take(self, n_tokens: int) -> np.ndarray:
        """Concatenated noise documents totalling at least ``n_tokens``."""
        out: list[np.ndarray] = []
        have = 0
        while have < n_tokens:
            if self._buf_i >= len(self._buf):
                self._refill()
            doc = self._buf[self._buf_i]
            self._buf_i += 1
            if doc.size == 0:
                continue
            out.append(doc)
            have += int(doc.size)
        return np.concatenate(out)[:n_tokens]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--babi-zip", required=True)
    ap.add_argument("--repo-dir", required=True, help="babilong repo (for prompts)")
    ap.add_argument("--noise-dir", required=True, help="FineWeb-Edu token shard dir")
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--tasks", nargs="+", default=list(SUPPORTED_TASKS))
    ap.add_argument("--n-train", type=int, default=3000)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument(
        "--lengths",
        nargs="+",
        type=int,
        default=[4000, 8000, 16000, 32000],
        help="noise-token targets sampled per document (the haystack length)",
    )
    ap.add_argument("--no-examples", action="store_true")
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    sys.path.insert(0, args.repo_dir)
    from babilong.prompts import DEFAULT_PROMPTS
    from transformers import AutoTokenizer

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_babilong_phase2 import compose_question

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)
    allowed = word_start_mask(tokenizer)
    noise_stream = NoiseStream(args.noise_dir, random.Random(args.seed + 1))

    # Per task, keep the first n_train questions for train and a DISJOINT
    # later slice for the in-distribution eval split. (The real eval remains
    # the BABILong benchmark arm; this one only tracks fit during training.)
    per_task = max(1, (args.n_train + args.n_eval) // max(1, len(args.tasks)))
    plan: list[tuple[str, object, str]] = []
    for task in args.tasks:
        samples = read_babi_zip(args.babi_zip, task, split="train")
        rng.shuffle(samples)
        n_tr = int(per_task * args.n_train / max(1, args.n_train + args.n_eval))
        for i, s in enumerate(samples[:per_task]):
            plan.append((task, s, "train" if i < n_tr else "eval"))
    rng.shuffle(plan)

    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}
    skipped = 0

    for idx, (task, sample, split) in enumerate(plan):
        cfg = DEFAULT_PROMPTS[task]
        target = rng.choice(args.lengths)
        runs = [
            np.asarray(tokenizer.encode(" " + fact, add_special_tokens=False), dtype=np.int32)
            for fact in sample.facts
        ]
        try:
            haystack, spans = build_haystack(noise_stream.take(target), runs, allowed, rng)
        except ValueError:
            skipped += 1
            continue
        # Span supervision should point at the sentence that CONTAINS the
        # answer word. For qa1 that is the single supporting fact; for qa2
        # the two supporting facts split the job ("Sandra left the apple." /
        # "Sandra journeyed to the hallway.") and only the second names the
        # location, so pick by content rather than by position.
        gold = None
        if sample.supporting:
            answer_low = sample.answer.lower()
            pick = next(
                (i for i in sample.supporting if answer_low in sample.facts[i].lower()),
                max(sample.supporting),
            )
            gold = spans[pick]

        doc_id = f"babilong_style:{task}:{idx:06d}"
        doc_ids.append(doc_id)
        doc_tokens.append(haystack)
        traj_rows.append(
            flat_trajectory_row(
                dataset_name=DATASET_NAME,
                doc_id=doc_id,
                question=compose_question(
                    cfg, sample.question, with_examples=not args.no_examples
                ),
                answer=format_answer(task, sample.question, sample.answer),
                scope_description=f"a long story transcript ({task})",
                split=split,
                group_id=doc_id,
                gold_span_json=span_to_json(gold),
                # Selection is query-conditioned: the bare question only.
                query=sample.question.strip(),
            )
        )
        key = f"{split}:{task}"
        stats[key] = stats.get(key, 0) + 1

    manifest = write_flat_phase2_dataset(
        dataset_name=DATASET_NAME,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        traj_rows=traj_rows,
        mmap_out=args.mmap_out,
        traj_out=args.traj_out,
        tokenizer_name=args.tokenizer,
    )
    lens = np.asarray([t.size for t in doc_tokens])
    print(
        f"{DATASET_NAME}: {manifest['row_count']} documents, "
        f"{manifest['total_tokens']:,} tokens "
        f"(median {int(np.median(lens)):,}, max {int(lens.max()):,}); "
        f"{len(traj_rows)} trajectories, {skipped} skipped"
    )
    print("by split:task:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
