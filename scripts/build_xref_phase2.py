#!/usr/bin/env python
"""Build the ``xref`` Phase-2 flat dataset: anchored cross-reference QA.

The formulation that escapes the guessability tension every other wide-net
family hits -- see :mod:`bgkit.data.xref_qa` for the measurements that
motivate it. Short answers (a symbol name, ~15 chars) so they survive
compression, drawn from a per-document vocabulary so no prior covers them,
and never named in the question so echoing cannot reach them.

THE CORPUS FILTERS RUN HERE, not in the generator: whether ``__init__`` is a
guessable answer is a property of the corpus, invisible to a generator
looking at one file. Candidates are collected across every document first,
filtered, and only then written.

Shares the corpus and the held-out-REPOS split rule with fileneedle and
reconstruct, so a repo in one family's eval split is never in another's
train split.

Usage:
    .venv/bin/python scripts/build_xref_phase2.py \
      --repos-dir $DATA_DIR/repos \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np

from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.repo_files import iter_repo_files
from bgkit.data.xref_qa import filter_candidates, generate_xref_samples

DATASET_NAME = "xref"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-repos", type=int, default=500)
    ap.add_argument("--files-per-repo", type=int, default=5)
    ap.add_argument("--min-bytes", type=int, default=20_000)
    ap.add_argument("--max-bytes", type=int, default=150_000)
    ap.add_argument("--max-file-tokens", type=int, default=45_000)
    ap.add_argument("--per-type", type=int, default=3)
    ap.add_argument("--eval-repo-fraction", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    repos: list[Path] = []
    for owner in sorted(Path(args.repos_dir).iterdir()):
        if owner.is_dir():
            repos.extend(p for p in sorted(owner.iterdir()) if p.is_dir())
    rng.shuffle(repos)

    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    doc_text: dict[str, str] = {}
    doc_split: dict[str, str] = {}
    doc_path: dict[str, str] = {}
    cand: list[dict] = []
    used_repos = 0

    for repo_path in repos:
        if used_repos >= args.n_repos:
            break
        repo_label = f"{repo_path.parent.name}/{repo_path.name}"
        digest = int(hashlib.sha256(repo_label.encode()).hexdigest(), 16)
        split = "eval" if (digest % 1000) < int(args.eval_repo_fraction * 1000) else "train"
        got_any = False
        for path, blob8, text in iter_repo_files(
            repo_path, min_bytes=args.min_bytes, max_bytes=args.max_bytes,
            max_files=args.files_per_repo,
        ):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > args.max_file_tokens or len(ids) < 512:
                continue
            samples = generate_xref_samples(
                text, source_label=f"{repo_label}:{path}", rng=rng,
                max_per_type=args.per_type,
            )
            if not samples:
                continue
            got_any = True
            del blob8
            doc_id = f"file:{repo_label}:{path}"
            doc_ids.append(doc_id)
            doc_tokens.append(np.asarray(ids, dtype=np.int32))
            doc_text[doc_id] = text
            doc_split[doc_id] = split
            doc_path[doc_id] = path
            for s in samples:
                cand.append({
                    "doc_id": doc_id, "qtype": s.qtype, "question": s.question,
                    "answer": s.answer, "span_text": s.span_text,
                    "group_id": repo_label,
                })
        if got_any:
            used_repos += 1

    before = len(cand)
    cand = filter_candidates(cand)
    print(f"guessability filters: {before} -> {len(cand)} candidates "
          f"({before - len(cand)} dropped)")

    # Documents that lost every candidate must not be written: an article no
    # question refers to is dead weight in the mmap and in the browse tree.
    live = {c["doc_id"] for c in cand}
    keep = [i for i, d in enumerate(doc_ids) if d in live]
    doc_ids = [doc_ids[i] for i in keep]
    doc_tokens = [doc_tokens[i] for i in keep]

    traj_rows: list[dict] = []
    stats: Counter = Counter()
    offsets_cache: dict[str, object] = {}
    for c in cand:
        doc_id = c["doc_id"]
        if doc_id not in offsets_cache:
            offsets_cache[doc_id] = article_offsets(tokenizer, doc_text[doc_id])
        span = answer_span_from_offsets(
            offsets_cache[doc_id], doc_text[doc_id], c["span_text"],
        )
        traj_rows.append(flat_trajectory_row(
            dataset_name=DATASET_NAME,
            doc_id=doc_id,
            question=c["question"],
            answer=c["answer"],
            scope_description=f"source file {doc_path[doc_id]}",
            split=doc_split[doc_id],
            group_id=c["group_id"],
            gold_span_json=span_to_json(span),
        ))
        stats[f"{doc_split[doc_id]}:{c['qtype']}"] += 1

    manifest = write_flat_phase2_dataset(
        dataset_name=DATASET_NAME,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        traj_rows=traj_rows,
        mmap_out=args.mmap_out,
        traj_out=args.traj_out,
        tokenizer_name=args.tokenizer,
    )
    answers = [c["answer"] for c in cand]
    uniq = len(set(answers)) / max(len(answers), 1)
    top20 = sum(n for _, n in Counter(answers).most_common(20)) / max(len(answers), 1)
    print(
        f"mmap: {manifest['row_count']} files, {manifest['total_tokens']:,} tokens; "
        f"trajectories: {len(traj_rows)} from {used_repos} repos"
    )
    print(f"answers: {uniq:.1%} unique, top-20 cover {top20:.1%}, "
          f"mean {sum(len(a) for a in answers) / max(len(answers), 1):.0f} chars")
    print("by split:qtype:", json.dumps(dict(sorted(stats.items()))))


if __name__ == "__main__":
    main()
