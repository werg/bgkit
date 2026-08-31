#!/usr/bin/env python
"""Build the ``fileneedle`` Phase-2 flat dataset from the bare-repo corpus.

Family B "large file read + needle" (`plans/capability_packaging_2026_08_20.md`
§5.1): each article is one real source file (a simulated big `read_file` tool
result); questions ask for definition lines, constant values, unique-identifier
lines, and a balanced present/absent presence check. Split = held-out REPOS.

Usage:
    .venv/bin/python scripts/build_fileneedle_phase2.py \
      --repos-dir $DATA_DIR/repos \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from bgkit.data.file_needle_qa import defined_symbols, generate_file_samples
from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.repo_files import iter_repo_files

DATASET_NAME = "fileneedle"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-repos", type=int, default=500)
    ap.add_argument("--files-per-repo", type=int, default=3)
    ap.add_argument("--min-bytes", type=int, default=20_000)
    ap.add_argument("--max-bytes", type=int, default=150_000)
    ap.add_argument("--max-file-tokens", type=int, default=45_000)
    ap.add_argument("--eval-repo-fraction", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=17)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    roots = sorted(Path(args.repos_dir).iterdir())
    repos: list[Path] = []
    for owner in roots:
        if not owner.is_dir():
            continue
        repos.extend(p for p in sorted(owner.iterdir()) if p.is_dir())
    rng.shuffle(repos)

    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}
    absent_pool: list[str] = []
    used_repos = 0

    for repo_path in repos:
        if used_repos >= args.n_repos:
            break
        repo_label = f"{repo_path.parent.name}/{repo_path.name}"
        digest = int(hashlib.sha256(repo_label.encode()).hexdigest(), 16)
        split = "eval" if (digest % 1000) < int(args.eval_repo_fraction * 1000) else "train"
        got_any = False
        for path, blob8, text in iter_repo_files(
            repo_path,
            min_bytes=args.min_bytes,
            max_bytes=args.max_bytes,
            max_files=args.files_per_repo,
        ):
            ids = tokenizer.encode(text, add_special_tokens=False)
            if len(ids) > args.max_file_tokens or len(ids) < 512:
                continue
            samples = generate_file_samples(
                text,
                source_label=f"{repo_label}:{path}",
                rng=rng,
                absent_symbols=rng.sample(absent_pool, min(20, len(absent_pool)))
                if absent_pool
                else None,
            )
            if not samples:
                continue
            got_any = True
            # Readable, content-derived id: one blob per (repo, path) at HEAD,
            # so the blob sha never disambiguated — it was ~8 random hex tokens
            # the decoder had to copy into every tool call (2026-08-23).
            del blob8
            doc_id = f"file:{repo_label}:{path}"
            doc_ids.append(doc_id)
            doc_tokens.append(np.asarray(ids, dtype=np.int32))
            offsets = article_offsets(tokenizer, text)
            for s in samples:
                # Key off span_text, not the qtype string: the answer of a
                # negative presence question is a statement ABOUT the file,
                # not a quotation FROM it, and asking for its span would
                # search the text for a string that is not there.
                span = (
                    None
                    if s.span_text is None
                    else answer_span_from_offsets(offsets, text, s.span_text)
                )
                traj_rows.append(
                    flat_trajectory_row(
                        dataset_name=DATASET_NAME,
                        doc_id=doc_id,
                        question=s.question,
                        answer=s.answer,
                        scope_description=f"source file {path}",
                        split=split,
                        group_id=repo_label,
                        gold_span_json=span_to_json(span),
                    )
                )
                stats[f"{split}:{s.qtype}"] = stats.get(f"{split}:{s.qtype}", 0) + 1
            # feed the absent-symbol pool from this file's defs
            absent_pool.extend(list(defined_symbols(text))[:5])
            if len(absent_pool) > 2000:
                del absent_pool[:1000]
        if got_any:
            used_repos += 1

    manifest = write_flat_phase2_dataset(
        dataset_name=DATASET_NAME,
        doc_ids=doc_ids,
        doc_tokens=doc_tokens,
        traj_rows=traj_rows,
        mmap_out=args.mmap_out,
        traj_out=args.traj_out,
        tokenizer_name=args.tokenizer,
    )
    print(
        f"mmap: {manifest['row_count']} files, {manifest['total_tokens']:,} tokens; "
        f"trajectories: {len(traj_rows)} from {used_repos} repos"
    )
    print("by split:qtype:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
