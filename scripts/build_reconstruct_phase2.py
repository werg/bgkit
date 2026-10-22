#!/usr/bin/env python
"""Build the ``reconstruct`` Phase-2 flat dataset from the bare-repo corpus.

The information-forcing family: each article is one real source file, and the
answer is a VERBATIM span of it that appears nowhere in the prompt. See
:mod:`bgkit.data.reconstruct_qa` for why the wide-net QA families could not
force the encoder to carry document content, and what each design choice here
is protecting against.

Shares the corpus and the split rule with ``fileneedle`` -- same repos, same
held-out-REPOS hash -- so the two families are directly comparable and a repo
in one family's eval split is never in the other's train split.

Usage:
    .venv/bin/python scripts/build_reconstruct_phase2.py \
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

from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.reconstruct_qa import generate_reconstruct_samples
from bgkit.data.repo_files import iter_repo_files

DATASET_NAME = "reconstruct"


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
    ap.add_argument("--samples-per-file", type=int, default=4)
    # Same 0.08 held-out-repo fraction and the same hash as fileneedle, so the
    # splits coincide rather than leaking across families.
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
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}
    answer_chars: list[int] = []
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
            samples = generate_reconstruct_samples(
                text,
                source_label=f"{repo_label}:{path}",
                rng=rng,
                n_samples=args.samples_per_file,
            )
            if not samples:
                continue
            got_any = True
            del blob8  # one blob per (repo, path) at HEAD; the sha never disambiguated
            doc_id = f"file:{repo_label}:{path}"
            doc_ids.append(doc_id)
            doc_tokens.append(np.asarray(ids, dtype=np.int32))
            offsets = article_offsets(tokenizer, text)
            for s in samples:
                span = answer_span_from_offsets(offsets, text, s.span_text)
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
                answer_chars.append(len(s.answer))
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
    chars = sorted(answer_chars)
    print(
        f"mmap: {manifest['row_count']} files, {manifest['total_tokens']:,} tokens; "
        f"trajectories: {len(traj_rows)} from {used_repos} repos"
    )
    if chars:
        print(
            f"answer chars: min {chars[0]} p50 {chars[len(chars) // 2]} "
            f"p95 {chars[int(len(chars) * 0.95)]} max {chars[-1]}"
        )
    print("by split:qtype:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
