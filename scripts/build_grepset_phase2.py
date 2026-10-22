#!/usr/bin/env python
"""Build the ``grepset`` Phase-2 flat dataset: wide search-result QA.

Family B "search/grep result sets" (`plans/capability_packaging_2026_08_20.md`
§5.2): each article simulates a repo-wide ``grep -rn`` result — matching
lines with ``path:lineno:`` prefixes across many files, widened with matches
of a common pattern so the tool result is large and distractor-heavy.
Questions: which file defines a symbol, quote the matching line, and
not-found negatives. Split = held-out repos.

Usage:
    .venv/bin/python scripts/build_grepset_phase2.py \
      --repos-dir $DATA_DIR/repos \
      --mmap-out $DATA_DIR/mmap/phase2 --traj-out $DATA_DIR/trajectories
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np

from bgkit.data.file_needle_qa import defined_symbols
from bgkit.data.flat_phase2_writer import flat_trajectory_row, write_flat_phase2_dataset
from bgkit.data.gold_span import answer_span_from_offsets, article_offsets, span_to_json
from bgkit.data.repo_files import iter_repo_files

DATASET_NAME = "grepset"
COMMON_PATTERNS = ["def ", "class ", "import ", "return ", "function ", "const "]


def grep_lines(files: list[tuple[str, str]], needle: str) -> list[str]:
    out = []
    for path, text in files:
        for i, line in enumerate(text.splitlines(), 1):
            if needle in line:
                out.append(f"{path}:{i}:{line.rstrip()}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repos-dir", required=True)
    ap.add_argument("--mmap-out", required=True)
    ap.add_argument("--traj-out", required=True)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--n-repos", type=int, default=400)
    ap.add_argument("--files-per-repo", type=int, default=20)
    ap.add_argument("--max-article-tokens", type=int, default=45_000)
    ap.add_argument("--eval-repo-fraction", type=float, default=0.08)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--max-answer-chars", type=int, default=400)
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    rng = random.Random(args.seed)

    roots = sorted(Path(args.repos_dir).iterdir())
    repos: list[Path] = []
    for owner in roots:
        if owner.is_dir():
            repos.extend(p for p in sorted(owner.iterdir()) if p.is_dir())
    rng.shuffle(repos)

    doc_ids: list[str] = []
    doc_tokens: list[np.ndarray] = []
    traj_rows: list[dict] = []
    stats: dict[str, int] = {}
    used = 0

    for repo_path in repos:
        if used >= args.n_repos:
            break
        repo_label = f"{repo_path.parent.name}/{repo_path.name}"
        digest = int(hashlib.sha256(repo_label.encode()).hexdigest(), 16)
        split = "eval" if (digest % 1000) < int(args.eval_repo_fraction * 1000) else "train"

        files = [
            (path, text)
            for path, _sha, text in iter_repo_files(
                repo_path, min_bytes=2_000, max_bytes=120_000, max_files=args.files_per_repo
            )
        ]
        if len(files) < 5:
            continue

        # symbols defined in exactly one file across the set
        sym_to_file: dict[str, str] = {}
        sym_seen: dict[str, int] = {}
        for path, text in files:
            for sym in defined_symbols(text):
                sym_seen[sym] = sym_seen.get(sym, 0) + 1
                sym_to_file[sym] = path
        rare_syms = [s for s, c in sym_seen.items() if c == 1 and len(s) >= 5]
        if not rare_syms:
            continue
        rng.shuffle(rare_syms)

        # article: wide grep output = common-pattern matches + the rare needles
        pattern = rng.choice(COMMON_PATTERNS)
        wide = grep_lines(files, pattern)
        needles = rare_syms[:3]
        needle_hits = {s: grep_lines(files, s) for s in needles}
        body_lines = wide + [ln for hits in needle_hits.values() for ln in hits]
        rng.shuffle(body_lines)
        article = f"$ grep -rn {pattern!r} + symbol search results\n" + "\n".join(body_lines)
        ids = tokenizer.encode(article, add_special_tokens=False)
        if not (1_000 <= len(ids) <= args.max_article_tokens):
            continue
        doc_id = f"grep:{repo_label}"  # one grep article per repo; no hash needed
        doc_ids.append(doc_id)
        doc_tokens.append(np.asarray(ids, dtype=np.int32))
        used += 1
        offsets = article_offsets(tokenizer, article)

        def add(
            question,
            answer,
            qtype,
            *,
            _doc=doc_id,
            _repo=repo_label,
            _split=split,
            _offsets=offsets,
            _article=article,
        ):
            span = (
                None if qtype == "absent" else answer_span_from_offsets(_offsets, _article, answer)
            )
            traj_rows.append(
                flat_trajectory_row(
                    dataset_name=DATASET_NAME,
                    doc_id=_doc,
                    question=question,
                    answer=answer,
                    scope_description=f"search results across {_repo}",
                    split=_split,
                    group_id=_repo,
                    gold_span_json=span_to_json(span),
                )
            )
            stats[f"{_split}:{qtype}"] = stats.get(f"{_split}:{qtype}", 0) + 1

        for sym in needles:
            hits = needle_hits[sym]
            if not hits:
                continue
            deff = [h for h in hits if re.search(r"\b(def|class|function|fn|func)\b", h)]
            target = deff[0] if deff else hits[0]
            add(
                f"According to these search results, which file defines `{sym}`?",
                sym_to_file[sym],
                "which_file",
            )
            # A quote target is ONE human-scale result line (minified sources
            # are already rejected by iter_repo_files; this guards data lines).
            if len(target) <= args.max_answer_chars:
                add(
                    f"Quote the search-result line where `{sym}` is defined.",
                    target,
                    "quote_line",
                )
        # not-found negative: a symbol from a different repo (unlikely present)
        fake = f"{rng.choice(rare_syms)}_{rng.randint(100, 999)}x"
        if fake not in article:
            add(
                f"Do these search results contain any match for `{fake}`?",
                f"No — there is no match for `{fake}` in these results.",
                "absent",
            )

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
        f"mmap: {manifest['row_count']} grep articles, {manifest['total_tokens']:,} tokens; "
        f"trajectories: {len(traj_rows)}"
    )
    print("by split:qtype:", json.dumps(stats, sort_keys=True))


if __name__ == "__main__":
    main()
