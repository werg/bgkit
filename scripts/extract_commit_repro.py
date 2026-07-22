#!/usr/bin/env python
"""Extract oldest-first commits per repo for the ``git_commit_repro`` dataset.

Per repo (``$DATA_DIR/repos/<owner>/<repo>``) we walk commits OLDEST-first
(chronological ascending from the root commit) and keep commits until the
cumulative diff-token count reaches a global per-repo token budget ``B``. ``B``
is chosen so the AVERAGE commit count across repos ≈ 64; the realized count
``N`` varies per repo because commit diffs vary in size.

Budget mechanism (see ``bgkit.data.commit_repro``):

1. Walk each repo (up to ``--max-walked`` commits), tokenizing each commit's
   diff. Record per-commit diff-token counts.
2. ``M`` = global median per-commit diff tokens across the calibration sample.
3. ``B = 64 * M`` (override with ``--budget``).
4. Per repo, take oldest commits until the cumulative diff-token count ≥ ``B``
   (always keep ≥ 1; cap at ``--max-commits``).

For FILE-STATE RECONSTRUCTION, per (commit, file) we ALSO capture the file's
full blob content at that commit (the gold) and mark it a valid TARGET when the
blob is decodable and within ``--max-gold-tokens``.

Output: a JSONL of commit records consumed by
``scripts/convert_commit_repro_to_mmap.py``,
``scripts/build_commit_repro_tree.py``, and
``scripts/build_commit_repro_trajectories.py``::

    {"repo", "sha", "ordinal", "window_idx", "message", "timestamp",
     "n_diff_tokens",
     "file_changes": [{"file_idx", "path", "diff_text", "n_tokens",
                       "blob_text", "n_blob_tokens", "is_target"}, ...]}
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
from pathlib import Path

_src = str(Path(__file__).resolve().parent.parent / "src")
if _src not in sys.path:
    sys.path.insert(0, _src)

from bgkit.data.commit_repro import (
    DATASET_NAME,
    FileChange,
    ReproCommit,
    diff_token_budget_from_median,
    summarize_distribution,
    walk_repo_commits_oldest_first,
)

log = logging.getLogger(__name__)


def _discover_repos(repos_dir: Path, max_repos: int | None) -> list[tuple[str, Path]]:
    """Return ``(repo_id, path)`` for every ``<owner>/<repo>`` git dir."""
    out: list[tuple[str, Path]] = []
    for owner in sorted(p for p in repos_dir.iterdir() if p.is_dir()):
        for repo in sorted(p for p in owner.iterdir() if p.is_dir()):
            if (repo / "HEAD").exists() or (repo / ".git").exists():
                out.append((f"{owner.name}/{repo.name}", repo))
                if max_repos is not None and len(out) >= max_repos:
                    return out
    return out


def _collect_repo(
    repo_id: str,
    repo_path: Path,
    tokenizer,
    max_walked: int,
    max_file_tokens: int,
    max_commit_tokens: int,
    max_gold_tokens: int,
) -> list[ReproCommit]:
    """Walk one repo oldest-first; tokenize per-file diffs AND, per (commit,
    file), the file's full blob (the reconstruction gold). No budgeting yet."""
    commits: list[ReproCommit] = []
    ordinal = 0
    for sha, message, ts, files in walk_repo_commits_oldest_first(
        str(repo_path), repo_id, max_walked=max_walked,
    ):
        file_changes: list[FileChange] = []
        total = 0
        skip = False
        for fi, (path, diff_text, blob_text) in enumerate(files):
            ids = tokenizer.encode(diff_text, add_special_tokens=False)
            if len(ids) > max_file_tokens:
                # A single file beyond the (lenient) per-file cap is a
                # generated/vendored blob, not a meaningful change — drop the
                # whole commit (keeps the diff leaves complete, never truncated).
                skip = True
                break
            # Gold blob: a (file, commit) is a valid reconstruction TARGET only
            # when the blob is decodable (not None — i.e. not deleted/binary)
            # AND within the gold-token cap. Larger/absent blobs still leave a
            # diff leaf for navigation in some other file's history.
            n_blob = 0
            is_target = False
            stored_blob = ""
            if blob_text is not None:
                n_blob = len(tokenizer.encode(blob_text, add_special_tokens=False))
                is_target = n_blob <= max_gold_tokens
                if is_target:
                    stored_blob = blob_text  # only persist gold for real targets
            file_changes.append(FileChange(
                file_idx=fi, path=path, diff_text=diff_text, n_tokens=len(ids),
                blob_text=stored_blob, n_blob_tokens=n_blob, is_target=is_target,
            ))
            total += len(ids)
        # Per-commit TOTAL diff cap is the primary gate. The per-file cap stays
        # lenient — one large-but-legitimate file shouldn't drop a whole commit.
        if skip or not file_changes or total > max_commit_tokens:
            continue
        commits.append(ReproCommit(
            repo=repo_id, sha=sha, ordinal=ordinal, message=message,
            timestamp=ts, file_changes=file_changes, n_diff_tokens=total,
        ))
        ordinal += 1
    return commits


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repos-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokenizer", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--max-repos", type=int, default=None)
    parser.add_argument(
        "--repo", action="append", default=None,
        help="Explicit owner/repo to include (repeatable). Overrides discovery.",
    )
    parser.add_argument("--max-walked", type=int, default=4000,
                        help="Max commits to inspect per repo (full-history walk "
                             "for windowing; also the calibration sample).")
    parser.add_argument("--max-windows-per-repo", type=int, default=1,
                        help="Cap on contiguous B-budget windows tiled per repo. "
                             "Default 1 (WINDOW-0-ONLY: the oldest B-budget window). "
                             "Raise to tile more of each repo's history; IMPORTANT "
                             "for balance — uncapped, a 10K-commit repo yields ~150 "
                             "windows and over-weights big repos.")
    parser.add_argument("--max-commits", type=int, default=2048,
                        help="Overall safety cap on commits kept per repo across "
                             "all windows.")
    parser.add_argument("--max-file-tokens", type=int, default=8192,
                        help="Lenient per-file cap: a single file beyond this is "
                             "treated as a generated/vendored blob → drop the commit.")
    parser.add_argument("--max-commit-tokens", type=int, default=16384,
                        help="Primary gate: skip a commit whose TOTAL diff "
                             "exceeds this.")
    parser.add_argument("--max-gold-tokens", type=int, default=8192,
                        help="A (file, commit) is a valid reconstruction TARGET "
                             "only if the file's blob at that commit is ≤ this.")
    parser.add_argument("--budget", type=int, default=None,
                        help="Override B (per-window token budget). "
                             "Default = 64 * median.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    if args.repo:
        repos = [(r, args.repos_dir / r) for r in args.repo]
    else:
        repos = _discover_repos(args.repos_dir, args.max_repos)
    log.info("discovered %d repos", len(repos))

    # Pass 1: collect (tokenized) commits per repo.
    collected: dict[str, list[ReproCommit]] = {}
    all_commit_tokens: list[int] = []
    for repo_id, path in repos:
        try:
            commits = _collect_repo(
                repo_id, path, tokenizer, args.max_walked, args.max_file_tokens,
                args.max_commit_tokens, args.max_gold_tokens,
            )
        except Exception as exc:
            log.warning("repo %s failed: %s", repo_id, exc)
            continue
        if not commits:
            continue
        collected[repo_id] = commits
        all_commit_tokens.extend(c.n_diff_tokens for c in commits)
        log.info("  %s: walked %d commits", repo_id, len(commits))

    if not all_commit_tokens:
        log.error("no commits collected — nothing to write")
        sys.exit(1)

    median_tokens = statistics.median(all_commit_tokens)
    budget = args.budget or diff_token_budget_from_median(median_tokens)
    log.info(
        "calibration: %d commits across %d repos; median per-commit diff "
        "tokens = %.1f; B = %d",
        len(all_commit_tokens), len(collected), median_tokens, budget,
    )

    # Pass 2: tile each repo's full history into contiguous B-budget windows
    # (oldest-first). Window k closes when its cumulative diff-token count ≥ B;
    # the next commit opens window k+1. Stop at --max-windows-per-repo or the
    # overall --max-commits safety cap. The final window may be short (history
    # exhausted mid-window) → a proportionally shallower subtree downstream.
    window_counts: list[int] = []  # windows per repo
    window_commit_counts: list[int] = []  # commits per window
    targets_per_repo: list[int] = []  # (file, commit) reconstruction targets per repo
    gold_token_sizes: list[int] = []  # blob token counts of all targets
    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_rows = 0
    n_targets = 0
    with args.output.open("w") as f:
        for _repo_id, commits in collected.items():
            window_idx = 0
            ord_in_window = 0
            cum = 0
            kept_total = 0
            per_window = [0]
            repo_targets = 0
            for c in commits:
                if kept_total >= args.max_commits:
                    break
                c.window_idx = window_idx
                c.ordinal = ord_in_window
                f.write(json.dumps(c.to_dict()) + "\n")
                n_rows += 1
                for fc in c.file_changes:
                    if fc.is_target:
                        repo_targets += 1
                        n_targets += 1
                        gold_token_sizes.append(fc.n_blob_tokens)
                kept_total += 1  # noqa: SIM113 - also a break guard, not just an index
                ord_in_window += 1
                per_window[-1] += 1
                cum += c.n_diff_tokens
                if cum >= budget:
                    window_idx += 1
                    if window_idx >= args.max_windows_per_repo:
                        break
                    ord_in_window = 0
                    cum = 0
                    per_window.append(0)
            per_window = [w for w in per_window if w > 0]
            window_counts.append(len(per_window))
            window_commit_counts.extend(per_window)
            targets_per_repo.append(repo_targets)

    summary = {
        "dataset": DATASET_NAME,
        "task": "file_state_reconstruction",
        "budget_B_tokens": budget,
        "median_per_commit_diff_tokens": round(median_tokens, 1),
        "max_windows_per_repo": args.max_windows_per_repo,
        "max_gold_tokens": args.max_gold_tokens,
        "n_commit_rows": n_rows,
        "n_targets": n_targets,
        "windows_per_repo": summarize_distribution(window_counts),
        "commits_per_window": summarize_distribution(window_commit_counts),
        "targets_per_repo": summarize_distribution(targets_per_repo),
        "gold_token_sizes": summarize_distribution(gold_token_sizes),
    }
    args.output.with_suffix(".calibration.json").write_text(
        json.dumps(summary, indent=2),
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
