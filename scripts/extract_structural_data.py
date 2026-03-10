#!/usr/bin/env python3
"""Extract structural metadata from repos using tree-sitter.

Walks data/repos/{owner}/{repo}/ directories, parses files with tree-sitter,
builds dependency graphs, and serializes three structural views (skeleton,
dependency, module_summary) into per-repo JSONL files.

Usage:
    python scripts/extract_structural_data.py \
        --repos-dir data/repos/ \
        --output-dir data/structural/ \
        --workers 8
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
import traceback
from concurrent.futures import BrokenExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path, PurePosixPath

import structlog

from bgkit.data.repo_processing import extract_repo_snapshot
from bgkit.data.structural.dependency_graph import build_dependency_graph
from bgkit.data.structural.parser import LANGUAGE_MAP, TIER_A_LANGUAGES, parse_repo
from bgkit.data.structural.serializer import (
    serialize_dependencies,
    serialize_module_summary,
    serialize_skeleton,
)

logger = structlog.get_logger()


def collect_repo_paths(
    repos_dir: Path,
    max_repos: int | None = None,
    shuffle_seed: int | None = None,
) -> list[Path]:
    """Collect all repo paths, optionally shuffled to avoid alphabetical bias."""
    from bgkit.utils.git_utils import collect_repo_paths as _collect

    return _collect(repos_dir, max_repos=max_repos, shuffle_seed=shuffle_seed)


def _get_module_paths(skeletons: list) -> list[str]:
    """Collect unique parent directories that contain parsed files."""
    dirs: set[str] = set()
    for skel in skeletons:
        parent = str(PurePosixPath(skel.path).parent)
        if parent and parent != ".":
            dirs.add(parent)
    return sorted(dirs)


def process_single_repo(
    repo_path: Path,
    output_dir: Path,
) -> dict:
    """Process a single repo: extract snapshot, parse, serialize, write JSONL.

    Returns a stats dict with counts for logging.
    """
    owner = repo_path.parent.name
    repo_name = repo_path.name
    rel_key = f"{owner}/{repo_name}"

    output_file = output_dir / f"{rel_key}.jsonl"

    # Skip if already done
    if output_file.exists():
        return {"repo": rel_key, "status": "skipped", "files": 0, "records": 0}

    try:
        # Extract files from HEAD
        snapshot = extract_repo_snapshot(str(repo_path))
        if not snapshot.files:
            return {"repo": rel_key, "status": "empty", "files": 0, "records": 0}

        # Parse all files with tree-sitter
        skeletons = parse_repo(snapshot)
        if not skeletons:
            return {"repo": rel_key, "status": "no_parseable", "files": len(snapshot.files),
                    "records": 0}

        # Build dependency graph
        file_paths = [f.path for f in snapshot.files]
        dep_graph = build_dependency_graph(skeletons, file_paths)

        # Collect module paths for module_summary
        module_paths = _get_module_paths(skeletons)

        # Build records
        records: list[dict] = []
        commit_sha = snapshot.commit_sha

        for skel in skeletons:
            # Skeleton view
            skeleton_text = serialize_skeleton(skel)
            if skeleton_text.strip():
                records.append({
                    "file_path": skel.path,
                    "commit_sha": commit_sha,
                    "language": skel.language,
                    "skeleton_text": skeleton_text,
                    "dependency_text": "",
                    "module_summary_text": "",
                })

            # Dependency view
            dep_text = serialize_dependencies(skel, dep_graph)
            if dep_text.strip():
                records.append({
                    "file_path": skel.path,
                    "commit_sha": commit_sha,
                    "language": skel.language,
                    "skeleton_text": "",
                    "dependency_text": dep_text,
                    "module_summary_text": "",
                })

        # Module summaries (one per directory)
        for mod_path in module_paths:
            mod_text = serialize_module_summary(skeletons, mod_path)
            if mod_text.strip():
                records.append({
                    "file_path": mod_path,
                    "commit_sha": commit_sha,
                    "language": "",
                    "skeleton_text": "",
                    "dependency_text": "",
                    "module_summary_text": mod_text,
                })

        if not records:
            return {"repo": rel_key, "status": "no_records", "files": len(snapshot.files),
                    "records": 0}

        # Atomic write: write to tmp, then rename
        output_file.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_path = tempfile.mkstemp(
            dir=str(output_file.parent), suffix=".jsonl.tmp"
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                for rec in records:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.rename(tmp_path, str(output_file))
        except BaseException:
            # Clean up temp file on failure
            with contextlib.suppress(OSError):
                os.unlink(tmp_path)
            raise

        # Compute tier stats
        tier_a = sum(1 for s in skeletons if s.tier == "A")
        tier_b = sum(1 for s in skeletons if s.tier == "B")

        return {
            "repo": rel_key,
            "status": "ok",
            "files": len(snapshot.files),
            "parsed": len(skeletons),
            "records": len(records),
            "tier_a": tier_a,
            "tier_b": tier_b,
        }

    except Exception:
        tb = traceback.format_exc()
        return {"repo": rel_key, "status": "error", "error": tb,
                "files": 0, "records": 0}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract structural metadata from repos using tree-sitter."
    )
    from bgkit.env import get_data_dir
    _data = get_data_dir(must_exist=False)
    parser.add_argument(
        "--repos-dir", type=Path, default=_data / "repos",
        help="Directory containing owner/repo/ subdirectories",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=_data / "structural",
        help="Output directory for per-repo JSONL files",
    )
    parser.add_argument(
        "--max-repos", type=int, default=None,
        help="Process at most this many repos",
    )
    parser.add_argument(
        "--workers", type=int, default=min(os.cpu_count() or 4, 8),
        help="Number of parallel workers (default: min(cpu_count, 8))",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="Seed for shuffling repo order (avoids alphabetical bias on partial runs)",
    )
    args = parser.parse_args()

    structlog.configure(
        processors=[
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )

    if not args.repos_dir.is_dir():
        print(f"ERROR: {args.repos_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    repo_paths = collect_repo_paths(args.repos_dir, args.max_repos, shuffle_seed=args.shuffle_seed)
    logger.info("collected_repos", count=len(repo_paths))

    # Track stats
    total_ok = 0
    total_skipped = 0
    total_errors = 0
    total_tier_a = 0
    total_tier_b = 0
    total_unsupported = 0
    total_files = 0
    total_parsed = 0

    done_count = 0
    remaining = list(repo_paths)

    while remaining:
        batch_done: set[Path] = set()
        try:
            with ProcessPoolExecutor(max_workers=args.workers) as executor:
                futures = {
                    executor.submit(process_single_repo, rp, args.output_dir): rp
                    for rp in remaining
                }

                for future in as_completed(futures):
                    repo_path = futures[future]
                    try:
                        result = future.result()
                    except BrokenExecutor:
                        # Worker process crashed (segfault, OOM). The executor is
                        # dead — break out and restart with remaining repos.
                        logger.error(
                            "worker_crash",
                            repo=f"{repo_path.parent.name}/{repo_path.name}",
                            msg="Worker process died; restarting pool for remaining repos",
                        )
                        total_errors += 1
                        batch_done.add(repo_path)
                        raise  # exit the for-loop via the outer except
                    except Exception as exc:
                        logger.error(
                            "future_error",
                            repo=f"{repo_path.parent.name}/{repo_path.name}",
                            error=str(exc)[:200],
                        )
                        total_errors += 1
                        batch_done.add(repo_path)
                        continue

                    batch_done.add(repo_path)
                    status = result["status"]

                    if status == "ok":
                        total_ok += 1
                        total_tier_a += result.get("tier_a", 0)
                        total_tier_b += result.get("tier_b", 0)
                        total_files += result.get("files", 0)
                        total_parsed += result.get("parsed", 0)
                    elif status == "skipped":
                        total_skipped += 1
                    elif status == "error":
                        total_errors += 1
                        logger.error(
                            "repo_failed", repo=result["repo"],
                            error=result.get("error", "")[:200],
                        )

                    done_count += 1
                    if done_count % 100 == 0 or done_count == len(repo_paths):
                        logger.info(
                            "progress",
                            done=done_count,
                            total=len(repo_paths),
                            ok=total_ok,
                            skipped=total_skipped,
                            errors=total_errors,
                        )
        except BrokenExecutor:
            # Pool crashed — remove completed repos and retry the rest
            remaining = [rp for rp in remaining if rp not in batch_done]
            logger.info(
                "pool_restart",
                remaining=len(remaining),
                completed_before_crash=len(batch_done),
            )
            continue

        # Normal completion — all remaining repos processed
        break

    # Compute unsupported as total_files - total_parsed (files that couldn't be parsed)
    total_unsupported = total_files - total_parsed

    # Coverage stats
    total_parsed_files = total_tier_a + total_tier_b
    if total_parsed_files > 0:
        pct_a = 100.0 * total_tier_a / total_parsed_files
        pct_b = 100.0 * total_tier_b / total_parsed_files
    else:
        pct_a = 0.0
        pct_b = 0.0

    pct_unsupported = 100.0 * total_unsupported / total_files if total_files > 0 else 0.0

    logger.info(
        "extraction_complete",
        repos_processed=total_ok,
        repos_skipped=total_skipped,
        repos_errored=total_errors,
        total_files=total_files,
        total_parsed=total_parsed,
        tier_a_files=total_tier_a,
        tier_b_files=total_tier_b,
        unsupported_files=total_unsupported,
    )
    print(f"\nCoverage: {pct_a:.1f}% Tier A, {pct_b:.1f}% Tier B, "
          f"{pct_unsupported:.1f}% unsupported")
    print(f"Tier A languages: {', '.join(sorted(TIER_A_LANGUAGES))}")
    print(f"All mapped languages: {len(LANGUAGE_MAP)}")


if __name__ == "__main__":
    main()
