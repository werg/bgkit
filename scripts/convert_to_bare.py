#!/usr/bin/env python3
"""Convert working-tree git repos to bare repos to save disk space.

Walks the repos directory (owner/repo/ structure), converts each repo
in-place by keeping only the .git contents, then optionally moves the
whole collection to a new destination and creates a symlink.

Usage:
    # Dry run — show what would happen:
    python scripts/convert_to_bare.py data/repos --dry-run

    # Convert in-place:
    python scripts/convert_to_bare.py data/repos

    # Convert and move to external drive, leaving a symlink:
    python scripts/convert_to_bare.py data/repos --move-to /mnt/external/repos
"""

from __future__ import annotations

import argparse
import logging
import shutil
import subprocess
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)


def is_bare_repo(path: Path) -> bool:
    """Check if a directory is already a bare git repo."""
    return (path / "HEAD").is_file() and (path / "objects").is_dir()


def is_worktree_repo(path: Path) -> bool:
    """Check if a directory is a standard (non-bare) git repo."""
    return (path / ".git").is_dir()


def convert_repo_to_bare(repo_path: Path, dry_run: bool = False) -> bool:
    """Convert a single working-tree repo to a bare repo in-place.

    Strategy: move .git/ contents up, remove working tree files.
    This avoids copying git objects (which can be large).

    Returns True if converted, False if skipped.
    """
    if is_bare_repo(repo_path):
        return False  # already bare

    if not is_worktree_repo(repo_path):
        log.warning("Not a git repo, skipping: %s", repo_path)
        return False

    if dry_run:
        log.info("[DRY RUN] Would convert: %s", repo_path)
        return True

    git_dir = repo_path / ".git"

    # 1. Remove all working tree files/dirs (everything except .git/)
    for entry in repo_path.iterdir():
        if entry.name == ".git":
            continue
        if entry.is_symlink() or not entry.is_dir():
            entry.unlink()
        else:
            shutil.rmtree(entry)

    # 2. Move .git/ contents up to repo_path/
    #    Use a temp dir to avoid name collisions during the move.
    tmp = repo_path / ".git_tmp_move"
    git_dir.rename(tmp)

    for entry in tmp.iterdir():
        entry.rename(repo_path / entry.name)

    tmp.rmdir()

    # 3. Mark as bare in config
    config_file = repo_path / "config"
    if config_file.exists():
        text = config_file.read_text()
        text = text.replace("bare = false", "bare = true")
        config_file.write_text(text)

    return True


def collect_repos(repos_dir: Path) -> list[Path]:
    """Collect all owner/repo/ directories."""
    repos = []
    for owner_dir in sorted(repos_dir.iterdir()):
        if not owner_dir.is_dir():
            continue
        for repo_dir in sorted(owner_dir.iterdir()):
            if not repo_dir.is_dir():
                continue
            if is_worktree_repo(repo_dir) or is_bare_repo(repo_dir):
                repos.append(repo_dir)
    return repos


def main():
    parser = argparse.ArgumentParser(description="Convert git repos to bare format")
    parser.add_argument("repos_dir", type=Path, help="Directory containing owner/repo/ structure")
    parser.add_argument(
        "--move-to",
        type=Path,
        default=None,
        help="Move converted repos to this directory and create a symlink at the original location",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen")
    args = parser.parse_args()

    repos_dir = args.repos_dir.resolve()
    if not repos_dir.is_dir():
        log.error("repos_dir does not exist: %s", repos_dir)
        sys.exit(1)

    repos = collect_repos(repos_dir)
    log.info("Found %d repos in %s", len(repos), repos_dir)

    already_bare = sum(1 for r in repos if is_bare_repo(r))
    to_convert = sum(1 for r in repos if is_worktree_repo(r))
    log.info("Already bare: %d, to convert: %d", already_bare, to_convert)

    if args.dry_run:
        log.info("[DRY RUN] Would convert %d repos", to_convert)
        if args.move_to:
            log.info("[DRY RUN] Would move %s -> %s and create symlink", repos_dir, args.move_to)
        return

    # Convert repos
    converted = 0
    failed = 0
    for i, repo_path in enumerate(repos):
        try:
            if convert_repo_to_bare(repo_path):
                converted += 1
            if (i + 1) % 500 == 0:
                log.info("Progress: %d / %d repos processed", i + 1, len(repos))
        except Exception:
            log.error("Failed to convert %s", repo_path, exc_info=True)
            failed += 1

    log.info("Converted %d repos, %d failed, %d were already bare", converted, failed, already_bare)

    # Move to external drive if requested
    if args.move_to:
        dest = args.move_to.resolve()
        if dest.exists():
            log.error("Destination already exists: %s", dest)
            log.error("Remove it first or choose a different path.")
            sys.exit(1)

        dest.parent.mkdir(parents=True, exist_ok=True)

        log.info("Moving %s -> %s (this may take a while)...", repos_dir, dest)
        # Use mv for efficiency (same-fs = instant rename, cross-fs = copy+delete)
        subprocess.run(["mv", str(repos_dir), str(dest)], check=True)

        # Create symlink at original location
        repos_dir.symlink_to(dest)
        log.info("Created symlink: %s -> %s", repos_dir, dest)

    log.info("Done!")


if __name__ == "__main__":
    main()
