"""Commit extraction with filtering.

Extracts commits from git repos and applies filtering rules to produce
clean training data. Each commit becomes a potential training example.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExtractedCommit:
    """A filtered commit ready for task construction."""

    repo_path: str
    sha: str
    parent_sha: str
    message: str
    diff_paths: list[str]
    diff_hunks: list[str]
    author: str
    timestamp: int


def extract_commits(repo_path: str, max_commits: int | None = None) -> list[ExtractedCommit]:
    """Extract and filter commits from a git repo.

    Args:
        repo_path: Path to the git repo.
        max_commits: Maximum number of commits to extract.

    Returns:
        List of filtered commits.
    """
    # TODO: Implement using gitpython/pygit2 + commit_filters
    raise NotImplementedError
