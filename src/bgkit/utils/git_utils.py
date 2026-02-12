"""Git checkout and diff parsing utilities."""

from __future__ import annotations


def get_file_at_commit(repo_path: str, commit_sha: str, file_path: str) -> str | None:
    """Get file contents at a specific commit.

    Args:
        repo_path: Path to git repo.
        commit_sha: Git commit SHA.
        file_path: Path relative to repo root.

    Returns:
        File contents as string, or None if file doesn't exist at that commit.
    """
    # TODO: Implement with pygit2 for bare repo support
    raise NotImplementedError


def parse_diff_hunks(diff_text: str) -> list[dict]:
    """Parse a unified diff into structured hunks.

    Args:
        diff_text: Raw unified diff text.

    Returns:
        List of hunk dicts with file_path, old_start, new_start, changes.
    """
    # TODO: Implement diff parsing
    raise NotImplementedError
