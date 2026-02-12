"""Git repo loading and file extraction.

Processes bare git repos on disk: extracts file trees at specific commits,
reads file contents, and produces tokenizable file records.
"""

from __future__ import annotations


def load_repo_files(repo_path: str, commit_sha: str) -> dict[str, str]:
    """Load all files from a git repo at a specific commit.

    Args:
        repo_path: Path to the bare git repo.
        commit_sha: Git commit SHA to check out.

    Returns:
        Dict mapping file paths to file contents.
    """
    # TODO: Implement using gitpython or pygit2
    raise NotImplementedError
