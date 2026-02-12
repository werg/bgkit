"""Build training tasks from commits.

Each training example treats a commit as supervised signal: check out
at parent commit, run BgKIT compression, use commit message as task prompt.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingTask:
    """A constructed training task from a commit."""

    repo_path: str
    parent_sha: str
    commit_sha: str
    prompt: str  # Commit message, optionally enriched with diff summary/PR desc
    target_files: list[str]  # Files modified in the commit
    tier: int  # 1=retrieval, 2=QA, 3=full agentic
    with_injection: bool  # Whether to include BgKIT tool-call frames


def build_tasks_from_commit(
    repo_path: str,
    commit_sha: str,
    parent_sha: str,
    message: str,
    diff_paths: list[str],
    tier: int = 1,
    no_injection_fraction: float = 0.3,
) -> list[TrainingTask]:
    """Build training tasks from a single commit.

    Args:
        repo_path: Path to git repo.
        commit_sha: The commit SHA.
        parent_sha: Parent commit SHA (checkout point).
        message: Commit message.
        diff_paths: Files changed in the commit.
        tier: Task tier (1/2/3).
        no_injection_fraction: Fraction of tasks without BgKIT injection.

    Returns:
        List of training tasks.
    """
    # TODO: Implement task construction logic
    raise NotImplementedError
