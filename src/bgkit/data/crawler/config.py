"""Configuration for the GitHub repository crawler."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional


@dataclass
class CrawlerConfig:
    """Configuration for crawling GitHub repositories.

    Attributes:
        target_repo_count: Target number of repositories to crawl.
        min_stars: Minimum star count for repository inclusion.
        storage_base_path: Base directory for cloned repositories.
        db_path: Path to SQLite database for crawl state.
        max_concurrent_downloads: Number of parallel git clones.
        clone_timeout_seconds: Maximum time per clone operation.
        retry_attempts: Number of retry attempts for failed operations.
        retry_delay_seconds: Delay between retry attempts.
        github_tokens: List of GitHub API tokens for rate limit distribution.
        exclude_forks: Whether to exclude forked repositories.
        exclude_archived: Whether to exclude archived repositories.
        max_repo_size_kb: Maximum repository size in KB (None for no limit).
    """

    target_repo_count: int = 100_000
    min_stars: int = 50  # Default for mainstream languages; niche/emerging use lower thresholds
    storage_base_path: Path = field(default_factory=lambda: Path("./data/repos"))
    db_path: Path = field(default_factory=lambda: Path("./data/crawl_state.db"))
    max_concurrent_downloads: int = 10
    clone_timeout_seconds: int = 3600
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    github_tokens: List[str] = field(default_factory=list)
    exclude_forks: bool = True
    exclude_archived: bool = False
    max_repo_size_kb: Optional[int] = None

    def __post_init__(self):
        """Ensure paths are Path objects and create directories."""
        if isinstance(self.storage_base_path, str):
            self.storage_base_path = Path(self.storage_base_path)
        if isinstance(self.db_path, str):
            self.db_path = Path(self.db_path)
