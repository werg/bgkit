"""Parallel repository downloader with retry logic.

Downloads repositories using git clone with:
- Configurable concurrency
- Timeout handling
- Retry with exponential backoff
- Resume capability via database state
"""

import asyncio
import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional

from bgkit.data.crawler.config import CrawlerConfig
from bgkit.data.crawler.database import CrawlDatabase, RepoMetadata


class FailureAction(Enum):
    """Action to take after a download failure."""

    RETRY = "retry"
    RETRY_LATER = "retry_later"
    RETRY_WITH_EXTENDED_TIMEOUT = "retry_extended"
    SKIP = "skip"


@dataclass
class DownloadResult:
    """Result of a single repository download."""

    full_name: str
    success: bool
    path: Optional[Path] = None
    size_bytes: int = 0
    error: Optional[str] = None


class RepoDownloader:
    """Parallel repository downloader with retry logic.

    Downloads repositories using git clone --single-branch for full history
    of the main branch only.
    """

    def __init__(self, config: CrawlerConfig, db: CrawlDatabase):
        self.config = config
        self.db = db
        self._semaphore = asyncio.Semaphore(config.max_concurrent_downloads)

    def _get_repo_path(self, repo: RepoMetadata) -> Path:
        """Generate storage path: /base/owner/repo/"""
        owner, name = repo.full_name.split("/")
        return self.config.storage_base_path / owner / name

    def _get_dir_size(self, path: Path) -> int:
        """Calculate total size of directory in bytes."""
        total = 0
        for dirpath, _, filenames in os.walk(path):
            for filename in filenames:
                filepath = Path(dirpath) / filename
                try:
                    total += filepath.stat().st_size
                except (OSError, FileNotFoundError):
                    pass
        return total

    async def _clone_repo(
        self, repo: RepoMetadata, dest_path: Path, timeout: int
    ) -> DownloadResult:
        """Execute git clone with timeout."""
        url = f"https://github.com/{repo.full_name}.git"

        # Ensure parent directory exists
        dest_path.parent.mkdir(parents=True, exist_ok=True)

        # Remove destination if it exists (partial clone)
        if dest_path.exists():
            shutil.rmtree(dest_path)

        cmd = [
            "git",
            "clone",
            "--single-branch",
            "--branch",
            repo.default_branch,
            url,
            str(dest_path),
        ]

        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=timeout
            )

            if process.returncode == 0:
                size_bytes = self._get_dir_size(dest_path)
                return DownloadResult(
                    full_name=repo.full_name,
                    success=True,
                    path=dest_path,
                    size_bytes=size_bytes,
                )
            else:
                error = stderr.decode().strip()
                return DownloadResult(
                    full_name=repo.full_name,
                    success=False,
                    error=error,
                )

        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            # Clean up partial clone
            if dest_path.exists():
                shutil.rmtree(dest_path)
            return DownloadResult(
                full_name=repo.full_name,
                success=False,
                error=f"Clone timeout after {timeout}s",
            )

    def _determine_failure_action(self, error: str) -> FailureAction:
        """Determine action based on error type."""
        error_lower = error.lower()

        if "repository not found" in error_lower or "404" in error_lower:
            # Repo deleted or made private
            return FailureAction.SKIP

        if "repository moved" in error_lower or "301" in error_lower:
            # Repo renamed - will need to resolve redirect
            return FailureAction.SKIP

        if "timeout" in error_lower:
            # Large repo timeout - could retry with extended timeout
            return FailureAction.RETRY_WITH_EXTENDED_TIMEOUT

        if "could not read" in error_lower or "ssl" in error_lower:
            # Transient network error
            return FailureAction.RETRY_LATER

        if "remote: " in error_lower and "error" in error_lower:
            # GitHub remote error
            return FailureAction.RETRY_LATER

        if "does not exist" in error_lower or "not found" in error_lower:
            # Branch doesn't exist
            return FailureAction.SKIP

        # Unknown error - retry
        return FailureAction.RETRY

    async def download_repo(
        self, repo: RepoMetadata, attempt: int = 1
    ) -> DownloadResult:
        """Download a single repository with retry logic."""
        async with self._semaphore:
            dest_path = self._get_repo_path(repo)

            # Check repo size before cloning (if configured)
            if self.config.max_repo_size_kb and repo.size_kb > self.config.max_repo_size_kb:
                error_msg = f"skipped: repo size {repo.size_kb}KB exceeds limit {self.config.max_repo_size_kb}KB"
                await self.db.mark_failed(repo.full_name, error_msg)
                return DownloadResult(
                    full_name=repo.full_name,
                    success=False,
                    error=error_msg,
                )

            # Mark as downloading in database
            await self.db.mark_downloading(repo.full_name)

            # Calculate timeout (increase on retries)
            timeout = self.config.clone_timeout_seconds
            if attempt > 1:
                timeout = int(timeout * 1.5)

            # Attempt clone
            result = await self._clone_repo(repo, dest_path, timeout)

            if result.success:
                # Record success
                await self.db.mark_downloaded(
                    repo.full_name,
                    result.size_bytes,
                    str(result.path),
                )
                return result

            # Handle failure
            action = self._determine_failure_action(result.error or "")

            if action == FailureAction.SKIP:
                # Mark as failed with skip reason (simplified state machine)
                await self.db.mark_failed(repo.full_name, f"skipped: {result.error or 'unknown'}")
                return result

            if attempt < self.config.retry_attempts:
                if action == FailureAction.RETRY_WITH_EXTENDED_TIMEOUT:
                    # Retry with extended timeout
                    await asyncio.sleep(self.config.retry_delay_seconds)
                    return await self.download_repo(repo, attempt + 1)
                elif action in (FailureAction.RETRY, FailureAction.RETRY_LATER):
                    # Standard retry
                    delay = self.config.retry_delay_seconds * attempt
                    await asyncio.sleep(delay)
                    return await self.download_repo(repo, attempt + 1)

            # Max retries exceeded
            await self.db.mark_failed(repo.full_name, result.error or "unknown")
            return result

    async def download_batch(
        self, repos: List[RepoMetadata], progress_callback: Optional[callable] = None
    ) -> List[DownloadResult]:
        """Download multiple repos in parallel.

        Args:
            repos: List of repos to download
            progress_callback: Optional callback(completed, total, result)

        Returns:
            List of download results
        """
        results = []
        total = len(repos)

        async def download_with_progress(repo: RepoMetadata, index: int):
            result = await self.download_repo(repo)
            if progress_callback:
                progress_callback(index + 1, total, result)
            return result

        tasks = [download_with_progress(repo, i) for i, repo in enumerate(repos)]
        results = await asyncio.gather(*tasks)

        return results

    async def download_all_queued(
        self,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
        randomize: bool = False,
    ) -> dict:
        """Download all queued repositories.

        Args:
            batch_size: Number of repos to fetch per batch
            progress_callback: Optional callback for progress
            randomize: If True, select repos randomly for balanced language coverage

        Returns:
            Summary statistics
        """
        stats = {
            "total_downloaded": 0,
            "total_failed": 0,
            "total_skipped": 0,
            "total_size_bytes": 0,
        }

        while True:
            # Get next batch
            repos = await self.db.get_pending_repos(
                batch_size=batch_size, randomize=randomize,
                max_size_kb=self.config.max_repo_size_kb,
            )

            if not repos:
                break

            print(f"Downloading batch of {len(repos)} repos...")

            results = await self.download_batch(repos, progress_callback)

            for result in results:
                if result.success:
                    stats["total_downloaded"] += 1
                    stats["total_size_bytes"] += result.size_bytes
                else:
                    if "skipped" in str(result.error):
                        stats["total_skipped"] += 1
                    else:
                        stats["total_failed"] += 1

            print(
                f"Batch complete: {stats['total_downloaded']} downloaded, "
                f"{stats['total_failed']} failed, {stats['total_skipped']} skipped"
            )

        return stats
