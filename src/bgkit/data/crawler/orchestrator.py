"""Main crawl orchestration.

Coordinates discovery and download phases with:
- Resume capability
- Progress tracking
- Statistics reporting
"""

import json
from dataclasses import asdict
from datetime import datetime
from typing import Optional

from bgkit.data.crawler.config import CrawlerConfig
from bgkit.data.crawler.database import CrawlDatabase, RepoStatus
from bgkit.data.crawler.discovery import RepoDiscovery
from bgkit.data.crawler.downloader import RepoDownloader
from bgkit.data.crawler.rate_limiter import RateLimitedClient, TokenRotator


class CrawlOrchestrator:
    """Main orchestrator for the GitHub repository crawl.

    Coordinates:
    - Repository discovery via GitHub API
    - Download queue management
    - Progress tracking and reporting
    - Resume from interruption
    """

    def __init__(self, config: CrawlerConfig):
        self.config = config
        self.db: Optional[CrawlDatabase] = None
        self.downloader: Optional[RepoDownloader] = None
        self.discovery: Optional[RepoDiscovery] = None
        self.token_rotator: Optional[TokenRotator] = None
        self._run_id: Optional[int] = None

    async def init(self) -> None:
        """Initialize all components."""
        # Ensure directories exist
        self.config.storage_base_path.mkdir(parents=True, exist_ok=True)
        self.config.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize database
        self.db = CrawlDatabase(self.config.db_path)
        await self.db.init()

        # Initialize token rotator and API client
        if self.config.github_tokens:
            self.token_rotator = TokenRotator(self.config.github_tokens)

        # Initialize downloader
        self.downloader = RepoDownloader(self.config, self.db)

    async def close(self) -> None:
        """Clean up resources."""
        if self.db:
            await self.db.close()

    async def __aenter__(self) -> "CrawlOrchestrator":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def discover_repos(
        self,
        target_count: Optional[int] = None,
    ) -> int:
        """Discover repositories via GitHub API and add to database.

        Args:
            target_count: Number of repos to discover (default from config)

        Returns:
            Number of repos discovered
        """
        target = target_count or self.config.target_repo_count

        if not self.token_rotator:
            raise RuntimeError("No GitHub tokens configured for API discovery")

        async with RateLimitedClient(self.token_rotator) as client:
            self.discovery = RepoDiscovery(client, self.db)
            return await self.discovery.discover_and_store(
                target_count=target,
                min_stars=self.config.min_stars,
                exclude_forks=self.config.exclude_forks,
            )

    async def queue_repos(self, count: Optional[int] = None) -> int:
        """Move discovered repos to download queue.

        Args:
            count: Number of repos to queue (default: all discovered)

        Returns:
            Number of repos queued
        """
        limit = count or self.config.target_repo_count
        return await self.db.queue_repos(
            limit=limit,
            min_stars=self.config.min_stars,
        )

    async def download_repos(
        self,
        batch_size: int = 100,
        progress_callback: Optional[callable] = None,
        randomize: bool = False,
    ) -> dict:
        """Download all queued repositories.

        Args:
            batch_size: Repos to process per batch
            progress_callback: Optional progress callback
            randomize: If True, select repos randomly for balanced language coverage

        Returns:
            Download statistics
        """
        # Reset any stuck downloads from previous run
        reset_count = await self.db.reset_stuck_downloads(timeout_minutes=120)
        if reset_count:
            print(f"Reset {reset_count} stuck downloads")

        return await self.downloader.download_all_queued(
            batch_size=batch_size,
            progress_callback=progress_callback,
            randomize=randomize,
        )

    async def run_full_crawl(
        self,
        batch_size: int = 100,
    ) -> dict:
        """Run complete crawl: discover + queue + download.

        Args:
            batch_size: Download batch size

        Returns:
            Complete statistics
        """
        # Start crawl run tracking
        config_json = json.dumps(asdict(self.config), default=str)
        self._run_id = await self.db.start_crawl_run(config_json)

        stats = {
            "started_at": datetime.now().isoformat(),
            "discovered": 0,
            "queued": 0,
            "downloaded": 0,
            "failed": 0,
            "skipped": 0,
        }

        try:
            # Check current state
            current_stats = await self.db.get_stats()
            pending_count = current_stats.get("pending", 0)
            downloading_count = current_stats.get("downloading", 0)

            # Discovery phase (skip if we have enough)
            total_available = pending_count + downloading_count
            if total_available < self.config.target_repo_count:
                print(f"Discovering repos (have {total_available}, need {self.config.target_repo_count})...")
                stats["discovered"] = await self.discover_repos()

            # Queue phase
            print("Queuing repos for download...")
            stats["queued"] = await self.queue_repos()

            # Download phase
            print("Starting downloads...")
            download_stats = await self.download_repos(batch_size=batch_size)
            stats["downloaded"] = download_stats["total_downloaded"]
            stats["failed"] = download_stats["total_failed"]
            stats["skipped"] = download_stats["total_skipped"]

        finally:
            # End crawl run
            stats["ended_at"] = datetime.now().isoformat()
            if self._run_id:
                await self.db.end_crawl_run(
                    self._run_id,
                    stats["discovered"],
                    stats["downloaded"],
                    stats["failed"],
                )

        return stats

    async def resume_crawl(self, batch_size: int = 100) -> dict:
        """Resume interrupted crawl.

        Continues downloading queued repos without re-discovering.
        Resets any in-flight downloads from previous run.

        Returns:
            Download statistics
        """
        # Reset ALL downloads that were in-flight when we crashed
        # (they will have partial clones that get cleaned up on retry)
        reset_count = await self.db.reset_all_downloading()
        if reset_count:
            print(f"Reset {reset_count} interrupted downloads from previous run")

        # Retry failed repos (up to max retries)
        retry_count = await self.db.retry_failed(max_failures=self.config.retry_attempts)
        if retry_count:
            print(f"Re-queued {retry_count} failed repos for retry")

        # Continue downloads
        return await self.download_repos(batch_size=batch_size)

    async def reconcile_missing(self) -> dict:
        """Find repos marked as downloaded/processed whose local files are missing.

        Marks them as failed with a descriptive reason. Returns statistics.
        """
        from pathlib import Path

        repos = await self.db.get_repos_with_local_paths()
        missing = []

        for full_name, local_path, status, size_kb in repos:
            if not local_path or not Path(local_path).exists():
                missing.append((full_name, local_path, status, size_kb))

        for full_name, local_path, status, size_kb in missing:
            size_mb = (size_kb or 0) / 1024
            await self.db.mark_failed(
                full_name,
                f"reconciled: local directory missing (was {status}, ~{size_mb:.0f} MB)",
            )

        return {
            "checked": len(repos),
            "missing": len(missing),
            "ok": len(repos) - len(missing),
        }

    async def get_progress(self) -> dict:
        """Get current crawl progress."""
        stats = await self.db.get_stats()

        # Calculate progress with simplified state machine
        total = sum(
            stats.get(s.value, 0)
            for s in [
                RepoStatus.PENDING,
                RepoStatus.DOWNLOADING,
                RepoStatus.DOWNLOADED,
                RepoStatus.PROCESSED,
                RepoStatus.FAILED,
            ]
        )

        completed = stats.get("downloaded", 0) + stats.get("processed", 0)
        failed = stats.get("failed", 0)
        pending = stats.get("pending", 0)
        in_progress = stats.get("downloading", 0)

        return {
            "total_repos": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "progress_percent": (completed / total * 100) if total else 0,
            "total_size_gb": stats.get("total_size_bytes", 0) / (1024**3),
            "total_commits": stats.get("total_commits", 0),
            "by_status": {
                status.value: stats.get(status.value, 0) for status in RepoStatus
            },
        }

    async def print_progress(self) -> None:
        """Print formatted progress report."""
        progress = await self.get_progress()

        print("\n" + "=" * 60)
        print("CRAWL PROGRESS")
        print("=" * 60)
        print(f"Total repos:     {progress['total_repos']:,}")
        print(f"Completed:       {progress['completed']:,} ({progress['progress_percent']:.1f}%)")
        print(f"Failed/Skipped:  {progress['failed']:,}")
        print(f"Pending:         {progress['pending']:,}")
        print(f"In Progress:     {progress['in_progress']:,}")
        print(f"Total Size:      {progress['total_size_gb']:.2f} GB")
        print("-" * 60)
        print("By Status:")
        for status, count in progress["by_status"].items():
            if count > 0:
                print(f"  {status:15} {count:,}")
        print("=" * 60 + "\n")


async def run_crawl(
    config: CrawlerConfig,
    resume: bool = False,
) -> dict:
    """Convenience function to run a crawl.

    Args:
        config: Crawler configuration
        resume: Resume interrupted crawl instead of starting fresh

    Returns:
        Crawl statistics
    """
    async with CrawlOrchestrator(config) as orchestrator:
        if resume:
            return await orchestrator.resume_crawl()
        else:
            return await orchestrator.run_full_crawl()
