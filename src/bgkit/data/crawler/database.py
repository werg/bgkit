"""SQLite database for crawl state management.

Provides persistent storage for:
- Repository metadata and crawl status
- API token rotation state
- Crawl run history
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path

import aiosqlite


class RepoStatus(Enum):
    """Repository crawl status.

    Simplified state machine:
    - PENDING: Discovered and ready for download
    - DOWNLOADING: Currently being cloned
    - DOWNLOADED: Successfully cloned, ready for processing
    - PROCESSING: Currently extracting commits
    - PROCESSED: Commits extracted
    - FAILED: Operation failed (with error reason)
    """

    PENDING = "pending"  # Ready for download (combines discovered + queued)
    DOWNLOADING = "downloading"  # Currently being cloned
    DOWNLOADED = "downloaded"  # Successfully cloned
    PROCESSING = "processing"  # Currently extracting commits
    PROCESSED = "processed"  # Commits extracted
    FAILED = "failed"  # Failed (check last_error for reason)


@dataclass
class RepoMetadata:
    """Repository metadata from GitHub."""

    full_name: str  # owner/repo
    stars: int
    language: str | None
    license_key: str | None
    size_kb: int
    default_branch: str
    fork: bool = False
    archived: bool = False
    created_at: datetime | None = None
    updated_at: datetime | None = None


SCHEMA = """
-- repositories: Main repo tracking table
CREATE TABLE IF NOT EXISTS repositories (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name TEXT UNIQUE NOT NULL,
    owner TEXT NOT NULL,
    name TEXT NOT NULL,

    -- Discovery metadata
    stars INTEGER,
    language TEXT,
    license_key TEXT,
    size_kb INTEGER,
    default_branch TEXT DEFAULT 'main',
    fork BOOLEAN DEFAULT FALSE,
    archived BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP,
    updated_at TIMESTAMP,
    discovered_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    metadata_updated_at TIMESTAMP,  -- Last time we refreshed from GitHub API

    -- Crawl state (simplified: pending -> downloading -> downloaded -> processed | failed)
    status TEXT DEFAULT 'pending',
    status_updated_at TIMESTAMP,

    -- Download info
    download_started_at TIMESTAMP,
    download_completed_at TIMESTAMP,
    actual_size_bytes INTEGER,
    local_path TEXT,

    -- Processing info
    commit_count INTEGER,
    processed_at TIMESTAMP,

    -- Failure tracking
    failure_count INTEGER DEFAULT 0,
    last_error TEXT
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_repos_status ON repositories(status);
CREATE INDEX IF NOT EXISTS idx_repos_stars ON repositories(stars DESC);
CREATE INDEX IF NOT EXISTS idx_repos_language ON repositories(language);
CREATE INDEX IF NOT EXISTS idx_repos_status_stars ON repositories(status, stars DESC);
CREATE INDEX IF NOT EXISTS idx_repos_discovered_at ON repositories(discovered_at);

-- discovery_progress: Track discovery cursor for resumability
CREATE TABLE IF NOT EXISTS discovery_progress (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    tier TEXT NOT NULL,           -- 'mainstream', 'niche', 'emerging'
    language TEXT NOT NULL,
    star_range_low INTEGER NOT NULL,
    star_range_high INTEGER,      -- NULL means "no upper bound"
    completed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    repos_found INTEGER DEFAULT 0,
    UNIQUE(tier, language, star_range_low)
);

-- crawl_runs: Track crawl sessions for debugging
CREATE TABLE IF NOT EXISTS crawl_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP,
    repos_discovered INTEGER DEFAULT 0,
    repos_downloaded INTEGER DEFAULT 0,
    repos_failed INTEGER DEFAULT 0,
    config_json TEXT
);
"""


class CrawlDatabase:
    """Async SQLite database for crawl state management."""

    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._db: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()

    async def init(self) -> None:
        """Initialize database and create tables."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self.db_path)
        self._db.row_factory = aiosqlite.Row
        await self._db.executescript(SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def __aenter__(self) -> "CrawlDatabase":
        await self.init()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def add_discovered_repos(self, repos: list[RepoMetadata]) -> int:
        """Bulk insert discovered repos. Returns number inserted."""
        async with self._lock:
            cursor = await self._db.executemany(
                """
                INSERT OR IGNORE INTO repositories
                (full_name, owner, name, stars, language, license_key,
                 size_kb, default_branch, fork, archived, created_at,
                 updated_at, status, status_updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', CURRENT_TIMESTAMP)
                """,
                [
                    (
                        r.full_name,
                        r.full_name.split("/")[0],
                        r.full_name.split("/")[1],
                        r.stars,
                        r.language,
                        r.license_key,
                        r.size_kb,
                        r.default_branch,
                        r.fork,
                        r.archived,
                        r.created_at,
                        r.updated_at,
                    )
                    for r in repos
                ],
            )
            await self._db.commit()
            return cursor.rowcount

    async def queue_repos(self, limit: int = 1000, min_stars: int = 0) -> int:
        """Count repos ready for download (no-op in simplified state machine).

        With the simplified state machine, repos are PENDING immediately after discovery.
        This method now just returns the count for backward compatibility.
        """
        async with self._db.execute(
            "SELECT COUNT(*) as count FROM repositories WHERE status = 'pending' AND stars >= ?",
            (min_stars,),
        ) as cursor:
            row = await cursor.fetchone()
            return min(row["count"], limit)

    async def get_pending_repos(
        self, batch_size: int = 100, randomize: bool = False,
        max_size_kb: int | None = None,
    ) -> list[RepoMetadata]:
        """Get next batch of repos to download.

        Args:
            batch_size: Number of repos to fetch
            randomize: If True, select randomly for balanced language coverage.
                       If False, order by stars descending (default).
            max_size_kb: If set, exclude repos larger than this (KB).
        """
        order_clause = "RANDOM()" if randomize else "stars DESC"
        size_filter = "AND (size_kb IS NULL OR size_kb <= ?)" if max_size_kb else ""
        params: list = []
        if max_size_kb:
            params.append(max_size_kb)
        params.append(batch_size)
        async with self._db.execute(
            f"""
            SELECT full_name, stars, language, license_key, size_kb,
                   default_branch, fork, archived
            FROM repositories
            WHERE status = 'pending'
            {size_filter}
            ORDER BY {order_clause}
            LIMIT ?
            """,
            params,
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                RepoMetadata(
                    full_name=row["full_name"],
                    stars=row["stars"],
                    language=row["language"],
                    license_key=row["license_key"],
                    size_kb=row["size_kb"] or 0,
                    default_branch=row["default_branch"] or "main",
                    fork=bool(row["fork"]),
                    archived=bool(row["archived"]),
                )
                for row in rows
            ]

    async def mark_downloading(self, full_name: str) -> None:
        """Mark repo as currently downloading."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'downloading',
                    download_started_at = CURRENT_TIMESTAMP,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (full_name,),
            )
            await self._db.commit()

    async def mark_downloaded(
        self, full_name: str, size_bytes: int, local_path: str
    ) -> None:
        """Mark repo as successfully downloaded."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'downloaded',
                    actual_size_bytes = ?,
                    local_path = ?,
                    download_completed_at = CURRENT_TIMESTAMP,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (size_bytes, local_path, full_name),
            )
            await self._db.commit()

    async def mark_processing(self, full_name: str) -> None:
        """Mark repo as currently being processed."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'processing',
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (full_name,),
            )
            await self._db.commit()

    async def mark_processed(self, full_name: str, commit_count: int) -> None:
        """Mark repo as processed with commit count."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'processed',
                    commit_count = ?,
                    processed_at = CURRENT_TIMESTAMP,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (commit_count, full_name),
            )
            await self._db.commit()

    async def reset_all_processing(self) -> int:
        """Reset ALL repos in 'processing' status back to 'downloaded'.

        Use this on startup to recover from crashes during processing.
        Note: This may result in some duplicate commits in output if crash
        happened after writing but before marking processed.
        """
        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE repositories
                SET status = 'downloaded',
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processing'
                """
            )
            await self._db.commit()
            return cursor.rowcount

    async def mark_failed(self, full_name: str, error: str) -> None:
        """Mark repo as failed with error message."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'failed',
                    failure_count = failure_count + 1,
                    last_error = ?,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (error, full_name),
            )
            await self._db.commit()

    async def reset_stuck_downloads(self, timeout_minutes: int = 120) -> int:
        """Reset downloads that have been stuck for too long back to pending."""
        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE repositories
                SET status = 'pending',
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE status = 'downloading'
                AND download_started_at < datetime('now', ? || ' minutes')
                """,
                (f"-{timeout_minutes}",),
            )
            await self._db.commit()
            return cursor.rowcount

    async def reset_all_downloading(self) -> int:
        """Reset ALL repos in 'downloading' status back to 'pending'.

        Use this on startup to recover from crashes.
        """
        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE repositories
                SET status = 'pending',
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE status = 'downloading'
                """
            )
            await self._db.commit()
            return cursor.rowcount

    async def retry_failed(self, max_failures: int = 3) -> int:
        """Re-queue failed repos that haven't exceeded max failures."""
        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE repositories
                SET status = 'pending',
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE status = 'failed'
                AND failure_count < ?
                """,
                (max_failures,),
            )
            await self._db.commit()
            return cursor.rowcount

    async def update_repo_name(self, old_name: str, new_name: str) -> None:
        """Update repo name after redirect."""
        owner, name = new_name.split("/")
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET full_name = ?, owner = ?, name = ?,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ?
                """,
                (new_name, owner, name, old_name),
            )
            await self._db.commit()

    async def get_stats(self) -> dict:
        """Get crawl progress statistics."""
        stats = {}

        for status in RepoStatus:
            async with self._db.execute(
                "SELECT COUNT(*) as count FROM repositories WHERE status = ?",
                (status.value,),
            ) as cursor:
                row = await cursor.fetchone()
                stats[status.value] = row["count"]

        async with self._db.execute(
            """
            SELECT
                COUNT(*) as total_repos,
                SUM(actual_size_bytes) as total_size_bytes,
                SUM(commit_count) as total_commits
            FROM repositories
            WHERE status IN ('downloaded', 'processed')
            """
        ) as cursor:
            row = await cursor.fetchone()
            stats["total_repos_with_data"] = row["total_repos"]
            stats["total_size_bytes"] = row["total_size_bytes"] or 0
            stats["total_commits"] = row["total_commits"] or 0

        async with self._db.execute(
            "SELECT COUNT(DISTINCT language) as count FROM repositories WHERE language IS NOT NULL"
        ) as cursor:
            row = await cursor.fetchone()
            stats["unique_languages"] = row["count"]

        return stats

    async def get_language_distribution(self) -> dict:
        """Get count of repos per language."""
        async with self._db.execute(
            """
            SELECT language, COUNT(*) as count
            FROM repositories
            WHERE language IS NOT NULL
            GROUP BY language
            ORDER BY count DESC
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return {row["language"]: row["count"] for row in rows}

    async def get_downloaded_repos(
        self, batch_size: int = 100, offset: int = 0
    ) -> list[tuple]:
        """Get downloaded repos for processing."""
        async with self._db.execute(
            """
            SELECT full_name, local_path, language
            FROM repositories
            WHERE status = 'downloaded'
            ORDER BY stars DESC
            LIMIT ? OFFSET ?
            """,
            (batch_size, offset),
        ) as cursor:
            rows = await cursor.fetchall()
            return [(row["full_name"], row["local_path"], row["language"]) for row in rows]

    async def get_repos_with_local_paths(self) -> list[tuple]:
        """Get all repos that claim to have a local clone on disk.

        Returns list of (full_name, local_path, status, size_kb) for repos
        in 'downloaded' or 'processed' status with a local_path set.
        """
        async with self._db.execute(
            """
            SELECT full_name, local_path, status, size_kb
            FROM repositories
            WHERE status IN ('downloaded', 'processed')
            AND local_path IS NOT NULL
            """
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                (row["full_name"], row["local_path"], row["status"], row["size_kb"])
                for row in rows
            ]

    async def count_by_status(self, status: RepoStatus) -> int:
        """Count repos with given status."""
        async with self._db.execute(
            "SELECT COUNT(*) as count FROM repositories WHERE status = ?",
            (status.value,),
        ) as cursor:
            row = await cursor.fetchone()
            return row["count"]

    # Crawl run tracking

    async def start_crawl_run(self, config_json: str) -> int:
        """Start a new crawl run and return its ID."""
        async with self._lock:
            cursor = await self._db.execute(
                "INSERT INTO crawl_runs (config_json) VALUES (?)",
                (config_json,),
            )
            await self._db.commit()
            return cursor.lastrowid

    async def end_crawl_run(
        self, run_id: int, discovered: int, downloaded: int, failed: int
    ) -> None:
        """End a crawl run with final statistics."""
        async with self._lock:
            await self._db.execute(
                """
                UPDATE crawl_runs
                SET ended_at = CURRENT_TIMESTAMP,
                    repos_discovered = ?,
                    repos_downloaded = ?,
                    repos_failed = ?
                WHERE id = ?
                """,
                (discovered, downloaded, failed, run_id),
            )
            await self._db.commit()

    # Discovery progress tracking for resumability

    async def mark_discovery_complete(
        self,
        tier: str,
        language: str,
        star_range_low: int,
        star_range_high: int | None,
        repos_found: int,
    ) -> None:
        """Mark a discovery query (tier x language x star_range) as complete."""
        async with self._lock:
            await self._db.execute(
                """
                INSERT OR REPLACE INTO discovery_progress
                (tier, language, star_range_low, star_range_high, repos_found, completed_at)
                VALUES (?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (tier, language, star_range_low, star_range_high, repos_found),
            )
            await self._db.commit()

    async def is_discovery_complete(
        self,
        tier: str,
        language: str,
        star_range_low: int,
    ) -> bool:
        """Check if a discovery query has been completed."""
        async with self._db.execute(
            """
            SELECT 1 FROM discovery_progress
            WHERE tier = ? AND language = ? AND star_range_low = ?
            """,
            (tier, language, star_range_low),
        ) as cursor:
            row = await cursor.fetchone()
            return row is not None

    async def clear_discovery_progress(self, tier: str | None = None) -> int:
        """Clear discovery progress to force re-discovery.

        Args:
            tier: If provided, only clear progress for this tier.
                  If None, clear all progress.

        Returns:
            Number of entries cleared.
        """
        async with self._lock:
            if tier:
                cursor = await self._db.execute(
                    "DELETE FROM discovery_progress WHERE tier = ?",
                    (tier,),
                )
            else:
                cursor = await self._db.execute("DELETE FROM discovery_progress")
            await self._db.commit()
            return cursor.rowcount

    async def get_discovery_stats(self) -> dict:
        """Get discovery progress statistics."""
        stats = {"tiers": {}, "total_queries_complete": 0, "total_repos_found": 0}

        async with self._db.execute(
            """
            SELECT tier, COUNT(*) as queries, SUM(repos_found) as repos
            FROM discovery_progress
            GROUP BY tier
            """
        ) as cursor:
            rows = await cursor.fetchall()
            for row in rows:
                stats["tiers"][row["tier"]] = {
                    "queries_complete": row["queries"],
                    "repos_found": row["repos"] or 0,
                }
                stats["total_queries_complete"] += row["queries"]
                stats["total_repos_found"] += row["repos"] or 0

        return stats

    # Metadata refresh for incremental updates

    async def upsert_repo(self, repo: RepoMetadata) -> bool:
        """Insert or update a repository, returning True if it was new.

        For existing repos, updates metadata (stars, size, etc.) but
        preserves status and download/processing state.
        """
        async with self._lock:
            # Check if exists
            async with self._db.execute(
                "SELECT id, status FROM repositories WHERE full_name = ?",
                (repo.full_name,),
            ) as cursor:
                existing = await cursor.fetchone()

            if existing:
                # Update metadata only, preserve status
                await self._db.execute(
                    """
                    UPDATE repositories
                    SET stars = ?,
                        language = ?,
                        license_key = ?,
                        size_kb = ?,
                        default_branch = ?,
                        fork = ?,
                        archived = ?,
                        updated_at = ?,
                        metadata_updated_at = CURRENT_TIMESTAMP
                    WHERE full_name = ?
                    """,
                    (
                        repo.stars,
                        repo.language,
                        repo.license_key,
                        repo.size_kb,
                        repo.default_branch,
                        repo.fork,
                        repo.archived,
                        repo.updated_at,
                        repo.full_name,
                    ),
                )
                await self._db.commit()
                return False
            else:
                # Insert new
                await self._db.execute(
                    """
                    INSERT INTO repositories
                    (full_name, owner, name, stars, language, license_key,
                     size_kb, default_branch, fork, archived, created_at,
                     updated_at, status, status_updated_at, metadata_updated_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                     'pending', CURRENT_TIMESTAMP, CURRENT_TIMESTAMP)
                    """,
                    (
                        repo.full_name,
                        repo.full_name.split("/")[0],
                        repo.full_name.split("/")[1],
                        repo.stars,
                        repo.language,
                        repo.license_key,
                        repo.size_kb,
                        repo.default_branch,
                        repo.fork,
                        repo.archived,
                        repo.created_at,
                        repo.updated_at,
                    ),
                )
                await self._db.commit()
                return True

    async def get_stale_repos(
        self, max_age_days: int = 30, limit: int = 1000
    ) -> list[str]:
        """Get repos whose metadata hasn't been updated recently.

        Useful for periodic refresh of star counts, etc.
        """
        async with self._db.execute(
            """
            SELECT full_name FROM repositories
            WHERE metadata_updated_at IS NULL
               OR metadata_updated_at < datetime('now', ? || ' days')
            ORDER BY stars DESC
            LIMIT ?
            """,
            (f"-{max_age_days}", limit),
        ) as cursor:
            rows = await cursor.fetchall()
            return [row["full_name"] for row in rows]

    async def reset_for_reprocessing(self, full_name: str) -> None:
        """Reset a processed repo back to downloaded for reprocessing.

        Useful when you want to re-extract commits (e.g., with new filters).
        """
        async with self._lock:
            await self._db.execute(
                """
                UPDATE repositories
                SET status = 'downloaded',
                    commit_count = NULL,
                    processed_at = NULL,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE full_name = ? AND status = 'processed'
                """,
                (full_name,),
            )
            await self._db.commit()

    async def reset_all_for_reprocessing(self) -> int:
        """Reset all processed repos back to downloaded.

        Returns number of repos reset.
        """
        async with self._lock:
            cursor = await self._db.execute(
                """
                UPDATE repositories
                SET status = 'downloaded',
                    commit_count = NULL,
                    processed_at = NULL,
                    status_updated_at = CURRENT_TIMESTAMP
                WHERE status = 'processed'
                """
            )
            await self._db.commit()
            return cursor.rowcount
