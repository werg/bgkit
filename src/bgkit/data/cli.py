"""Command-line interface for the data crawler.

Usage:
    bgkit-data discover --target 100000
    bgkit-data download --concurrency 10
    bgkit-data stats

Environment Variables (set in .env file):
    GITHUB_TOKEN      - Single GitHub personal access token
    GITHUB_TOKENS     - Multiple tokens, comma-separated (for higher throughput)
    DATA_DIR          - Root data directory (required, set in .env)
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

from bgkit.env import get_db_path, get_repos_dir

# Lazy defaults — resolved after arg parsing so --help works without .env
_SENTINEL = object()



def get_github_tokens() -> list:
    """Get GitHub tokens from environment."""
    tokens = []

    # Single token
    if os.environ.get("GITHUB_TOKEN"):
        tokens.append(os.environ["GITHUB_TOKEN"])

    # Multiple tokens (comma-separated)
    if os.environ.get("GITHUB_TOKENS"):
        tokens.extend(os.environ["GITHUB_TOKENS"].split(","))

    return [t.strip() for t in tokens if t.strip()]


async def cmd_discover(args):
    """Discover repositories from GitHub API."""
    from bgkit.data.crawler.database import CrawlDatabase
    from bgkit.data.crawler.discovery import (
        EMERGING_LANGUAGES,
        MAINSTREAM_LANGUAGES,
        NICHE_LANGUAGES,
        RepoDiscovery,
    )
    from bgkit.data.crawler.rate_limiter import RateLimitedClient, TokenRotator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found.")
        print("Set GITHUB_TOKEN or GITHUB_TOKENS environment variable.")
        print("Get tokens from: https://github.com/settings/tokens")
        return 1

    async with CrawlDatabase(Path(args.db)) as db:
        print(f"Discovering {args.target} repos via GitHub API...")
        print(f"Using {len(tokens)} token(s) for rate limit distribution")

        # Show what we'll query
        lang_tiers = []
        if not args.mainstream_only:
            lang_tiers.append(f"{len(MAINSTREAM_LANGUAGES)} mainstream (50+ stars)")
            if args.include_niche:
                lang_tiers.append(f"{len(NICHE_LANGUAGES)} niche (20+ stars)")
            if args.include_emerging:
                lang_tiers.append(f"{len(EMERGING_LANGUAGES)} emerging (10+ stars)")
        else:
            lang_tiers.append(f"{len(MAINSTREAM_LANGUAGES)} mainstream ({args.min_stars}+ stars)")

        print(f"Language tiers: {', '.join(lang_tiers)}")
        print("Note: 2.5s delay between search requests for rate limit compliance")

        if args.fresh:
            print("Starting fresh discovery (clearing previous progress)")
        else:
            print("Discovery is resumable - will skip already-completed queries")

        rotator = TokenRotator(tokens)
        async with RateLimitedClient(rotator) as client:
            discovery = RepoDiscovery(client, db)
            count = await discovery.discover_and_store(
                target_count=args.target,
                min_stars=args.min_stars,
                exclude_forks=not args.include_forks,
                include_niche=args.include_niche and not args.mainstream_only,
                include_emerging=args.include_emerging and not args.mainstream_only,
                resume=not args.fresh,
                force_refresh=args.fresh,
            )

        print(f"\nDiscovered {count} new repositories")

        # Show discovery progress
        disc_stats = await db.get_discovery_stats()
        print(f"Discovery progress: {disc_stats['total_queries_complete']} queries complete")

        # Show total pending
        stats = await db.get_stats()
        print(f"Total pending for download: {stats.get('pending', 0)}")

    return 0


async def cmd_download(args):
    """Download queued repositories."""
    from bgkit.data.crawler.config import CrawlerConfig
    from bgkit.data.crawler.orchestrator import CrawlOrchestrator

    tokens = get_github_tokens()

    config = CrawlerConfig(
        db_path=Path(args.db),
        storage_base_path=Path(args.output),
        max_concurrent_downloads=args.concurrency,
        clone_timeout_seconds=args.timeout,
        github_tokens=tokens,
        max_repo_size_kb=args.max_size * 1024 if args.max_size else None,
    )

    async with CrawlOrchestrator(config) as orchestrator:
        if config.max_repo_size_kb:
            print(f"Filtering repos larger than {args.max_size} MB (GitHub API estimate)")

        if args.resume:
            print("Resuming interrupted crawl...")
            stats = await orchestrator.resume_crawl(batch_size=args.batch_size)
        else:
            if args.randomize:
                print("Starting download (randomized for balanced language coverage)...")
            else:
                print("Starting download (ordered by stars)...")
            stats = await orchestrator.download_repos(
                batch_size=args.batch_size,
                randomize=args.randomize,
            )

        print("\nDownload complete:")
        print(f"  Downloaded: {stats.get('total_downloaded', 0)}")
        print(f"  Failed:     {stats.get('total_failed', 0)}")
        print(f"  Skipped:    {stats.get('total_skipped', 0)}")
        print(f"  Total size: {stats.get('total_size_bytes', 0) / (1024**3):.2f} GB")

    return 0


async def cmd_stats(args):
    """Show crawl statistics."""
    from bgkit.data.crawler.database import CrawlDatabase

    db_path = Path(args.db)

    if not db_path.exists():
        print(f"Database not found: {db_path}")
        return 1

    print("=" * 60)
    print("CRAWL DATABASE STATISTICS")
    print("=" * 60)

    async with CrawlDatabase(db_path) as db:
        stats = await db.get_stats()

        print("\nRepository Status:")
        for status, count in stats.items():
            if isinstance(count, int) and count > 0:
                print(f"  {status:20} {count:,}")

        print("\nLanguage Distribution (top 20):")
        lang_dist = await db.get_language_distribution()
        for lang, count in list(lang_dist.items())[:20]:
            print(f"  {lang:20} {count:,}")

        # Show discovery progress
        disc_stats = await db.get_discovery_stats()
        if disc_stats["total_queries_complete"] > 0:
            print("\nDiscovery Progress:")
            print(f"  Total queries complete: {disc_stats['total_queries_complete']}")
            print(f"  Total repos found:      {disc_stats['total_repos_found']:,}")
            for tier, tier_stats in disc_stats["tiers"].items():
                print(
                    f"    {tier}: {tier_stats['queries_complete']} queries,"
                    f" {tier_stats['repos_found']:,} repos"
                )

    return 0


async def cmd_refresh(args):
    """Refresh discovery to find new repos and update metadata.

    This is designed for periodic re-runs to:
    1. Re-discover all language/star combinations to find new repos
    2. Update metadata (star counts, etc.) for existing repos
    """
    from bgkit.data.crawler.database import CrawlDatabase
    from bgkit.data.crawler.discovery import RepoDiscovery
    from bgkit.data.crawler.rate_limiter import RateLimitedClient, TokenRotator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found.")
        return 1

    async with CrawlDatabase(Path(args.db)) as db:
        print("=== REFRESH MODE ===")
        print("Re-discovering to find new repos and update metadata...")

        # Clear discovery progress to force re-query of all combinations
        cleared = await db.clear_discovery_progress()
        print(f"Cleared {cleared} discovery progress entries")

        # Get current stats
        stats = await db.get_stats()
        print(f"Current database: {stats.get('pending', 0)} pending, "
              f"{stats.get('downloaded', 0)} downloaded")

        rotator = TokenRotator(tokens)
        async with RateLimitedClient(rotator) as client:
            discovery = RepoDiscovery(client, db)

            # Re-discover with resume=False since we cleared progress
            new_count = await discovery.discover_and_store(
                target_count=args.target,
                min_stars=50,  # Use default thresholds
                exclude_forks=True,
                include_niche=True,
                include_emerging=True,
                resume=False,
                force_refresh=False,  # Already cleared above
            )

        print(f"\nRefresh complete: {new_count} new repos discovered")

        # Show updated stats
        stats = await db.get_stats()
        print(f"Updated database: {stats.get('pending', 0)} pending, "
              f"{stats.get('downloaded', 0)} downloaded")

        # Show discovery stats
        disc_stats = await db.get_discovery_stats()
        print(f"Discovery queries complete: {disc_stats['total_queries_complete']}")

    return 0


async def cmd_reconcile(args):
    """Check for repos whose local directories are missing and fix DB state."""
    from bgkit.data.crawler.config import CrawlerConfig
    from bgkit.data.crawler.orchestrator import CrawlOrchestrator

    config = CrawlerConfig(
        db_path=Path(args.db),
    )

    async with CrawlOrchestrator(config) as orchestrator:
        print("Checking for repos with missing local directories...")
        stats = await orchestrator.reconcile_missing()

        print("\nReconciliation complete:")
        print(f"  Checked:  {stats['checked']}")
        print(f"  OK:       {stats['ok']}")
        print(f"  Missing:  {stats['missing']} (marked as failed)")

    return 0


async def cmd_full(args):
    """Run full crawl pipeline: discover + download."""
    from bgkit.data.crawler.config import CrawlerConfig
    from bgkit.data.crawler.orchestrator import CrawlOrchestrator

    tokens = get_github_tokens()
    if not tokens:
        print("Error: No GitHub tokens found. Set GITHUB_TOKEN or GITHUB_TOKENS env var.")
        return 1

    config = CrawlerConfig(
        target_repo_count=args.target,
        min_stars=args.min_stars,
        db_path=Path(args.db),
        storage_base_path=Path(args.output),
        max_concurrent_downloads=args.concurrency,
        github_tokens=tokens,
        exclude_forks=not args.include_forks,
        max_repo_size_kb=args.max_size * 1024 if args.max_size else None,
    )

    async with CrawlOrchestrator(config) as orchestrator:
        print("Starting full crawl pipeline...")
        stats = await orchestrator.run_full_crawl()

        print("\nCrawl complete:")
        print(json.dumps(stats, indent=2))

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="GitHub repository crawler for training data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--db",
        default=_SENTINEL,
        help="Path to SQLite database (default: DATA_DIR/crawl_state.db)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Discover command
    discover_parser = subparsers.add_parser("discover", help="Discover repositories")
    discover_parser.add_argument(
        "--target", type=int, default=100000, help="Target number of repos"
    )
    discover_parser.add_argument(
        "--min-stars", type=int, default=50,
        help="Minimum star count for mainstream languages (default: 50)",
    )
    discover_parser.add_argument(
        "--include-forks", action="store_true", help="Include forked repos"
    )
    discover_parser.add_argument(
        "--include-niche", action="store_true", default=True,
        help="Include niche languages (Haskell, OCaml, etc.) with 20+ stars (default: True)"
    )
    discover_parser.add_argument(
        "--no-niche", action="store_false", dest="include_niche",
        help="Exclude niche languages"
    )
    discover_parser.add_argument(
        "--include-emerging", action="store_true", default=True,
        help="Include emerging languages (Zig, Gleam, etc.) with 10+ stars (default: True)"
    )
    discover_parser.add_argument(
        "--no-emerging", action="store_false", dest="include_emerging",
        help="Exclude emerging languages"
    )
    discover_parser.add_argument(
        "--mainstream-only", action="store_true",
        help="Only discover mainstream languages (Python, JS, etc.)"
    )
    discover_parser.add_argument(
        "--fresh", action="store_true",
        help="Clear discovery progress and start fresh (default: resume from last run)"
    )

    # Download command
    download_parser = subparsers.add_parser("download", help="Download repositories")
    download_parser.add_argument(
        "--output",
        default=_SENTINEL,
        help="Output directory (default: DATA_DIR/repos)",
    )
    download_parser.add_argument(
        "--concurrency", type=int, default=10, help="Parallel downloads"
    )
    download_parser.add_argument(
        "--timeout", type=int, default=3600, help="Clone timeout (seconds)"
    )
    download_parser.add_argument(
        "--batch-size", type=int, default=100, help="Batch size"
    )
    download_parser.add_argument(
        "--resume", action="store_true", help="Resume interrupted crawl"
    )
    download_parser.add_argument(
        "--randomize", action="store_true",
        help="Select repos randomly for balanced language coverage (default: order by stars)"
    )
    download_parser.add_argument(
        "--max-size", type=int, default=None,
        help="Maximum repo size in MB (based on GitHub API size estimate)."
        " Repos larger than this are skipped.",
    )

    # Stats command
    subparsers.add_parser("stats", help="Show statistics")

    # Full command
    full_parser = subparsers.add_parser("full", help="Run full pipeline (discover + download)")
    full_parser.add_argument(
        "--target", type=int, default=100000, help="Target number of repos"
    )
    full_parser.add_argument(
        "--min-stars", type=int, default=50,
        help="Minimum star count for mainstream languages (default: 50)",
    )
    full_parser.add_argument(
        "--output",
        default=_SENTINEL,
        help="Output directory (default: DATA_DIR/repos)",
    )
    full_parser.add_argument(
        "--concurrency", type=int, default=10, help="Parallel downloads"
    )
    full_parser.add_argument(
        "--include-forks", action="store_true", help="Include forked repos"
    )
    full_parser.add_argument(
        "--include-niche", action="store_true", default=True,
        help="Include niche languages (default: True)"
    )
    full_parser.add_argument(
        "--no-niche", action="store_false", dest="include_niche",
        help="Exclude niche languages"
    )
    full_parser.add_argument(
        "--include-emerging", action="store_true", default=True,
        help="Include emerging languages (default: True)"
    )
    full_parser.add_argument(
        "--no-emerging", action="store_false", dest="include_emerging",
        help="Exclude emerging languages"
    )
    full_parser.add_argument(
        "--max-size", type=int, default=None,
        help="Maximum repo size in MB (based on GitHub API size estimate)."
        " Repos larger than this are skipped.",
    )

    # Reconcile command
    subparsers.add_parser(
        "reconcile",
        help="Check for repos marked as downloaded whose local files are missing,"
        " and mark them as failed",
    )

    # Refresh command - for periodic re-runs
    refresh_parser = subparsers.add_parser(
        "refresh",
        help="Re-discover repos to find new ones and update metadata"
    )
    refresh_parser.add_argument(
        "--target", type=int, default=200000,
        help="Target number of repos (default: 200000 for refresh)"
    )

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 1

    # Resolve lazy defaults (deferred so --help works without .env)
    if args.db is _SENTINEL:
        args.db = str(get_db_path())
    if hasattr(args, "output") and args.output is _SENTINEL:
        args.output = str(get_repos_dir(must_exist=False))

    # Run the appropriate command
    if args.command == "discover":
        return asyncio.run(cmd_discover(args))
    elif args.command == "download":
        return asyncio.run(cmd_download(args))
    elif args.command == "stats":
        return asyncio.run(cmd_stats(args))
    elif args.command == "full":
        return asyncio.run(cmd_full(args))
    elif args.command == "refresh":
        return asyncio.run(cmd_refresh(args))
    elif args.command == "reconcile":
        return asyncio.run(cmd_reconcile(args))

    return 1


if __name__ == "__main__":
    sys.exit(main())
