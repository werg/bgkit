"""Repository discovery via GitHub API.

Uses adaptive star-range subdivision to discover repositories while minimizing
API calls. Only subdivides queries when results exceed GitHub's 1000-result limit.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from bgkit.data.crawler.database import CrawlDatabase, RepoMetadata
from bgkit.data.crawler.rate_limiter import RateLimitedClient

# GitHub Search API hard limit - cannot retrieve more than 1000 results per query
GITHUB_SEARCH_LIMIT = 1000

# Star thresholds for subdivision when results exceed 1000
# Used to split queries into smaller ranges
SUBDIVISION_THRESHOLDS = [100000, 50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20, 10]

# Starting star thresholds by language popularity
# Avoids wasting API calls checking 100K+ stars for niche languages
LANGUAGE_START_THRESHOLD: dict[str, int] = {
    # Mega-popular: start at 100K
    "python": 100000,
    "javascript": 100000,
    "typescript": 50000,
    "java": 50000,
    # Popular: start at 20K
    "go": 20000,
    "rust": 20000,
    "cpp": 20000,
    "c": 20000,
    "csharp": 20000,
    "ruby": 20000,
    "php": 20000,
    "swift": 20000,
    "kotlin": 10000,
    "scala": 10000,
    "shell": 10000,
    # Established niche: start at 5K
    "haskell": 5000,
    "elixir": 5000,
    "erlang": 2000,
    "clojure": 5000,
    "ocaml": 2000,
    "fsharp": 2000,
    "r": 5000,
    "julia": 5000,
    "lua": 5000,
    "perl": 2000,
    # Small ecosystems: start at 1K
    "elm": 1000,
    "purescript": 500,
    "nim": 1000,
    "crystal": 1000,
    "zig": 2000,
    "d": 500,
    "ada": 500,
    "fortran": 500,
    "racket": 500,
    "scheme": 500,
    "commonlisp": 500,
    # Very small/new: start at 200
    "gleam": 200,
    "roc": 100,
    "unison": 100,
    "koka": 100,
    "vale": 100,
    "odin": 200,
    "pony": 200,
    "idris": 200,
    "agda": 200,
    "coq": 500,
    "lean": 500,
}

# Default starting threshold for unlisted languages
DEFAULT_START_THRESHOLD = 1000

# Mainstream languages (high activity, many repos)
# Use standard star threshold (50+)
MAINSTREAM_LANGUAGES: list[str] = [
    "python",
    "javascript",
    "typescript",
    "java",
    "go",
    "rust",
    "cpp",
    "c",
    "ruby",
    "php",
    "csharp",  # C#
    "swift",
    "kotlin",
    "scala",
    "shell",  # bash/shell scripts
]

# Niche high-quality languages (functional, academic, specialized)
# Use lower star threshold (20+) as there are fewer repos
NICHE_LANGUAGES: list[str] = [
    # Functional programming
    "haskell",
    "ocaml",
    "fsharp",  # F#
    "elixir",
    "erlang",
    "clojure",
    "elm",
    "purescript",
    # Lisp family
    "commonlisp",
    "scheme",
    "racket",
    # Proof assistants & dependently typed
    "agda",
    "idris",
    "coq",
    "lean",
    # ML/Data Science
    "r",
    "julia",
    "matlab",
    # Systems/Low-level
    "nim",
    "d",
    "ada",
    "fortran",
    "assembly",
    # Logic programming
    "prolog",
    # Other specialized
    "lua",
    "perl",
    "tcl",
    "forth",
    "smalltalk",
    "crystal",
    "v",  # Vlang
]

# Emerging/new languages (growing ecosystems)
# Use lowest star threshold (10+) as many are very new
EMERGING_LANGUAGES: list[str] = [
    "zig",
    "gleam",
    "mojo",
    "pony",
    "roc",
    "unison",
    "koka",
    "vale",
    "odin",
    "carbon",  # Google's Carbon
    "bun",  # Bun's Zig-based runtime files
]

# Star thresholds by language category
MAINSTREAM_MIN_STARS = 50
NICHE_MIN_STARS = 20
EMERGING_MIN_STARS = 10

# Combined for backward compatibility
LANGUAGES: list[str] = MAINSTREAM_LANGUAGES


@dataclass
class DiscoveryResult:
    """Result of repository discovery."""

    repos: list[RepoMetadata]
    total_found: int


class RepoDiscovery:
    """Repository discovery using GitHub API.

    Uses star-range pagination to bypass GitHub's 1000-result limit per search.
    For each star range, queries up to 10 pages (1000 results).
    """

    def __init__(self, client: RateLimitedClient, db: CrawlDatabase):
        self.client = client
        self.db = db

    def _parse_repo(self, data: dict[str, Any]) -> RepoMetadata:
        """Parse GitHub API response into RepoMetadata."""
        return RepoMetadata(
            full_name=data["full_name"],
            stars=data["stargazers_count"],
            language=data.get("language"),
            license_key=data.get("license", {}).get("key") if data.get("license") else None,
            size_kb=data.get("size", 0),
            default_branch=data.get("default_branch", "main"),
            fork=data.get("fork", False),
            archived=data.get("archived", False),
            created_at=datetime.fromisoformat(data["created_at"].replace("Z", "+00:00"))
            if data.get("created_at")
            else None,
            updated_at=datetime.fromisoformat(data["updated_at"].replace("Z", "+00:00"))
            if data.get("updated_at")
            else None,
        )

    async def discover_by_star_range(
        self,
        min_stars: int,
        max_stars: int | None = None,
        language: str | None = None,
        exclude_forks: bool = True,
        max_pages: int = 10,
    ) -> tuple[list[RepoMetadata], int]:
        """Discover repos within a star range.

        Returns:
            Tuple of (list of repos, total_count from API)
            total_count indicates how many repos match (may exceed 1000 limit)
        """
        # Build query
        query = (
            f"stars:{min_stars}..{max_stars}" if max_stars
            else f"stars:>={min_stars}"
        )

        if language:
            query += f" language:{language}"

        if exclude_forks:
            query += " fork:false"

        try:
            results, total_count = await self.client.search_repos_with_count(
                query=query,
                sort="stars",
                order="desc",
                max_pages=max_pages,
            )
            return [self._parse_repo(r) for r in results], total_count
        except Exception as e:
            print(f"Error searching {query}: {e}")
            return [], 0

    async def discover_top_repos(
        self,
        target_count: int = 100000,
        min_stars: int = 100,
        exclude_forks: bool = True,
        progress_callback: Callable | None = None,
    ) -> DiscoveryResult:
        """Discover top repositories by stars.

        Uses adaptive subdivision - starts with broad query and only subdivides
        when results exceed GitHub's 1000-result limit.

        Args:
            target_count: Target number of repos to discover
            min_stars: Minimum star threshold
            exclude_forks: Whether to exclude forked repositories
            progress_callback: Optional callback(discovered, target) for progress

        Returns:
            DiscoveryResult with discovered repos
        """
        all_repos: list[RepoMetadata] = []
        seen_names: set = set()

        # Start with a broad query
        repos, total_count = await self.discover_by_star_range(
            min_stars=min_stars,
            exclude_forks=exclude_forks,
        )

        # Deduplicate
        for repo in repos:
            if repo.full_name not in seen_names:
                seen_names.add(repo.full_name)
                all_repos.append(repo)

        if progress_callback:
            progress_callback(len(all_repos), target_count)

        # If we got all results (total_count <= 1000), we're done
        # Otherwise, we need to subdivide by star ranges
        if total_count > GITHUB_SEARCH_LIMIT and len(all_repos) < target_count:
            print(f"Query has {total_count} results, subdividing by star ranges...")
            all_repos = await self._discover_with_subdivision(
                min_stars=min_stars,
                exclude_forks=exclude_forks,
                seen_names=seen_names,
                target_count=target_count,
                progress_callback=progress_callback,
            )

        # Sort by stars descending
        all_repos.sort(key=lambda r: r.stars, reverse=True)

        # Trim to target
        if len(all_repos) > target_count:
            all_repos = all_repos[:target_count]

        return DiscoveryResult(
            repos=all_repos,
            total_found=len(all_repos),
        )

    async def _discover_with_subdivision(
        self,
        min_stars: int,
        exclude_forks: bool,
        seen_names: set,
        target_count: int,
        language: str | None = None,
        progress_callback: Callable | None = None,
    ) -> list[RepoMetadata]:
        """Recursively subdivide star ranges when results exceed 1000.

        Strategy: binary search through star thresholds to find ranges
        with <1000 results each.
        """
        all_repos: list[RepoMetadata] = list()
        for _name in seen_names:
            pass  # seen_names already tracked, repos added below

        # Get thresholds above our min_stars
        thresholds = [t for t in SUBDIVISION_THRESHOLDS if t >= min_stars]
        if not thresholds or thresholds[-1] > min_stars:
            thresholds.append(min_stars)

        # Create ranges from thresholds: [(100000, None), (50000, 99999), ...]
        ranges = []
        for i, threshold in enumerate(thresholds):
            if i == 0:
                ranges.append((threshold, None))  # top range: threshold+
            else:
                prev_threshold = thresholds[i - 1]
                ranges.append((threshold, prev_threshold - 1))

        for range_min, range_max in ranges:
            if len(all_repos) >= target_count:
                break

            repos, _total_count = await self.discover_by_star_range(
                min_stars=range_min,
                max_stars=range_max,
                language=language,
                exclude_forks=exclude_forks,
            )

            # Deduplicate and add
            for repo in repos:
                if repo.full_name not in seen_names:
                    seen_names.add(repo.full_name)
                    all_repos.append(repo)

            if progress_callback:
                progress_callback(len(all_repos), target_count)

            # Small delay between queries
            await asyncio.sleep(0.1)

        return all_repos

    async def _discover_language_adaptive(
        self,
        language: str,
        min_stars: int,
        max_stars: int | None,
        exclude_forks: bool,
        seen_names: set,
        tier_name: str,
        target_count: int | None = None,
        current_total: int = 0,
    ) -> tuple[int, int]:
        """Discover repos for a single language, starting high and backing off.

        Strategy:
        1. Start at language-appropriate threshold (100K for Python, 200 for Gleam)
        2. Work down through progressively lower thresholds
        3. Stop when we hit the target OR a threshold returns 0 results

        Args:
            language: Language to query
            min_stars: Minimum star threshold (floor)
            max_stars: Maximum star threshold (None for unbounded)
            exclude_forks: Whether to exclude forks
            seen_names: Set of already-seen repo names (modified in place)
            tier_name: Name for logging
            target_count: Optional target to stop early
            current_total: Current total repos discovered

        Returns:
            Tuple of (new_repos_stored, total_repos_found)
        """
        # Get language-specific starting threshold
        start_threshold = LANGUAGE_START_THRESHOLD.get(language.lower(), DEFAULT_START_THRESHOLD)

        # Get thresholds from high to low, filtered by min_stars and start_threshold
        thresholds = sorted(
            [t for t in SUBDIVISION_THRESHOLDS if t >= min_stars and t <= start_threshold],
            reverse=True
        )
        if not thresholds or thresholds[-1] > min_stars:
            thresholds.append(min_stars)

        total_new = 0
        total_found = 0

        # Track the upper bound as we descend through thresholds
        upper_bound = max_stars

        for _i, threshold in enumerate(thresholds):
            # Check if we've hit our target
            if target_count and (current_total + total_new) >= target_count:
                break

            # Check if this threshold was already completed
            if await self.db.is_discovery_complete(tier_name, language, threshold):
                # Update upper_bound for next iteration
                upper_bound = threshold - 1 if threshold > min_stars else None
                continue

            # Query this star range
            repos, query_total = await self.discover_by_star_range(
                min_stars=threshold,
                max_stars=upper_bound,
                language=language,
                exclude_forks=exclude_forks,
            )

            # Store results
            new_count = 0
            for repo in repos:
                if repo.full_name not in seen_names:
                    seen_names.add(repo.full_name)
                    is_new = await self.db.upsert_repo(repo)
                    if is_new:
                        new_count += 1

            total_new += new_count
            total_found += len(repos)

            # Mark this range as complete
            await self.db.mark_discovery_complete(
                tier=tier_name,
                language=language,
                star_range_low=threshold,
                star_range_high=upper_bound,
                repos_found=len(repos),
            )

            # Log progress
            if query_total > GITHUB_SEARCH_LIMIT:
                print(
                    f"    {threshold}-{upper_bound or '∞'} stars:"
                    f" {len(repos)} retrieved ({query_total} total, truncated)"
                )
            elif len(repos) > 0:
                print(f"    {threshold}-{upper_bound or '∞'} stars: {len(repos)} repos")

            # If this range returned nothing, no point going lower
            if len(repos) == 0 and query_total == 0:
                print(f"    No more repos at {threshold}+ stars, stopping")
                break

            # Update upper bound for next iteration
            upper_bound = threshold - 1

        return total_new, total_found

    async def _discover_language_tier(
        self,
        languages: list[str],
        min_stars: int,
        exclude_forks: bool,
        seen_names: set,
        target_count: int,
        current_total: int,
        tier_name: str,
        resume: bool = True,
    ) -> int:
        """Discover repos for a specific language tier.

        Strategy: Start with high star thresholds and work down until we
        hit the target or run out of repos.

        Args:
            languages: List of languages to query
            min_stars: Minimum star threshold (floor) for this tier
            exclude_forks: Whether to exclude forks
            seen_names: Set of already-seen repo names (modified in place)
            target_count: Target total repos
            current_total: Current count of stored repos
            tier_name: Name for logging (e.g., "mainstream", "niche")
            resume: If True, skip already-completed queries (default: True)

        Returns:
            Number of new repos stored in this tier
        """
        tier_stored = 0
        skipped_languages = 0

        print(f"\n=== Discovering {tier_name} languages (min {min_stars} stars) ===")

        for lang in languages:
            if current_total + tier_stored >= target_count:
                break

            # Check if this language was already fully completed (resumability)
            # We check if the lowest threshold (min_stars) is complete
            if resume and await self.db.is_discovery_complete(tier_name, lang, min_stars):
                skipped_languages += 1
                continue

            print(f"Discovering {lang} repos (down to {min_stars} stars)...")

            new_count, total_found = await self._discover_language_adaptive(
                language=lang,
                min_stars=min_stars,
                max_stars=None,
                exclude_forks=exclude_forks,
                seen_names=seen_names,
                tier_name=tier_name,
                target_count=target_count,
                current_total=current_total + tier_stored,
            )

            tier_stored += new_count
            print(
                f"  Total for {lang}: {total_found} found, {new_count} new"
                f" (tier: {tier_stored}, total: {current_total + tier_stored})"
            )

        if skipped_languages > 0:
            print(f"  (Skipped {skipped_languages} already-completed languages)")

        return tier_stored

    async def discover_and_store(
        self,
        target_count: int = 100000,
        min_stars: int = 50,
        exclude_forks: bool = True,
        languages: list[str] | None = None,
        include_niche: bool = True,
        include_emerging: bool = True,
        resume: bool = True,
        force_refresh: bool = False,
    ) -> int:
        """Discover repos and store in database.

        Uses a tiered approach with different star thresholds per language category:
        - Mainstream languages (Python, JS, etc.): 50+ stars
        - Niche languages (Haskell, OCaml, etc.): 20+ stars
        - Emerging languages (Zig, Gleam, etc.): 10+ stars

        This enables discovering high-quality repos even in smaller ecosystems.

        The discovery process is resumable - if interrupted, it will skip
        already-completed queries on the next run.

        Args:
            target_count: Target number of repos
            min_stars: Base minimum star threshold (used for mainstream)
            exclude_forks: Whether to exclude forks
            languages: Override languages to query (disables tiered approach)
            include_niche: Include niche/academic languages
            include_emerging: Include emerging/new languages
            resume: If True, skip already-completed queries (default: True)
            force_refresh: If True, clear discovery progress and start fresh

        Returns:
            Total number of NEW repos stored (excludes metadata updates)
        """
        # Optionally clear progress to force re-discovery
        if force_refresh:
            cleared = await self.db.clear_discovery_progress()
            print(f"Cleared {cleared} discovery progress entries for fresh start")

        # Show discovery progress stats
        if resume:
            disc_stats = await self.db.get_discovery_stats()
            if disc_stats["total_queries_complete"] > 0:
                print(
                    f"Resuming discovery: {disc_stats['total_queries_complete']}"
                    " queries already complete"
                )
                for tier, stats in disc_stats["tiers"].items():
                    print(
                        f"  {tier}: {stats['queries_complete']} queries,"
                        f" {stats['repos_found']} repos"
                    )

        total_stored = 0
        seen_names: set = set()

        # If custom languages provided, use adaptive approach per language
        if languages:
            print(f"\n=== Discovering custom languages (down to {min_stars} stars) ===")

            for lang in languages:
                if total_stored >= target_count:
                    break

                print(f"Discovering {lang} repos (down to {min_stars} stars)...")

                new_count, total_found = await self._discover_language_adaptive(
                    language=lang,
                    min_stars=min_stars,
                    max_stars=None,
                    exclude_forks=exclude_forks,
                    seen_names=seen_names,
                    tier_name="custom",
                    target_count=target_count,
                    current_total=total_stored,
                )

                total_stored += new_count
                print(
                    f"  Total for {lang}: {total_found} found,"
                    f" {new_count} new (total: {total_stored})"
                )

            return total_stored

        # Tiered approach: different star thresholds per language category

        # 1. Mainstream languages (highest quality threshold)
        mainstream_stars = max(min_stars, MAINSTREAM_MIN_STARS)
        stored = await self._discover_language_tier(
            languages=MAINSTREAM_LANGUAGES,
            min_stars=mainstream_stars,
            exclude_forks=exclude_forks,
            seen_names=seen_names,
            target_count=target_count,
            current_total=total_stored,
            tier_name="mainstream",
            resume=resume,
        )
        total_stored += stored

        # 2. Niche languages (lower threshold for smaller ecosystems)
        if include_niche and total_stored < target_count:
            stored = await self._discover_language_tier(
                languages=NICHE_LANGUAGES,
                min_stars=NICHE_MIN_STARS,
                exclude_forks=exclude_forks,
                seen_names=seen_names,
                target_count=target_count,
                current_total=total_stored,
                tier_name="niche",
                resume=resume,
            )
            total_stored += stored

        # 3. Emerging languages (lowest threshold for new ecosystems)
        if include_emerging and total_stored < target_count:
            stored = await self._discover_language_tier(
                languages=EMERGING_LANGUAGES,
                min_stars=EMERGING_MIN_STARS,
                exclude_forks=exclude_forks,
                seen_names=seen_names,
                target_count=target_count,
                current_total=total_stored,
                tier_name="emerging",
                resume=resume,
            )
            total_stored += stored

        print(f"\n=== Discovery complete: {total_stored} new repos stored ===")
        return total_stored

    async def validate_repo(self, full_name: str) -> RepoMetadata | None:
        """Validate a repo still exists and get current metadata.

        Returns None if repo doesn't exist or is inaccessible.
        """
        try:
            data = await self.client.get_repo(full_name)
            return self._parse_repo(data)
        except Exception:
            return None

    async def resolve_redirect(self, full_name: str) -> str | None:
        """Check if repo has been renamed and return new name."""
        try:
            data = await self.client.get_repo(full_name)
            current_name = data["full_name"]
            if current_name.lower() != full_name.lower():
                return current_name
            return None
        except Exception:
            return None
