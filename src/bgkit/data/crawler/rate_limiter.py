"""GitHub API rate limiting with multi-token rotation.

Manages multiple GitHub API tokens to maximize throughput while respecting rate limits.

GitHub Rate Limits:
- Core API: 5000 requests/hour per token
- Search API: 30 requests/minute (separate limit, per-IP not per-token)
- Secondary limits: Undocumented, can trigger 403/429 for rapid requests

This module implements conservative throttling to avoid secondary rate limits.
"""

import asyncio
import hashlib
import random
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import httpx


# Search API has a separate 30 req/min limit. Be conservative.
SEARCH_API_MIN_DELAY = 2.5  # seconds between search requests
# Additional jitter to avoid synchronized bursts
SEARCH_API_JITTER = 0.5  # random 0-0.5s added to delay


@dataclass
class TokenState:
    """State of a single GitHub API token."""

    token: str
    token_hash: str
    remaining: int = 5000  # Default GitHub rate limit
    reset_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_available(self) -> bool:
        """Check if token has available requests."""
        if self.remaining > 0:
            return True
        return datetime.now(timezone.utc) >= self.reset_at


class TokenRotator:
    """Manages multiple GitHub tokens for rate limit distribution.

    Strategy:
    - Use tokens with highest remaining quota first
    - Track per-token rate limits from API responses
    - Automatically rotate to next available token
    - Sleep when all tokens exhausted
    """

    def __init__(self, tokens: List[str]):
        self.tokens: List[TokenState] = []
        self._lock = asyncio.Lock()

        for token in tokens:
            token_hash = hashlib.sha256(token.encode()).hexdigest()[:16]
            self.tokens.append(
                TokenState(
                    token=token,
                    token_hash=token_hash,
                )
            )

    async def get_token(self) -> str:
        """Get next available token, waiting if necessary."""
        async with self._lock:
            while True:
                # Sort by remaining requests (descending)
                available = [t for t in self.tokens if t.is_available]

                if available:
                    # Return token with most remaining requests
                    available.sort(key=lambda t: t.remaining, reverse=True)
                    token = available[0]
                    # Decrement to avoid thundering herd
                    token.remaining = max(0, token.remaining - 1)
                    return token.token

                # All tokens exhausted - calculate wait time
                if not self.tokens:
                    raise RuntimeError("No API tokens configured")

                next_reset = min(t.reset_at for t in self.tokens)
                wait_seconds = (next_reset - datetime.now(timezone.utc)).total_seconds()

                if wait_seconds > 0:
                    print(f"Rate limited. Waiting {wait_seconds:.0f}s until reset.")
                    await asyncio.sleep(wait_seconds + 1)

    async def update_limits(
        self, token: str, remaining: int, reset_timestamp: int
    ) -> None:
        """Update token state from API response headers."""
        reset_at = datetime.fromtimestamp(reset_timestamp, tz=timezone.utc)

        async with self._lock:
            for t in self.tokens:
                if t.token == token:
                    t.remaining = remaining
                    t.reset_at = reset_at
                    break

    def get_status(self) -> List[Dict[str, Any]]:
        """Get status of all tokens (for debugging/monitoring)."""
        return [
            {
                "hash": t.token_hash,
                "remaining": t.remaining,
                "reset_at": t.reset_at.isoformat(),
                "available": t.is_available,
            }
            for t in self.tokens
        ]


class RateLimitedClient:
    """GitHub API client with automatic rate limit handling.

    Features:
    - Automatic token rotation
    - Rate limit header parsing
    - Retry on rate limit errors with exponential backoff
    - Pagination support
    - Search API throttling (separate 30 req/min limit)
    - Secondary rate limit detection and handling
    """

    BASE_URL = "https://api.github.com"
    MAX_RETRIES = 3
    SECONDARY_RATE_LIMIT_BACKOFF = [60, 120, 300]  # seconds

    def __init__(self, token_rotator: TokenRotator, timeout: float = 30.0):
        self.rotator = token_rotator
        self.timeout = timeout
        self._client: Optional[httpx.AsyncClient] = None
        self._last_search_time: float = 0

    async def __aenter__(self) -> "RateLimitedClient":
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._client:
            await self._client.aclose()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def _throttle_search(self) -> None:
        """Enforce minimum delay between search API requests."""
        import time
        now = time.monotonic()
        elapsed = now - self._last_search_time
        delay = SEARCH_API_MIN_DELAY + random.uniform(0, SEARCH_API_JITTER)
        if elapsed < delay:
            await asyncio.sleep(delay - elapsed)
        self._last_search_time = time.monotonic()

    def _is_secondary_rate_limit(self, response: httpx.Response) -> bool:
        """Detect secondary rate limit (abuse detection)."""
        if response.status_code in (403, 429):
            # Check for secondary rate limit indicators
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                return True
            # Check response body for abuse message
            try:
                data = response.json()
                message = data.get("message", "").lower()
                if "abuse" in message or "secondary rate" in message:
                    return True
            except Exception:
                pass
        return False

    async def request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        retry_count: int = 0,
        **kwargs,
    ) -> httpx.Response:
        """Make rate-limited API request with retry logic."""
        client = await self._ensure_client()
        token = await self.rotator.get_token()

        headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github.v3+json",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        url = f"{self.BASE_URL}{endpoint}"

        response = await client.request(
            method, url, headers=headers, params=params, **kwargs
        )

        # Update rate limits from headers
        remaining = int(response.headers.get("X-RateLimit-Remaining", 0))
        reset_timestamp = int(response.headers.get("X-RateLimit-Reset", 0))

        await self.rotator.update_limits(token, remaining, reset_timestamp)

        # Handle secondary rate limits (abuse detection)
        if self._is_secondary_rate_limit(response):
            if retry_count < self.MAX_RETRIES:
                backoff = self.SECONDARY_RATE_LIMIT_BACKOFF[min(retry_count, len(self.SECONDARY_RATE_LIMIT_BACKOFF) - 1)]
                retry_after = response.headers.get("Retry-After")
                if retry_after:
                    try:
                        backoff = int(retry_after)
                    except ValueError:
                        pass
                print(f"Secondary rate limit hit. Backing off for {backoff}s (attempt {retry_count + 1}/{self.MAX_RETRIES})")
                await asyncio.sleep(backoff)
                return await self.request(method, endpoint, params=params, retry_count=retry_count + 1, **kwargs)
            else:
                print(f"Secondary rate limit: max retries exceeded for {endpoint}")

        # Handle primary rate limit errors
        if response.status_code == 403 and remaining == 0:
            # Rate limited - retry with different token
            return await self.request(method, endpoint, params=params, retry_count=retry_count, **kwargs)

        return response

    async def get(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> httpx.Response:
        """GET request."""
        return await self.request("GET", endpoint, params=params)

    async def get_json(
        self, endpoint: str, params: Optional[Dict[str, Any]] = None
    ) -> Any:
        """GET request returning JSON."""
        response = await self.get(endpoint, params=params)
        response.raise_for_status()
        return response.json()

    async def get_paginated(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 10,
    ) -> List[Any]:
        """GET request with pagination, returning all results."""
        if params is None:
            params = {}
        params.setdefault("per_page", 100)

        all_results = []
        page = 1

        while page <= max_pages:
            params["page"] = page
            response = await self.get(endpoint, params=params)
            response.raise_for_status()

            data = response.json()
            if not data:
                break

            if isinstance(data, dict) and "items" in data:
                # Search API returns {items: [...], total_count: ...}
                all_results.extend(data["items"])
                if len(data["items"]) < params["per_page"]:
                    break
            elif isinstance(data, list):
                all_results.extend(data)
                if len(data) < params["per_page"]:
                    break
            else:
                all_results.append(data)
                break

            page += 1

        return all_results

    async def get_repo(self, full_name: str) -> Dict[str, Any]:
        """Get repository details."""
        return await self.get_json(f"/repos/{full_name}")

    async def search_repos(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 100,
        max_pages: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search repositories with pagination and throttling.

        The Search API has a separate rate limit of 30 requests/minute.
        This method enforces delays between requests to stay within limits.
        """
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page,
        }
        results, _ = await self.search_repos_with_count(query, sort, order, per_page, max_pages)
        return results

    async def search_repos_with_count(
        self,
        query: str,
        sort: str = "stars",
        order: str = "desc",
        per_page: int = 100,
        max_pages: int = 10,
    ) -> Tuple[List[Dict[str, Any]], int]:
        """Search repositories and return (results, total_count).

        Returns:
            Tuple of (list of repo dicts, total_count from API)
            total_count indicates how many repos match the query (may exceed 1000)
        """
        params = {
            "q": query,
            "sort": sort,
            "order": order,
            "per_page": per_page,
        }
        return await self.get_paginated_search_with_count("/search/repositories", params, max_pages)

    async def get_paginated_search(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 10,
    ) -> List[Any]:
        """GET request with pagination for search endpoints (with throttling)."""
        if params is None:
            params = {}
        params.setdefault("per_page", 100)

        all_results = []
        page = 1

        while page <= max_pages:
            # Throttle search requests
            await self._throttle_search()

            params["page"] = page
            response = await self.get(endpoint, params=params)

            # Handle errors gracefully for search
            if response.status_code != 200:
                print(f"Search API error: {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            if isinstance(data, dict) and "items" in data:
                # Search API returns {items: [...], total_count: ...}
                all_results.extend(data["items"])
                if len(data["items"]) < params["per_page"]:
                    break
            elif isinstance(data, list):
                all_results.extend(data)
                if len(data) < params["per_page"]:
                    break
            else:
                all_results.append(data)
                break

            page += 1

        return all_results

    async def get_paginated_search_with_count(
        self,
        endpoint: str,
        params: Optional[Dict[str, Any]] = None,
        max_pages: int = 10,
    ) -> Tuple[List[Any], int]:
        """GET request with pagination for search endpoints, returning (results, total_count).

        Returns:
            Tuple of (list of results, total_count from first API response)
            total_count indicates how many items match the query (may exceed 1000 limit)
        """
        if params is None:
            params = {}
        params.setdefault("per_page", 100)

        all_results = []
        total_count = 0
        page = 1

        while page <= max_pages:
            # Throttle search requests
            await self._throttle_search()

            params["page"] = page
            response = await self.get(endpoint, params=params)

            # Handle errors gracefully for search
            if response.status_code != 200:
                print(f"Search API error: {response.status_code}")
                break

            data = response.json()
            if not data:
                break

            if isinstance(data, dict) and "items" in data:
                # Search API returns {items: [...], total_count: ...}
                if page == 1:
                    total_count = data.get("total_count", 0)
                all_results.extend(data["items"])
                if len(data["items"]) < params["per_page"]:
                    break
            elif isinstance(data, list):
                all_results.extend(data)
                if len(data) < params["per_page"]:
                    break
            else:
                all_results.append(data)
                break

            page += 1

        return all_results, total_count

    async def check_rate_limit(self) -> Dict[str, Any]:
        """Check current rate limit status."""
        return await self.get_json("/rate_limit")
