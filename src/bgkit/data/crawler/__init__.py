"""GitHub repository crawler for collecting training data."""

from bgkit.data.crawler.config import CrawlerConfig
from bgkit.data.crawler.database import CrawlDatabase, RepoMetadata, RepoStatus
from bgkit.data.crawler.discovery import RepoDiscovery
from bgkit.data.crawler.downloader import DownloadResult, RepoDownloader
from bgkit.data.crawler.orchestrator import CrawlOrchestrator
from bgkit.data.crawler.rate_limiter import RateLimitedClient, TokenRotator

__all__ = [
    "CrawlDatabase",
    "CrawlOrchestrator",
    "CrawlerConfig",
    "DownloadResult",
    "RateLimitedClient",
    "RepoDiscovery",
    "RepoDownloader",
    "RepoMetadata",
    "RepoStatus",
    "TokenRotator",
]
