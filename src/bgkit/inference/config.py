"""Configuration for the llama-server inference client."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class InferenceConfig:
    """Configuration for a LlamaClient instance."""

    base_url: str = "http://localhost:8080"
    max_concurrent: int = 32
    timeout: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 2.0
    max_new_tokens: int = 512
    temperature: float = 0.0
