"""Configuration for the inference client (llama-server and vLLM)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bgkit.inference.models import ModelProfile


@dataclass
class InferenceConfig:
    """Configuration for a LlamaClient instance."""

    base_url: str = "http://localhost:8080"
    max_concurrent: int = 16
    timeout: float = 120.0
    max_retries: int = 3
    retry_base_delay: float = 2.0
    max_new_tokens: int = 512
    temperature: float = 0.0
    model_profile: ModelProfile | None = None
    backend_type: str = "auto"  # "llama", "vllm", or "auto" (probe /version)
