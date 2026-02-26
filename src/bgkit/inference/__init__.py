"""Reusable inference client for llama-server (OpenAI-compatible API)."""

from bgkit.inference.client import LlamaClient
from bgkit.inference.config import InferenceConfig

__all__ = ["InferenceConfig", "LlamaClient"]
