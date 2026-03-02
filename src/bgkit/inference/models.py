"""Model profiles for inference behavior configuration.

Each ModelProfile captures model-specific settings (thinking tag stripping,
chat_template_kwargs) so the client doesn't need ad-hoc flags.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

_THINK_PATTERN = r"<think>.*?</think>\s*"
_THINKING_DISABLED: tuple[tuple[str, Any], ...] = (("enable_thinking", False),)


@dataclass(frozen=True)
class ModelProfile:
    """Model-specific inference behavior."""

    name: str
    thinking_tag_pattern: str | None = None
    # Immutable storage — use get_chat_template_kwargs() to get a dict copy
    _chat_template_kwargs: tuple[tuple[str, Any], ...] | None = None
    # Extra fields merged into every request payload (e.g. include_reasoning)
    _extra_body: tuple[tuple[str, Any], ...] | None = None

    def get_chat_template_kwargs(self) -> dict[str, Any] | None:
        """Return chat_template_kwargs as a fresh dict, or None."""
        if self._chat_template_kwargs is None:
            return None
        return dict(self._chat_template_kwargs)

    def get_extra_body(self) -> dict[str, Any] | None:
        """Return extra request body fields as a fresh dict, or None."""
        if self._extra_body is None:
            return None
        return dict(self._extra_body)

    def compile_thinking_re(self) -> re.Pattern[str] | None:
        """Compile the thinking tag regex, or None if no pattern."""
        if self.thinking_tag_pattern is None:
            return None
        return re.compile(self.thinking_tag_pattern, re.DOTALL)


MODEL_PROFILES: dict[str, ModelProfile] = {
    "Qwen3.5": ModelProfile(
        name="Qwen3.5",
        thinking_tag_pattern=_THINK_PATTERN,
        _chat_template_kwargs=_THINKING_DISABLED,
    ),
    "Qwen3": ModelProfile(
        name="Qwen3",
        thinking_tag_pattern=_THINK_PATTERN,
        _chat_template_kwargs=_THINKING_DISABLED,
    ),
    "GLM-4": ModelProfile(
        name="GLM-4",
        thinking_tag_pattern=_THINK_PATTERN,
        _chat_template_kwargs=_THINKING_DISABLED,
    ),
    "GPT-OSS": ModelProfile(
        name="GPT-OSS",
        _extra_body=(("include_reasoning", False), ("reasoning_effort", "low")),
    ),
}

DEFAULT_PROFILE = ModelProfile(name="default")


def resolve_profile(model_filename: str) -> ModelProfile:
    """Match a model identifier to a known profile.

    Works with both GGUF filenames (e.g. ``Qwen3-0.6B-Q8_0.gguf``) and
    HuggingFace model IDs (e.g. ``Qwen/Qwen3.5-35B-A3B-FP8``,
    ``openai/gpt-oss-20b``).  Candidates are sorted by key length descending
    so ``Qwen3.5`` matches before ``Qwen3``.  Case-insensitive.
    Returns DEFAULT_PROFILE if no match.
    """
    lower = model_filename.lower()
    for key in sorted(MODEL_PROFILES, key=len, reverse=True):
        if key.lower() in lower:
            return MODEL_PROFILES[key]
    logger.warning("No model profile matched for %r, using default", model_filename)
    return DEFAULT_PROFILE
