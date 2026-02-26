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

    def get_chat_template_kwargs(self) -> dict[str, Any] | None:
        """Return chat_template_kwargs as a fresh dict, or None."""
        if self._chat_template_kwargs is None:
            return None
        return dict(self._chat_template_kwargs)

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
}

DEFAULT_PROFILE = ModelProfile(name="default")


def resolve_profile(model_filename: str) -> ModelProfile:
    """Match GGUF filename to known profile.

    Candidates sorted by key length descending so "Qwen3.5" matches before "Qwen3".
    Case-insensitive. Returns DEFAULT_PROFILE if no match.
    """
    lower = model_filename.lower()
    for key in sorted(MODEL_PROFILES, key=len, reverse=True):
        if key.lower() in lower:
            return MODEL_PROFILES[key]
    logger.warning("No model profile matched for %r, using default", model_filename)
    return DEFAULT_PROFILE
