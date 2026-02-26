"""Tests for bgkit.inference.models — profile resolution and immutability."""

from __future__ import annotations

from bgkit.inference.models import DEFAULT_PROFILE, MODEL_PROFILES, ModelProfile, resolve_profile


def test_resolve_qwen35():
    profile = resolve_profile("Qwen3.5-35B-A3B-Q4_K_M.gguf")
    assert profile.name == "Qwen3.5"


def test_resolve_qwen3():
    profile = resolve_profile("Qwen3-0.6B-Q8_0.gguf")
    assert profile.name == "Qwen3"


def test_resolve_glm():
    profile = resolve_profile("GLM-4.7-Flash-Q4_K_M.gguf")
    assert profile.name == "GLM-4"


def test_resolve_unknown():
    profile = resolve_profile("mystery-model.gguf")
    assert profile is DEFAULT_PROFILE
    assert profile.name == "default"
    assert profile.thinking_tag_pattern is None
    assert profile.get_chat_template_kwargs() is None


def test_qwen35_not_matched_as_qwen3():
    """Longest-match-first: 'Qwen3.5' key (7 chars) beats 'Qwen3' (5 chars)."""
    profile = resolve_profile("Qwen3.5-35B-A3B-Q4_K_M.gguf")
    assert profile.name == "Qwen3.5"
    assert profile is MODEL_PROFILES["Qwen3.5"]


def test_case_insensitive():
    profile = resolve_profile("qwen3.5-35b-a3b-q4_k_m.gguf")
    assert profile.name == "Qwen3.5"


def test_default_profile_immutable():
    assert DEFAULT_PROFILE.thinking_tag_pattern is None
    assert DEFAULT_PROFILE.get_chat_template_kwargs() is None


def test_get_chat_template_kwargs_returns_copy():
    """Mutating the returned dict doesn't affect the profile."""
    profile = MODEL_PROFILES["GLM-4"]
    kwargs1 = profile.get_chat_template_kwargs()
    assert kwargs1 == {"enable_thinking": False}
    # Mutate the returned dict
    kwargs1["enable_thinking"] = True
    kwargs1["extra_key"] = "bad"
    # Original profile unchanged
    kwargs2 = profile.get_chat_template_kwargs()
    assert kwargs2 == {"enable_thinking": False}


def test_compile_thinking_re():
    """Profile with a pattern compiles a regex; default returns None."""
    profile = MODEL_PROFILES["GLM-4"]
    regex = profile.compile_thinking_re()
    assert regex is not None
    assert regex.sub("", "<think>stuff</think>\nreal content").strip() == "real content"

    assert DEFAULT_PROFILE.compile_thinking_re() is None
