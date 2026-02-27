"""Tests for backend dispatch logic in generate_descriptions.py."""

from __future__ import annotations

import itertools
from unittest.mock import patch

import httpx
import pytest

from bgkit.inference.client import LlamaClient
from bgkit.inference.config import InferenceConfig


def _make_completion_response(content: str) -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
    }


def _make_mock_client(name: str) -> LlamaClient:
    """Create a LlamaClient with a mock transport that tags responses with the client name."""

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(
                200, json=_make_completion_response(f"[{name}] response")
            )
        return httpx.Response(404)

    config = InferenceConfig(base_url="http://test", max_concurrent=4)
    client = LlamaClient(config)
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        timeout=5.0,
    )
    return client


def _patch_all_clients(gd, large, small, tiny):
    """Context manager to patch all three llama clients."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(gd, "_llama_client_large", large))
    stack.enter_context(patch.object(gd, "_llama_client_small", small))
    stack.enter_context(patch.object(gd, "_llama_client_tiny", tiny))
    return stack


class TestTierRouting:
    """Test _is_tiny_routable and _pick_tier classification."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_json_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("config.json", "JSON", 2000)

    def test_yaml_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("deploy.yaml", "YAML", 1000)

    def test_markdown_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("README.md", "Markdown", 5000)

    def test_test_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("tests/test_auth.py", "Python", 3000)

    def test_test_suffix_routes_tiny(self):
        assert self.gd._is_tiny_routable("src/auth_test.go", "Go", 2000)

    def test_init_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("src/__init__.py", "Python", 100)

    def test_lockfile_routes_tiny(self):
        assert self.gd._is_tiny_routable("yarn.lock", None, 50000)

    def test_very_short_file_routes_tiny(self):
        assert self.gd._is_tiny_routable("complex_logic.py", "Python", 200)

    def test_migrations_routes_tiny(self):
        assert self.gd._is_tiny_routable("db/migrations/001_init.py", "Python", 2000)

    def test_pyproject_toml_routes_tiny(self):
        assert self.gd._is_tiny_routable("pyproject.toml", "TOML", 3000)

    def test_complex_python_not_tiny(self):
        assert not self.gd._is_tiny_routable("src/engine.py", "Python", 4000)

    def test_complex_rust_not_tiny(self):
        assert not self.gd._is_tiny_routable("src/main.rs", "Rust", 2000)

    def test_pick_tier_tiny(self):
        assert self.gd._pick_tier(1000, "config.json", "JSON", 1000) == "tiny"

    def test_pick_tier_small_big_source(self):
        assert self.gd._pick_tier(5000, "src/engine.py", "Python", 5000) == "small"

    def test_pick_tier_large_moderate_source(self):
        assert self.gd._pick_tier(2000, "src/engine.py", "Python", 2000) == "large"

    def test_pick_tier_tiny_beats_small_threshold(self):
        # A large JSON file should still go to tiny, not small
        assert self.gd._pick_tier(5000, "data.json", "JSON", 5000) == "tiny"


class TestCallLocal:
    """Test that call_local routes to the correct tier client."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_moderate_source_routes_large(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            result, backend = self.gd.call_local(
                "describe", content_chars=2000, file_path="src/app.py",
                language="Python", size_bytes=2000,
            )
            assert "[large]" in result
            assert backend == "local-large"

    def test_big_source_routes_small(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            result, backend = self.gd.call_local(
                "describe", content_chars=5000, file_path="src/app.py",
                language="Python", size_bytes=5000,
            )
            assert "[small]" in result
            assert backend == "local-small"

    def test_config_file_routes_tiny(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            result, backend = self.gd.call_local(
                "describe", content_chars=1000, file_path="config.json",
                language="JSON", size_bytes=1000,
            )
            assert "[tiny]" in result
            assert backend == "local-tiny"

    def test_batch_routes_by_tier(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            results = self.gd.call_local_batch(
                ["p1", "p2", "p3"],
                tiers=["large", "small", "tiny"],
            )
            assert len(results) == 3
            assert results[0][1] == "local-large"
            assert results[1][1] == "local-small"
            assert results[2][1] == "local-tiny"
            assert "[large]" in results[0][0]
            assert "[small]" in results[1][0]
            assert "[tiny]" in results[2][0]

    def test_batch_defaults_to_large_without_tiers(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            results = self.gd.call_local_batch(["p1", "p2"])
            assert all(b == "local-large" for _, b in results)


class TestGenerateDescription:
    """Test the backend dispatcher."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_local_backend_source_file(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            result, backend = self.gd.generate_description(
                "test prompt", "local", content_chars=2000,
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            assert backend == "local-large"
            assert "[large]" in result

    def test_local_backend_config_file(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")

        with _patch_all_clients(self.gd, large, small, tiny):
            result, backend = self.gd.generate_description(
                "test prompt", "local", content_chars=500,
                file_path="config.yaml", language="YAML", size_bytes=500,
            )
            assert backend == "local-tiny"
            assert "[tiny]" in result

    def test_haiku_backend(self):
        with patch.object(self.gd, "call_claude", return_value="haiku response"):
            result, backend = self.gd.generate_description("test prompt", "haiku")
            assert backend == "haiku"
            assert result == "haiku response"

    def test_mixed_backend_cycles(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        tiny = _make_mock_client("tiny")
        cycle = itertools.cycle(["haiku", "local"])

        with _patch_all_clients(self.gd, large, small, tiny), \
             patch.object(self.gd, "call_claude", return_value="haiku response"):
            _r1, b1 = self.gd.generate_description(
                "p1", "mixed", cycle, content_chars=2000,
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            _r2, b2 = self.gd.generate_description(
                "p2", "mixed", cycle, content_chars=2000,
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            assert b1 == "haiku"
            assert b2 == "local-large"


class TestReadinessGate:
    """Test that init_local_client exits when server is unavailable."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_exits_when_large_server_unavailable(self):
        with patch.object(
            LlamaClient,
            "wait_ready_sync",
            return_value=False,
        ):
            with pytest.raises(SystemExit) as exc_info:
                self.gd.init_local_client(
                    url_large="http://localhost:9999", readiness_timeout=1.0,
                )
            assert exc_info.value.code == 1
