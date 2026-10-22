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


def _patch_all_clients(gd, primary, fast):
    """Context manager to patch both inference clients."""
    from contextlib import ExitStack

    stack = ExitStack()
    stack.enter_context(patch.object(gd, "_client_primary", primary))
    stack.enter_context(patch.object(gd, "_client_fast", fast))
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

    def test_pick_tier_fast(self):
        assert self.gd._pick_tier("config.json", "JSON", 1000) == "fast"

    def test_pick_tier_primary_big_source(self):
        assert self.gd._pick_tier("src/engine.py", "Python", 5000) == "primary"

    def test_pick_tier_primary_moderate_source(self):
        assert self.gd._pick_tier("src/engine.py", "Python", 2000) == "primary"

    def test_pick_tier_fast_beats_size(self):
        # A large JSON file should still go to fast
        assert self.gd._pick_tier("data.json", "JSON", 5000) == "fast"


class TestCallLocal:
    """Test that call_local routes to the correct tier client."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_source_file_routes_primary(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            result, backend = self.gd.call_local(
                "describe", file_path="src/app.py",
                language="Python", size_bytes=2000,
            )
            assert "[primary]" in result
            assert backend == "local-primary"

    def test_big_source_routes_primary(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            result, backend = self.gd.call_local(
                "describe", file_path="src/app.py",
                language="Python", size_bytes=5000,
            )
            assert "[primary]" in result
            assert backend == "local-primary"

    def test_config_file_routes_fast(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            result, backend = self.gd.call_local(
                "describe", file_path="config.json",
                language="JSON", size_bytes=1000,
            )
            assert "[fast]" in result
            assert backend == "local-fast"

    def test_batch_routes_by_tier(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            results = self.gd.call_local_batch(
                ["p1", "p2"],
                tiers=["primary", "fast"],
            )
            assert len(results) == 2
            assert results[0][1] == "local-primary"
            assert results[1][1] == "local-fast"
            assert "[primary]" in results[0][0]
            assert "[fast]" in results[1][0]

    def test_batch_defaults_to_primary_without_tiers(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            results = self.gd.call_local_batch(["p1", "p2"])
            assert all(b == "local-primary" for _, b in results)


class TestGenerateDescription:
    """Test the backend dispatcher."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_local_backend_source_file(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            result, backend = self.gd.generate_description(
                "test prompt", "local",
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            assert backend == "local-primary"
            assert "[primary]" in result

    def test_local_backend_config_file(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")

        with _patch_all_clients(self.gd, primary, fast):
            result, backend = self.gd.generate_description(
                "test prompt", "local", file_path="config.yaml", language="YAML", size_bytes=500,
            )
            assert backend == "local-fast"
            assert "[fast]" in result

    def test_haiku_backend(self):
        with patch.object(self.gd, "call_claude", return_value="haiku response"):
            result, backend = self.gd.generate_description("test prompt", "haiku")
            assert backend == "haiku"
            assert result == "haiku response"

    def test_mixed_backend_cycles(self):
        primary = _make_mock_client("primary")
        fast = _make_mock_client("fast")
        cycle = itertools.cycle(["haiku", "local"])

        with _patch_all_clients(self.gd, primary, fast), \
             patch.object(self.gd, "call_claude", return_value="haiku response"):
            _r1, b1 = self.gd.generate_description(
                "p1", "mixed", cycle,
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            _r2, b2 = self.gd.generate_description(
                "p2", "mixed", cycle,
                file_path="src/app.py", language="Python", size_bytes=2000,
            )
            assert b1 == "haiku"
            assert b2 == "local-primary"


class TestReadinessGate:
    """Test that init_local_client exits when server is unavailable."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_exits_when_primary_server_unavailable(self):
        with patch.object(
            LlamaClient,
            "wait_ready_sync",
            return_value=False,
        ):
            with pytest.raises(SystemExit) as exc_info:
                self.gd.init_local_client(
                    url_primary="http://localhost:9999", readiness_timeout=1.0,
                )
            assert exc_info.value.code == 1
