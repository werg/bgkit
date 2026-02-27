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


class TestCallLocal:
    """Test that call_local routes to large/small clients by content size."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_call_local_small_content_routes_large(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            result, backend = self.gd.call_local("describe this file", content_chars=100)
            assert result is not None
            assert "[large]" in result
            assert backend == "local-large"

    def test_call_local_big_content_routes_small(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            result, backend = self.gd.call_local("describe this file", content_chars=5000)
            assert result is not None
            assert "[small]" in result
            assert backend == "local-small"

    def test_batch_routes_by_content_length(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            results = self.gd.call_local_batch(
                ["short", "long"],
                content_lengths=[100, 5000],
            )
            assert len(results) == 2
            desc0, b0 = results[0]
            desc1, b1 = results[1]
            assert "[large]" in desc0
            assert b0 == "local-large"
            assert "[small]" in desc1
            assert b1 == "local-small"

    def test_batch_defaults_to_large_without_lengths(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            results = self.gd.call_local_batch(["p1", "p2"])
            assert all(b == "local-large" for _, b in results)


class TestGenerateDescription:
    """Test the backend dispatcher."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_local_backend_small_content(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            result, backend = self.gd.generate_description(
                "test prompt", "local", content_chars=100,
            )
            assert backend == "local-large"
            assert result is not None
            assert "[large]" in result

    def test_local_backend_big_content(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small):
            result, backend = self.gd.generate_description(
                "test prompt", "local", content_chars=5000,
            )
            assert backend == "local-small"
            assert result is not None
            assert "[small]" in result

    def test_haiku_backend(self):
        with patch.object(self.gd, "call_claude", return_value="haiku response"):
            result, backend = self.gd.generate_description("test prompt", "haiku")
            assert backend == "haiku"
            assert result == "haiku response"

    def test_mixed_backend_cycles(self):
        large = _make_mock_client("large")
        small = _make_mock_client("small")
        cycle = itertools.cycle(["haiku", "local"])

        with patch.object(self.gd, "_llama_client_large", large), \
             patch.object(self.gd, "_llama_client_small", small), \
             patch.object(self.gd, "call_claude", return_value="haiku response"):
            _r1, b1 = self.gd.generate_description("p1", "mixed", cycle, content_chars=100)
            _r2, b2 = self.gd.generate_description("p2", "mixed", cycle, content_chars=100)
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
                self.gd.init_local_client(url="http://localhost:9999", readiness_timeout=1.0)
            assert exc_info.value.code == 1
