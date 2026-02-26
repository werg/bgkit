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
    """Test that call_local uses the llama client."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_call_local_returns_response(self):
        client = _make_mock_client("llama")

        with patch.object(self.gd, "_llama_client", client):
            result = self.gd.call_local("describe this file")
            assert result is not None
            assert "[llama]" in result

    def test_no_client_returns_none(self):
        with patch.object(self.gd, "_llama_client", None):
            result = self.gd.call_local("test")
            assert result is None

    def test_batch_returns_results(self):
        client = _make_mock_client("llama")

        with patch.object(self.gd, "_llama_client", client):
            results = self.gd.call_local_batch(["prompt1", "prompt2"])
            assert len(results) == 2
            assert all(r is not None and "[llama]" in r for r in results)

    def test_batch_no_client_returns_nones(self):
        with patch.object(self.gd, "_llama_client", None):
            results = self.gd.call_local_batch(["p1", "p2"])
            assert results == [None, None]


class TestGenerateDescription:
    """Test the backend dispatcher."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_local_backend(self):
        client = _make_mock_client("llama")

        with patch.object(self.gd, "_llama_client", client):
            result, backend = self.gd.generate_description("test prompt", "local")
            assert backend == "local"
            assert result is not None
            assert "[llama]" in result

    def test_haiku_backend(self):
        with patch.object(self.gd, "call_claude", return_value="haiku response"):
            result, backend = self.gd.generate_description("test prompt", "haiku")
            assert backend == "haiku"
            assert result == "haiku response"

    def test_mixed_backend_cycles(self):
        client = _make_mock_client("llama")
        cycle = itertools.cycle(["haiku", "local"])

        with patch.object(self.gd, "_llama_client", client), patch.object(
            self.gd, "call_claude", return_value="haiku response"
        ):
            _r1, b1 = self.gd.generate_description("p1", "mixed", cycle)
            _r2, b2 = self.gd.generate_description("p2", "mixed", cycle)
            assert b1 == "haiku"
            assert b2 == "local"


class TestReadinessGate:
    """Test that init_local_client exits when server is unavailable."""

    def setup_method(self):
        import scripts.generate_descriptions as gd

        self.gd = gd

    def test_exits_when_server_unavailable(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503)

        with patch.object(
            LlamaClient,
            "wait_ready_sync",
            return_value=False,
        ):
            with pytest.raises(SystemExit) as exc_info:
                self.gd.init_local_client(url="http://localhost:9999", readiness_timeout=1.0)
            assert exc_info.value.code == 1
