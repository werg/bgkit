"""Tests for bgkit.inference.client using httpx MockTransport."""

from __future__ import annotations

import json

import httpx
import pytest

from bgkit.inference.client import LlamaClient, _run_sync
from bgkit.inference.config import InferenceConfig


def _make_completion_response(content: str = "Hello world") -> dict:
    return {
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


def _ok_handler(request: httpx.Request) -> httpx.Response:
    """Mock handler that returns a successful completion."""
    if request.url.path == "/health":
        return httpx.Response(200, json={"status": "ok"})
    if request.url.path == "/v1/chat/completions":
        body = json.loads(request.content)
        prompt = body["messages"][-1]["content"]
        return httpx.Response(200, json=_make_completion_response(f"Response to: {prompt}"))
    return httpx.Response(404)


def _make_client(handler, **config_kwargs) -> LlamaClient:
    """Create a LlamaClient with a mock transport."""
    config = InferenceConfig(base_url="http://test", **config_kwargs)
    client = LlamaClient(config)
    # Replace the async client with one using mock transport
    client._async_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="http://test",
        timeout=5.0,
    )
    return client


def test_generate_success():
    client = _make_client(_ok_handler)
    result = _run_sync(client.generate("What is Python?"))
    assert result is not None
    assert "Response to: What is Python?" in result


def test_generate_with_system_prompt():
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            received["body"] = json.loads(request.content)
            return httpx.Response(200, json=_make_completion_response("ok"))
        return httpx.Response(404)

    client = _make_client(handler)
    _run_sync(client.generate("Hello", system="You are helpful"))
    assert received["body"]["messages"][0] == {"role": "system", "content": "You are helpful"}
    assert received["body"]["messages"][1] == {"role": "user", "content": "Hello"}


def test_retry_on_503():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/v1/chat/completions":
            call_count += 1
            if call_count <= 2:
                return httpx.Response(503, json={"error": "busy"})
            return httpx.Response(200, json=_make_completion_response("finally"))
        return httpx.Response(404)

    client = _make_client(handler, max_retries=3, retry_base_delay=0.01)
    result = _run_sync(client.generate("test"))
    assert result == "finally"
    assert call_count == 3


def test_retry_on_429():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/v1/chat/completions":
            call_count += 1
            if call_count == 1:
                return httpx.Response(429, json={"error": "rate limited"})
            return httpx.Response(200, json=_make_completion_response("ok"))
        return httpx.Response(404)

    client = _make_client(handler, max_retries=3, retry_base_delay=0.01)
    result = _run_sync(client.generate("test"))
    assert result == "ok"
    assert call_count == 2


def test_returns_none_after_max_retries():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(503, json={"error": "busy"})
        return httpx.Response(404)

    client = _make_client(handler, max_retries=2, retry_base_delay=0.01)
    result = _run_sync(client.generate("test"))
    assert result is None


def test_generate_batch():
    client = _make_client(_ok_handler, max_concurrent=4)
    results = _run_sync(client.generate_batch(["a", "b", "c"]))
    assert len(results) == 3
    assert all(r is not None and "Response to:" in r for r in results)


def test_wait_ready_success():
    client = _make_client(_ok_handler)
    ready = _run_sync(client.wait_ready(timeout=5.0))
    assert ready is True


def test_wait_ready_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(503)
        return httpx.Response(404)

    client = _make_client(handler)
    ready = _run_sync(client.wait_ready(timeout=0.5))
    assert ready is False


def test_warmup_idempotent():
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/v1/chat/completions":
            call_count += 1
            return httpx.Response(200, json=_make_completion_response("warm"))
        return httpx.Response(404)

    client = _make_client(handler)
    _run_sync(client.warmup())
    _run_sync(client.warmup())
    _run_sync(client.warmup())
    assert call_count == 1


def test_strips_thinking_blocks():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            content = "<think>\nLet me analyze this...\n</think>\nThis file handles authentication."
            return httpx.Response(200, json=_make_completion_response(content))
        return httpx.Response(404)

    client = _make_client(handler)
    result = client.generate_sync("describe this")
    assert result == "This file handles authentication."
    assert "<think>" not in result


def test_generate_sync():
    client = _make_client(_ok_handler)
    result = client.generate_sync("Hello sync")
    assert result is not None
    assert "Response to: Hello sync" in result


def test_empty_choices_returns_none():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/chat/completions":
            return httpx.Response(200, json={"choices": []})
        return httpx.Response(404)

    client = _make_client(handler)
    result = client.generate_sync("test")
    assert result is None


def test_bad_request_returns_none():
    """400 responses return None immediately without retrying."""
    call_count = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal call_count
        if request.url.path == "/v1/chat/completions":
            call_count += 1
            return httpx.Response(400, json={
                "error": {
                    "code": 400,
                    "message": "request (8000 tokens) exceeds context (4096 tokens)",
                    "type": "exceed_context_size_error",
                    "n_prompt_tokens": 8000,
                    "n_ctx": 4096,
                }
            })
        return httpx.Response(404)

    client = _make_client(handler)
    result = client.generate_sync("test")
    assert result is None
    assert call_count == 1  # no retries on 400


def test_generate_sync_thread_safety():
    """Verify multiple threads can call generate_sync concurrently."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    client = _make_client(_ok_handler, max_concurrent=8)
    prompts = [f"prompt-{i}" for i in range(20)]

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(client.generate_sync, p): p for p in prompts}
        results = {}
        for f in as_completed(futures):
            prompt = futures[f]
            results[prompt] = f.result()

    assert len(results) == 20
    for prompt, result in results.items():
        assert result is not None
        assert f"Response to: {prompt}" in result
