"""Async and sync inference client for llama-server's OpenAI-compatible API."""

from __future__ import annotations

import asyncio
import logging
import re
import threading
from typing import TYPE_CHECKING, Any

import httpx

from bgkit.inference.config import InferenceConfig

if TYPE_CHECKING:
    from bgkit.inference.models import ModelProfile

logger = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    """Raised when a request exceeds the server's per-slot context size."""

    def __init__(self, n_tokens: int | None = None, n_ctx: int | None = None):
        self.n_tokens = n_tokens
        self.n_ctx = n_ctx
        super().__init__(f"request ({n_tokens} tokens) exceeds context ({n_ctx} tokens)")

# Shared background event loop for all sync callers.
_bg_loop: asyncio.AbstractEventLoop | None = None
_bg_thread: threading.Thread | None = None
_bg_lock = threading.Lock()


def _ensure_background_loop() -> asyncio.AbstractEventLoop:
    """Start a background thread running an event loop (once, lazily)."""
    global _bg_loop, _bg_thread
    if _bg_loop is not None and _bg_loop.is_running():
        return _bg_loop
    with _bg_lock:
        if _bg_loop is not None and _bg_loop.is_running():
            return _bg_loop
        _bg_loop = asyncio.new_event_loop()

        def _run() -> None:
            asyncio.set_event_loop(_bg_loop)
            _bg_loop.run_forever()

        _bg_thread = threading.Thread(target=_run, daemon=True, name="llama-client-loop")
        _bg_thread.start()
        return _bg_loop


def _run_sync(coro: Any) -> Any:
    """Submit a coroutine to the shared background loop and block until done."""
    loop = _ensure_background_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    return future.result()


class LlamaClient:
    """HTTP client for llama-server's /v1/chat/completions endpoint.

    Supports async batch generation with concurrency control and retry logic.
    Thread-safe sync wrappers use a shared background event loop.
    """

    def __init__(self, config: InferenceConfig | None = None) -> None:
        self.config = config or InferenceConfig()
        self._semaphore: asyncio.Semaphore | None = None
        self._async_client: httpx.AsyncClient | None = None
        self._warmed_up = False
        # Cache compiled regex from model profile
        self._think_re: re.Pattern[str] | None = None
        self._chat_template_kwargs: dict[str, Any] | None = None
        profile = self.config.model_profile
        if profile is not None:
            self._think_re = profile.compile_thinking_re()
            self._chat_template_kwargs = profile.get_chat_template_kwargs()

    async def _get_client(self) -> httpx.AsyncClient:
        if self._async_client is None or self._async_client.is_closed:
            self._async_client = httpx.AsyncClient(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
            )
        return self._async_client

    async def _get_semaphore(self) -> asyncio.Semaphore:
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.config.max_concurrent)
        return self._semaphore

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._async_client is not None and not self._async_client.is_closed:
            await self._async_client.aclose()
            self._async_client = None

    async def generate(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Generate a completion via /v1/chat/completions.

        Returns the generated text, or None on failure after retries.
        """
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tokens or self.config.max_new_tokens,
            "temperature": temperature if temperature is not None else self.config.temperature,
        }
        if self._chat_template_kwargs is not None:
            payload["chat_template_kwargs"] = self._chat_template_kwargs

        sem = await self._get_semaphore()
        client = await self._get_client()

        for attempt in range(self.config.max_retries):
            try:
                async with sem:
                    resp = await client.post("/v1/chat/completions", json=payload)

                if resp.status_code in (500, 503, 429):
                    delay = self.config.retry_base_delay * (2**attempt)
                    logger.warning(
                        "llama_server_busy (status=%d attempt=%d delay=%.1fs)",
                        resp.status_code,
                        attempt + 1,
                        delay,
                    )
                    await asyncio.sleep(delay)
                    continue

                if resp.status_code == 400:
                    try:
                        err = resp.json().get("error", {})
                        logger.warning(
                            "llama_server_bad_request: %s",
                            err.get("message", resp.text[:200]),
                        )
                        if err.get("type") == "exceed_context_size_error":
                            raise ContextOverflowError(
                                err.get("n_prompt_tokens"),
                                err.get("n_ctx"),
                            )
                    except ContextOverflowError:
                        raise
                    except Exception:
                        logger.warning("llama_server_bad_request: %s", resp.text[:200])
                    return None

                resp.raise_for_status()
                data = resp.json()
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content and self._think_re is not None:
                        content = self._think_re.sub("", content).strip()
                    return content if content else None
                return None

            except httpx.TimeoutException:
                delay = self.config.retry_base_delay * (2**attempt)
                logger.warning(
                    "llama_server_timeout (attempt=%d delay=%.1fs)", attempt + 1, delay
                )
                await asyncio.sleep(delay)
            except httpx.HTTPStatusError as e:
                logger.warning("llama_server_error (status=%d): %s", e.response.status_code, e)
                return None
            except httpx.HTTPError as e:
                logger.warning("llama_server_connection_error: %s", e)
                delay = self.config.retry_base_delay * (2**attempt)
                await asyncio.sleep(delay)

        return None

    async def generate_batch(
        self,
        prompts: list[str],
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> list[str | None]:
        """Generate completions for multiple prompts concurrently."""
        tasks = [
            self.generate(p, system=system, max_tokens=max_tokens, temperature=temperature)
            for p in prompts
        ]
        return await asyncio.gather(*tasks)

    def generate_sync(
        self,
        prompt: str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str | None:
        """Synchronous wrapper around generate(). Thread-safe."""
        return _run_sync(
            self.generate(prompt, system=system, max_tokens=max_tokens, temperature=temperature)
        )

    async def wait_ready(self, timeout: float = 120.0) -> bool:
        """Poll /health until the server is ready or timeout expires."""
        client = await self._get_client()
        deadline = asyncio.get_event_loop().time() + timeout
        while asyncio.get_event_loop().time() < deadline:
            try:
                resp = await client.get("/health")
                if resp.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass
            await asyncio.sleep(2.0)
        return False

    def wait_ready_sync(self, timeout: float = 120.0) -> bool:
        """Synchronous wrapper around wait_ready(). Thread-safe."""
        return _run_sync(self.wait_ready(timeout))

    def apply_profile(self, profile: ModelProfile) -> None:
        """Update the client's model profile, re-caching regex and template kwargs."""
        self.config.model_profile = profile
        self._think_re = profile.compile_thinking_re()
        self._chat_template_kwargs = profile.get_chat_template_kwargs()

    async def detect_model(self) -> str | None:
        """Query /v1/models to get the loaded model identifier.

        Returns the model id string (typically the GGUF filename), or None
        if the endpoint is unavailable or returns unexpected data.
        """
        client = await self._get_client()
        try:
            resp = await client.get("/v1/models")
            if resp.status_code != 200:
                return None
            data = resp.json()
            models = data.get("data", [])
            if models:
                return models[0].get("id")
        except (httpx.HTTPError, ValueError, KeyError):
            pass
        return None

    def detect_model_sync(self) -> str | None:
        """Synchronous wrapper around detect_model(). Thread-safe."""
        return _run_sync(self.detect_model())

    async def warmup(self) -> None:
        """Fire one short completion to JIT-compile CUDA kernels. No-op after first call."""
        if self._warmed_up:
            return
        logger.info("Warming up llama-server at %s", self.config.base_url)
        await self.generate("Hi", max_tokens=1)
        self._warmed_up = True
        logger.info("Warmup complete for %s", self.config.base_url)

    def warmup_sync(self) -> None:
        """Synchronous wrapper around warmup(). Thread-safe."""
        _run_sync(self.warmup())
