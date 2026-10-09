"""
Ollama provider implementation.

Ollama runs models locally via a REST API (default: http://localhost:11434).
It is the "free" fallback when API cost matters most.

Decision: target Ollama's /api/chat endpoint (v0.3+) which returns OpenAI-style
JSON rather than the streaming /generate endpoint. Keeps response handling uniform.

Tradeoff: Ollama must be running locally; not suitable for all deployments.
Future: add a 'type' field to ProviderConfig to distinguish local vs remote.
"""
from __future__ import annotations

import time

import httpx

from theriad.core.logging.logger import get_logger

from .interfaces import CompletionRequest, CompletionResponse, ProviderError, ProviderProtocol

logger = get_logger("providers.ollama")

_DEFAULT_BASE = "http://localhost:11434"


class OllamaProvider(ProviderProtocol):
    """
    Ollama local inference provider.

    No API key required. Requires Ollama to be running locally.
    """

    def __init__(self, base_url: str = _DEFAULT_BASE, timeout: float = 60.0):
        self._base_url = base_url.rstrip("/")
        self._timeout = timeout

    @property
    def name(self) -> str:
        return "ollama"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        # Ollama model names don't have org prefixes — strip if present
        model = request.model.split("/")[-1] if "/" in request.model else request.model

        payload = {
            "model": model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "stream": False,
            "options": {"temperature": request.temperature},
        }
        if request.max_tokens:
            payload["options"]["num_predict"] = request.max_tokens

        t0 = time.monotonic()
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
        except httpx.TimeoutException as exc:
            raise ProviderError(self.name, f"timeout after {self._timeout}s") from exc
        except httpx.RequestError as exc:
            raise ProviderError(self.name, f"request error: {exc}") from exc

        latency_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            raise ProviderError(self.name, resp.text, status_code=resp.status_code)

        data = resp.json()
        try:
            content = data["message"]["content"]
            usage = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)
            return CompletionResponse(
                model=model,
                content=content,
                prompt_tokens=usage,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                provider=self.name,
                raw=data,
            )
        except (KeyError, IndexError) as exc:
            raise ProviderError(self.name, f"unexpected response shape: {data}") from exc

    async def health_check(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base_url}/api/tags")
            return resp.status_code == 200
        except Exception:
            return False
