"""
Ollama benchmark lane.

Connects to a locally-running Ollama instance (default http://localhost:11434).
Auto-discovers available models via GET /api/tags. Free inference — zero cost.

If Ollama is not running or has no models, the lane is silently disabled.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from theriad.benchmark.allowlist import filter_models
from theriad.benchmark.pricing import OLLAMA_PRICE, estimate
from theriad.benchmark.providers.base import BenchmarkLane
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult, _make_output_preview
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.ollama")

_DEFAULT_BASE_URL = "http://localhost:11434"
_DISCOVERY_TIMEOUT = 3.0   # fast ping — don't block startup
_INFERENCE_TIMEOUT = 120.0  # local models can be slow


class OllamaLane(BenchmarkLane):
    """Ollama local-inference benchmark lane."""

    def __init__(
        self,
        base_url: str = _DEFAULT_BASE_URL,
        models: list[str] | None = None,
        allow_large: bool = False,
    ):
        self._base_url = base_url.rstrip("/")
        self._explicit_models = models  # None = auto-discover
        self._allow_large = allow_large
        self._discovered: list[str] | None = None

    @property
    def provider_name(self) -> str:
        return "ollama"

    def models(self) -> list[str]:
        if self._explicit_models is not None:
            # Still apply allowlist filtering to explicit models
            return filter_models(list(self._explicit_models), allow_large=self._allow_large)
        if self._discovered is None:
            self._discovered = self._discover_models()
        return self._discovered

    def _filter_and_log(self, raw: list[str]) -> list[str]:
        return filter_models(raw, allow_large=self._allow_large)

    def _discover_models(self) -> list[str]:
        """
        Query GET /api/tags and return model names.
        Returns [] if Ollama is unreachable.
        """
        try:
            with httpx.Client(timeout=_DISCOVERY_TIMEOUT) as client:
                resp = client.get(f"{self._base_url}/api/tags")
            if resp.status_code != 200:
                logger.debug("Ollama /api/tags returned %s", resp.status_code)
                return []
            data = resp.json()
            raw_names = [m["name"] for m in data.get("models", [])]
            names = filter_models(raw_names, allow_large=self._allow_large)
            if names:
                logger.info("Ollama models available: %s", ", ".join(names))
                blocked = set(raw_names) - set(names)
                if blocked:
                    logger.info("Ollama models blocked by allowlist: %s", ", ".join(sorted(blocked)))
            else:
                logger.info("Ollama is running but has no allowed models")
            return names
        except Exception as exc:
            logger.debug("Ollama not available: %s", exc)
            return []

    async def run_one(self, model: str, benchmark_prompt: BenchmarkPrompt) -> BenchmarkResult:
        ts = datetime.now(timezone.utc).isoformat()
        t0 = time.monotonic()

        options: dict = {"temperature": 0.0}
        if benchmark_prompt.max_output_tokens is not None:
            options["num_predict"] = benchmark_prompt.max_output_tokens
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": benchmark_prompt.prompt}],
            "options": options,
            "stream": False,
        }

        try:
            async with httpx.AsyncClient(timeout=_INFERENCE_TIMEOUT) as client:
                resp = await client.post(
                    f"{self._base_url}/api/chat",
                    json=payload,
                )
            total_ms = (time.monotonic() - t0) * 1000

            if resp.status_code != 200:
                return self._error(
                    ts, model, benchmark_prompt, total_ms,
                    f"HTTP {resp.status_code}: {resp.text[:300]}",
                )

            data = resp.json()
            content = (data.get("message") or {}).get("content") or ""
            finish_reason = "stop" if not data.get("done") is False else "length"

            # Ollama reports token counts in the response
            prompt_tokens = data.get("prompt_eval_count", 0)
            completion_tokens = data.get("eval_count", 0)

            # Ollama reports eval_duration in nanoseconds
            eval_ns = data.get("eval_duration", 0)
            provider_ms = eval_ns / 1_000_000 if eval_ns else total_ms
            tps = (
                completion_tokens / (provider_ms / 1000)
                if provider_ms > 0 and completion_tokens > 0 else None
            )

            preview, sha256 = _make_output_preview(content) if content else ("", "")

            return BenchmarkResult(
                timestamp=ts,
                provider=self.provider_name,
                model=model,
                tier=None,
                prompt_id=benchmark_prompt.id,
                category=benchmark_prompt.category,
                input_tokens=prompt_tokens,
                output_tokens=completion_tokens,
                actual_cost_usd=estimate(OLLAMA_PRICE, prompt_tokens, completion_tokens),
                provider_latency_ms=provider_ms,
                total_latency_ms=total_ms,
                tokens_per_second=tps,
                finish_reason=finish_reason,
                status="success",
                error=None,
                quality_pass=benchmark_prompt.quality_check(content) if content else False,
                output_preview=preview or None,
                output_sha256=sha256 or None,
            )

        except httpx.TimeoutException:
            total_ms = (time.monotonic() - t0) * 1000
            return self._error(
                ts, model, benchmark_prompt, total_ms,
                f"timeout after {_INFERENCE_TIMEOUT}s",
                status="timeout",
            )
        except Exception as exc:
            total_ms = (time.monotonic() - t0) * 1000
            return self._error(ts, model, benchmark_prompt, total_ms, str(exc))

    def _error(
        self, ts: str, model: str, bp: BenchmarkPrompt,
        total_ms: float, error: str, status: str = "error",
    ) -> BenchmarkResult:
        return BenchmarkResult(
            timestamp=ts, provider=self.provider_name, model=model, tier=None,
            prompt_id=bp.id, category=bp.category,
            input_tokens=0, output_tokens=0, actual_cost_usd=0.0,
            provider_latency_ms=total_ms, total_latency_ms=total_ms,
            tokens_per_second=None, finish_reason=None,
            status=status, error=error, quality_pass=None,
        )
