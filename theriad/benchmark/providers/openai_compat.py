"""
Shared OpenAI-compatible HTTP adapter for benchmark provider lanes.

Used by Groq and Cerebras — both expose /v1/chat/completions.
Not used for OpenRouter (which goes through the main theriad provider)
or Ollama (which has a slightly different response shape).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

import httpx

from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult
from theriad.benchmark.pricing import ProviderPrice, estimate
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.openai_compat")

_DEFAULT_TIMEOUT = 60.0


async def call_openai_compat(
    *,
    provider_name: str,
    base_url: str,
    api_key: str,
    model: str,
    benchmark_prompt: BenchmarkPrompt,
    price: ProviderPrice | None = None,
    timeout: float = _DEFAULT_TIMEOUT,
    tier: str | None = None,
    extra_headers: dict[str, str] | None = None,
) -> BenchmarkResult:
    """
    Call an OpenAI-compatible /v1/chat/completions endpoint and return
    a BenchmarkResult. All errors are captured — never raises.
    """
    ts = datetime.now(timezone.utc).isoformat()
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    payload: dict = {
        "model": model,
        "messages": [{"role": "user", "content": benchmark_prompt.prompt}],
        "temperature": 0.0,
    }
    if benchmark_prompt.max_output_tokens is not None:
        payload["max_tokens"] = benchmark_prompt.max_output_tokens

    t0 = time.monotonic()
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(
                f"{base_url.rstrip('/')}/v1/chat/completions",
                headers=headers,
                json=payload,
            )
        total_ms = (time.monotonic() - t0) * 1000

        if resp.status_code != 200:
            return _error_result(
                ts, provider_name, model, tier, benchmark_prompt,
                total_ms,
                f"HTTP {resp.status_code}: {resp.text[:300]}",
            )

        data = resp.json()
        choice = data["choices"][0]
        usage = data.get("usage", {})
        content = (choice.get("message") or {}).get("content") or ""
        finish_reason = choice.get("finish_reason")
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)

        # Provider latency: prefer x-response-time header, fall back to total
        provider_ms = total_ms
        if "x-response-time" in resp.headers:
            try:
                provider_ms = float(resp.headers["x-response-time"])
            except (ValueError, TypeError):
                pass

        tps = (
            completion_tokens / (provider_ms / 1000)
            if provider_ms > 0 and completion_tokens > 0 else None
        )
        actual_cost = estimate(price, prompt_tokens, completion_tokens)

        return BenchmarkResult(
            timestamp=ts,
            provider=provider_name,
            model=model,
            tier=tier,
            prompt_id=benchmark_prompt.id,
            category=benchmark_prompt.category,
            input_tokens=prompt_tokens,
            output_tokens=completion_tokens,
            actual_cost_usd=actual_cost,
            provider_latency_ms=provider_ms,
            total_latency_ms=total_ms,
            tokens_per_second=tps,
            finish_reason=finish_reason,
            status="success",
            error=None,
            quality_pass=benchmark_prompt.quality_check(content) if content else False,
        )

    except httpx.TimeoutException:
        total_ms = (time.monotonic() - t0) * 1000
        return _error_result(
            ts, provider_name, model, tier, benchmark_prompt, total_ms,
            f"timeout after {timeout}s",
            status="timeout",
        )
    except Exception as exc:
        total_ms = (time.monotonic() - t0) * 1000
        return _error_result(
            ts, provider_name, model, tier, benchmark_prompt, total_ms, str(exc)
        )


def _error_result(
    ts: str,
    provider: str,
    model: str,
    tier: str | None,
    bp: BenchmarkPrompt,
    total_ms: float,
    error: str,
    status: str = "error",
) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=ts,
        provider=provider,
        model=model,
        tier=tier,
        prompt_id=bp.id,
        category=bp.category,
        input_tokens=0,
        output_tokens=0,
        actual_cost_usd=None,
        provider_latency_ms=total_ms,
        total_latency_ms=total_ms,
        tokens_per_second=None,
        finish_reason=None,
        status=status,
        error=error,
        quality_pass=None,
    )
