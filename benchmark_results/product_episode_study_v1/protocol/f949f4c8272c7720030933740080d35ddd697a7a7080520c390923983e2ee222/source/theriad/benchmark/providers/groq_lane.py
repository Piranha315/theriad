"""
Groq benchmark lane.

Uses Groq's OpenAI-compatible /openai/v1/chat/completions endpoint.
Requires GROQ_API_KEY.

Models benchmarked:
- llama-3.3-70b-versatile  (strong, low cost)
- llama-3.1-8b-instant     (fast, very cheap)
"""
from __future__ import annotations

import os

from theriad.benchmark.pricing import GROQ_PRICES
from theriad.benchmark.providers.base import BenchmarkLane
from theriad.benchmark.providers.openai_compat import call_openai_compat
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.groq")

_BASE_URL = "https://api.groq.com/openai"
_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]
_TIER_MAP = {
    "llama-3.3-70b-versatile": "code_strong",
    "llama-3.1-8b-instant":    "cheap_bulk",
}


class GroqLane(BenchmarkLane):
    """Groq inference benchmark lane."""

    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self._api_key = api_key or os.environ.get("GROQ_API_KEY", "")
        self._models = models or _MODELS

    @property
    def provider_name(self) -> str:
        return "groq"

    def models(self) -> list[str]:
        if not self._api_key:
            logger.warning("GROQ_API_KEY not set — groq lane disabled")
            return []
        return list(self._models)

    async def run_one(self, model: str, benchmark_prompt: BenchmarkPrompt) -> BenchmarkResult:
        return await call_openai_compat(
            provider_name=self.provider_name,
            base_url=_BASE_URL,
            api_key=self._api_key,
            model=model,
            benchmark_prompt=benchmark_prompt,
            price=GROQ_PRICES.get(model),
            tier=_TIER_MAP.get(model),
        )
