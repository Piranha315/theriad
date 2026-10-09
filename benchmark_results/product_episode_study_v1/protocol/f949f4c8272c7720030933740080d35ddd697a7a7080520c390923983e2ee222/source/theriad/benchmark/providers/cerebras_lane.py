"""
Cerebras benchmark lane.

Uses Cerebras's OpenAI-compatible /v1/chat/completions endpoint.
Requires CEREBRAS_API_KEY.

Models benchmarked:
- llama3.3-70b  (strong, very fast chip-native inference)
- llama3.1-8b   (fast, extremely cheap)
"""
from __future__ import annotations

import os

from theriad.benchmark.pricing import CEREBRAS_PRICES
from theriad.benchmark.providers.base import BenchmarkLane
from theriad.benchmark.providers.openai_compat import call_openai_compat
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.cerebras")

_BASE_URL = "https://api.cerebras.ai"
_MODELS = [
    "llama3.3-70b",
    "llama3.1-8b",
]
_TIER_MAP = {
    "llama3.3-70b": "code_strong",
    "llama3.1-8b":  "cheap_bulk",
}


class CerebrasLane(BenchmarkLane):
    """Cerebras inference benchmark lane."""

    def __init__(self, api_key: str | None = None, models: list[str] | None = None):
        self._api_key = api_key or os.environ.get("CEREBRAS_API_KEY", "")
        self._models = models or _MODELS

    @property
    def provider_name(self) -> str:
        return "cerebras"

    def models(self) -> list[str]:
        if not self._api_key:
            logger.warning("CEREBRAS_API_KEY not set — cerebras lane disabled")
            return []
        return list(self._models)

    async def run_one(self, model: str, benchmark_prompt: BenchmarkPrompt) -> BenchmarkResult:
        return await call_openai_compat(
            provider_name=self.provider_name,
            base_url=_BASE_URL,
            api_key=self._api_key,
            model=model,
            benchmark_prompt=benchmark_prompt,
            price=CEREBRAS_PRICES.get(model),
            tier=_TIER_MAP.get(model),
            extra_headers={"User-Agent": "theriad-benchmark/1.0"},
        )
