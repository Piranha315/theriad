"""
BenchmarkLane protocol — the interface every provider lane must implement.

A lane represents one provider (e.g. Groq) and exposes zero or more
(provider, model) pairs that will be benchmarked.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult


class BenchmarkLane(ABC):
    """Abstract base for a benchmark provider lane."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Short identifier used in result records, e.g. 'groq'."""

    @abstractmethod
    def models(self) -> list[str]:
        """
        List of model slugs this lane will benchmark.

        May return an empty list when the provider is not reachable or
        not configured (no API key, Ollama not running, etc.).
        """

    @abstractmethod
    async def run_one(
        self,
        model: str,
        benchmark_prompt: BenchmarkPrompt,
    ) -> BenchmarkResult:
        """
        Execute one prompt against one model and return a BenchmarkResult.

        Must not raise — all errors should be captured in the result's
        status/error fields.
        """

    def is_available(self) -> bool:
        """Return True if the lane is configured and reachable enough to start."""
        return len(self.models()) > 0
