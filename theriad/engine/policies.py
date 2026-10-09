"""
Execution policies: retry, fallback, timeout.

Design decisions:
- Policies are pure dataclasses — no logic, just config. Logic lives in the
  pipeline so it can be tested independently of providers.
- FallbackPolicy is an ordered list of provider+model pairs. On failure, each
  entry is tried in sequence.
- RetryPolicy uses exponential backoff with a cap. Fixed delays rejected
  (Principle 7 is simplicity, not naivety — jitter helps under burst load).
- Timeout is per-attempt, not total, so retries get full time budget each time.
  Tradeoff: total wall time can be (max_attempts * timeout). Acceptable for v1.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    backoff_base_seconds: float = 0.5   # wait = base * 2^attempt
    backoff_cap_seconds: float = 10.0


@dataclass
class FallbackPolicy:
    """Ordered list of (provider_name, model) pairs to try after primary fails."""
    chain: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class ExecutionPolicy:
    timeout_seconds: float = 30.0
    retry: RetryPolicy = field(default_factory=RetryPolicy)
    fallback: FallbackPolicy = field(default_factory=FallbackPolicy)
