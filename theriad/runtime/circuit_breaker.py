"""Process-wide bounded circuit breakers for external providers.

The registry contains no credentials or response bodies. It records only
failure classes, latency, counters, and cooldown timestamps so readiness can be
reported safely and repeated provider outages do not create failure loops.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass


class CircuitOpenError(RuntimeError):
    def __init__(self, key: str, retry_after_seconds: float):
        self.key = key
        self.retry_after_seconds = max(0.0, retry_after_seconds)
        super().__init__(
            f"{key} is temporarily unavailable; retry after "
            f"{self.retry_after_seconds:.1f}s"
        )


@dataclass
class CircuitState:
    key: str
    consecutive_failures: int = 0
    total_successes: int = 0
    total_failures: int = 0
    opened_until: float = 0.0
    half_open_in_flight: bool = False
    last_error_type: str | None = None
    last_latency_ms: float | None = None
    last_checked_at: float | None = None


class CircuitBreakerRegistry:
    def __init__(
        self,
        *,
        failure_threshold: int | None = None,
        cooldown_seconds: float | None = None,
    ) -> None:
        self.failure_threshold = max(
            1,
            int(
                failure_threshold
                if failure_threshold is not None
                else os.environ.get("THERIAD_CIRCUIT_FAILURE_THRESHOLD", 2)
            ),
        )
        self.cooldown_seconds = max(
            0.01,
            float(
                cooldown_seconds
                if cooldown_seconds is not None
                else os.environ.get("THERIAD_CIRCUIT_COOLDOWN_SECONDS", 30)
            ),
        )
        self._states: dict[str, CircuitState] = {}
        self._lock = threading.RLock()

    def _state(self, key: str) -> CircuitState:
        return self._states.setdefault(key, CircuitState(key=key))

    def before_call(self, key: str) -> None:
        now = time.time()
        with self._lock:
            state = self._state(key)
            if state.opened_until > now:
                raise CircuitOpenError(key, state.opened_until - now)
            if state.opened_until and state.half_open_in_flight:
                raise CircuitOpenError(key, self.cooldown_seconds)
            if state.opened_until:
                state.half_open_in_flight = True

    def record_success(self, key: str, *, latency_ms: float | None = None) -> None:
        with self._lock:
            state = self._state(key)
            state.consecutive_failures = 0
            state.total_successes += 1
            state.opened_until = 0.0
            state.half_open_in_flight = False
            state.last_error_type = None
            state.last_latency_ms = latency_ms
            state.last_checked_at = time.time()

    def record_failure(
        self,
        key: str,
        error: BaseException,
        *,
        latency_ms: float | None = None,
        transient: bool = True,
    ) -> None:
        with self._lock:
            state = self._state(key)
            state.total_failures += 1
            state.last_error_type = type(error).__name__
            state.last_latency_ms = latency_ms
            state.last_checked_at = time.time()
            state.half_open_in_flight = False
            if transient:
                state.consecutive_failures += 1
                if state.consecutive_failures >= self.failure_threshold:
                    state.opened_until = time.time() + self.cooldown_seconds
            else:
                state.consecutive_failures = 0
                state.opened_until = 0.0

    def reset(self, key: str | None = None) -> None:
        with self._lock:
            if key is None:
                self._states.clear()
            else:
                self._states.pop(key, None)

    def snapshot(self, prefix: str | None = None) -> list[dict]:
        now = time.time()
        with self._lock:
            rows = []
            for key, state in sorted(self._states.items()):
                if prefix and not key.startswith(prefix):
                    continue
                data = asdict(state)
                if state.opened_until > now:
                    status = "open"
                elif state.opened_until:
                    status = "half_open"
                elif state.last_checked_at is None:
                    status = "unknown"
                else:
                    status = "ready" if state.last_error_type is None else "degraded"
                data.update({
                    "status": status,
                    "retry_after_seconds": round(max(0.0, state.opened_until - now), 3),
                })
                rows.append(data)
            return rows


_REGISTRY = CircuitBreakerRegistry()


def get_circuit_registry() -> CircuitBreakerRegistry:
    return _REGISTRY


def model_circuit_key(provider: str, model: str) -> str:
    return f"model:{provider}:{model}"


def unavailable_model_slugs(candidates) -> frozenset[str]:
    """Return candidates whose model-specific circuit is currently open."""

    registry = get_circuit_registry()
    unavailable = set()
    for candidate in candidates:
        key = model_circuit_key(candidate.provider, candidate.model)
        state = next((row for row in registry.snapshot(prefix=key) if row["key"] == key), None)
        if state and state["status"] == "open":
            unavailable.add(candidate.model)
    return frozenset(unavailable)
