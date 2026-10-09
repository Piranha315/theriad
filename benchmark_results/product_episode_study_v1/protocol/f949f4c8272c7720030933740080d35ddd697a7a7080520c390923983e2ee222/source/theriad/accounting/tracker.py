"""
CostTracker — records InferenceRecords and persists them as JSONL.

Design decisions:
- JSONL chosen over SQLite: append-only, zero schema migrations, grep-able,
  trivially importable by pandas/duckdb for analysis. Tradeoff: no indexed
  queries, but querying patterns at v1 don't need indexes.
- Path defaults to ~/.theriad/metrics.jsonl — configurable via TheriadConfig.
- Thread/task safety: writes are synchronized with a threading.Lock so the
  tracker can be shared across async tasks without corruption.
- InferenceRecord is a dataclass (not Pydantic) to keep it lightweight.
  to_dict() provides the serialization path.
"""
from __future__ import annotations

import json
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from theriad.core.logging.logger import get_logger
from theriad.models.registry import DEFAULT_BASELINE_MODEL

from .pricing import PricingTable, estimate_cost

logger = get_logger("accounting.tracker")


@dataclass
class InferenceRecord:
    """One complete inference event."""
    timestamp: str              # ISO-8601 UTC
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    estimated_cost_usd: float
    routing_layer: str          # which layer made the routing decision
    routing_reason: str
    used_fallback: bool = False
    tags: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class CostTracker:
    """
    Records inference metrics and persists them to a JSONL file.

    Usage:
        tracker = CostTracker()
        tracker.record(response, routing_result)

        stats = tracker.summary()
    """

    def __init__(
        self,
        metrics_path: str = "~/.theriad/metrics.jsonl",
        pricing_table: PricingTable | None = None,
    ):
        self._path = Path(metrics_path).expanduser()
        self._pricing = pricing_table or PricingTable()
        self._lock = threading.Lock()
        self._records: list[InferenceRecord] = []
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def record(
        self,
        *,
        provider: str,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        latency_ms: float,
        routing_layer: str = "unknown",
        routing_reason: str = "",
        used_fallback: bool = False,
        tags: dict[str, str] | None = None,
    ) -> InferenceRecord:
        """Create, store, and persist an InferenceRecord."""
        cost = estimate_cost(model, prompt_tokens, completion_tokens, self._pricing)
        rec = InferenceRecord(
            timestamp=datetime.now(timezone.utc).isoformat(),
            provider=provider,
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
            routing_layer=routing_layer,
            routing_reason=routing_reason,
            used_fallback=used_fallback,
            tags=tags or {},
        )
        with self._lock:
            self._records.append(rec)
            self._append_to_disk(rec)

        logger.debug(
            "inference recorded",
            extra={
                "model": model,
                "tokens": rec.total_tokens,
                "cost_usd": cost,
                "latency_ms": latency_ms,
            },
        )
        return rec

    def _append_to_disk(self, rec: InferenceRecord) -> None:
        try:
            with open(self._path, "a") as fh:
                fh.write(json.dumps(rec.to_dict()) + "\n")
        except OSError as exc:
            logger.error("failed to write metrics", extra={"error": str(exc)})

    def summary(self) -> dict:
        """Return aggregate statistics across all in-memory records."""
        if not self._records:
            return {
                "total_requests": 0,
                "total_tokens": 0,
                "total_cost_usd": 0.0,
                "avg_latency_ms": 0.0,
                "fallback_rate": 0.0,
                "models_used": {},
                "baseline_model": DEFAULT_BASELINE_MODEL,
                "baseline_cost_usd": 0.0,
                "estimated_savings_usd": 0.0,
                "estimated_savings_pct": 0.0,
            }
        total_tokens = sum(r.total_tokens for r in self._records)
        total_cost = sum(r.estimated_cost_usd for r in self._records)
        avg_latency = sum(r.latency_ms for r in self._records) / len(self._records)
        fallback_count = sum(1 for r in self._records if r.used_fallback)
        models: dict[str, int] = {}
        for r in self._records:
            models[r.model] = models.get(r.model, 0) + 1
        baseline_model = DEFAULT_BASELINE_MODEL
        baseline_cost = sum(
            estimate_cost(
                baseline_model,
                r.prompt_tokens,
                r.completion_tokens,
                self._pricing,
            )
            for r in self._records
        )
        savings = baseline_cost - total_cost
        savings_pct = savings / baseline_cost * 100 if baseline_cost > 0 else 0.0

        return {
            "total_requests": len(self._records),
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "fallback_rate": round(fallback_count / len(self._records), 4),
            "models_used": models,
            "baseline_model": baseline_model,
            "baseline_cost_usd": round(baseline_cost, 6),
            "estimated_savings_usd": round(savings, 6),
            "estimated_savings_pct": round(savings_pct, 1),
        }

    def records(self) -> list[InferenceRecord]:
        """Return a copy of all in-memory records."""
        with self._lock:
            return list(self._records)

    @classmethod
    def load_from_disk(cls, path: str) -> list[dict]:
        """Load all records from a JSONL file. Returns raw dicts."""
        p = Path(path).expanduser()
        if not p.exists():
            return []
        records = []
        with open(p) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        logger.warning("corrupt metrics line", extra={"line": line[:80]})
        return records
