"""Tenant-scoped metric records derived from the durable runs table.

The JSONL tracker remains a local/CLI compatibility surface. HTTP analytics
must read the database instead so a hosted account can never see a process
global metrics file.
"""
from __future__ import annotations

from theriad.storage.database import get_db


def load_scoped_records(scope_id: str | None) -> list[dict]:
    """Return metrics-compatible records for one user scope."""
    rows = get_db().execute(
        """SELECT id, model, provider, tier, input_tokens, output_tokens,
                  actual_cost_usd, total_latency_ms, fallback_used, status,
                  error, created_at, routing_source, routing_reason
           FROM runs
           WHERE user_id IS ?
           ORDER BY created_at ASC""",
        (scope_id,),
    ).fetchall()
    return [_row_to_record(row) for row in rows]


def _row_to_record(row) -> dict:
    input_tokens = int(row["input_tokens"] or 0)
    output_tokens = int(row["output_tokens"] or 0)
    return {
        "id": row["id"],
        "timestamp": row["created_at"],
        "model": row["model"] or "unknown",
        "provider": row["provider"] or "unknown",
        "prompt_tokens": input_tokens,
        "completion_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "estimated_cost_usd": (
            float(row["actual_cost_usd"])
            if row["actual_cost_usd"] is not None else None
        ),
        "latency_ms": float(row["total_latency_ms"] or 0.0),
        "routing_layer": row["routing_source"] or row["tier"] or "unknown",
        "routing_reason": row["routing_reason"] or "(no prompt recorded)",
        "used_fallback": bool(row["fallback_used"]),
        "status": row["status"] or "unknown",
        "error": row["error"],
    }
