"""Optional shared Supabase mirror for Router V2 observations."""
from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass(frozen=True)
class SupabaseObservationConfig:
    url: str
    secret_key: str
    timeout_seconds: float = 10.0

    @classmethod
    def from_environment(cls) -> SupabaseObservationConfig | None:
        url = (os.environ.get("SUPABASE_URL") or "").strip().rstrip("/")
        secret = (
            os.environ.get("SUPABASE_SECRET_KEY")
            or os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
            or ""
        ).strip()
        if not url or not secret:
            return None
        return cls(url=url, secret_key=secret)


def _finite(value: Any) -> float | None:
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def shared_observation_mode() -> str:
    mode = (os.environ.get("THERIAD_ROUTING_OBSERVATION_STORE") or "auto").strip().lower()
    return mode if mode in {"auto", "local", "supabase"} else "local"


def mirror_to_supabase(
    rows: list[dict[str, Any]], *, run_id: str, principal_id: str | None
) -> bool:
    """Mirror rows with the service key; return False when disabled/unavailable.

    The local SQLite transaction is authoritative for the request path. A
    shared-store outage must not discard a paid model response or turn a
    successful chat into HTTP 500, so callers log a False result and continue.
    """

    mode = shared_observation_mode()
    if mode == "local" or not rows:
        return False
    config = SupabaseObservationConfig.from_environment()
    if config is None:
        return False
    payload = []
    numeric = {
        "propensity",
        "p_success",
        "p_success_lower",
        "expected_output_tokens",
        "expected_cost_usd",
        "expected_latency_ms",
        "uncertainty",
        "realized_success",
        "realized_score",
        "realized_cost_usd",
        "realized_latency_ms",
    }
    for row in rows:
        serialized = {
            key: (_finite(value) if key in numeric else value)
            for key, value in row.items()
            if key != "user_id"
        }
        serialized.update(
            {
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "principal_id": principal_id,
            }
        )
        payload.append(serialized)
    headers = {
        "apikey": config.secret_key,
        "Content-Type": "application/json",
        "Prefer": "return=minimal",
    }
    if config.secret_key.startswith("eyJ"):
        headers["Authorization"] = f"Bearer {config.secret_key}"
    try:
        response = httpx.post(
            f"{config.url}/rest/v1/routing_observations",
            json=payload,
            headers=headers,
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        return True
    except httpx.HTTPError:
        return False
