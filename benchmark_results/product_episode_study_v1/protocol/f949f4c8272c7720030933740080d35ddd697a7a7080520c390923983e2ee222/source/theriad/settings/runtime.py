"""Per-account runtime controls for chat execution and spending limits."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from fastapi import Request

from theriad.api.auth import user_scope_id
from theriad.storage.database import get_db

LOCAL_SCOPE = "__local__"

DEFAULT_RUNTIME_SETTINGS: dict[str, Any] = {
    "regular_chat_returns_answer": True,
    "allow_follow_up_questions": False,
    "route_only_available": False,
    "disable_premium_models": False,
    "model_origin_policy": "any",
    "max_cost_per_request_usd": 0.05,
    "max_daily_spend_usd": 5.0,
}

_SETTING_KEYS = tuple(DEFAULT_RUNTIME_SETTINGS)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _scope_key(request: Request) -> tuple[str, str | None]:
    """Return the settings key and the nullable runs.user_id scope."""
    scope = user_scope_id(request)
    return scope or LOCAL_SCOPE, scope


def _row_for(scope_key: str) -> dict[str, Any]:
    db = get_db()
    row = db.execute(
        "SELECT * FROM runtime_settings WHERE scope_id = ?", (scope_key,)
    ).fetchone()
    if row is None:
        now = _now()
        db.execute(
            """INSERT INTO runtime_settings (
                scope_id, regular_chat_returns_answer, allow_follow_up_questions,
                route_only_available,
                disable_premium_models, model_origin_policy,
                max_cost_per_request_usd, max_daily_spend_usd, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                scope_key,
                int(DEFAULT_RUNTIME_SETTINGS["regular_chat_returns_answer"]),
                int(DEFAULT_RUNTIME_SETTINGS["allow_follow_up_questions"]),
                int(DEFAULT_RUNTIME_SETTINGS["route_only_available"]),
                int(DEFAULT_RUNTIME_SETTINGS["disable_premium_models"]),
                DEFAULT_RUNTIME_SETTINGS["model_origin_policy"],
                DEFAULT_RUNTIME_SETTINGS["max_cost_per_request_usd"],
                DEFAULT_RUNTIME_SETTINGS["max_daily_spend_usd"],
                now,
            ),
        )
        db.commit()
        row = db.execute(
            "SELECT * FROM runtime_settings WHERE scope_id = ?", (scope_key,)
        ).fetchone()

    assert row is not None
    return dict(row)


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "regular_chat_returns_answer": bool(row["regular_chat_returns_answer"]),
        "allow_follow_up_questions": bool(row["allow_follow_up_questions"]),
        "route_only_available": bool(row["route_only_available"]),
        "disable_premium_models": bool(row["disable_premium_models"]),
        "model_origin_policy": row["model_origin_policy"] or "any",
        "max_cost_per_request_usd": row["max_cost_per_request_usd"],
        "max_daily_spend_usd": row["max_daily_spend_usd"],
        "updated_at": row["updated_at"],
    }


def get_runtime_settings(request: Request) -> dict[str, Any]:
    scope_key, _ = _scope_key(request)
    return _payload(_row_for(scope_key))


def update_runtime_settings(
    request: Request, updates: dict[str, Any],
) -> dict[str, Any]:
    scope_key, _ = _scope_key(request)
    current = _row_for(scope_key)
    values = {
        key: updates[key] if key in updates else current[key]
        for key in _SETTING_KEYS
    }
    if values["model_origin_policy"] not in {"any", "us_only"}:
        raise ValueError("model_origin_policy must be 'any' or 'us_only'")
    db = get_db()
    db.execute(
        """UPDATE runtime_settings SET
            regular_chat_returns_answer = ?,
            allow_follow_up_questions = ?,
            route_only_available = ?,
            disable_premium_models = ?,
            model_origin_policy = ?,
            max_cost_per_request_usd = ?,
            max_daily_spend_usd = ?,
            updated_at = ?
        WHERE scope_id = ?""",
        (
            int(values["regular_chat_returns_answer"]),
            int(values["allow_follow_up_questions"]),
            int(values["route_only_available"]),
            int(values["disable_premium_models"]),
            values["model_origin_policy"],
            values["max_cost_per_request_usd"],
            values["max_daily_spend_usd"],
            _now(),
            scope_key,
        ),
    )
    db.commit()
    return _payload(_row_for(scope_key))


def execution_budget_error(
    request: Request, estimated_cost_usd: float | None,
) -> dict[str, Any] | None:
    """Return a structured refusal when a request would exceed a live cap."""
    settings = get_runtime_settings(request)
    estimated = max(0.0, float(estimated_cost_usd or 0.0))

    per_request = settings["max_cost_per_request_usd"]
    if per_request is not None and estimated > float(per_request):
        return {
            "code": "max_cost_per_request_exceeded",
            "message": (
                f"Estimated cost ${estimated:.4f} exceeds the per-request limit "
                f"of ${float(per_request):.2f}."
            ),
            "estimated_cost_usd": estimated,
            "max_cost_per_request_usd": float(per_request),
        }

    daily_limit = settings["max_daily_spend_usd"]
    if daily_limit is None:
        return None

    _, user_scope = _scope_key(request)
    today = datetime.now(timezone.utc).date().isoformat()
    row = get_db().execute(
        """SELECT COALESCE(SUM(COALESCE(actual_cost_usd, 0)), 0)
           AS spent
           FROM runs
           WHERE user_id IS ? AND status = 'success'
             AND substr(created_at, 1, 10) = ?""",
        (user_scope, today),
    ).fetchone()
    spent = float(row["spent"] or 0.0)
    if spent + estimated > float(daily_limit):
        return {
            "code": "max_daily_spend_exceeded",
            "message": (
                f"This request would bring today's estimated spend to "
                f"${spent + estimated:.2f}, above the daily limit of "
                f"${float(daily_limit):.2f}."
            ),
            "estimated_cost_usd": estimated,
            "spent_today_usd": spent,
            "max_daily_spend_usd": float(daily_limit),
        }
    return None
