"""Private site-traffic analytics with anonymous first-party collection.

The collection endpoint intentionally accepts no authentication: the public
landing page must be measurable before a visitor creates an account. The
summary endpoint is separately gated by ``THERIAD_ANALYTICS_OWNER_USERNAME``
and never returns raw visitor/session identifiers.
"""
from __future__ import annotations

import os
import re
import uuid
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from theriad.api.auth import get_request_user
from theriad.storage.database import get_db

router = APIRouter(prefix="/api/analytics", tags=["analytics"])

_VISITOR_COOKIE = "theriad_visitor_id"
_VISITOR_COOKIE_MAX_AGE = 60 * 60 * 24 * 365 * 2
_EXCLUSION_COOKIE = "theriad_analytics_excluded"
_EXCLUSION_COOKIE_MAX_AGE = 60 * 60 * 24 * 365 * 10
_EVENT_TYPES = {"page_view", "heartbeat", "session_end"}
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]{8,96}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _valid_id(value: str) -> bool:
    return bool(_SAFE_ID.fullmatch(value))


def _visitor_id(request: Request) -> tuple[str, bool]:
    existing = (request.cookies.get(_VISITOR_COOKIE) or "").strip()
    if existing and _valid_id(existing):
        return existing, False
    return uuid.uuid4().hex, True


def _clean_path(path: str) -> str:
    value = (path or "/").strip()
    # Query strings are not useful for traffic totals and can contain user
    # supplied identifiers. The client already omits them; enforce that here
    # too for callers that send directly to the public collector.
    value = value.split("?", 1)[0]
    if not value.startswith("/"):
        value = "/" + value
    return value[:512]


def _clean_referrer(referrer: str | None) -> str | None:
    """Keep only the referring origin, never a path/query that may contain PII."""
    value = (referrer or "").strip()
    if not value:
        return None
    try:
        parsed = urlparse(value)
    except ValueError:
        return None
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        return None
    return f"{parsed.scheme}://{parsed.hostname}"[:256]


class AnalyticsEvent(BaseModel):
    event: str = Field(min_length=1, max_length=32)
    session_id: str = Field(min_length=8, max_length=96)
    path: str = Field(default="/", max_length=512)
    referrer: str | None = Field(default=None, max_length=2048)


class AnalyticsExclusionRequest(BaseModel):
    enabled: bool


def _require_owner(request: Request) -> dict:
    user = get_request_user(request)
    if user is None:
        raise HTTPException(status_code=401, detail="Sign in to view site analytics.")

    configured_owner = os.environ.get("THERIAD_ANALYTICS_OWNER_USERNAME", "").strip()
    if not configured_owner:
        raise HTTPException(
            status_code=503,
            detail=(
                "Site analytics is not configured. Set "
                "THERIAD_ANALYTICS_OWNER_USERNAME to the owner account username."
            ),
        )
    if user.get("is_guest") or str(user.get("username") or "").casefold() != configured_owner.casefold():
        raise HTTPException(status_code=403, detail="You are not allowed to view site analytics.")
    return user


@router.get("/exclusion")
def exclusion_status(request: Request):
    """Return whether this browser is excluded from future collection."""
    _require_owner(request)
    return {"excluded": request.cookies.get(_EXCLUSION_COOKIE) == "1"}


@router.post("/exclusion")
def set_exclusion(payload: AnalyticsExclusionRequest, request: Request):
    """Enable or disable owner-browser exclusion without storing an IP address."""
    _require_owner(request)
    response = JSONResponse({"ok": True, "excluded": payload.enabled})
    if payload.enabled:
        response.set_cookie(
            key=_EXCLUSION_COOKIE,
            value="1",
            max_age=_EXCLUSION_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            path="/",
        )
    else:
        response.delete_cookie(key=_EXCLUSION_COOKIE, path="/")
    return response


@router.post("/collect")
def collect(event: AnalyticsEvent, request: Request):
    """Record one anonymous page view, heartbeat, or session end."""
    if event.event not in _EVENT_TYPES:
        raise HTTPException(status_code=422, detail="Unsupported analytics event.")
    if not _valid_id(event.session_id):
        raise HTTPException(status_code=422, detail="Invalid analytics session ID.")

    # The owner can opt this browser out without linking traffic to an account
    # or persisting an IP address. Existing historical rows remain unchanged.
    if request.cookies.get(_EXCLUSION_COOKIE) == "1":
        return {"ok": True, "excluded": True}

    visitor_id, set_cookie = _visitor_id(request)
    path = _clean_path(event.path)
    referrer = _clean_referrer(event.referrer)
    now = _now()
    now_text = _timestamp(now)
    db = get_db()

    db.execute(
        """INSERT INTO analytics_visitors (visitor_id, first_seen, last_seen)
           VALUES (?, ?, ?)
           ON CONFLICT(visitor_id) DO UPDATE SET last_seen = excluded.last_seen""",
        (visitor_id, now_text, now_text),
    )
    db.execute(
        """INSERT INTO analytics_sessions
               (session_id, visitor_id, started_at, last_seen, landing_path, landing_referrer)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(session_id) DO UPDATE SET
               last_seen = excluded.last_seen,
               ended_at = CASE
                   WHEN ? = 'session_end' THEN excluded.last_seen
                   ELSE analytics_sessions.ended_at
               END""",
        (event.session_id, visitor_id, now_text, now_text, path, referrer, event.event),
    )
    # Keep the durable event log bounded: heartbeats update the session's
    # last_seen timestamp, while only page views are useful for page/referrer
    # aggregation and bounce-rate calculations.
    if event.event == "page_view":
        db.execute(
            """INSERT INTO analytics_events
                   (id, visitor_id, session_id, event_type, path, referrer, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (uuid.uuid4().hex, visitor_id, event.session_id, event.event, path, referrer, now_text),
        )
    db.commit()

    response = JSONResponse({"ok": True})
    if set_cookie:
        response.set_cookie(
            key=_VISITOR_COOKIE,
            value=visitor_id,
            max_age=_VISITOR_COOKIE_MAX_AGE,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https",
            path="/",
        )
    return response


def _duration_sql() -> str:
    """Return a bounded session duration expression in seconds."""
    return "MIN(1800.0, MAX(0.0, (julianday(COALESCE(ended_at, last_seen)) - julianday(started_at)) * 86400.0))"


@router.get("/summary")
def summary(request: Request, days: int = Query(default=30, ge=7, le=90)):
    """Return aggregate site traffic for the configured owner."""
    owner = _require_owner(request)
    db = get_db()
    now = _now()
    start = now - timedelta(days=days - 1)
    start_text = _timestamp(start)
    active_cutoff = _timestamp(now - timedelta(seconds=120))

    visitor_count = db.execute("SELECT COUNT(*) AS n FROM analytics_visitors").fetchone()["n"]
    session_count = db.execute("SELECT COUNT(*) AS n FROM analytics_sessions").fetchone()["n"]
    page_view_count = db.execute(
        "SELECT COUNT(*) AS n FROM analytics_events WHERE event_type = 'page_view'"
    ).fetchone()["n"]
    session_stats = db.execute(
        f"SELECT AVG({_duration_sql()}) AS average_seconds, "
        "SUM(CASE WHEN (SELECT COUNT(*) FROM analytics_events e "
        "          WHERE e.session_id = s.session_id AND e.event_type = 'page_view') = 1 "
        "         THEN 1 ELSE 0 END) AS bounced "
        "FROM analytics_sessions s"
    ).fetchone()
    active_now = db.execute(
        """SELECT COUNT(*) AS n FROM analytics_sessions
           WHERE ended_at IS NULL AND last_seen >= ?""",
        (active_cutoff,),
    ).fetchone()["n"]

    daily_rows = db.execute(
        """SELECT substr(created_at, 1, 10) AS day,
                  COUNT(DISTINCT visitor_id) AS visitors,
                  COUNT(DISTINCT session_id) AS visits,
                  COUNT(*) AS page_views
           FROM analytics_events
           WHERE event_type = 'page_view' AND created_at >= ?
           GROUP BY substr(created_at, 1, 10)
           ORDER BY day""",
        (start_text,),
    ).fetchall()
    daily_by_day = {row["day"]: dict(row) for row in daily_rows}
    daily = []
    for offset in range(days):
        day = (start + timedelta(days=offset)).date().isoformat()
        daily.append({
            "day": day,
            "visitors": int(daily_by_day.get(day, {}).get("visitors", 0)),
            "visits": int(daily_by_day.get(day, {}).get("visits", 0)),
            "page_views": int(daily_by_day.get(day, {}).get("page_views", 0)),
        })

    top_pages = db.execute(
        """SELECT path, COUNT(*) AS views
           FROM analytics_events
           WHERE event_type = 'page_view' AND created_at >= ?
           GROUP BY path ORDER BY views DESC, path LIMIT 10""",
        (start_text,),
    ).fetchall()
    top_referrers = db.execute(
        """SELECT COALESCE(NULLIF(landing_referrer, ''), 'Direct / none') AS referrer,
                  COUNT(*) AS visits
           FROM analytics_sessions
           WHERE started_at >= ?
           GROUP BY referrer ORDER BY visits DESC, referrer LIMIT 10""",
        (start_text,),
    ).fetchall()

    total_sessions = int(session_count)
    bounced = int(session_stats["bounced"] or 0)
    return {
        "owner": owner.get("username"),
        "window_days": days,
        "generated_at": _timestamp(now),
        "total_visitors": int(visitor_count),
        "total_visits": total_sessions,
        "total_page_views": int(page_view_count),
        "active_now": int(active_now),
        "average_session_seconds": round(float(session_stats["average_seconds"] or 0.0), 1),
        "bounce_rate": round((bounced / total_sessions) if total_sessions else 0.0, 4),
        "daily": daily,
        "top_pages": [dict(row) for row in top_pages],
        "top_referrers": [dict(row) for row in top_referrers],
        "tracking": {
            "identity": "anonymous first-party cookie",
            "session_timeout_minutes": 30,
            "duration_note": "Duration is based on page-view and 30-second heartbeat activity.",
        },
    }
