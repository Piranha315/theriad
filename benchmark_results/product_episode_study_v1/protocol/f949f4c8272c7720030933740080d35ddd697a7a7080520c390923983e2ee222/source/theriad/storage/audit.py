"""Append-only, tenant-scoped audit event persistence."""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone

from theriad.storage.database import get_db


def record_event(
    event_type: str,
    *,
    organization_id: str | None = None,
    user_id: str | None = None,
    resource_type: str | None = None,
    resource_id: str | None = None,
    metadata: dict | None = None,
) -> str:
    event_id = str(uuid.uuid4())
    get_db().execute(
        """INSERT INTO audit_events
           (id, organization_id, user_id, event_type, resource_type, resource_id,
            metadata_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            event_id,
            organization_id,
            user_id,
            event_type,
            resource_type,
            resource_id,
            json.dumps(metadata or {}, separators=(",", ":"), sort_keys=True),
            datetime.now(timezone.utc).isoformat(),
        ),
    )
    get_db().commit()
    return event_id


def list_events(*, organization_id: str | None, user_id: str | None, limit: int = 100) -> list[dict]:
    rows = get_db().execute(
        """SELECT id, organization_id, user_id, event_type, resource_type,
                  resource_id, metadata_json, created_at
           FROM audit_events
           WHERE organization_id IS ? AND user_id IS ?
           ORDER BY created_at DESC LIMIT ?""",
        (organization_id, user_id, max(1, min(limit, 500))),
    ).fetchall()
    result = []
    for row in rows:
        try:
            metadata = json.loads(row["metadata_json"] or "{}")
        except json.JSONDecodeError:
            metadata = {}
        result.append({
            "id": row["id"],
            "event_type": row["event_type"],
            "resource_type": row["resource_type"],
            "resource_id": row["resource_id"],
            "metadata": metadata,
            "created_at": row["created_at"],
        })
    return result
