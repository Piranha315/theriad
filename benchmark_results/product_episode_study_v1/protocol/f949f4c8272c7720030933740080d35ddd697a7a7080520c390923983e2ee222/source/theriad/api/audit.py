"""Read-only owner-scoped audit event API."""
from __future__ import annotations

from fastapi import APIRouter, Request

from theriad.api.auth_context import require_auth_context
from theriad.storage.audit import list_events

router = APIRouter(prefix="/api/audit", tags=["audit"])


@router.get("/events")
def audit_events(request: Request, limit: int = 100):
    context = require_auth_context(request)
    return {
        "events": list_events(
            organization_id=context.organization_id,
            user_id=context.user_id,
            limit=limit,
        )
    }
