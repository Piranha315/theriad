"""Canonical HTTP contract for policy-gated browser sessions.

This module deliberately wraps the existing session store rather than adding a
second executor.  It is used by the product FastAPI application and by legacy
compatibility handlers.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .browser_agent import BrowserAgentStartRequest, STORE


class SessionCommand(BaseModel):
    session_id: str


router = APIRouter(prefix="/api/browser/session", tags=["browser-session"])


def _session(session_id: str):
    try:
        return STORE.get(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="unknown session_id") from exc


@router.post("/start")
def start_browser_session(request: BrowserAgentStartRequest) -> dict:
    """Start the mock session adapter; it never claims a real browser exists."""

    return STORE.start(request).to_dict(include_diagnostics=False)


@router.get("/{session_id}")
def browser_session_status(session_id: str, diagnostics: bool = False) -> dict:
    return _session(session_id).to_dict(include_diagnostics=diagnostics)


@router.post("/{session_id}/approve")
def approve_browser_session(session_id: str) -> dict:
    return _session(session_id).approve() or _session(session_id).to_dict(include_diagnostics=False)


@router.post("/{session_id}/deny")
def deny_browser_session(session_id: str) -> dict:
    return _session(session_id).deny() or _session(session_id).to_dict(include_diagnostics=False)


@router.post("/{session_id}/stop")
def stop_browser_session(session_id: str) -> dict:
    return _session(session_id).stop() or _session(session_id).to_dict(include_diagnostics=False)


@router.post("/{session_id}/use-api-search")
def use_api_search(session_id: str) -> dict:
    return _session(session_id).use_api_search() or _session(session_id).to_dict(include_diagnostics=False)
