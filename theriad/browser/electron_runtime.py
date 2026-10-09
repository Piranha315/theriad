"""
Electron-backed browser runtime for Theriad.

Connects to the Theriad Browser Electron app's local HTTP RPC server on 127.0.0.1:47200.
The Electron BrowserView can retain its own persistent cookies and localStorage.
It is not represented as undetectable automation.

Usage:
  1. Start the Electron app:
       cd apps/theriad-extension/electron-browser && npm run dev
  2. The app starts the RPC server on port 47200 automatically.
  3. This runtime connects and drives the BrowserView over HTTP.

Session model:
  The Electron app has a single BrowserView. Multiple concurrent Python sessions
  share that view (last-writer-wins for navigation). For true isolation, run
  multiple Electron instances on different ports via THERIAD_ELECTRON_RPC_PORT.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_PORT = 47200
_NAV_TIMEOUT = 20.0    # seconds — navigation can be slow
_CONNECT_TIMEOUT = 2.0  # health-check / availability probe


def _port() -> int:
    return int(os.environ.get("THERIAD_ELECTRON_RPC_PORT", _DEFAULT_PORT))


def _token() -> str | None:
    configured = os.environ.get("THERIAD_ELECTRON_RPC_TOKEN", "").strip()
    if configured:
        return configured
    path = Path.home() / ".theriad" / "electron-rpc-token"
    try:
        return path.read_text().strip() or None
    except OSError:
        return None


def _headers(*, json_body: bool = False) -> dict[str, str]:
    headers = {"Authorization": f"Bearer {_token() or ''}"}
    if json_body:
        headers["Content-Type"] = "application/json"
    return headers


def _url(path: str, port: int) -> str:
    return f"http://127.0.0.1:{port}{path}"


def _http_get(path: str, *, port: int, timeout: float = _CONNECT_TIMEOUT) -> dict[str, Any]:
    req = urllib.request.Request(_url(path, port), headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except Exception:
        return {"ok": False, "error": "Connection refused"}


def _http_post(path: str, body: dict[str, Any], *, port: int, timeout: float = _NAV_TIMEOUT) -> dict[str, Any]:
    data = json.dumps(body).encode()
    req = urllib.request.Request(
        _url(path, port),
        data=data,
        headers=_headers(json_body=True),
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as exc:
        try:
            return json.loads(exc.read())
        except Exception:
            return {"ok": False, "error": f"HTTP {exc.code}"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


# ── Session state ─────────────────────────────────────────────────────────────

@dataclass
class ElectronSession:
    session_id: str
    user_prompt: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    current_url: str = "about:blank"
    action_log: list[dict[str, Any]] = field(default_factory=list)


# ── Runtime ───────────────────────────────────────────────────────────────────

class ElectronBrowserRuntime:
    """
    Drive the Theriad Browser Electron BrowserView over localhost HTTP.

    Has the same public interface as LocalBrowserRuntime so adapters.py can
    swap backends without changing call sites.
    """

    def __init__(self, port: int | None = None) -> None:
        self._port: int = port if port is not None else _port()
        self.sessions: dict[str, ElectronSession] = {}
        self._lock = asyncio.Lock()

    # ── Availability ──────────────────────────────────────────────────────────

    def availability(self) -> dict[str, Any]:
        result = _http_get("/health", port=self._port)
        if result.get("ok"):
            return {"available": True, "error": None}
        return {
            "available": False,
            "error": (
                "Electron browser RPC server is not running or authentication failed. "
                f"Detail: {result.get('error') or 'unavailable'}. "
                "Start it with: cd apps/theriad-extension/electron-browser && npm run dev"
            ),
        }

    # ── Session lifecycle ─────────────────────────────────────────────────────

    async def start_session(
        self,
        *,
        session_id: str | None = None,
        visible: bool = True,
        start_url: str | None = None,
        user_prompt: str = "",
        allow_placeholder_navigation: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        availability = self.availability()
        if not availability["available"]:
            return {"ok": False, "status": "unavailable", **availability}

        import uuid
        sid = session_id or str(uuid.uuid4())

        async with self._lock:
            if sid in self.sessions:
                return await self.status(sid)
            self.sessions[sid] = ElectronSession(session_id=sid, user_prompt=user_prompt)

        if start_url:
            nav = await asyncio.to_thread(_http_post, "/navigate", {"url": start_url}, port=self._port)
            if not nav.get("ok"):
                async with self._lock:
                    self.sessions.pop(sid, None)
                return {"ok": False, "status": "failed", "error": nav.get("error", "Navigation failed")}
            self.sessions[sid].current_url = nav.get("url", start_url)

        return {
            "ok": True,
            "status": "running",
            "session_id": sid,
            "backend": "electron",
            "browser": "chromium",
            "url": self.sessions[sid].current_url,
        }

    async def stop_session(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return {"ok": False, "status": "not_found", "error": "Session not found"}
        session.status = "stopped"
        return {"ok": True, "status": "stopped", "session_id": session_id}

    async def stop_all(self) -> None:
        for sid in list(self.sessions):
            await self.stop_session(sid)

    def _session(self, session_id: str) -> ElectronSession:
        s = self.sessions.get(session_id)
        if s is None:
            raise KeyError("Browser session not found.")
        return s

    # ── Browser operations ────────────────────────────────────────────────────

    async def navigate(self, session_id: str, url: str) -> dict[str, Any]:
        session = self._session(session_id)
        result = await asyncio.to_thread(_http_post, "/navigate", {"url": url}, port=self._port)
        ok = result.get("ok", False)
        if ok:
            session.current_url = result.get("url", url)
        self._log(session, "navigate", "executed" if ok else "failed", target=url,
                  error=result.get("error") if not ok else None)
        if not ok:
            return result
        return await self.observe(session_id)

    async def observe(self, session_id: str, *, screenshot: bool = False) -> dict[str, Any]:
        session = self._session(session_id)
        result = await asyncio.to_thread(
            _http_post, "/observe", {"screenshot": screenshot}, port=self._port
        )
        if result.get("ok"):
            session.current_url = result.get("url", session.current_url)
        from theriad.agents.browser_observation import detect_captcha
        captcha_detected, captcha_reason = detect_captcha(
            result.get("url", session.current_url),
            result.get("title", ""),
            result.get("visible_text", ""),
        )
        # Normalise keys to match LocalBrowserRuntime's output shape.
        return {
            **result,
            "session_id": session_id,
            "backend": "electron",
            "browser": "chromium",
            "visible": True,
            "page_title": result.get("title", ""),
            "current_url": result.get("url", session.current_url),
            "visible_text_summary": result.get("visible_text", ""),
            "screenshot_base64": result.get("screenshot_base64"),
            "clickable_elements": result.get("clickable_elements", []),
            "input_fields": result.get("input_fields", []),
            "accessibility_tree_summary": result.get("accessibility_tree_summary", []),
            "action_log": list(session.action_log),
            "status": session.status,
            "captcha_detected": captcha_detected,
            "captcha_reason": captcha_reason,
            "pending_action": (
                {"action": "captcha_handoff", "requires_human": True}
                if captcha_detected else None
            ),
            "untrusted_content": True,
            "original_user_prompt": session.user_prompt,
        }

    async def status(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            try:
                session = self._session(session_id)
            except KeyError as exc:
                return {"ok": False, "status": "not_found", "error": str(exc)}
            return {
                "ok": True,
                "status": session.status,
                "session_id": session_id,
                "current_url": session.current_url,
                "backend": "electron",
                "browser": "chromium",
                "action_log": list(session.action_log),
            }
        availability = self.availability()
        return {
            "ok": True,
            "status": "available" if availability["available"] else "unavailable",
            "backend": "electron",
            "browser": "chromium",
            "active_sessions": len(self.sessions),
            **availability,
        }

    async def action(
        self,
        session_id: str,
        action: dict[str, Any] | None = None,
        *,
        approved: bool = False,
        deny: bool = False,
    ) -> dict[str, Any]:
        session = self._session(session_id)
        action = dict(action or {})
        kind = str(action.get("action") or "").strip().lower()
        if not kind:
            return {"ok": False, "status": "failed", "error": "action is required"}

        if kind == "navigate":
            return await self.navigate(session_id, str(action["url"]))

        if kind == "click":
            payload: dict[str, Any] = {}
            if "element_id" in action:
                payload["element_id"] = action["element_id"]
            elif "x" in action and "y" in action:
                payload["x"] = action["x"]
                payload["y"] = action["y"]
            result = await asyncio.to_thread(_http_post, "/action/click", payload, port=self._port)

        elif kind in {"type", "fill"}:
            result = await asyncio.to_thread(
                _http_post,
                "/action/fill",
                {"element_id": action.get("element_id"), "text": str(action.get("text") or "")},
                port=self._port,
            )

        elif kind == "scroll":
            result = await asyncio.to_thread(
                _http_post,
                "/action/scroll",
                {"amount": int(action.get("amount", 300))},
                port=self._port,
            )

        elif kind == "press_key":
            result = await asyncio.to_thread(
                _http_post,
                "/action/key",
                {"key": str(action.get("key") or "Return")},
                port=self._port,
            )

        elif kind == "wait":
            ms = min(int(action.get("milliseconds", 500)), 10_000)
            await asyncio.sleep(ms / 1000)
            return await self.observe(session_id)

        else:
            return {"ok": False, "status": "failed", "error": f"Unsupported action: {kind}"}

        ok = result.get("ok", False)
        self._log(
            session, kind, "executed" if ok else "failed",
            target=action.get("element_id") or action.get("url"),
            error=result.get("error") if not ok else None,
        )
        if not ok:
            return {**result, "status": "failed"}

        if result.get("url"):
            session.current_url = result["url"]

        return {
            **result,
            "session_id": session_id,
            "backend": "electron",
            "status": "running",
            "current_url": result.get("url", session.current_url),
            "page_title": result.get("title", ""),
            "visible_text_summary": result.get("visible_text", ""),
            "action_log": list(session.action_log),
        }

    @staticmethod
    def _log(
        session: ElectronSession,
        action: str,
        status: str,
        *,
        target: Any = None,
        error: str | None = None,
    ) -> None:
        session.action_log.append({
            "timestamp": time.time(),
            "action": action,
            "status": status,
            "target": target,
            "url": session.current_url,
            "error": error,
        })
        session.action_log[:] = session.action_log[-100:]


# ── Module-level singleton ────────────────────────────────────────────────────

_runtime: ElectronBrowserRuntime | None = None


def get_electron_browser_runtime() -> ElectronBrowserRuntime:
    global _runtime
    if _runtime is None:
        _runtime = ElectronBrowserRuntime()
    return _runtime
