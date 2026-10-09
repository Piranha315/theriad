"""BrowserBase cloud browser runtime.

Connects to a managed Chromium instance via the BrowserBase CDP endpoint, using
the official ``browserbase`` Python SDK to create/destroy sessions and Playwright
to drive the page.

Required env vars:
    BROWSERBASE_API_KEY      — from https://www.browserbase.com/settings
    BROWSERBASE_PROJECT_ID   — from the same settings page (optional; inferred
                               from the API key when omitted)

Install:
    pip install browserbase
    pip install playwright && playwright install chromium
"""
from __future__ import annotations

import asyncio
import base64
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from theriad.agents.browser_observation import detect_captcha

logger = logging.getLogger(__name__)

_MAX_TEXT = 4_000
_MAX_ELEMENTS = 60

# ---------------------------------------------------------------------------
# Lazy dependency references — set on first use; patchable in tests.
# ---------------------------------------------------------------------------
_Browserbase: Any = None        # browserbase.Browserbase
_async_playwright: Any = None   # playwright.async_api.async_playwright


def _load_deps() -> tuple[bool, str | None]:
    """Import optional deps and populate module-level refs. Returns (ok, error)."""
    global _Browserbase, _async_playwright
    if _Browserbase is None:
        try:
            from browserbase import Browserbase  # type: ignore[import]
            _Browserbase = Browserbase
        except ImportError:
            return False, "browserbase SDK not installed. Run: pip install browserbase"
    if _async_playwright is None:
        try:
            from playwright.async_api import async_playwright  # type: ignore[import]
            _async_playwright = async_playwright
        except ImportError:
            return False, (
                "Playwright not installed. "
                "Run: pip install playwright && playwright install chromium"
            )
    return True, None

# ---------------------------------------------------------------------------
# DOM observation script (identical to LocalBrowserRuntime for key parity)
# ---------------------------------------------------------------------------
_OBSERVE_JS = """(limit) => {
    const visible = (el) => {
      const s = getComputedStyle(el);
      const r = el.getBoundingClientRect();
      return s.visibility !== 'hidden' && s.display !== 'none' &&
        r.width > 0 && r.height > 0;
    };
    const labelFor = (el) => {
      const aria = el.getAttribute('aria-label');
      if (aria) return aria;
      if (el.id) {
        const label = document.querySelector(`label[for="${CSS.escape(el.id)}"]`);
        if (label) return (label.innerText || label.textContent || '').trim();
      }
      return (el.closest('label')?.innerText || '').trim();
    };
    const candidates = [...document.querySelectorAll(
      'a[href], button, [role="button"], input, textarea, select'
    )].filter(visible).slice(0, limit);
    const clickables = [];
    const inputs = [];
    candidates.forEach((el, index) => {
      let id = el.getAttribute('data-theriad-id');
      if (!id) {
        id = `ae-${index + 1}`;
        el.setAttribute('data-theriad-id', id);
      }
      const tag = el.tagName.toLowerCase();
      const type = (el.type || el.getAttribute('type') || '').toLowerCase();
      const label = labelFor(el);
      const text = (el.innerText || el.textContent || el.value || '').trim().slice(0, 160);
      const common = {
        id, tag, type, text, label,
        in_form: Boolean(el.form),
        aria_label: el.getAttribute('aria-label') || '',
        placeholder: el.getAttribute('placeholder') || '',
      };
      if (['input', 'textarea', 'select'].includes(tag)) {
        if (type !== 'hidden') inputs.push(common);
      }
      if (tag === 'a' || tag === 'button' || el.getAttribute('role') === 'button' ||
          ['button', 'submit', 'reset'].includes(type)) {
        clickables.push({
          ...common,
          href: el.href || '',
          download: el.hasAttribute('download'),
        });
      }
    });
    const roles = [...document.querySelectorAll(
      'main, nav, header, footer, form, [role], h1, h2, h3'
    )].filter(visible).slice(0, 30).map(el => ({
      role: el.getAttribute('role') || el.tagName.toLowerCase(),
      name: (el.getAttribute('aria-label') || el.innerText || '')
        .trim().replace(/\\s+/g, ' ').slice(0, 180),
    }));
    return {
      visible_text: (document.body?.innerText || '').trim().slice(0, 4000),
      clickable_elements: clickables,
      input_fields: inputs,
      accessibility_tree_summary: roles,
    };
}"""


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class BrowserBaseSession:
    session_id: str
    bb_session_id: str          # BrowserBase's own session ID
    playwright: Any             # AsyncPlaywright instance
    browser: Any                # RemoteBrowser from connect_over_cdp
    context: Any                # Default browser context
    page: Any                   # Default page
    current_url: str = ""
    user_prompt: str = ""
    created_at: float = field(default_factory=time.time)
    status: str = "running"
    elements: dict[str, dict[str, Any]] = field(default_factory=dict)
    action_log: list[dict[str, Any]] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runtime
# ---------------------------------------------------------------------------

class BrowserBaseRuntime:
    """
    Lifecycle manager for BrowserBase cloud browser sessions.

    One instance per process (see module singleton below). Sessions are
    isolated Chromium contexts hosted by BrowserBase; Playwright drives them
    via a CDP WebSocket URL.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        project_id: str | None = None,
    ) -> None:
        self._api_key = api_key or os.environ.get("BROWSERBASE_API_KEY", "")
        if not self._api_key:
            try:
                from theriad.settings.secrets import get_browserbase_api_key as _get_bb_key
                self._api_key = _get_bb_key() or ""
            except ImportError:
                pass
        self._project_id = project_id or os.environ.get("BROWSERBASE_PROJECT_ID", "")
        self.sessions: dict[str, BrowserBaseSession] = {}
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Availability
    # ------------------------------------------------------------------

    def availability(self) -> dict[str, Any]:
        """Return ``{"available": bool, "error": str|None}``."""
        ok, error = _load_deps()
        if not ok:
            return {"available": False, "error": error}
        # Re-resolve the key so keys saved via the Settings UI are picked up
        # without restarting the singleton.
        api_key = self._api_key
        if not api_key:
            try:
                from theriad.settings.secrets import get_browserbase_api_key as _get_bb_key
                api_key = _get_bb_key() or ""
            except ImportError:
                pass
        if not api_key:
            return {
                "available": False,
                "error": "No BROWSERBASE_API_KEY configured. Add one in Settings → Browser Backend.",
            }
        return {"available": True, "error": None}

    # ------------------------------------------------------------------
    # Session lifecycle
    # ------------------------------------------------------------------

    async def start_session(
        self,
        *,
        session_id: str | None = None,
        start_url: str | None = None,
        user_prompt: str = "",
        stealth: bool = False,
        solve_captchas: bool = False,
        keep_alive: bool = False,
        timeout: int = 300,
    ) -> dict[str, Any]:
        avail = self.availability()
        if not avail["available"]:
            return {"ok": False, "status": "unavailable", **avail}

        sid = session_id or str(uuid.uuid4())
        async with self._lock:
            if sid in self.sessions:
                return await self.status(sid)

        try:
            bb = _Browserbase(api_key=self._api_key)
            if stealth or solve_captchas:
                logger.warning(
                    "Ignoring stealth/CAPTCHA-solving request; Theriad requires transparent automation and human challenge handoff."
                )
            create_kwargs: dict[str, Any] = {
                "browser_settings": {
                    "solveCaptchas": False,
                    "advancedStealth": False,
                },
                "keep_alive": keep_alive,
                "timeout": timeout,
            }
            if self._project_id:
                create_kwargs["project_id"] = self._project_id
            bb_session = bb.sessions.create(**create_kwargs)
        except Exception as exc:
            logger.error("BrowserBase session creation failed: %s", exc)
            return {"ok": False, "status": "failed", "error": f"BrowserBase session creation failed: {exc}"}

        try:
            pw = await _async_playwright().start()
            browser = await pw.chromium.connect_over_cdp(bb_session.connect_url)
            context = browser.contexts[0]
            page = context.pages[0]
        except Exception as exc:
            logger.error("Playwright CDP connect failed: %s", exc)
            return {"ok": False, "status": "failed", "error": f"Playwright CDP connect failed: {exc}"}

        session = BrowserBaseSession(
            session_id=sid,
            bb_session_id=bb_session.id,
            playwright=pw,
            browser=browser,
            context=context,
            page=page,
            user_prompt=user_prompt,
        )
        async with self._lock:
            self.sessions[sid] = session

        if start_url:
            nav = await self.navigate(sid, start_url)
            if not nav.get("ok"):
                await self.stop_session(sid)
                return nav

        session.current_url = page.url
        logger.info("BrowserBase session started: %s (bb=%s)", sid, bb_session.id)
        return {
            "ok": True,
            "status": "running",
            "session_id": sid,
            "bb_session_id": bb_session.id,
            "backend": "browserbase",
            "current_url": session.current_url,
        }

    async def stop_session(self, session_id: str) -> dict[str, Any]:
        async with self._lock:
            session = self.sessions.pop(session_id, None)
        if session is None:
            return {"ok": False, "status": "not_found", "error": "Session not found."}
        session.status = "stopped"
        try:
            await session.browser.disconnect()
        except Exception as exc:
            logger.warning("Browser disconnect error (session %s): %s", session_id, exc)
        try:
            await session.playwright.stop()
        except Exception as exc:
            logger.warning("Playwright stop error (session %s): %s", session_id, exc)
        logger.info("BrowserBase session stopped: %s", session_id)
        return {"ok": True, "status": "stopped", "session_id": session_id}

    async def stop_all(self) -> None:
        for sid in list(self.sessions):
            await self.stop_session(sid)

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    async def observe(self, session_id: str, *, screenshot: bool = False) -> dict[str, Any]:
        session = self._session(session_id)
        try:
            dom = await session.page.evaluate(_OBSERVE_JS, _MAX_ELEMENTS)
        except Exception as exc:
            return {"ok": False, "status": "failed", "error": f"DOM evaluation failed: {exc}"}

        session.elements = {
            item["id"]: item
            for item in [*dom["clickable_elements"], *dom["input_fields"]]
        }

        screenshot_b64 = None
        if screenshot:
            try:
                raw = await session.page.screenshot(type="png", full_page=False)
                screenshot_b64 = base64.b64encode(raw).decode("ascii")
            except Exception as exc:
                logger.warning("Screenshot failed (session %s): %s", session_id, exc)

        try:
            page_title = await session.page.title()
        except Exception:
            page_title = ""

        session.current_url = session.page.url

        captcha_detected, captcha_reason = detect_captcha(
            session.current_url, page_title, dom["visible_text"]
        )

        return {
            "ok": True,
            "status": session.status,
            "session_id": session_id,
            "backend": "browserbase",
            "current_url": session.current_url,
            "page_title": page_title,
            "screenshot_base64": screenshot_b64,
            "accessibility_tree_summary": dom["accessibility_tree_summary"],
            "visible_text_summary": dom["visible_text"],
            "clickable_elements": dom["clickable_elements"],
            "input_fields": dom["input_fields"],
            "captcha_detected": captcha_detected,
            "captcha_reason": captcha_reason,
            "action_log": list(session.action_log),
            "bb_session_id": session.bb_session_id,
        }

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    async def navigate(self, session_id: str, url: str) -> dict[str, Any]:
        session = self._session(session_id)
        try:
            await session.page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        except Exception as exc:
            if session.page.url in ("about:blank", ""):
                self._log(session, "navigate", "failed", error=str(exc))
                return {"ok": False, "status": "failed", "error": f"Navigation failed: {exc}"}
            # Partial load — continue
        session.current_url = session.page.url
        self._log(session, "navigate", "executed", target=url)
        return await self.observe(session_id)

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    async def action(self, session_id: str, action: dict[str, Any]) -> dict[str, Any]:
        session = self._session(session_id)
        kind = str(action.get("action") or "").strip().lower()
        if not kind:
            return {"ok": False, "status": "failed", "error": "action key is required"}

        try:
            if kind == "navigate":
                return await self.navigate(session_id, str(action["url"]))
            elif kind == "click":
                element_id = str(action.get("element_id") or "")
                await session.page.locator(
                    f'[data-theriad-id="{element_id}"]'
                ).click(timeout=10_000)
            elif kind in {"fill", "type"}:
                element_id = str(action.get("element_id") or "")
                await session.page.locator(
                    f'[data-theriad-id="{element_id}"]'
                ).fill(str(action.get("text") or ""), timeout=10_000)
            elif kind == "press_key":
                element_id = action.get("element_id")
                locator = (
                    session.page.locator(f'[data-theriad-id="{element_id}"]')
                    if element_id else session.page.locator("body")
                )
                await locator.press(str(action.get("key") or "Enter"))
            elif kind == "scroll":
                amount = int(action.get("amount", 700))
                await session.page.evaluate("(amt) => window.scrollBy(0, amt)", amount)
            elif kind == "wait":
                ms = min(int(action.get("milliseconds", 500)), 10_000)
                await session.page.wait_for_timeout(ms)
            else:
                return {"ok": False, "status": "failed", "error": f"Unsupported action: {kind}"}
        except Exception as exc:
            self._log(session, kind, "failed", error=str(exc))
            return {"ok": False, "status": "failed", "error": str(exc)}

        self._log(session, kind, "executed", target=action.get("element_id") or action.get("url"))
        return await self.observe(session_id)

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------

    async def status(self, session_id: str | None = None) -> dict[str, Any]:
        if session_id:
            session = self.sessions.get(session_id)
            if session is None:
                return {"ok": False, "status": "not_found", "error": "Session not found."}
            return {
                "ok": True,
                "status": session.status,
                "session_id": session_id,
                "bb_session_id": session.bb_session_id,
                "current_url": session.current_url,
                "backend": "browserbase",
                "action_log": list(session.action_log),
            }
        avail = self.availability()
        return {
            "ok": True,
            "status": "available" if avail["available"] else "unavailable",
            "backend": "browserbase",
            "active_sessions": len(self.sessions),
            **avail,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _session(self, session_id: str) -> BrowserBaseSession:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError(f"BrowserBase session not found: {session_id}")
        return session

    def _log(
        self,
        session: BrowserBaseSession,
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


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------

_runtime: BrowserBaseRuntime | None = None


def get_browserbase_runtime() -> BrowserBaseRuntime:
    global _runtime
    if _runtime is None:
        _runtime = BrowserBaseRuntime()
    return _runtime
