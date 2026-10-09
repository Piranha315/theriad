"""Persistent local browser runtime backed by Playwright's bundled Chromium."""
from __future__ import annotations

import asyncio
import base64
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import urlsplit

from .policy import BrowserPolicy
from theriad.search import (
    is_placeholder_url,
    is_unresolved_search_url,
    resolve_initial_navigation,
)

logger = logging.getLogger(__name__)

# Errors that indicate the page navigated mid-evaluation and the JS execution
# context was torn down.  These are transient — waiting for the new page to
# settle and retrying usually succeeds.
_NAVIGATION_ERROR_FRAGMENTS = (
    "execution context was destroyed",
    "navigating",
    "target closed",
    "frame was detached",
    "context was destroyed",
)

_OBSERVE_MAX_RETRIES = 3
_OBSERVE_RETRY_DELAY_S = 0.6

_MAX_TEXT = 4_000
_MAX_ELEMENTS = 60
_SENSITIVE_FIELD_TERMS = (
    "password", "passcode", "api key", "apikey", "secret", "token",
    "private key", "credential",
)
_APPROVAL_TEXT_ACTIONS = (
    ("delete", "delete"),
    ("remove", "delete"),
    ("purchase", "purchase"),
    ("buy", "purchase"),
    ("checkout", "payment"),
    ("pay", "payment"),
    ("send", "send_message"),
    ("submit", "submit_form"),
    ("sign in", "login"),
    ("log in", "login"),
    ("login", "login"),
    ("download", "download"),
    ("upload", "upload"),
)


def _normalize_navigation_url(url: str) -> tuple[str | None, str | None]:
    """Allow HTTP(S) destinations and reject page-controlled schemes."""
    candidate = (url or "").strip()
    if candidate == "about:blank" or candidate.startswith("data:text/html,"):
        return candidate, None
    parsed = urlsplit(candidate)
    if parsed.scheme:
        if parsed.scheme.lower() not in {"http", "https"}:
            return None, f"Navigation blocked: unsupported URL scheme '{parsed.scheme}:'."
        if not parsed.netloc:
            return None, "Navigation blocked: HTTP(S) URL must include a host."
        return candidate, None
    if candidate.startswith("//"):
        return f"https:{candidate}", None
    normalized = f"https://{candidate}"
    if not urlsplit(normalized).netloc:
        return None, "Navigation blocked: URL must include a host."
    return normalized, None


@dataclass
class LocalBrowserSession:
    session_id: str
    playwright: Any
    browser: Any
    context: Any
    page: Any
    visible: bool
    created_at: float
    owner_user_id: str | None = None
    owner_organization_id: str | None = None
    user_prompt: str = ""
    resolved_initial_url: str | None = None
    navigation_reason: str | None = None
    allow_placeholder_navigation: bool = False
    status: str = "running"
    elements: dict[str, dict[str, Any]] = field(default_factory=dict)
    pending_action: dict[str, Any] | None = None
    action_log: list[dict[str, Any]] = field(default_factory=list)


class LocalBrowserRuntime:
    """Own isolated Chromium contexts and execute policy-gated UI actions."""

    def __init__(self, policy: BrowserPolicy | None = None) -> None:
        self.policy = policy or BrowserPolicy()
        self.sessions: dict[str, LocalBrowserSession] = {}
        self._lock = asyncio.Lock()

    @staticmethod
    def availability() -> dict[str, Any]:
        try:
            import importlib
            importlib.import_module("playwright.async_api")
            return {"available": True, "error": None}
        except ImportError:
            return {
                "available": False,
                "error": (
                    "Playwright is unavailable. Install with: "
                    "pip install playwright && playwright install chromium"
                ),
            }

    async def start_session(
        self,
        *,
        session_id: str | None = None,
        visible: bool = False,
        start_url: str | None = None,
        viewport: dict[str, int] | None = None,
        user_prompt: str = "",
        allow_placeholder_navigation: bool = False,
        user_data_dir: str | None = None,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> dict[str, Any]:
        availability = self.availability()
        if not availability["available"]:
            return {"ok": False, "status": "unavailable", **availability}

        from playwright.async_api import async_playwright

        sid = session_id or str(uuid.uuid4())
        vp = viewport or {"width": 1280, "height": 900}
        async with self._lock:
            if sid in self.sessions:
                return await self.status(sid, owner_user_id=owner_user_id)
            pw = await async_playwright().start()
            try:
                if user_data_dir:
                    context = await pw.chromium.launch_persistent_context(
                        user_data_dir,
                        headless=not visible,
                        args=["--disable-dev-shm-usage"],
                        viewport=vp,
                        accept_downloads=False,
                        bypass_csp=False,
                    )
                    browser = context.browser
                    page = context.pages[0] if context.pages else await context.new_page()
                else:
                    browser = await pw.chromium.launch(
                        headless=not visible,
                        args=["--disable-extensions", "--disable-dev-shm-usage"],
                    )
                    context = await browser.new_context(
                        viewport=vp,
                        accept_downloads=False,
                        bypass_csp=False,
                    )
                    page = await context.new_page()
            except Exception as exc:
                await pw.stop()
                return {
                    "ok": False,
                    "status": "unavailable",
                    "error": f"Chromium could not launch: {exc}",
                    "available": False,
                }
            import time
            session = LocalBrowserSession(
                session_id=sid,
                playwright=pw,
                browser=browser,
                context=context,
                page=page,
                visible=visible,
                created_at=time.time(),
                owner_user_id=owner_user_id,
                owner_organization_id=owner_organization_id,
                user_prompt=user_prompt,
                allow_placeholder_navigation=allow_placeholder_navigation,
            )
            self.sessions[sid] = session

        if start_url:
            result = await self.navigate(sid, start_url, owner_user_id=owner_user_id)
            if not result.get("ok"):
                await self.stop_session(sid, owner_user_id=owner_user_id)
                return result
        return {
            "ok": True,
            "status": "running",
            "session_id": sid,
            "visible": visible,
            "backend": "playwright",
            "browser": "chromium",
            "url": session.page.url,
        }

    async def stop_session(
        self, session_id: str, *, owner_user_id: str | None = None
    ) -> dict[str, Any]:
        async with self._lock:
            session = self.sessions.get(session_id)
            if session is not None and not self._owned(session, owner_user_id):
                session = None
            if session is not None:
                self.sessions.pop(session_id, None)
        if session is None:
            return {"ok": False, "status": "not_found", "error": "Browser session not found."}
        session.status = "stopped"
        try:
            await session.context.close()
        finally:
            try:
                await session.browser.close()
            finally:
                await session.playwright.stop()
        return {"ok": True, "status": "stopped", "session_id": session_id}

    async def stop_all(self) -> None:
        for session_id in list(self.sessions):
            await self.stop_session(session_id)

    @staticmethod
    def _owned(session: LocalBrowserSession, owner_user_id: str | None) -> bool:
        return session.owner_user_id == owner_user_id and (
            session.owner_user_id is None or owner_user_id is not None
        )

    def _session(
        self, session_id: str, *, owner_user_id: str | None = None
    ) -> LocalBrowserSession:
        session = self.sessions.get(session_id)
        if session is None or not self._owned(session, owner_user_id):
            raise KeyError("Browser session not found.")
        return session

    async def navigate(
        self, session_id: str, url: str, *, owner_user_id: str | None = None
    ) -> dict[str, Any]:
        session = self._session(session_id, owner_user_id=owner_user_id)
        requested_url = url
        url, invalid_url_reason = _normalize_navigation_url(url)
        if invalid_url_reason:
            self._log(session, "navigate", "blocked", target=requested_url, error=invalid_url_reason)
            return {"ok": False, "status": "blocked", "error": invalid_url_reason}
        assert url is not None
        should_rewrite_placeholder = (
            is_placeholder_url(url) and not session.allow_placeholder_navigation
        )
        should_rewrite_search_homepage = is_unresolved_search_url(url)
        if should_rewrite_placeholder or should_rewrite_search_homepage:
            navigation = resolve_initial_navigation(
                session.user_prompt or "Search the web"
            )
            url = navigation.resolved_url
            session.resolved_initial_url = url
            if should_rewrite_placeholder:
                session.navigation_reason = (
                    "Blocked placeholder navigation to example.com. "
                    f"{navigation.reason}"
                )
                logger.warning("Blocked placeholder navigation to example.com")
            else:
                session.navigation_reason = (
                    "Replaced unresolved search homepage with a query URL. "
                    f"{navigation.reason}"
                )
                logger.warning("Replaced unresolved search homepage with query URL")
            self._log(
                session,
                "navigate",
                "rewritten",
                target=requested_url,
                details={
                    "corrected_url": url,
                    "reason": session.navigation_reason,
                },
            )
        try:
            await session.page.goto(url, wait_until="domcontentloaded", timeout=20_000)
        except Exception as exc:
            # A timeout can still leave a useful partially loaded document.
            if session.page.url in ("about:blank", ""):
                return {"ok": False, "status": "failed", "error": f"Navigation failed: {exc}"}
        self._log(session, "navigate", "executed", target=url)
        session.resolved_initial_url = session.resolved_initial_url or url
        return await self.observe(session_id, owner_user_id=owner_user_id)

    @staticmethod
    def _is_navigation_error(exc: Exception) -> bool:
        """True when *exc* looks like a Playwright context-destruction error."""
        msg = str(exc).lower()
        return any(frag in msg for frag in _NAVIGATION_ERROR_FRAGMENTS)

    async def _wait_for_stable_page(self, session: LocalBrowserSession) -> None:
        """Best-effort wait for the page to reach a usable load state."""
        try:
            await session.page.wait_for_load_state(
                "domcontentloaded", timeout=8_000,
            )
        except Exception:
            # The page may still be usable even if this times out.
            pass

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = True,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(session_id, owner_user_id=owner_user_id)

        # --- DOM extraction with retry on navigation-caused context destruction ---
        dom: dict[str, Any] | None = None
        last_exc: Exception | None = None
        for attempt in range(_OBSERVE_MAX_RETRIES):
            try:
                dom = await session.page.evaluate(
                    """(limit) => {
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
                    }""",
                    _MAX_ELEMENTS,
                )
                break  # success
            except Exception as exc:
                last_exc = exc
                if not self._is_navigation_error(exc) or attempt == _OBSERVE_MAX_RETRIES - 1:
                    raise
                logger.warning(
                    "observe: page.evaluate failed (attempt %d/%d): %s — waiting for page to settle",
                    attempt + 1, _OBSERVE_MAX_RETRIES, exc,
                )
                await asyncio.sleep(_OBSERVE_RETRY_DELAY_S)
                await self._wait_for_stable_page(session)

        assert dom is not None  # the loop either set dom or raised

        session.elements = {
            item["id"]: item
            for item in [*dom["clickable_elements"], *dom["input_fields"]]
        }

        # --- Screenshot (also retry-protected) ---
        screenshot_b64 = None
        if screenshot:
            for attempt in range(_OBSERVE_MAX_RETRIES):
                try:
                    raw = await session.page.screenshot(type="png", full_page=False)
                    screenshot_b64 = base64.b64encode(raw).decode("ascii")
                    break
                except Exception as exc:
                    if not self._is_navigation_error(exc) or attempt == _OBSERVE_MAX_RETRIES - 1:
                        raise
                    await asyncio.sleep(_OBSERVE_RETRY_DELAY_S)
                    await self._wait_for_stable_page(session)

        # --- Page title (also retry-protected) ---
        from theriad.agents.browser_observation import detect_captcha
        page_title = ""
        for attempt in range(_OBSERVE_MAX_RETRIES):
            try:
                page_title = await session.page.title()
                break
            except Exception as exc:
                if not self._is_navigation_error(exc) or attempt == _OBSERVE_MAX_RETRIES - 1:
                    page_title = ""
                    break  # non-fatal — degrade gracefully
                await asyncio.sleep(_OBSERVE_RETRY_DELAY_S)
                await self._wait_for_stable_page(session)

        captcha_detected, captcha_reason = detect_captcha(
            session.page.url, page_title, dom["visible_text"]
        )
        final_answer = None
        if captcha_detected:
            session.status = "needs_approval"
            session.pending_action = {
                "action": "captcha_handoff",
                "current_url": session.page.url,
                "page_title": page_title,
            }
            final_answer = (
                "CAPTCHA encountered. Theriad tried automatic protection handling. "
                "Complete it in the temporary task browser or switch to API search."
            )
        return {
            "ok": True,
            "status": session.status,
            "session_id": session_id,
            "backend": "playwright",
            "browser": "chromium",
            "visible": session.visible,
            "current_url": session.page.url,
            "page_title": page_title,
            "screenshot_base64": screenshot_b64,
            "screenshot_path": None,
            "accessibility_tree_summary": dom["accessibility_tree_summary"],
            "visible_text_summary": dom["visible_text"],
            "clickable_elements": dom["clickable_elements"],
            "input_fields": dom["input_fields"],
            "pending_action": session.pending_action,
            "captcha_detected": captcha_detected,
            "captcha_reason": captcha_reason,
            "final_answer": final_answer,
            "action_log": list(session.action_log),
            "untrusted_content": True,
            "original_user_prompt": session.user_prompt,
            "resolved_initial_url": session.resolved_initial_url,
            "navigation_reason": session.navigation_reason,
        }

    async def status(
        self, session_id: str | None = None, *, owner_user_id: str | None = None
    ) -> dict[str, Any]:
        if session_id:
            try:
                session = self._session(session_id, owner_user_id=owner_user_id)
            except KeyError as exc:
                return {"ok": False, "status": "not_found", "error": str(exc)}
            return {
                "ok": True,
                "status": session.status,
                "session_id": session_id,
                "current_url": session.page.url,
                "visible": session.visible,
                "pending_action": session.pending_action,
                "action_log": list(session.action_log),
                "backend": "playwright",
                "browser": "chromium",
                "original_user_prompt": session.user_prompt,
                "resolved_initial_url": session.resolved_initial_url,
                "navigation_reason": session.navigation_reason,
            }
        availability = self.availability()
        active_sessions = (
            sum(self._owned(session, owner_user_id) for session in self.sessions.values())
            if owner_user_id is not None
            else len(self.sessions)
        )
        return {
            "ok": True,
            "status": "available" if availability["available"] else "unavailable",
            "backend": "playwright",
            "browser": "chromium",
            "active_sessions": active_sessions,
            **availability,
        }

    async def action(
        self,
        session_id: str,
        action: dict[str, Any] | None = None,
        *,
        approved: bool = False,
        deny: bool = False,
        owner_user_id: str | None = None,
    ) -> dict[str, Any]:
        session = self._session(session_id, owner_user_id=owner_user_id)
        if session.status == "needs_approval" and not (approved or deny):
            return {
                "ok": False,
                "status": "needs_approval",
                "reason": "CAPTCHA requires human handoff or API search.",
                "pending_action": session.pending_action,
            }
        if deny:
            pending = session.pending_action
            session.pending_action = None
            session.status = "running"
            self._log(session, (pending or {}).get("action", "unknown"), "denied")
            return {"ok": True, "status": "denied", "session_id": session_id}
        if (
            approved
            and session.pending_action
            and session.pending_action.get("action") == "captcha_handoff"
        ):
            session.pending_action = None
            session.status = "running"
            observed = await self.observe(session_id, owner_user_id=owner_user_id)
            if observed.get("captcha_detected"):
                return observed
            self._log(session, "captcha_handoff", "continued")
            return observed
        if approved and session.pending_action:
            action = session.pending_action
            session.pending_action = None
        action = dict(action or {})
        kind = str(action.get("action") or "").strip().lower()
        if not kind:
            return {"ok": False, "status": "failed", "error": "action is required"}

        policy_action, blocked_reason = self._policy_action(session, kind, action)
        if blocked_reason:
            self._log(session, kind, "blocked", error=blocked_reason)
            return {"ok": False, "status": "blocked", "error": blocked_reason}
        decision = self.policy.classify(policy_action, approved=approved)
        if decision.status == "approval_required":
            session.pending_action = action
            session.status = "waiting_for_approval"
            self._log(session, kind, "approval_required", target=action.get("element_id"))
            return {
                "ok": False,
                "status": "approval_required",
                "reason": decision.reason,
                "pending_action": action,
                "session_id": session_id,
            }
        if not decision.allowed:
            self._log(session, kind, decision.status, error=decision.reason)
            return {"ok": False, "status": decision.status, "error": decision.reason}

        session.status = "running"
        try:
            await self._execute(session, kind, action)
        except Exception as exc:
            self._log(session, kind, "failed", error=str(exc))
            return {"ok": False, "status": "failed", "error": str(exc)}
        self._log(session, kind, "executed", target=action.get("element_id") or action.get("url"))
        return await self.observe(session_id, owner_user_id=owner_user_id)

    def _policy_action(
        self,
        session: LocalBrowserSession,
        kind: str,
        action: dict[str, Any],
    ) -> tuple[str, str | None]:
        if kind in {"extract_password", "extract_api_key", "read_secret", "copy_secret"}:
            return "credential_theft", "Password, API-key, and secret extraction is blocked."
        element = session.elements.get(str(action.get("element_id") or ""), {})
        field_context = " ".join(
            str(element.get(key) or "") for key in ("type", "label", "aria_label", "placeholder")
        ).lower()
        if kind in {"type", "fill"} and any(term in field_context for term in _SENSITIVE_FIELD_TERMS):
            return "credential_theft", "Typing into password, API-key, token, or secret fields is blocked."
        if kind == "type":
            return "fill", None
        if kind == "click":
            text = " ".join(
                str(element.get(key) or "")
                for key in ("text", "label", "aria_label")
            ).lower()
            if element.get("in_form"):
                text = f"{text} {element.get('type', '')}"
            if element.get("download"):
                return "download", None
            for cue, action_type in _APPROVAL_TEXT_ACTIONS:
                if cue in text:
                    return action_type, None
        return kind, None

    async def _execute(
        self,
        session: LocalBrowserSession,
        kind: str,
        action: dict[str, Any],
    ) -> None:
        page = session.page
        if kind == "navigate":
            await page.goto(str(action["url"]), wait_until="domcontentloaded", timeout=20_000)
        elif kind == "click":
            await page.locator(
                f'[data-theriad-id="{str(action["element_id"])}"]'
            ).click(timeout=10_000)
        elif kind in {"type", "fill"}:
            await page.locator(
                f'[data-theriad-id="{str(action["element_id"])}"]'
            ).fill(str(action.get("text") or ""), timeout=10_000)
        elif kind == "press_key":
            element_id = action.get("element_id")
            locator = (
                page.locator(f'[data-theriad-id="{str(element_id)}"]')
                if element_id else page.locator("body")
            )
            await locator.press(str(action.get("key") or "Enter"))
        elif kind == "scroll":
            amount = int(action.get("amount", 700))
            await page.evaluate("(amount) => window.scrollBy(0, amount)", amount)
        elif kind == "wait":
            await page.wait_for_timeout(min(int(action.get("milliseconds", 500)), 10_000))
        else:
            raise ValueError(f"Unsupported browser action: {kind}")

    @staticmethod
    def _log(
        session: LocalBrowserSession,
        action: str,
        status: str,
        *,
        target: Any = None,
        error: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        import time
        session.action_log.append({
            "timestamp": time.time(),
            "action": action,
            "status": status,
            "target": target,
            "url": session.page.url,
            "error": error,
            **(details or {}),
        })
        session.action_log[:] = session.action_log[-100:]


_runtime = LocalBrowserRuntime()


def get_local_browser_runtime() -> LocalBrowserRuntime:
    return _runtime
