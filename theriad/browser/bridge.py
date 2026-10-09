"""Compatibility facade over the OpenSandbox browser runtime.

This module keeps the small bridge interface used by older internal callers.
It always checks OpenSandbox first; only an explicitly legacy bridge seam may
fall back to its injected/selected adapter when OpenSandbox is unavailable.
The default facade never silently creates a synthetic or local browser.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any

from theriad.agents.browser_events import ObservationEvent

from .opensandbox_runtime import (
    OPEN_SANDBOX_BACKEND,
    OpenSandboxBrowserRuntime,
    get_opensandbox_browser_runtime,
)
from .policy import BrowserPolicy


_LEGACY_BACKENDS = frozenset({
    "mock",
    "playwright",
    "playwright_mcp",
    "browserbase",
    "electron",
    "chrome_mcp",
    "extension",
})


@dataclass
class BridgeResult:
    ok: bool
    status: str
    backend: str
    session_id: str | None = None
    url: str | None = None
    title: str | None = None
    text: str | None = None
    dom_summary: dict[str, Any] | None = None
    screenshot_available: bool = False
    error: str | None = None
    action_log_entry: dict[str, Any] | None = None
    observation: ObservationEvent | None = field(default=None, repr=False)


class BrowserBridge:
    """Stable browser facade with OpenSandbox-first compatibility routing.

    New callers use the OpenSandbox runtime directly.  Older internal callers
    still construct this facade with an observer or a legacy backend name, so
    those calls are retained as a bounded fallback.  The fallback is selected
    only after an OpenSandbox availability check; a healthy OpenSandbox always
    takes precedence over the requested legacy backend.
    """

    def __init__(
        self,
        *,
        backend: str = OPEN_SANDBOX_BACKEND,
        observer: Any = None,
        chrome_mcp: Any = None,
        policy: Any = None,
        runtime: OpenSandboxBrowserRuntime | None = None,
        force_legacy: bool = False,
    ) -> None:
        self.requested_backend = str(backend or OPEN_SANDBOX_BACKEND).lower()
        self.policy = policy or BrowserPolicy()
        self.runtime = runtime or get_opensandbox_browser_runtime()
        self._force_legacy = bool(force_legacy)
        # An observer/backend argument is an explicit legacy compatibility
        # request.  Construct the adapter now so old embedders can inspect its
        # availability, but do not use it until OpenSandbox has been checked.
        self._legacy_requested = bool(
            observer is not None
            or chrome_mcp is not None
            or self.requested_backend in _LEGACY_BACKENDS
        )
        self._legacy_active = False
        self._backend_decided = False
        self.adapter = self._build_legacy_adapter(observer, chrome_mcp)
        self.backend = (
            getattr(self.adapter, "backend", self.requested_backend)
            if self.adapter is not None
            else OPEN_SANDBOX_BACKEND
        )

    def _build_legacy_adapter(self, observer: Any, chrome_mcp: Any) -> Any:
        if not self._legacy_requested:
            return None
        from . import adapters
        from theriad.agents.browser_observation import observe_page

        legacy_backend = self.requested_backend
        # Passing an observer to the default OpenSandbox facade is the legacy
        # BrowserBridge test/embedding seam and historically meant mock mode.
        if legacy_backend == OPEN_SANDBOX_BACKEND:
            # An injected Chrome MCP object is an explicit legacy seam. Do
            # not discard it merely because the caller omitted the old
            # backend name; the normal (non-injected) default still maps to
            # the deterministic mock adapter for old in-process callers.
            legacy_backend = "chrome_mcp" if chrome_mcp is not None else "mock"
        observer = observer or observe_page
        adapter_types = {
            "mock": adapters.MockBrowserAdapter,
            "playwright": adapters.PlaywrightBrowserAdapter,
            "browserbase": adapters.BrowserBaseBrowserAdapter,
            "electron": adapters.ElectronBrowserAdapter,
            "chrome_mcp": adapters.ChromeMCPBrowserAdapter,
            "extension": adapters.ExtensionBrowserAdapter,
        }
        # There is no standalone Playwright-MCP adapter in the current
        # compatibility package; its callers must use the same async
        # Playwright shape while they migrate to OpenSandbox.
        adapter_type = adapter_types.get(
            legacy_backend,
            adapters.PlaywrightBrowserAdapter
            if legacy_backend == "playwright_mcp"
            else adapters.MockBrowserAdapter,
        )
        if legacy_backend == "chrome_mcp":
            return adapter_type(observer, chrome_mcp)
        return adapter_type(observer)

    async def _use_legacy(self) -> bool:
        """Resolve the backend once, preferring a usable OpenSandbox path."""
        if not self._legacy_requested:
            return False
        if self._backend_decided:
            return self._legacy_active
        self._backend_decided = True
        if self._force_legacy:
            self._legacy_active = True
            self.backend = getattr(self.adapter, "backend", self.requested_backend)
            return True
        try:
            state = await self.runtime.availability()
            opensandbox_available = bool(
                isinstance(state, dict) and state.get("available")
            )
        except Exception:
            opensandbox_available = False
        if opensandbox_available:
            self._legacy_active = False
            self.backend = OPEN_SANDBOX_BACKEND
            return False
        self._legacy_active = True
        self.backend = getattr(self.adapter, "backend", self.requested_backend)
        return True

    @staticmethod
    def _legacy_result(
        *,
        ok: bool,
        status: str,
        backend: str,
        action: str,
        session_id: str | None = None,
        url: str | None = None,
        title: str | None = None,
        text: str | None = None,
        observation: ObservationEvent | None = None,
        error: str | None = None,
        target: str | None = None,
    ) -> BridgeResult:
        return BridgeResult(
            ok=ok,
            status=status,
            backend=backend,
            session_id=session_id,
            url=url,
            title=title,
            text=text,
            dom_summary=(
                {
                    "clickable_elements": observation.clickable_elements,
                    "forms": observation.forms,
                    "aria_labels": observation.aria_labels,
                }
                if observation is not None
                else None
            ),
            screenshot_available=bool(
                observation is not None
                and (observation.screenshot_description or observation.raw_dom_snippet)
            ),
            error=error,
            action_log_entry={
                "action": action,
                "target": target,
                "status": status,
                **({"error": error} if error else {}),
            },
            observation=observation,
        )

    def _legacy_error(self, error: str, *, action: str = "unknown") -> BridgeResult:
        return self._legacy_result(
            ok=False,
            status="failed",
            backend=getattr(self.adapter, "backend", self.backend),
            action=action,
            error=error,
        )

    async def _legacy_observe(
        self,
        session_id: str,
        *,
        action: str,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> BridgeResult:
        if self.adapter is None:
            return self._legacy_error("Legacy browser adapter is unavailable.", action=action)
        try:
            observation = await self.adapter.observe(
                session_id,
                screenshot=screenshot,
                forced_page_type=forced_page_type,
            )
        except Exception as exc:
            return self._legacy_error(str(exc), action=action)
        return self._legacy_result(
            ok=True,
            status="success",
            backend=self.adapter.backend,
            action=action,
            session_id=session_id,
            url=observation.url,
            title=observation.title,
            text=observation.visible_text,
            observation=observation,
        )

    async def start_session(
        self,
        config: dict[str, Any] | None = None,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> BridgeResult:
        config = dict(config or {})
        if await self._use_legacy():
            if self.adapter is None or not getattr(self.adapter, "available", False):
                return self._legacy_error(
                    getattr(self.adapter, "unavailable_error", None)
                    or f"{self.backend} is unavailable.",
                    action="start_session",
                )
            session_id = str(config.get("session_id") or uuid.uuid4())
            try:
                await self.adapter.start_session(session_id, config)
            except Exception as exc:
                return self._legacy_error(str(exc), action="start_session")
            state = self.adapter.sessions.get(session_id)
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="start_session",
                session_id=session_id,
                url=getattr(state, "url", None),
            )
        result = await self.runtime.start_session(
            session_id=config.get("session_id"),
            start_url=config.get("start_url"),
            user_prompt=str(config.get("user_prompt") or ""),
            visible=bool(config.get("visible", False)),
            viewport=config.get("viewport"),
            allowed_domains=allowed_domains or config.get("allowed_domains"),
            owner_user_id=owner_user_id or config.get("owner_user_id"),
            owner_organization_id=owner_organization_id or config.get("owner_organization_id"),
        )
        return self._from_payload(result, action="start_session")

    async def ensure_session(
        self,
        session_id: str,
        *,
        start_url: str | None = None,
        user_prompt: str = "",
        visible: bool = False,
        user_data_dir: str | None = None,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> BridgeResult:
        del user_data_dir
        if await self._use_legacy():
            if self.adapter is not None and session_id in self.adapter.sessions:
                if start_url:
                    try:
                        observation = await self.adapter.navigate(session_id, start_url)
                    except Exception as exc:
                        return self._legacy_error(str(exc), action="ensure_session")
                    return self._legacy_result(
                        ok=True,
                        status="success",
                        backend=self.adapter.backend,
                        action="ensure_session",
                        session_id=session_id,
                        url=observation.url,
                        title=observation.title,
                        text=observation.visible_text,
                        observation=observation,
                    )
                return await self._legacy_observe(
                    session_id, action="ensure_session"
                )
            return await self.start_session({
                "session_id": session_id,
                "start_url": start_url,
                "user_prompt": user_prompt,
                "visible": visible,
                "allowed_domains": allowed_domains,
            })
        if session_id in self.runtime.sessions:
            if start_url:
                result = await self.runtime.navigate(
                    session_id,
                    start_url,
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
            else:
                result = await self.runtime.status(
                    session_id,
                    owner_user_id=owner_user_id,
                    owner_organization_id=owner_organization_id,
                )
            return self._from_payload(result, action="ensure_session")
        return await self.start_session({
            "session_id": session_id,
            "start_url": start_url,
            "user_prompt": user_prompt,
            "visible": visible,
            "allowed_domains": allowed_domains,
            "owner_user_id": owner_user_id,
            "owner_organization_id": owner_organization_id,
        })

    async def close_session(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action="close_session")
            try:
                await self.adapter.close_session(session_id)
            except Exception as exc:
                return self._legacy_error(str(exc), action="close_session")
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="close_session",
                session_id=session_id,
            )
        result = await self.runtime.stop_session(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="close_session")

    async def observe(
        self,
        session_id: str,
        *,
        forced_page_type: str | None = None,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            return await self._legacy_observe(
                session_id,
                action="observe",
                forced_page_type=forced_page_type,
            )
        del forced_page_type
        result = await self.runtime.observe(
            session_id,
            screenshot=False,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="observe")

    async def read_page(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            return await self._legacy_observe(session_id, action="read_page")
        result = await self.runtime.observe(
            session_id,
            screenshot=False,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="read_page")

    async def navigate(
        self,
        session_id: str,
        url: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action="navigate")
            try:
                observation = await self.adapter.navigate(session_id, url)
            except Exception as exc:
                return self._legacy_error(str(exc), action="navigate")
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="navigate",
                session_id=session_id,
                url=observation.url,
                title=observation.title,
                text=observation.visible_text,
                observation=observation,
                target=url,
            )
        result = await self.runtime.navigate(
            session_id,
            url,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="navigate", target=url)

    async def click(
        self,
        session_id: str,
        selector_or_target: str,
        *,
        action_type: str = "click",
        approved: bool = False,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action=action_type)
            decision = self.policy.classify(action_type, approved=approved)
            if not decision.allowed:
                return self._legacy_result(
                    ok=False,
                    status=decision.status,
                    backend=self.adapter.backend,
                    action=action_type,
                    error=decision.reason,
                    target=selector_or_target,
                )
            try:
                observation = await self.adapter.click(session_id, selector_or_target)
            except Exception as exc:
                return self._legacy_error(str(exc), action=action_type)
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action=action_type,
                session_id=session_id,
                url=observation.url,
                title=observation.title,
                text=observation.visible_text,
                observation=observation,
                target=selector_or_target,
            )
        result = await self.runtime.action(
            session_id,
            {"action": action_type, "element_id": selector_or_target},
            approved=approved,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action=action_type, target=selector_or_target)

    async def scroll(
        self,
        session_id: str,
        direction_or_amount: str | int,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action="scroll")
            try:
                observation = await self.adapter.scroll(session_id, direction_or_amount)
            except Exception as exc:
                return self._legacy_error(str(exc), action="scroll")
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="scroll",
                session_id=session_id,
                url=observation.url,
                title=observation.title,
                text=observation.visible_text,
                observation=observation,
                target=str(direction_or_amount),
            )
        if isinstance(direction_or_amount, int):
            action = {"action": "scroll", "amount": direction_or_amount}
        else:
            action = {"action": "scroll", "direction": direction_or_amount}
        result = await self.runtime.action(
            session_id,
            action,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="scroll", target=str(direction_or_amount))

    async def fill(
        self,
        session_id: str,
        target: str,
        text: str,
        *,
        approved: bool = False,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action="fill")
            decision = self.policy.classify("fill", approved=approved)
            if not decision.allowed:
                return self._legacy_result(
                    ok=False,
                    status=decision.status,
                    backend=self.adapter.backend,
                    action="fill",
                    error=decision.reason,
                    target=target,
                )
            try:
                observation = await self.adapter.fill(
                    session_id,
                    target,
                    text,
                    approved=approved,
                )
            except Exception as exc:
                return self._legacy_error(str(exc), action="fill")
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="fill",
                session_id=session_id,
                url=observation.url,
                title=observation.title,
                text=observation.visible_text,
                observation=observation,
                target=target,
            )
        result = await self.runtime.action(
            session_id,
            {"action": "fill", "element_id": target, "text": text},
            approved=approved,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="fill", target=target)

    async def screenshot(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            return await self._legacy_observe(
                session_id, action="screenshot", screenshot=True
            )
        result = await self.runtime.observe(
            session_id,
            screenshot=True,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="screenshot")

    async def get_status(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
        if await self._use_legacy():
            if self.adapter is None:
                return self._legacy_error("Legacy browser adapter is unavailable.", action="get_status")
            state = self.adapter.sessions.get(session_id)
            if state is None:
                return self._legacy_error(
                    f"Unknown browser bridge session: {session_id}",
                    action="get_status",
                )
            return self._legacy_result(
                ok=True,
                status="success",
                backend=self.adapter.backend,
                action="get_status",
                session_id=session_id,
                url=state.url,
            )
        result = await self.runtime.status(
            session_id,
            owner_user_id=owner_user_id,
            owner_organization_id=owner_organization_id,
        )
        return self._from_payload(result, action="get_status")

    def _from_payload(
        self,
        payload: dict[str, Any],
        *,
        action: str,
        target: str | None = None,
    ) -> BridgeResult:
        status = str(payload.get("status") or "failed")
        ok = bool(payload.get("ok", False)) and status not in {
            "blocked",
            "failed",
            "unavailable",
            "waiting_for_approval",
            "not_found",
        }
        error = payload.get("error")
        if not ok and not error and status == "waiting_for_approval":
            error = "Browser action requires explicit approval."
        elements = payload.get("elements") or []
        observation = ObservationEvent(
            url=str(payload.get("current_url") or "about:blank"),
            title=str(payload.get("page_title") or ""),
            observation_type="both" if payload.get("screenshot_base64") else "dom",
            visible_text=str(
                payload.get("visible_text_summary") or payload.get("current_page_summary") or ""
            ),
            clickable_elements=list(payload.get("clickable_elements") or elements),
            forms=[{"inputs": list(payload.get("input_fields") or [])}]
            if payload.get("input_fields")
            else [],
            aria_labels=[
                str(item.get("label") or item.get("text") or "")
                for item in elements
                if isinstance(item, dict) and (item.get("label") or item.get("text"))
            ],
            screenshot_description=(
                "OpenSandbox screenshot captured." if payload.get("screenshot_base64") else None
            ),
            captcha_detected=bool(payload.get("captcha_detected")),
            captcha_reason=payload.get("captcha_reason"),
        )
        return BridgeResult(
            ok=ok,
            status=status,
            backend=OPEN_SANDBOX_BACKEND,
            session_id=payload.get("session_id"),
            url=payload.get("current_url"),
            title=payload.get("page_title"),
            text=payload.get("visible_text_summary") or payload.get("current_page_summary"),
            dom_summary={
                "elements": payload.get("elements") or [],
                "visible_text": payload.get("visible_text_summary") or "",
            },
            screenshot_available=bool(payload.get("screenshot_base64")),
            error=error,
            action_log_entry={
                "action": action,
                "target": target,
                "status": status,
            },
            observation=observation,
        )
