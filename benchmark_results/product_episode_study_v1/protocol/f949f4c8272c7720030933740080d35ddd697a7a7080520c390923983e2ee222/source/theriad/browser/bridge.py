"""Compatibility facade over the OpenSandbox browser runtime.

This module keeps the small bridge interface used by older internal callers,
but it no longer constructs a mock, local Playwright, Browserbase, Electron,
Chrome-MCP, or extension adapter. Every bridge operation reaches the same
OpenSandbox-managed Chromium session API.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from theriad.agents.browser_events import ObservationEvent

from .opensandbox_runtime import (
    OPEN_SANDBOX_BACKEND,
    OpenSandboxBrowserRuntime,
    get_opensandbox_browser_runtime,
)


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
    """Stable bridge API whose sole backend is OpenSandbox."""

    def __init__(
        self,
        *,
        backend: str = OPEN_SANDBOX_BACKEND,
        observer: Any = None,
        chrome_mcp: Any = None,
        policy: Any = None,
        runtime: OpenSandboxBrowserRuntime | None = None,
    ) -> None:
        # These arguments remain accepted so old callers keep importing, but
        # no legacy backend is instantiated or consulted.
        del observer, chrome_mcp, policy
        self.requested_backend = backend
        self.backend = OPEN_SANDBOX_BACKEND
        self.runtime = runtime or get_opensandbox_browser_runtime()

    async def start_session(
        self,
        config: dict[str, Any] | None = None,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
        allowed_domains: list[str] | None = None,
    ) -> BridgeResult:
        config = dict(config or {})
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

    async def screenshot(
        self,
        session_id: str,
        *,
        owner_user_id: str | None = None,
        owner_organization_id: str | None = None,
    ) -> BridgeResult:
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
