"""Browser adapter implementations used by :mod:`theriad.browser.bridge`."""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

from theriad.agents.browser_events import ObservationEvent

Observer = Callable[..., Awaitable[ObservationEvent]]


@dataclass
class AdapterSession:
    url: str = "about:blank"
    config: dict[str, Any] = field(default_factory=dict)


class BrowserAdapter:
    backend = "mock"
    available = True
    unavailable_error: str | None = None

    def __init__(self, observer: Observer) -> None:
        self.observer = observer
        self.sessions: dict[str, AdapterSession] = {}

    async def start_session(self, session_id: str, config: dict[str, Any]) -> None:
        self.sessions[session_id] = AdapterSession(
            url=config.get("start_url") or "about:blank",
            config=dict(config),
        )

    async def close_session(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def session(self, session_id: str) -> AdapterSession:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown browser bridge session: {session_id}")
        return self.sessions[session_id]

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        raise NotImplementedError

    async def navigate(self, session_id: str, url: str) -> ObservationEvent:
        state = self.session(session_id)
        state.url = url
        return await self.observe(session_id)

    async def click(self, session_id: str, target: str) -> ObservationEvent:
        state = self.session(session_id)
        if target.startswith(("http://", "https://", "/")):
            state.url = target
        return await self.observe(session_id)

    async def scroll(self, session_id: str, amount: str | int) -> ObservationEvent:
        return await self.observe(session_id)

    async def fill(
        self,
        session_id: str,
        target: str,
        text: str,
        *,
        approved: bool = False,
    ) -> ObservationEvent:
        """Compatibility hook for typed actions on adapters that support them."""
        del target, text, approved
        return await self.observe(session_id)


class MockBrowserAdapter(BrowserAdapter):
    backend = "mock"

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        state = self.session(session_id)
        return await self.observer(
            state.url,
            use_screenshot=screenshot,
            mock_mode=True,
            real_observation_enabled=False,
            forced_page_type=forced_page_type,
        )


class PlaywrightBrowserAdapter(BrowserAdapter):
    backend = "playwright"

    def __init__(self, observer: Observer) -> None:
        super().__init__(observer)
        from .local_runtime import get_local_browser_runtime
        self.runtime = get_local_browser_runtime()
        # Tests and embedders may inject a deterministic observer. Preserve
        # that seam; production uses the canonical observation module and the
        # persistent Chromium runtime below.
        self._use_injected_observer = (
            getattr(observer, "__module__", "") != "theriad.agents.browser_observation"
        )
        # Skip the Playwright availability check when an injected observer is
        # provided — the injected observer handles observation itself, so
        # Playwright does not need to be installed.
        if not self._use_injected_observer:
            try:
                import importlib
                importlib.import_module("playwright.async_api")
            except ImportError:
                self.available = False
                self.unavailable_error = (
                    "Playwright is unavailable. Install with: "
                    "pip install playwright && playwright install chromium"
                )

    async def start_session(self, session_id: str, config: dict[str, Any]) -> None:
        await super().start_session(session_id, config)
        if self._use_injected_observer:
            return
        result = await self.runtime.start_session(
            session_id=session_id,
            visible=bool(config.get("visible", False)),
            start_url=config.get("start_url"),
            user_prompt=str(config.get("user_prompt") or ""),
            allow_placeholder_navigation=bool(
                config.get("allow_placeholder_navigation", False)
            ),
            user_data_dir=config.get("user_data_dir"),
        )
        if not result.get("ok"):
            self.sessions.pop(session_id, None)
            raise RuntimeError(result.get("error") or "Chromium session failed to start.")

    async def close_session(self, session_id: str) -> None:
        if not self._use_injected_observer:
            await self.runtime.stop_session(session_id)
        await super().close_session(session_id)

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        if self._use_injected_observer:
            state = self.session(session_id)
            return await self.observer(
                state.url,
                use_screenshot=screenshot,
                mock_mode=False,
                real_observation_enabled=True,
            )
        data = await self.runtime.observe(session_id, screenshot=screenshot)
        self.session(session_id).url = data["current_url"]
        return ObservationEvent(
            url=data["current_url"],
            title=data["page_title"],
            observation_type="both" if data.get("screenshot_base64") else "dom",
            visible_text=data["visible_text_summary"],
            clickable_elements=data["clickable_elements"],
            forms=[{"inputs": data["input_fields"]}] if data["input_fields"] else [],
            aria_labels=[
                item["name"] for item in data["accessibility_tree_summary"]
                if item.get("name")
            ],
            raw_dom_snippet=data.get("screenshot_base64"),
            screenshot_description=(
                "[Screenshot captured from persistent Chromium session]"
                if data.get("screenshot_base64") else None
            ),
            captcha_detected=bool(data.get("captcha_detected")),
            captcha_reason=data.get("captcha_reason"),
        )

    async def navigate(self, session_id: str, url: str) -> ObservationEvent:
        if self._use_injected_observer:
            return await super().navigate(session_id, url)
        result = await self.runtime.navigate(session_id, url)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Navigation failed.")
        self.session(session_id).url = result["current_url"]
        return await self.observe(session_id)

    async def click(self, session_id: str, target: str) -> ObservationEvent:
        if self._use_injected_observer:
            return await super().click(session_id, target)
        if target.startswith(("http://", "https://")):
            return await self.navigate(session_id, target)
        current = await self.runtime.observe(session_id, screenshot=False)
        match = next(
            (
                item for item in current["clickable_elements"]
                if target in (item.get("id"), item.get("text"), item.get("label"))
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"No clickable element matched '{target}'.")
        result = await self.runtime.action(
            session_id,
            {"action": "click", "element_id": match["id"]},
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("reason") or result.get("error") or "Click failed.")
        return await self.observe(session_id)

    async def scroll(self, session_id: str, amount: str | int) -> ObservationEvent:
        if self._use_injected_observer:
            return await super().scroll(session_id, amount)
        pixels = int(amount) if str(amount).lstrip("-").isdigit() else 700
        result = await self.runtime.action(
            session_id,
            {"action": "scroll", "amount": pixels},
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Scroll failed.")
        return await self.observe(session_id)


class ChromeMCPBrowserAdapter(BrowserAdapter):
    backend = "chrome_mcp"

    def __init__(self, observer: Observer, chrome_mcp: Any = None) -> None:
        super().__init__(observer)
        self.chrome_mcp = chrome_mcp
        if chrome_mcp is None:
            # Use the adapter's complete configuration check rather than the
            # legacy RuntimeConfig flag alone.  A caller may intentionally
            # configure Chrome MCP with CHROME_MCP_WS_URL and omit the old
            # *_ENABLED boolean; both forms must reach the same compatibility
            # path.  This check is metadata-only and does not spawn Chrome.
            from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter

            configured = ChromeDevToolsAdapter()
            self.available = bool(configured.available)
            self.unavailable_error = (
                configured.error
                if not self.available
                else None
            )

    async def start_session(self, session_id: str, config: dict[str, Any]) -> None:
        if self.chrome_mcp is None:
            from theriad.tools.chrome_mcp import build_chrome_mcp
            self.chrome_mcp = await asyncio.to_thread(build_chrome_mcp, session_id)
        await super().start_session(session_id, config)
        try:
            new_page = getattr(self.chrome_mcp, "new_page", None)
            if callable(new_page):
                await asyncio.to_thread(new_page)
            start_url = str(config.get("start_url") or "").strip()
            if start_url and start_url != "about:blank":
                await asyncio.to_thread(self.chrome_mcp.navigate, start_url)
        except Exception:
            self.sessions.pop(session_id, None)
            close = getattr(self.chrome_mcp, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
            raise

    async def close_session(self, session_id: str) -> None:
        try:
            close = getattr(self.chrome_mcp, "close", None)
            if callable(close):
                await asyncio.to_thread(close)
        finally:
            await super().close_session(session_id)

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        data = await asyncio.to_thread(self.chrome_mcp.read_dom)
        shot = await asyncio.to_thread(self.chrome_mcp.screenshot) if screenshot else None
        url = str(data.get("url") or self.session(session_id).url)
        self.session(session_id).url = url
        return ObservationEvent(
            url=url,
            title=str(data.get("title") or ""),
            observation_type="both" if screenshot else "dom",
            visible_text=str(data.get("visible_text") or ""),
            clickable_elements=list(data.get("links") or []) + list(data.get("buttons") or []),
            screenshot_description="[Chrome MCP screenshot available]" if shot else None,
        )

    async def navigate(self, session_id: str, url: str) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        await asyncio.to_thread(self.chrome_mcp.navigate, url)
        self.session(session_id).url = url
        return await self.observe(session_id)

    async def click(self, session_id: str, target: str) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        await asyncio.to_thread(self.chrome_mcp.click, target)
        return await self.observe(session_id)

    async def scroll(self, session_id: str, amount: str | int) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        expression = f"window.scrollBy(0, {int(amount) if str(amount).lstrip('-').isdigit() else 700})"
        await asyncio.to_thread(self.chrome_mcp.evaluate_js, expression)
        return await self.observe(session_id)

    async def fill(
        self,
        session_id: str,
        target: str,
        text: str,
        *,
        approved: bool = False,
    ) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        await asyncio.to_thread(
            self.chrome_mcp.type,
            target,
            text,
            approved=approved,
        )
        return await self.observe(session_id)


class ExtensionBrowserAdapter(BrowserAdapter):
    """Reserved interface for the future user-session browser extension."""

    backend = "extension"
    available = False
    unavailable_error = "Browser extension adapter is not implemented."

    async def observe(self, session_id: str, **_: Any) -> ObservationEvent:
        raise RuntimeError(self.unavailable_error)


class BrowserBaseBrowserAdapter(BrowserAdapter):
    """
    Drives a BrowserBase cloud Chromium session via Playwright CDP.

    Managed cloud Chromium with explicit CAPTCHA detection and human handoff.

    Requires:
        pip install browserbase
        BROWSERBASE_API_KEY env var (and optionally BROWSERBASE_PROJECT_ID)
    """

    backend = "browserbase"

    def __init__(self, observer: Observer) -> None:
        super().__init__(observer)
        from .browserbase_runtime import get_browserbase_runtime
        self.runtime = get_browserbase_runtime()
        avail = self.runtime.availability()
        if not avail["available"]:
            self.available = False
            self.unavailable_error = avail["error"]

    async def start_session(self, session_id: str, config: dict[str, Any]) -> None:
        avail = self.runtime.availability()
        if not avail["available"]:
            self.available = False
            self.unavailable_error = avail["error"]
            raise RuntimeError(self.unavailable_error)
        await super().start_session(session_id, config)
        result = await self.runtime.start_session(
            session_id=session_id,
            start_url=config.get("start_url"),
            user_prompt=str(config.get("user_prompt") or ""),
            stealth=False,
            solve_captchas=False,
            keep_alive=bool(config.get("keep_alive", False)),
        )
        if not result.get("ok"):
            self.sessions.pop(session_id, None)
            raise RuntimeError(result.get("error") or "BrowserBase session failed to start.")

    async def close_session(self, session_id: str) -> None:
        await self.runtime.stop_session(session_id)
        await super().close_session(session_id)

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        if not self.available:
            raise RuntimeError(self.unavailable_error)
        data = await self.runtime.observe(session_id, screenshot=screenshot)
        if not data.get("ok"):
            raise RuntimeError(data.get("error") or "BrowserBase observe failed.")
        self.session(session_id).url = data["current_url"]
        return ObservationEvent(
            url=data["current_url"],
            title=data.get("page_title", ""),
            observation_type="both" if data.get("screenshot_base64") else "dom",
            visible_text=data.get("visible_text_summary", ""),
            clickable_elements=data.get("clickable_elements", []),
            forms=[{"inputs": data["input_fields"]}] if data.get("input_fields") else [],
            aria_labels=[
                item["name"]
                for item in data.get("accessibility_tree_summary", [])
                if item.get("name")
            ],
            raw_dom_snippet=data.get("screenshot_base64"),
            screenshot_description=(
                "[Screenshot from BrowserBase cloud Chromium]"
                if data.get("screenshot_base64") else None
            ),
            captcha_detected=bool(data.get("captcha_detected")),
            captcha_reason=data.get("captcha_reason"),
        )

    async def navigate(self, session_id: str, url: str) -> ObservationEvent:
        result = await self.runtime.navigate(session_id, url)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Navigation failed.")
        self.session(session_id).url = result["current_url"]
        return await self.observe(session_id)

    async def click(self, session_id: str, target: str) -> ObservationEvent:
        if target.startswith(("http://", "https://")):
            return await self.navigate(session_id, target)
        current = await self.runtime.observe(session_id, screenshot=False)
        match = next(
            (
                item for item in current.get("clickable_elements", [])
                if target in (item.get("id"), item.get("text"), item.get("label"))
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"No clickable element matched '{target}'.")
        result = await self.runtime.action(
            session_id,
            {"action": "click", "element_id": match["id"]},
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Click failed.")
        return await self.observe(session_id)

    async def scroll(self, session_id: str, amount: str | int) -> ObservationEvent:
        pixels = int(amount) if str(amount).lstrip("-").isdigit() else 700
        result = await self.runtime.action(
            session_id, {"action": "scroll", "amount": pixels}
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Scroll failed.")
        return await self.observe(session_id)


class ElectronBrowserAdapter(BrowserAdapter):
    """
    Drives the Theriad Browser Electron BrowserView via the local HTTP RPC server.

    Electron's BrowserView can retain its own persistent cookies and localStorage.
    It is not represented as undetectable automation.

    Requires the Electron app to be running:
      cd apps/theriad-extension/electron-browser && npm run dev
    """

    backend = "electron"

    def __init__(self, observer: Observer) -> None:
        super().__init__(observer)
        from .electron_runtime import get_electron_browser_runtime
        self.runtime = get_electron_browser_runtime()
        # Availability is checked at start_session time (app may start later).
        # We mark available=True here so the adapter can be registered; the
        # runtime itself returns an error dict if Electron isn't up yet.

    async def start_session(self, session_id: str, config: dict[str, Any]) -> None:
        availability = self.runtime.availability()
        if not availability["available"]:
            self.available = False
            self.unavailable_error = availability["error"]
            raise RuntimeError(self.unavailable_error)
        await super().start_session(session_id, config)
        result = await self.runtime.start_session(
            session_id=session_id,
            visible=True,
            start_url=config.get("start_url"),
            user_prompt=str(config.get("user_prompt") or ""),
        )
        if not result.get("ok"):
            self.sessions.pop(session_id, None)
            raise RuntimeError(result.get("error") or "Electron session failed to start.")

    async def close_session(self, session_id: str) -> None:
        await self.runtime.stop_session(session_id)
        await super().close_session(session_id)

    async def observe(
        self,
        session_id: str,
        *,
        screenshot: bool = False,
        forced_page_type: str | None = None,
    ) -> ObservationEvent:
        data = await self.runtime.observe(session_id, screenshot=screenshot)
        if not data.get("ok"):
            raise RuntimeError(data.get("error") or "Electron observe failed.")
        self.session(session_id).url = data["current_url"]
        return ObservationEvent(
            url=data["current_url"],
            title=data.get("page_title", ""),
            observation_type="both" if data.get("screenshot_base64") else "dom",
            visible_text=data.get("visible_text_summary", ""),
            clickable_elements=data.get("clickable_elements", []),
            forms=[{"inputs": data["input_fields"]}] if data.get("input_fields") else [],
            aria_labels=[
                item["name"]
                for item in data.get("accessibility_tree_summary", [])
                if item.get("name")
            ],
            raw_dom_snippet=data.get("screenshot_base64"),
            screenshot_description=(
                "[Screenshot from Electron BrowserView — persistent Chromium, real cookies]"
                if data.get("screenshot_base64") else None
            ),
        )

    async def navigate(self, session_id: str, url: str) -> ObservationEvent:
        result = await self.runtime.navigate(session_id, url)
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Navigation failed.")
        self.session(session_id).url = result["current_url"]
        return await self.observe(session_id)

    async def click(self, session_id: str, target: str) -> ObservationEvent:
        if target.startswith(("http://", "https://")):
            return await self.navigate(session_id, target)
        # Observe first to inject data-theriad-id attributes into the page.
        current = await self.runtime.observe(session_id, screenshot=False)
        match = next(
            (
                item for item in current.get("clickable_elements", [])
                if target in (item.get("id"), item.get("text"), item.get("label"))
            ),
            None,
        )
        if match is None:
            raise RuntimeError(f"No clickable element matched '{target}'.")
        result = await self.runtime.action(
            session_id,
            {"action": "click", "element_id": match["id"]},
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Click failed.")
        return await self.observe(session_id)

    async def scroll(self, session_id: str, amount: str | int) -> ObservationEvent:
        pixels = int(amount) if str(amount).lstrip("-").isdigit() else 700
        result = await self.runtime.action(
            session_id, {"action": "scroll", "amount": pixels}
        )
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "Scroll failed.")
        return await self.observe(session_id)
