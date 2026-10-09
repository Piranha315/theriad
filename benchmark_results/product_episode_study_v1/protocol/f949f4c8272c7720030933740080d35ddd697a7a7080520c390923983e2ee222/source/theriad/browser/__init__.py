"""The single active browser execution boundary.

OpenSandbox owns browser isolation and Chromium lifecycle. Playwright is used
only as the CDP client inside :mod:`theriad.browser.opensandbox_runtime`.
Legacy local, mock, Browserbase, Electron, and MCP adapters remain in their
old modules only so older imports can be retired deliberately; they are not
exported or selected by the application.
"""

from .bridge import BrowserBridge, BridgeResult
from .opensandbox_runtime import (
    OPEN_SANDBOX_BACKEND,
    OpenSandboxBrowserRuntime,
    OpenSandboxBrowserSession,
    get_opensandbox_browser_runtime,
)
from .policy import BrowserPolicy, BrowserPolicyDecision

__all__ = [
    "OPEN_SANDBOX_BACKEND",
    "BrowserBridge",
    "BridgeResult",
    "BrowserPolicy",
    "BrowserPolicyDecision",
    "OpenSandboxBrowserRuntime",
    "OpenSandboxBrowserSession",
    "get_opensandbox_browser_runtime",
]
