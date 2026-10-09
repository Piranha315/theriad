"""
ToolBroker — compatibility dispatcher for non-browser agent tools.

Every tool an agent can use is registered here by name. The model/planner
never gets raw shell, raw filesystem, or raw browser access — it asks the
broker, the broker enforces policy, logs the call, and returns a structured
ToolResult.

Browser execution is canonicalized through TaskEngine's OpenSandbox tools.
The former browser, extension, DevTools, Playwright, mock, Electron, and
Browserbase paths are deprecated and are never instantiated here.

Initial tool set:
  browser.opensandbox.start
  browser.opensandbox.observe
  browser.opensandbox.screenshot
  browser.opensandbox.navigate
  browser.opensandbox.action
  browser.opensandbox.stop
  terminal.run_mock    — policy-checked, logged, NEVER-executed command
  terminal.run         — policy-gated command (mock/safe per configuration)
  project.search       — substring search across project files (root-jailed)
  project.read_file    — read one file inside the project root

Design invariants:
  - Async interface (browser tools await the async Playwright chain).
  - Unknown tool → error result, never an exception into the agent loop.
  - Every call is logged: tool, args summary, ok/error, latency.
  - Path-taking tools are jailed to the project root.
"""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from .terminal_tool import TerminalTool, get_project_root

logger = logging.getLogger(__name__)

_MAX_FILE_BYTES = 200_000
_MAX_SEARCH_RESULTS = 50
_SEARCHABLE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".md", ".txt", ".json", ".toml",
    ".yaml", ".yml", ".css", ".html", ".sh", ".cfg", ".ini",
}

_OPENSANDBOX_BROWSER_TOOLS = (
    "browser.opensandbox.start",
    "browser.opensandbox.observe",
    "browser.opensandbox.screenshot",
    "browser.opensandbox.navigate",
    "browser.opensandbox.action",
    "browser.opensandbox.stop",
)
_DEPRECATED_BROWSER_TOOL_PREFIXES = (
    "browser.extension.",
    "browser.devtools.",
)
_DEPRECATED_BROWSER_TOOLS = frozenset({
    "browser.navigate",
    "browser.read_page",
    "browser.click",
    "browser.screenshot",
})


@dataclass
class ToolResult:
    tool: str
    ok: bool
    output: Any = None
    error: Optional[str] = None
    latency_ms: float = 0.0
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "tool": self.tool,
            "ok": self.ok,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "meta": self.meta,
        }


def _jail_path(rel_path: str, root: str) -> Optional[str]:
    """Resolve rel_path inside root; None when it escapes."""
    resolved = os.path.realpath(os.path.join(root, os.path.expanduser(rel_path)))
    real_root = os.path.realpath(root)
    if resolved == real_root or resolved.startswith(real_root + os.sep):
        return resolved
    return None


class ToolBroker:
    """
    Central registry + dispatcher for agent tools.

    Usage::

        broker = ToolBroker()
        result = await broker.call("browser.read_page", {"url": "https://..."},
                                   session=browser_session)
    """

    def __init__(
        self,
        project_root: Optional[str] = None,
        terminal_tool: Optional[TerminalTool] = None,
    ) -> None:
        self.project_root = os.path.abspath(project_root or get_project_root())
        self.terminal = terminal_tool or TerminalTool(project_root=self.project_root)
        self._tools: dict[str, Callable[..., Awaitable[ToolResult]]] = {}
        self._register_defaults()

    # ── public API ─────────────────────────────────────────────────────────────

    def available_tools(self) -> list[str]:
        return sorted(self._tools)

    async def call(self, tool: str, args: Optional[dict] = None, **context) -> ToolResult:
        """Dispatch a tool call. Never raises — errors come back as results."""
        args = args or {}
        start = time.monotonic()
        if (
            tool in _DEPRECATED_BROWSER_TOOLS
            or tool.startswith(_DEPRECATED_BROWSER_TOOL_PREFIXES)
        ):
            result = ToolResult(
                tool=tool,
                ok=False,
                error=(
                    f"{tool} is deprecated. Use the canonical browser.opensandbox.* "
                    "tools through TaskEngine."
                ),
                meta={"deprecated": True, "replacement": "browser.opensandbox.*"},
            )
            result.latency_ms = (time.monotonic() - start) * 1000
            return result
        handler = self._tools.get(tool)
        if handler is None:
            result = ToolResult(tool=tool, ok=False, error=f"Unknown tool: {tool}")
        else:
            try:
                result = await handler(args, **context)
            except Exception as exc:
                result = ToolResult(tool=tool, ok=False, error=f"{type(exc).__name__}: {exc}")
        result.latency_ms = (time.monotonic() - start) * 1000
        logger.info(
            "[tool_broker] tool=%s ok=%s latency=%.0fms error=%s",
            tool, result.ok, result.latency_ms, result.error,
        )
        return result

    # ── registration ───────────────────────────────────────────────────────────

    def register(self, name: str, handler: Callable[..., Awaitable[ToolResult]]) -> None:
        self._tools[name] = handler

    def _register_defaults(self) -> None:
        for tool in _OPENSANDBOX_BROWSER_TOOLS:
            self.register(tool, self._browser_handler(tool))
        self.register("terminal.run_mock", self._terminal_run_mock)
        self.register("terminal.run", self._terminal_run)
        self.register("project.search", self._project_search)
        self.register("project.read_file", self._project_read_file)
        self.register("web.search", self._web_search)

    # ── browser tools (TaskEngine -> OpenSandbox only) ────────────────────────

    def _browser_handler(self, tool: str) -> Callable[..., Awaitable[ToolResult]]:
        async def handler(args: dict, **context) -> ToolResult:
            return await self._browser_opensandbox(tool, args, **context)

        return handler

    async def _browser_opensandbox(self, tool: str, args: dict, **context) -> ToolResult:
        """Delegate browser work to the canonical TaskEngine boundary.

        A task id is mandatory so the broker cannot bypass task ownership,
        domain allowlists, approvals, redaction, or durable audit events.
        """
        task = context.get("task")
        task_id = context.get("task_id") or getattr(task, "task_id", None)
        if not task_id:
            return ToolResult(
                tool=tool,
                ok=False,
                error="OpenSandbox browser tools require a canonical TaskEngine task_id.",
            )
        from .task_engine import ToolCall, get_task_engine

        call = ToolCall(
            tool,
            dict(args or {}),
            approved=bool(context.get("approved", False)),
        )
        engine = context.get("engine") or get_task_engine()
        result = await engine.execute_async(str(task_id), call)
        return ToolResult(
            tool=tool,
            ok=result.ok,
            output=result.output,
            error=result.error,
            meta={
                "canonical": True,
                "backend": "opensandbox",
                "approval_required": result.approval_required,
            },
        )

    # ── terminal tools ─────────────────────────────────────────────────────────

    async def _terminal_run_mock(self, args: dict, **_) -> ToolResult:
        """Stage 1: policy-check + log the command; NEVER execute."""
        command = args.get("command", "")
        mock_tool = TerminalTool(mode="mock", project_root=self.project_root)
        result = mock_tool.run(command)
        return ToolResult(tool="terminal.run_mock", ok=result.verdict != "blocked",
                          output=result.to_dict(),
                          error=result.reason if result.verdict == "blocked" else None)

    async def _terminal_run(self, args: dict, **_) -> ToolResult:
        """Policy-gated run in the configured mode (mock by default)."""
        command = args.get("command", "")
        approved = bool(args.get("approved", False))
        result = self.terminal.run(command, approved=approved)
        return ToolResult(tool="terminal.run", ok=result.verdict != "blocked",
                          output=result.to_dict(),
                          error=result.reason if result.verdict == "blocked" else None)

    # ── project tools (root-jailed, read-only) ─────────────────────────────────

    async def _project_search(self, args: dict, **_) -> ToolResult:
        query = args.get("query", "")
        if not query:
            return ToolResult(tool="project.search", ok=False, error="query is required")
        matches: list[dict] = []
        skip_dirs = {".git", "node_modules", "dist", "__pycache__", ".venv", "venv"}
        for dirpath, dirnames, filenames in os.walk(self.project_root):
            dirnames[:] = [d for d in dirnames if d not in skip_dirs]
            for fn in filenames:
                if os.path.splitext(fn)[1] not in _SEARCHABLE_EXTS:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    with open(path, "r", errors="ignore") as fh:
                        for lineno, line in enumerate(fh, 1):
                            if query in line:
                                matches.append({
                                    "file": os.path.relpath(path, self.project_root),
                                    "line": lineno,
                                    "text": line.strip()[:200],
                                })
                                if len(matches) >= _MAX_SEARCH_RESULTS:
                                    return ToolResult(tool="project.search", ok=True,
                                                      output={"matches": matches, "truncated": True})
                except OSError:
                    continue
        return ToolResult(tool="project.search", ok=True,
                          output={"matches": matches, "truncated": False})

    async def _project_read_file(self, args: dict, **_) -> ToolResult:
        rel = args.get("path", "")
        if not rel:
            return ToolResult(tool="project.read_file", ok=False, error="path is required")
        resolved = _jail_path(rel, self.project_root)
        if resolved is None:
            return ToolResult(tool="project.read_file", ok=False,
                              error=f"path '{rel}' is outside the project root")
        if not os.path.isfile(resolved):
            return ToolResult(tool="project.read_file", ok=False, error=f"not a file: {rel}")
        with open(resolved, "r", errors="ignore") as fh:
            content = fh.read(_MAX_FILE_BYTES)
        return ToolResult(tool="project.read_file", ok=True,
                          output={"path": rel, "content": content})

    # ── web search (canonical provider stack) ─────────────────────────────────

    async def _web_search(self, args: dict, **_) -> ToolResult:
        """
        Search through the same API-first provider stack used by chat research.

        Args:
            query        (str, required) — search query
            num_results  (int, default 5) — max results to return (cap: 8)

        Returns ToolResult with output:
            {
              "query": str,
              "results": [{"title", "url", "snippet", "source"}, ...],
              "provider": str,
            }

        A configured SearXNG or Brave failure gets one bounded keyless fallback.
        Errors surface as ToolResult(ok=False); no browser is opened implicitly.
        """
        query = args.get("query", "").strip()
        if not query:
            return ToolResult(tool="web.search", ok=False, error="query is required")

        from theriad.search.service import SearchService

        limit = max(1, min(int(args.get("num_results", 5)), 8))
        try:
            rows = (await SearchService().search(query))[:limit]
        except Exception as exc:  # noqa: BLE001
            return ToolResult(tool="web.search", ok=False,
                              error=f"web.search unavailable: {exc}")

        return ToolResult(
            tool="web.search",
            ok=True,
            output={
                "query": query,
                "results": [row.to_dict() for row in rows],
                "provider": rows[0].source if rows else None,
                "number_of_results": len(rows),
            },
            meta={"num_results": len(rows)},
        )
