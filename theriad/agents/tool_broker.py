"""
ToolBroker — compatibility dispatcher for non-browser agent tools.

Every tool an agent can use is registered here by name. The model/planner
never gets raw shell, raw filesystem, or raw browser access — it asks the
broker, the broker enforces policy, logs the call, and returns a structured
ToolResult.

Browser execution is canonicalized through TaskEngine's OpenSandbox tools.
OpenSandbox is the preferred browser path. Legacy browser names remain
registered as translation aliases: task-scoped calls are forwarded to the
canonical OpenSandbox tools, while an explicitly requested ``mock_mode`` may
use the old deterministic read-only seam for local compatibility.

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
import uuid
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
_LEGACY_BROWSER_ALIASES = {
    "browser.navigate": "browser.opensandbox.navigate",
    "browser.read_page": "browser.opensandbox.observe",
    "browser.click": "browser.opensandbox.action",
    "browser.screenshot": "browser.opensandbox.screenshot",
    "browser.scroll": "browser.opensandbox.action",
}


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
        if tool.startswith(_DEPRECATED_BROWSER_TOOL_PREFIXES):
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
        if tool in _LEGACY_BROWSER_ALIASES:
            result = await self._legacy_browser_handler(
                tool,
                args,
                canonical_tool=_LEGACY_BROWSER_ALIASES[tool],
                **context,
            )
            result.latency_ms = (time.monotonic() - start) * 1000
            logger.info(
                "[tool_broker] tool=%s alias=%s ok=%s latency=%.0fms",
                tool, _LEGACY_BROWSER_ALIASES[tool], result.ok, result.latency_ms,
            )
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
        # Older planners still advertise browser.* names. Keep them
        # discoverable, but resolve them through the compatibility handler so
        # they cannot bypass the OpenSandbox TaskEngine boundary.
        for tool in _LEGACY_BROWSER_ALIASES:
            self.register(tool, self._browser_handler(tool))
        self.register("terminal.run_mock", self._terminal_run_mock)
        self.register("terminal.run", self._terminal_run)
        self.register("project.search", self._project_search)
        self.register("project.list", self._project_list)
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

    async def _legacy_browser_handler(
        self,
        tool: str,
        args: dict,
        *,
        canonical_tool: str,
        **context,
    ) -> ToolResult:
        """Translate legacy browser calls without creating a second live path."""
        task = context.get("task")
        task_id = context.get("task_id") or getattr(task, "task_id", None)
        if task_id:
            result = await self._browser_opensandbox(
                canonical_tool,
                args,
                **context,
            )
            result.tool = tool
            result.meta.update({
                "deprecated": True,
                "replacement": canonical_tool,
                "canonical": True,
            })
            return result

        # Without a task, only an explicitly requested deterministic mock is
        # allowed. This keeps old in-process integrations useful while never
        # silently presenting synthetic content as live browser evidence.
        if not bool(args.get("mock_mode", False)):
            return ToolResult(
                tool=tool,
                ok=False,
                error=(
                    f"{tool} is a compatibility alias for {canonical_tool} and "
                    "requires a canonical TaskEngine task_id. Set mock_mode=true "
                    "only for local deterministic tests."
                ),
                meta={
                    "deprecated": True,
                    "replacement": canonical_tool,
                },
            )
        return await self._legacy_browser_mock(tool, args, canonical_tool)

    async def _legacy_browser_mock(
        self,
        tool: str,
        args: dict,
        canonical_tool: str,
    ) -> ToolResult:
        from theriad.agents.browser_observation import observe_page
        from theriad.browser.bridge import BrowserBridge

        # ``mock_mode`` is an explicit deterministic-test request.  Preserve
        # it even on a machine where OpenSandbox happens to be healthy; the
        # normal legacy alias path never sets this flag and remains
        # OpenSandbox-first.
        bridge = BrowserBridge(
            backend="mock",
            observer=observe_page,
            force_legacy=True,
        )
        session_id = str(args.get("session_id") or uuid.uuid4())
        requested_url = str(
            args.get("url") or args.get("start_url") or "about:blank"
        )
        started = await bridge.start_session({
            "session_id": session_id,
            "start_url": requested_url,
            "user_prompt": str(args.get("user_prompt") or requested_url),
        })
        if not started.ok:
            return ToolResult(
                tool=tool,
                ok=False,
                error=started.error or "Local compatibility browser could not start.",
                meta={"deprecated": True, "replacement": canonical_tool},
            )

        if tool == "browser.read_page":
            result = await bridge.read_page(session_id)
        elif tool == "browser.screenshot":
            result = await bridge.screenshot(session_id)
        elif tool == "browser.navigate":
            result = await bridge.navigate(session_id, requested_url)
        elif tool == "browser.click":
            target = str(
                args.get("element_id")
                or args.get("selector")
                or args.get("target")
                or ""
            )
            result = await bridge.click(session_id, target)
        else:  # browser.scroll
            result = await bridge.scroll(
                session_id,
                args.get("amount") or args.get("direction") or "down",
            )

        try:
            await bridge.close_session(session_id)
        except Exception:
            pass
        if not result.ok:
            return ToolResult(
                tool=tool,
                ok=False,
                error=result.error or f"{tool} compatibility call failed.",
                meta={"deprecated": True, "replacement": canonical_tool},
            )
        observation = result.observation
        output = {
            "session_id": result.session_id or session_id,
            "url": result.url,
            "title": result.title or "",
            "text": result.text or "",
            "visible_text": result.text or "",
            "dom_summary": result.dom_summary,
            "screenshot_available": result.screenshot_available,
            "backend": result.backend,
            "mock_mode": True,
        }
        if observation is not None:
            output.update({
                "clickable_elements": observation.clickable_elements,
                "forms": observation.forms,
                "aria_labels": observation.aria_labels,
            })
        return ToolResult(
            tool=tool,
            ok=True,
            output=output,
            meta={
                "deprecated": True,
                "replacement": canonical_tool,
                "canonical": False,
                "compatibility_fallback": "mock",
                "actions_mocked": True,
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

    async def _project_list(self, args: dict, **_) -> ToolResult:
        from .project_tools import list_paths
        try:
            result = list_paths(self.project_root, args.get("path", "."),
                                recursive=args.get("recursive", False), limit=args.get("limit", 500))
            return ToolResult(tool="project.list", ok=True, output=result)
        except (ValueError, OSError) as exc:
            return ToolResult(tool="project.list", ok=False, error=str(exc))

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
