from dataclasses import dataclass, field
from pathlib import Path
import re
from typing import Any


class ToolPolicyError(RuntimeError):
    pass


@dataclass
class ToolResult:
    tool: str
    status: str
    output: str = ""
    requires_approval: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)


class ToolBroker:
    SAFE_COMMANDS = {
        "pwd", "ls", "find", "grep", "git status", "git diff --stat",
        "npm run build",
    }
    WRITE_PREFIXES = ("apply_patch", "npm install", "pip install", "git checkout", "git commit", "git push", "rm ", "mv ")
    HARD_BLOCK_PATTERNS = (
        r"\bsudo\b", r"rm\s+-rf\s+/", r"~/.ssh", r"curl\b.*\|\s*bash",
        r"chmod\s+-R\s+777", r"/\.env", r"\bscp\b", r"\brsync\b.*:",
    )

    def __init__(self, project_root: str | Path, terminal_mode: str = "mock", permission_mode: str = "ask_high_risk") -> None:
        self.project_root = Path(project_root).resolve()
        self.terminal_mode = terminal_mode
        self.permission_mode = permission_mode
        self.log: list[dict[str, Any]] = []

    def call(self, tool: str, **kwargs: Any) -> ToolResult:
        if tool == "terminal.run_mock":
            return self.terminal_run_mock(kwargs.get("command", ""), kwargs.get("cwd"))
        if tool == "project.search":
            return ToolResult(tool, "mocked", metadata=kwargs)
        if tool == "project.read_file":
            return ToolResult(tool, "mocked", metadata=kwargs)
        if tool.startswith("browser."):
            return ToolResult(tool, "queued", metadata=kwargs)
        raise ToolPolicyError(f"Unknown tool: {tool}")

    def _ensure_inside_project(self, cwd: str | None) -> Path:
        path = (self.project_root / (cwd or ".")).resolve() if cwd else self.project_root
        if path != self.project_root and self.project_root not in path.parents:
            raise ToolPolicyError("Commands must run inside the configured project root")
        return path

    def classify_command(self, command: str, cwd: str | None = None) -> str:
        self._ensure_inside_project(cwd)
        stripped = command.strip()
        if any(re.search(pattern, stripped) for pattern in self.HARD_BLOCK_PATTERNS):
            return "blocked"
        if any(stripped.startswith(prefix) for prefix in self.WRITE_PREFIXES):
            return "requires_approval"
        if stripped.startswith("python -m pytest"):
            return "safe"
        if stripped in self.SAFE_COMMANDS or any(stripped.startswith(cmd + " ") for cmd in ("ls", "find", "grep")):
            return "safe"
        return "requires_approval"

    def terminal_run_mock(self, command: str, cwd: str | None = None) -> ToolResult:
        classification = self.classify_command(command, cwd)
        entry = {"tool": "terminal.run_mock", "command": command, "cwd": str(self._ensure_inside_project(cwd)), "classification": classification}
        self.log.append(entry)
        if classification == "blocked":
            raise ToolPolicyError("Command is hard-blocked by terminal policy")
        return ToolResult("terminal.run_mock", "mocked", output=f"Proposed command: {command}", requires_approval=classification == "requires_approval", metadata=entry)
