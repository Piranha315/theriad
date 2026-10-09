"""
Terminal tool — staged, policy-gated command execution for chat agents.

Stages (Part D of chat-integrated agents):
  Stage 1 (default): mode="mock" — commands are policy-checked and LOGGED,
                     never executed.
  Stage 2: mode="safe" — read-only allowlisted commands execute inside the
           configured project root with a timeout; stdout/stderr captured
           and redacted.
  Stage 3: write-class commands (installs, git mutations, file writes) are
           classified as needs_approval in every mode. Approved commands
           execute only in "safe" mode via the same runner.

Hard blocks (never executed, never approvable):
  sudo, rm -rf /, reading ~/.ssh, reading env files outside the project,
  curl|bash-style pipes to a shell, chmod -R 777, obvious secret
  exfiltration, and any path that escapes the project root.

Design invariants:
  - The model NEVER gets raw shell access. Every command goes through
    evaluate_command() and, when executed, runs WITHOUT a shell
    (shlex-split argv) so pipes/redirection/substitution are impossible
    in safe mode — commands containing shell metacharacters are never
    auto-executed.
  - Deterministic policy: pure string/regex rules, no model calls.
  - Every decision is logged with the reason.
"""
from __future__ import annotations

import logging
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass, field
from typing import Literal, Optional

logger = logging.getLogger(__name__)

# ── Configuration ─────────────────────────────────────────────────────────────

_ENV_TERMINAL_MODE = "THERIAD_TERMINAL_MODE"      # "mock" (default) | "safe"
_ENV_PROJECT_ROOT = "THERIAD_PROJECT_ROOT"        # default: cwd

DEFAULT_TIMEOUT_SECONDS = 30

TerminalMode = Literal["mock", "safe"]
Verdict = Literal["allowed", "needs_approval", "blocked"]


def get_terminal_mode() -> TerminalMode:
    raw = os.environ.get(_ENV_TERMINAL_MODE, "mock").strip().lower()
    return raw if raw in ("mock", "safe") else "mock"


def get_project_root() -> str:
    return os.path.abspath(os.environ.get(_ENV_PROJECT_ROOT, "").strip() or os.getcwd())


# ── Policy tables ─────────────────────────────────────────────────────────────

# Hard blocks: (compiled regex, reason). Checked against the raw command
# string first — these are never executed and never approvable.
_HARD_BLOCK_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"(^|\s)sudo(\s|$)"), "sudo is not permitted"),
    (re.compile(r"rm\s+(-[a-z]*[rf][a-z]*\s+)+(/|~)(\s|$)"), "recursive delete of / or ~ is not permitted"),
    (re.compile(r"rm\s+-rf\s+/"), "rm -rf / is not permitted"),
    (re.compile(r"(~|\$HOME|/home/[^/\s]+|/Users/[^/\s]+)/\.ssh"), "access to ~/.ssh is not permitted"),
    (re.compile(r"chmod\s+(-[a-zA-Z]*R[a-zA-Z]*\s+)?777"), "chmod 777 is not permitted"),
    (re.compile(r"(curl|wget)[^|;&]*\|\s*(ba|z|da|)sh"), "piping downloads into a shell is not permitted"),
    (re.compile(r"\b(printenv|env)\b[^|;&]*\|\s*(curl|wget|nc|ncat)"), "exfiltrating environment variables is not permitted"),
    (re.compile(r"\b(curl|wget|nc|ncat|scp|rsync)\b.*(\.env|id_rsa|\.pem|secrets?)"), "uploading secrets/credentials is not permitted"),
    (re.compile(r"(^|[\s=/])(/etc/passwd|/etc/shadow)"), "reading system credential files is not permitted"),
    (re.compile(r"mkfs|dd\s+if="), "raw disk operations are not permitted"),
]

# Safe (read-only) allowlist: argv-prefix patterns. A command is "allowed"
# only if its shlex-split argv starts with one of these prefixes AND contains
# no shell metacharacters AND every path argument stays inside project root.
_SAFE_PREFIXES: list[tuple[str, ...]] = [
    ("pwd",),
    ("ls",),
    ("find",),
    ("grep",),
    ("rg",),
    ("cat",),
    ("head",),
    ("tail",),
    ("wc",),
    ("git", "status"),
    ("git", "log"),
    ("git", "branch"),
    ("git", "diff", "--stat"),
    ("python", "-m", "pytest"),
    ("python3", "-m", "pytest"),
    ("pytest",),
    ("npm", "run", "build"),
    ("npm", "test"),
]

# Write-class commands: legitimate but state-changing → always needs_approval
# (Part E hard-stop list: destructive file changes, dependency installs,
# git push, etc.). Matched on argv prefix.
_APPROVAL_PREFIXES: list[tuple[str, ...]] = [
    ("apply_patch",),
    ("npm", "install"),
    ("npm", "i"),
    ("pip", "install"),
    ("pip3", "install"),
    ("python", "-m", "pip", "install"),
    ("git", "checkout"),
    ("git", "commit"),
    ("git", "push"),
    ("git", "add"),
    ("rm",),
    ("mv",),
    ("cp",),
    ("mkdir",),
    ("touch",),
    ("tee",),
]

# Shell metacharacters: their presence means the command cannot be safely
# executed as a plain argv — treat as needs_approval at best.
_SHELL_META_RE = re.compile(r"[|;&<>`$(){}]")

# Env-file reads: allowed only for files inside the project root.
_ENV_FILE_RE = re.compile(r"(^|[\s=/])((/|~)[^\s]*\.env[^\s]*|~/[^\s]*\.env[^\s]*)")

# ── Redaction ─────────────────────────────────────────────────────────────────

_REDACTION_RULES: list[re.Pattern] = [
    re.compile(r"\b(sk|pk)-[A-Za-z0-9_-]{12,}\b"),                      # API keys (sk-...)
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),                                # AWS access key
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),                      # GitHub tokens
    re.compile(r"\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),  # JWTs
    re.compile(r"(?i)\b(api[_-]?key|token|password|passwd|secret|authorization)\s*[=:]\s*\S+"),
]


def redact_secrets(text: str) -> str:
    """Replace obvious secrets in command output with [REDACTED]."""
    if not text:
        return text
    for rule in _REDACTION_RULES:
        text = rule.sub("[REDACTED]", text)
    return text


# ── Policy evaluation ─────────────────────────────────────────────────────────

@dataclass
class CommandDecision:
    verdict: Verdict
    reason: str
    argv: list[str] = field(default_factory=list)


def _path_escapes_root(token: str, project_root: str) -> bool:
    """True when a path-looking token resolves outside the project root."""
    if token.startswith("-"):
        return False
    looks_like_path = token.startswith(("/", "~", "./", "../")) or "/../" in token
    if not looks_like_path:
        return False
    expanded = os.path.expanduser(token)
    resolved = os.path.realpath(
        expanded if os.path.isabs(expanded) else os.path.join(project_root, expanded)
    )
    root = os.path.realpath(project_root)
    return not (resolved == root or resolved.startswith(root + os.sep))


def evaluate_command(command: str, *, project_root: Optional[str] = None) -> CommandDecision:
    """
    Deterministically classify a command:
      blocked        — hard-blocked, never executed, never approvable
      needs_approval — write-class / ambiguous; requires explicit user approval
      allowed        — read-only allowlisted command confined to project root
    """
    root = os.path.abspath(project_root or get_project_root())
    cmd = (command or "").strip()
    if not cmd:
        return CommandDecision("blocked", "empty command")

    # 1. Hard blocks on the raw string
    for rule, reason in _HARD_BLOCK_RULES:
        if rule.search(cmd):
            return CommandDecision("blocked", f"Hard-blocked: {reason}.")

    # 2. Env-file reads outside the project root are blocked
    m = _ENV_FILE_RE.search(cmd)
    if m and _path_escapes_root(m.group(2), root):
        return CommandDecision("blocked", "Hard-blocked: reading env files outside the project root is not permitted.")

    # 3. Parse argv; unparseable commands are never auto-executed
    try:
        argv = shlex.split(cmd)
    except ValueError as exc:
        return CommandDecision("needs_approval", f"Unparseable command ({exc}); requires approval.")
    if not argv:
        return CommandDecision("blocked", "empty command")

    # 4. Paths must stay inside the project root (applies to every class)
    for token in argv[1:]:
        if _path_escapes_root(token, root):
            return CommandDecision(
                "blocked",
                f"Hard-blocked: path '{token}' is outside the project root.",
                argv,
            )

    # 5. Write-class commands → approval
    for prefix in _APPROVAL_PREFIXES:
        if tuple(argv[: len(prefix)]) == prefix:
            return CommandDecision(
                "needs_approval",
                f"'{' '.join(prefix)}' changes state; user approval required.",
                argv,
            )

    # 6. Shell metacharacters cannot be executed as plain argv
    if _SHELL_META_RE.search(cmd):
        return CommandDecision(
            "needs_approval",
            "Command uses shell operators (pipes/redirection); requires approval and is never auto-executed.",
            argv,
        )

    # 7. Safe read-only allowlist
    for prefix in _SAFE_PREFIXES:
        if tuple(argv[: len(prefix)]) == prefix:
            return CommandDecision("allowed", f"Read-only allowlisted command: {' '.join(prefix)}.", argv)

    # 8. Unknown commands: not blocked, but never auto-executed
    return CommandDecision(
        "needs_approval",
        f"'{argv[0]}' is not on the read-only allowlist; requires approval.",
        argv,
    )


# ── Execution ─────────────────────────────────────────────────────────────────

@dataclass
class TerminalResult:
    command: str
    verdict: Verdict
    reason: str
    executed: bool
    mode: TerminalMode
    exit_code: Optional[int] = None
    stdout: str = ""
    stderr: str = ""
    duration_ms: float = 0.0
    timed_out: bool = False

    def to_dict(self) -> dict:
        return {
            "command": self.command,
            "verdict": self.verdict,
            "reason": self.reason,
            "executed": self.executed,
            "mode": self.mode,
            "exit_code": self.exit_code,
            "stdout": self.stdout,
            "stderr": self.stderr,
            "duration_ms": round(self.duration_ms, 1),
            "timed_out": self.timed_out,
        }


class TerminalTool:
    """
    Policy-gated terminal runner.

    mode="mock" (default): never executes; returns the policy decision and
    logs the proposed command.
    mode="safe": executes ONLY verdict=allowed commands, argv-exec (no
    shell), cwd=project_root, with a timeout and output redaction.
    needs_approval commands execute only when approved=True is passed AND
    mode="safe" — approval is granted by the user through the API layer,
    never by the model.
    """

    def __init__(
        self,
        mode: Optional[TerminalMode] = None,
        project_root: Optional[str] = None,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        self.mode: TerminalMode = mode or get_terminal_mode()
        self.project_root: str = os.path.abspath(project_root or get_project_root())
        self.timeout_seconds = timeout_seconds

    def run(self, command: str, *, approved: bool = False) -> TerminalResult:
        decision = evaluate_command(command, project_root=self.project_root)
        logger.info(
            "[terminal_tool] mode=%s verdict=%s cmd=%r reason=%s",
            self.mode, decision.verdict, command, decision.reason,
        )

        if decision.verdict == "blocked":
            return TerminalResult(
                command=command, verdict="blocked", reason=decision.reason,
                executed=False, mode=self.mode,
            )

        if self.mode == "mock":
            # Stage 1: log the proposal, never execute.
            return TerminalResult(
                command=command, verdict=decision.verdict,
                reason=f"[mock mode] proposed only, not executed. {decision.reason}",
                executed=False, mode=self.mode,
            )

        if decision.verdict == "needs_approval" and not approved:
            return TerminalResult(
                command=command, verdict="needs_approval", reason=decision.reason,
                executed=False, mode=self.mode,
            )

        # Approved write commands still can't be argv-executed if they rely
        # on shell operators — refuse rather than silently shell out.
        if _SHELL_META_RE.search(command):
            return TerminalResult(
                command=command, verdict="needs_approval",
                reason="Commands with shell operators are never executed by the agent, even after approval.",
                executed=False, mode=self.mode,
            )

        return self._execute(command, decision)

    # -- internals -------------------------------------------------------------

    def _execute(self, command: str, decision: CommandDecision) -> TerminalResult:
        start = time.monotonic()
        try:
            proc = subprocess.run(
                decision.argv,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                shell=False,
            )
            duration = (time.monotonic() - start) * 1000
            return TerminalResult(
                command=command, verdict=decision.verdict, reason=decision.reason,
                executed=True, mode=self.mode, exit_code=proc.returncode,
                stdout=redact_secrets(proc.stdout[-20_000:]),
                stderr=redact_secrets(proc.stderr[-20_000:]),
                duration_ms=duration,
            )
        except subprocess.TimeoutExpired:
            return TerminalResult(
                command=command, verdict=decision.verdict,
                reason=f"timed out after {self.timeout_seconds}s",
                executed=True, mode=self.mode, timed_out=True,
                duration_ms=(time.monotonic() - start) * 1000,
            )
        except FileNotFoundError:
            return TerminalResult(
                command=command, verdict=decision.verdict,
                reason=f"executable not found: {decision.argv[0]}",
                executed=False, mode=self.mode, exit_code=127,
            )
        except Exception as exc:
            return TerminalResult(
                command=command, verdict=decision.verdict,
                reason=f"execution error: {exc}",
                executed=False, mode=self.mode,
            )
