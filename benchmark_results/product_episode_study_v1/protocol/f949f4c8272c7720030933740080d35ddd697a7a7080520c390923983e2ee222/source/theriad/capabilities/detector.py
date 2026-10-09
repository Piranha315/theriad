"""
Capability Detector — rule-based, deterministic.

Inspects a prompt and returns a CapabilityResult that describes:
  - what execution capabilities are required (14 boolean flags)
  - the recommended execution mode
  - risk level
  - whether user confirmation is required before starting

Design principles
-----------------
- No model calls.  No embeddings.  No imports beyond stdlib.
- Conservative defaults: ambiguous prompts fall through to text_only.
- Every flag that raises risk level is documented so the logic can be audited.
- Adding new rules = adding keyword lists; no architectural changes needed.

Execution modes
---------------
text_only          — LLM completion only, no preview, no agents
route_only         — dry-run route analysis, no execution
text_plus_preview  — LLM completion + inline HTML/CSS artifact preview
browser_agent      — requires live web navigation via browser runtime
local_tool_agent   — requires local shell/file operations
workflow_agent     — multi-step workflow with tool use
hybrid             — combination of the above

Risk levels
-----------
low    — reversible, no account access, no sensitive data
medium — file edits, code execution, data extraction, form interaction
high   — account access, sensitive info, financial/legal/medical portals,
         paid API calls, explicit destructive operations
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Literal

# ── Types ─────────────────────────────────────────────────────────────────────

ExecutionMode = Literal[
    "text_only",
    "route_only",
    "text_plus_preview",
    "browser_agent",
    "local_tool_agent",
    "workflow_agent",
    "hybrid",
]

RiskLevel = Literal["low", "medium", "high"]


@dataclass
class CapabilityResult:
    """Output of CapabilityDetector.detect()."""

    recommended_execution_mode: ExecutionMode
    confidence: float
    capabilities: dict[str, bool]
    reason: str
    risk_level: RiskLevel
    requires_confirmation_before_start: bool
    # Structured list of matched signals, useful for diagnostics / tests
    matched_signals: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "recommended_execution_mode": self.recommended_execution_mode,
            "confidence": round(self.confidence, 3),
            "capabilities": self.capabilities,
            "reason": self.reason,
            "risk_level": self.risk_level,
            "requires_confirmation_before_start": self.requires_confirmation_before_start,
            "matched_signals": self.matched_signals,
        }


# ── Keyword tables ────────────────────────────────────────────────────────────
#
# Each table maps a capability flag to a list of keyword/phrase fragments.
# Matching is case-insensitive substring search on the lowercased prompt.
# Order within a list does not matter.

_CURRENT_WEB_KEYWORDS: list[str] = [
    "current", "today", "right now", "latest", "in stock", "near me",
    "nearby", "availability", "available today", "open now", "prices",
    "find online", "book", "reserve", "on sale", "currently",
    "this week", "this month", "news", "trending", "live",
    # Inherently point-in-time subjects. Without these a bare "what is the
    # weather in Denver?" carries no generic recency word, so the detector
    # reported text_only and the question was answered from model training
    # data instead of the web. theriad/search/service.py already treats these
    # as search triggers; the two lists must agree or the search path is
    # never reached.
    "weather", "forecast", "temperature",
]

_BROWSER_NAV_KEYWORDS: list[str] = [
    "go to", "open website", "open the website", "navigate to",
    "find on", "search for", "search online", "look up online",
    "compare listings", "check availability", "browse",
    "click", "scroll", "open page", "visit", "load the page",
    "find me", "look for near",
]

_VISUAL_OBSERVATION_KEYWORDS: list[str] = [
    "screenshot", "what does it look like", "describe the page",
    "visual appearance", "appearance", "layout of the page",
    "what is on the screen",
]

_DOM_OBSERVATION_KEYWORDS: list[str] = [
    "extract from the page", "scrape", "read the page",
    "get the html", "inspect the dom", "parse the page",
    "what is in the table", "pull data from",
]

_FILE_EDITING_KEYWORDS: list[str] = [
    "edit file", "modify file", "fix the codebase", "change implementation",
    "commit", "write to file", "update the code", "refactor",
    "patch", "apply the fix", "create a file", "delete the file",
    "rename", "move file", "add to repo", "modify repo",
    # "fix" in agent/tool contexts almost always means editing a file
    "fix any", "fix the bug", "fix errors", "fix failures", "fix all",
    "fix it", "fix them", "fix that", "fix this",
]

_LOCAL_COMMAND_KEYWORDS: list[str] = [
    "run tests", "run the tests", "execute", "install", "start server",
    "start the server", "build project", "build the project",
    "lint", "pytest", "npm run", "npm install", "pip install",
    "make ", "cargo ", "go build", "docker ", "kubectl ",
    "shell command", "terminal", "bash", "compile",
]

_CODE_EXECUTION_KEYWORDS: list[str] = [
    "run this code", "execute the script", "run the script",
    "run python", "run node", "run the program", "evaluate",
    "run this snippet", "test this function",
    # running tests is a form of code execution
    "run tests", "run the tests",
]

# These are intentionally complete, action-oriented phrases.  A bare mention
# of "CSS", "Tailwind", or "frontend" is often an explanatory question and
# must not turn an ordinary answer into a visual artifact.
_ARTIFACT_PREVIEW_KEYWORDS: list[str] = [
    "build a website", "make a website", "create a website", "design a website",
    "create html", "single-file html", "html page", "css page",
    "make a page", "design a page", "make a ui", "build a ui",
    "create a component", "html component",
]

# Natural-language visual requests are less formulaic than the exact phrases
# above.  Detect a construction action together with a renderable target, then
# ask the provider for a self-contained HTML artifact that the UI can actually
# render.  The separate format matcher catches requests such as "write CSS for
# a pricing card" without classifying questions such as "explain CSS".
_RENDERABLE_ACTION_RE = re.compile(
    r"\b(?:build|make|create|design|generate|prototype|mock(?:\s|-)?up|"
    r"implement|craft|render|write|code|style|turn|convert)\b"
)
# Declarative requests are common in chat (for example, "I need a landing
# page"). Keep this separate from the construction verbs above so "I need
# help understanding CSS" remains a normal text request.
_RENDERABLE_REQUEST_RE = re.compile(
    r"\b(?:i|we)\s+(?:want|need|would\s+like)\b|"
    r"\b(?:can|could)\s+you\s+(?:show|give|make|build|create|design|generate)\b|"
    r"\b(?:please\s+)?(?:show|give)\s+me\b"
)
_RENDERABLE_TARGET_RE = re.compile(
    r"\b(?:website|web\s*site|web\s*page|landing\s+page|home\s+page|"
    r"web\s+app(?:lication)?|(?:user\s+)?interface|ui|dashboard|"
    r"wireframe|mockup|prototype|component|card|button|form|navbar|"
    r"navigation|hero(?:\s+(?:section|banner))?|footer|svg|diagram|chart|"
    r"graph|infographic|poster|logo)\b"
)
_RENDERABLE_FORMAT_RE = re.compile(r"\b(?:html|css|svg|tailwind|canvas)\b")

_FORM_INTERACTION_KEYWORDS: list[str] = [
    "submit", "fill in", "fill out", "complete the form",
    "checkout", "purchase", "buy ", "add to cart", "place order",
    "subscribe", "sign up", "create account",
]

_ACCOUNT_ACCESS_KEYWORDS: list[str] = [
    "login", "log in", "log into", "sign in", "sign into",
    "my account", "my dashboard", "account dashboard", "portal", "authenticate",
    "access my", "open my",
]

_SENSITIVE_INFO_KEYWORDS: list[str] = [
    "bank", "banking", "financial", "tax", "taxes", "tax return",
    "medical", "health record", "statement", "password",
    "credit card", "ssn", "social security", "legal",
    "attorney", "lawyer", "court", "insurance", "paycheck",
    "salary", "download statement", "account balance",
]

_DATA_EXTRACTION_KEYWORDS: list[str] = [
    "extract", "scrape", "collect data", "gather data",
    "pull data", "download data", "export", "get all",
    "list of ", "compile a list", "fetch all", "aggregate",
]

_USER_APPROVAL_KEYWORDS: list[str] = [
    "send message", "send email", "send an email", "email someone",
    "message someone", "post to", "tweet", "upload", "delete",
    "purchase", "checkout", "buy ", "subscribe", "submit form",
    "grant permission", "authorize", "confirm", "approve",
]

_PAID_CALL_KEYWORDS: list[str] = [
    "use gpt-4", "use claude", "use gemini", "use openai",
    "premium model", "best model", "most capable",
]

# URL pattern — presence of a URL in the prompt is a weak signal for browser navigation
_URL_RE = re.compile(r'https?://[^\s<>"\']+', re.IGNORECASE)


# ── Detector ──────────────────────────────────────────────────────────────────

class CapabilityDetector:
    """
    Rule-based capability detector.

    Usage::

        result = CapabilityDetector().detect(prompt)
        print(result.recommended_execution_mode)  # "browser_agent"
    """

    # ---------- public API ---------------------------------------------------

    def detect(self, prompt: str) -> CapabilityResult:
        """
        Analyse *prompt* and return a CapabilityResult.

        Guaranteed to never raise; errors are swallowed and the result
        defaults to text_only.
        """
        try:
            return self._detect(prompt)
        except Exception as exc:  # pragma: no cover — safety net only
            return CapabilityResult(
                recommended_execution_mode="text_only",
                confidence=0.0,
                capabilities=self._empty_flags(),
                reason=f"Capability detection error (defaulting to text_only): {exc}",
                risk_level="low",
                requires_confirmation_before_start=False,
            )

    # ---------- internals ----------------------------------------------------

    def _detect(self, prompt: str) -> CapabilityResult:
        low = prompt.lower()
        signals: list[str] = []
        flags = self._empty_flags()

        # ── Flag detection ────────────────────────────────────────────────────

        flags["requires_current_web"] = self._match(
            low, _CURRENT_WEB_KEYWORDS, signals, "current_web"
        )
        flags["requires_browser_navigation"] = self._match(
            low, _BROWSER_NAV_KEYWORDS, signals, "browser_navigation"
        ) or (bool(_URL_RE.search(prompt)) and self._prompt_wants_interaction(low))
        flags["requires_dom_observation"] = self._match(
            low, _DOM_OBSERVATION_KEYWORDS, signals, "dom_observation"
        )
        flags["requires_file_editing"] = self._match(
            low, _FILE_EDITING_KEYWORDS, signals, "file_editing"
        )
        flags["requires_local_command"] = self._match(
            low, _LOCAL_COMMAND_KEYWORDS, signals, "local_command"
        )
        flags["requires_code_execution"] = self._match(
            low, _CODE_EXECUTION_KEYWORDS, signals, "code_execution"
        )
        flags["requires_artifact_preview"] = self._matches_artifact_preview_request(
            low, signals
        )
        # A visual artifact is rendered locally; it is not a request to inspect
        # an existing browser page. Evaluate it first so terms such as
        # "visual dashboard" do not accidentally start a browser route.
        flags["requires_visual_observation"] = (
            not flags["requires_artifact_preview"]
            and self._match(low, _VISUAL_OBSERVATION_KEYWORDS, signals, "visual_observation")
        )
        flags["requires_form_interaction"] = self._match(
            low, _FORM_INTERACTION_KEYWORDS, signals, "form_interaction"
        )
        flags["requires_account_access"] = self._match(
            low, _ACCOUNT_ACCESS_KEYWORDS, signals, "account_access"
        )
        flags["requires_sensitive_info"] = self._match(
            low, _SENSITIVE_INFO_KEYWORDS, signals, "sensitive_info"
        )
        flags["requires_data_extraction"] = self._match(
            low, _DATA_EXTRACTION_KEYWORDS, signals, "data_extraction"
        )
        flags["requires_user_approval"] = self._match(
            low, _USER_APPROVAL_KEYWORDS, signals, "user_approval"
        ) or flags["requires_account_access"] or flags["requires_form_interaction"]
        flags["requires_paid_call"] = self._match(
            low, _PAID_CALL_KEYWORDS, signals, "paid_call"
        )

        # ── Implied rules ────────────────────────────────────────────────────
        # Account access and live-web tasks always imply browser navigation
        # (you can't log in or fetch live data without navigating).
        if (flags["requires_account_access"] or flags["requires_current_web"]) \
                and not flags["requires_browser_navigation"]:
            flags["requires_browser_navigation"] = True
            signals.append("browser_navigation:implied")

        # Live web data tasks always imply data extraction
        # (fetching current availability / prices / results = extracting web data).
        if flags["requires_current_web"] and not flags["requires_data_extraction"]:
            flags["requires_data_extraction"] = True
            signals.append("data_extraction:implied")

        # ── URL-only prompts (read without interaction → text_only) ───────────
        has_url = bool(_URL_RE.search(prompt))
        if has_url and not self._prompt_wants_interaction(low):
            # "Summarise this URL" — treat as text_only (the model can read URLs)
            if not any([
                flags["requires_current_web"],
                flags["requires_browser_navigation"],
                flags["requires_data_extraction"],
                flags["requires_dom_observation"],
            ]):
                signals.append("url_read_only")

        # ── Mode resolution ───────────────────────────────────────────────────
        mode, confidence, reason = self._resolve_mode(flags, low, has_url, signals)

        # ── Risk level ────────────────────────────────────────────────────────
        risk = self._resolve_risk(flags)

        # ── Confirmation gate ─────────────────────────────────────────────────
        needs_confirmation = self._needs_confirmation(mode, risk, flags)

        return CapabilityResult(
            recommended_execution_mode=mode,
            confidence=confidence,
            capabilities=flags,
            reason=reason,
            risk_level=risk,
            requires_confirmation_before_start=needs_confirmation,
            matched_signals=signals,
        )

    # ---------- helpers -------------------------------------------------------

    @staticmethod
    def _empty_flags() -> dict[str, bool]:
        return {
            "requires_current_web":        False,
            "requires_browser_navigation": False,
            "requires_visual_observation": False,
            "requires_dom_observation":    False,
            "requires_file_editing":       False,
            "requires_local_command":      False,
            "requires_code_execution":     False,
            "requires_artifact_preview":   False,
            "requires_form_interaction":   False,
            "requires_account_access":     False,
            "requires_sensitive_info":     False,
            "requires_data_extraction":    False,
            "requires_user_approval":      False,
            "requires_paid_call":          False,
        }

    @staticmethod
    def _match(
        low: str, keywords: list[str], signals: list[str], label: str
    ) -> bool:
        for kw in keywords:
            if kw in low:
                signals.append(f"{label}:{kw.strip()}")
                return True
        return False

    @staticmethod
    def _matches_artifact_preview_request(low: str, signals: list[str]) -> bool:
        """Return True only when the user is asking to *make* a visual artifact.

        The preview renderer consumes complete HTML documents.  The detector
        therefore needs to distinguish a construction request ("design a
        website", "write CSS for a card") from a normal conceptual question
        that merely happens to mention CSS or a web page.
        """
        if CapabilityDetector._match(
            low, _ARTIFACT_PREVIEW_KEYWORDS, signals, "artifact_preview"
        ):
            return True

        has_construction_action = bool(_RENDERABLE_ACTION_RE.search(low))
        has_render_request = bool(_RENDERABLE_REQUEST_RE.search(low))

        if _RENDERABLE_TARGET_RE.search(low) and (
            has_construction_action or has_render_request
        ):
            signals.append("artifact_preview:renderable_visual_target")
            return True

        if has_construction_action and _RENDERABLE_FORMAT_RE.search(low):
            signals.append("artifact_preview:renderable_format_request")
            return True

        return False

    @staticmethod
    def _prompt_wants_interaction(low: str) -> bool:
        """Return True when the user wants to *do* something on a URL, not just read it."""
        interaction_words = [
            "click", "fill", "submit", "login", "log in", "sign in",
            "interact", "navigate", "browse", "scroll", "extract",
            "scrape", "download", "check", "find", "search",
        ]
        return any(w in low for w in interaction_words)

    def _resolve_mode(
        self,
        flags: dict[str, bool],
        low: str,
        has_url: bool,
        signals: list[str],
    ) -> tuple[ExecutionMode, float, str]:
        """
        Return (mode, confidence, reason) from the flags.

        Priority (highest to lowest):
          1. browser_agent  — any web/browser/account/sensitive flag
          2. local_tool_agent — local command or file editing alone
          3. workflow_agent — multi-step combination
          4. text_plus_preview — artifact preview
          5. text_only — default
        """
        c = flags  # shorthand

        # ── Browser agent path ────────────────────────────────────────────────
        browser_flags = [
            c["requires_current_web"],
            c["requires_browser_navigation"],
            c["requires_visual_observation"],
            c["requires_dom_observation"],
            c["requires_data_extraction"] and (c["requires_current_web"] or c["requires_browser_navigation"]),
            c["requires_form_interaction"],
            c["requires_account_access"],
        ]
        if any(browser_flags):
            confidence = min(0.95, 0.65 + sum(browser_flags) * 0.06)
            parts = []
            if c["requires_current_web"]:         parts.append("current web data")
            if c["requires_browser_navigation"]:  parts.append("browser navigation")
            if c["requires_visual_observation"]:  parts.append("visual observation")
            if c["requires_dom_observation"]:     parts.append("DOM inspection")
            if c["requires_form_interaction"]:    parts.append("form interaction")
            if c["requires_account_access"]:      parts.append("account access")
            if c["requires_data_extraction"]:     parts.append("data extraction")
            reason = f"Task requires {', '.join(parts)}."
            return "browser_agent", confidence, reason

        # ── Local tool / workflow agent path ──────────────────────────────────
        local_flags = [
            c["requires_local_command"],
            c["requires_file_editing"],
            c["requires_code_execution"],
        ]
        local_count = sum(local_flags)

        if local_count >= 2:
            # Multiple local flags → workflow agent (multi-step)
            confidence = min(0.92, 0.70 + local_count * 0.07)
            parts = []
            if c["requires_local_command"]:  parts.append("local command execution")
            if c["requires_file_editing"]:   parts.append("file editing")
            if c["requires_code_execution"]: parts.append("code execution")
            reason = f"Multi-step task requiring {', '.join(parts)}."
            return "workflow_agent", confidence, reason

        if local_count == 1:
            confidence = 0.75
            if c["requires_local_command"]:
                return "local_tool_agent", confidence, "Task requires running a local command."
            if c["requires_file_editing"]:
                return "local_tool_agent", confidence, "Task requires editing files on disk."
            if c["requires_code_execution"]:
                return "local_tool_agent", confidence, "Task requires executing code locally."

        # ── Artifact preview path ─────────────────────────────────────────────
        if c["requires_artifact_preview"]:
            return "text_plus_preview", 0.85, "Task will produce an HTML/CSS artifact for preview."

        # ── Data extraction without browser → still needs browser ─────────────
        if c["requires_data_extraction"] and not c["requires_artifact_preview"]:
            return "browser_agent", 0.70, "Task requires web data extraction."

        # ── Default: text only ────────────────────────────────────────────────
        return "text_only", 0.95, "No special execution capabilities detected."

    @staticmethod
    def _resolve_risk(flags: dict[str, bool]) -> RiskLevel:
        """
        Risk rules (any match escalates):
          high   → account_access OR sensitive_info OR (form_interaction AND account_access)
                   OR paid_call
          medium → file_editing OR local_command OR code_execution OR form_interaction
                   OR data_extraction OR user_approval
          low    → everything else
        """
        c = flags
        if (
            c["requires_account_access"]
            or c["requires_sensitive_info"]
            or c["requires_paid_call"]
        ):
            return "high"
        if (
            c["requires_file_editing"]
            or c["requires_local_command"]
            or c["requires_code_execution"]
            or c["requires_form_interaction"]
            or c["requires_data_extraction"]
            or c["requires_user_approval"]
        ):
            return "medium"
        return "low"

    @staticmethod
    def _needs_confirmation(
        mode: ExecutionMode, risk: RiskLevel, flags: dict[str, bool]
    ) -> bool:
        """
        Confirmation is required when:
        - mode is browser_agent, local_tool_agent, or workflow_agent
        - risk is medium or high
        - any of the high-risk flags is set (always require, regardless of mode)
        """
        if flags["requires_account_access"] or flags["requires_sensitive_info"]:
            return True
        if mode in ("browser_agent", "local_tool_agent", "workflow_agent"):
            return True
        if risk in ("medium", "high"):
            return True
        return False
