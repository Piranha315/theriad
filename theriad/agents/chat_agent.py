"""
Chat-integrated agents (Part A/B of chat-agent architecture).

Main Chat → intent detector → agent runtime → ToolBroker → tools → chat.

This module owns:
  - classify_chat_intent(): keyword guardrail → MRL embedding → default pipeline
      normal_chat | browser_agent | terminal_agent | multi_tool_agent
  - BrowserIntentDetector: mirrors WorkflowDetector's rule → embedding → default
    pipeline but for the browser-vs-chat boundary.
  - ChatAgentRun / ChatAgentRunStore: in-memory registry of chat-initiated
    agent runs, linking a chat message to the underlying browser session
    and/or terminal plan, with a compact UI-facing status payload.
  - propose_terminal_commands(): deterministic prompt → command plan for
    the terminal agent (Stage 1: proposals only).

Detection pipeline (browser intent):
  1. Keyword guardrail — fast substring match; confident hits (≥ 0.80) are final.
  2. MRL embedding — cosine similarity to browser/chat prototype corpora at
     dims [64, 128, 256, full]; first decisive margin (≥ 0.08) wins.
  3. CapabilityDetector fallback — legacy rule-based signal from requires_browser_*.
  4. Default → normal_chat.

The browser runtime itself is NOT duplicated here — browser intents create
a normal BrowserAgentSession (same runtime profile, backends, permission
mode, and budgets as the Agent page) and the Agent page remains the
detailed trace/debug console.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Literal, Optional

import numpy as np

from theriad.capabilities.detector import CapabilityDetector
from theriad.layers.layer2_embedding.backends import (
    EmbeddingBackend,
    SentenceTransformerBackend,
)
from theriad.core.logging.logger import get_logger

_logger = get_logger("agents.chat_intent")

ChatIntent = Literal["normal_chat", "browser_agent", "terminal_agent", "multi_tool_agent"]

# ── Keyword guardrail ─────────────────────────────────────────────────────────
# High-confidence substring cues. Any match → browser_agent at confidence 0.92.
# Keep these specific enough that false-positive rate is near zero.

_BROWSER_CUES: list[str] = [
    "go to", "open website", "open the website", "visit", "browse",
    "click", "webpage", "web page", "summarize this site", "summarise this site",
    "inspect page", "inspect the page", "navigate to", "open site",
    "search the web", "research online", "look up", "find listings",
    "find options", "find products", "find restaurants", "find events",
    "find real estate", "real estate listings",
    # Search categories that are commonly phrased without "for" or "online".
    # These are deliberately domain-qualified so a programming question such
    # as "how do I search a list?" remains ordinary chat.
    "find flights", "search flights", "flight options", "flight prices",
    "find hotels", "search hotels", "hotel options", "find accommodations",
    "find accommodation", "movies are playing", "playing tonight",
    "who won last night",
    # Live article/news research
    "find an article", "find a random article", "search for an article",
    "find a recent article", "find recent articles", "recent article about",
    "search for articles", "find a news article", "latest news",
    "current news", "today's news", "today’s news", "search online",
    "on the web", "from the web",
    # Property / housing search
    "find a house", "find houses", "find a home", "find homes",
    "find an apartment", "find apartments", "find a property", "find properties",
    "find a condo", "find condos",
    # General local / activity search
    "find fun things", "find things to do", "find good things",
    "find a good", "find me a", "find me the",
    "look for a house", "look for a home", "look for an apartment",
    "look for a", "search for a house", "search for a home",
]

# ── MRL embedding classifier ──────────────────────────────────────────────────
# Prototype corpora for cosine-similarity classification.
# "Browser" prototypes: tasks that genuinely need live web data.
# "Chat" prototypes: tasks where an LLM can answer without browsing.

_BROWSER_PROTOTYPES: list[str] = [
    "Find houses for sale near downtown Seattle under $900k.",
    "Can you look up current prices for flights from JFK to LAX?",
    "Search for the best Italian restaurants in my neighborhood.",
    "Find apartments for rent in Brooklyn under $2500 per month.",
    "What are the latest news headlines about artificial intelligence?",
    "Go to airbnb.com and find rentals in Austin for next weekend.",
    "Find fun things to do in Chicago this weekend.",
    "Look up the current weather in Miami.",
    "Search for used Tesla Model 3 listings under $30,000.",
    "Find a good plumber near me with high reviews.",
    "Navigate to the company website and extract their pricing page.",
    "Look up what movies are playing tonight at theaters near me.",
    "Find good hiking trails near Denver with parking.",
    "Search for open job postings for software engineers in NYC.",
    "What are homes selling for in the Mission District right now?",
]

_CHAT_PROTOTYPES: list[str] = [
    "Explain how transformers work in machine learning.",
    "Write me a short poem about autumn leaves.",
    "What is the capital of France?",
    "Help me rewrite this paragraph more clearly.",
    "Summarize the key ideas in this text I've pasted.",
    "What are the pros and cons of React vs Vue?",
    "Can you review this Python function for bugs?",
    "Translate this sentence into Spanish.",
    "What does the word 'ephemeral' mean?",
    "Explain the difference between TCP and UDP.",
    "Write a cover letter for a software engineering role.",
    "How do I reverse a linked list in Python?",
    "Give me three ideas for a birthday party theme.",
    "What causes inflation?",
    "Help me brainstorm names for my new startup.",
]

# Min |margin| at a given MRL dim to treat the embedding decision as confident
_MARGIN_THRESHOLD: float = 0.08
# Keyword hits above this confidence skip the embedding step entirely
_KEYWORD_CONFIDENCE: float = 0.92
# Confidence threshold below which we fall through to CapabilityDetector
_EMBEDDING_CONFIDENCE_THRESHOLD: float = 0.75
# MRL dims to try smallest-first; None = full dim
_MRL_DIMS: list[int | None] = [64, 128, 256, None]


class BrowserIntentDetector:
    """
    Two-stage browser-vs-chat classifier: keyword guardrail → MRL embedding.

    Mirrors the WorkflowDetector pattern but for the browser/chat boundary.
    Inject ``embedding_backend`` in tests to avoid model downloads.
    """

    def __init__(self, embedding_backend: EmbeddingBackend | None = None) -> None:
        self._backend: EmbeddingBackend = (
            embedding_backend or SentenceTransformerBackend("all-MiniLM-L6-v2")
        )
        self._browser_cache: np.ndarray | None = None
        self._chat_cache: np.ndarray | None = None

    def _load_prototypes(self) -> tuple[np.ndarray, np.ndarray]:
        if self._browser_cache is None:
            self._browser_cache = self._backend.embed_batch(_BROWSER_PROTOTYPES)
            self._chat_cache = self._backend.embed_batch(_CHAT_PROTOTYPES)
        return self._browser_cache, self._chat_cache

    @staticmethod
    def _truncate(arr: np.ndarray, dims: int | None) -> np.ndarray:
        if dims is None or dims >= arr.shape[-1]:
            return arr
        sliced = arr[..., :dims].copy()
        if sliced.ndim == 1:
            n = float(np.linalg.norm(sliced))
            return sliced / n if n > 0 else sliced
        norms = np.linalg.norm(sliced, axis=-1, keepdims=True)
        return np.where(norms > 0, sliced / norms, sliced)

    def is_browser_intent(
        self, message: str
    ) -> tuple[bool, float, str, float | None]:
        """
        Return (is_browser, confidence, decision_source, semantic_margin).

        decision_source is one of: "keyword", "mrl_embedding", "capability", "default".
        semantic_margin is the cosine margin (browser_sim - chat_sim) or None for keyword.
        """
        low = (message or "").lower()

        # ── 1. Keyword guardrail ──────────────────────────────────────────────
        matched = [c for c in _BROWSER_CUES if c in low]
        if matched:
            _logger.debug(
                "browser intent: keyword match",
                extra={"signals": matched[:4], "confidence": _KEYWORD_CONFIDENCE},
            )
            return True, _KEYWORD_CONFIDENCE, "keyword", None

        # ── 2. MRL embedding ──────────────────────────────────────────────────
        try:
            full_emb = self._backend.embed(message)
            browser_full, chat_full = self._load_prototypes()
        except (ImportError, ModuleNotFoundError):
            _logger.debug("sentence_transformers unavailable — skipping embedding step")
            return False, 0.5, "default", None
        except Exception:
            # The model weights are fetched from the Hub on first use, long
            # after the import succeeds. A blocked network, a Hub outage or an
            # evicted cache therefore raises something that is NOT an
            # ImportError -- it used to propagate out of the request and 500
            # the whole chat turn. Falling back to the keyword path keeps
            # answers flowing; scripts/deploy.sh pre-warms the model so this
            # is a genuine runtime incident rather than the normal case, and
            # WARNING (not debug) is what makes it visible when it happens.
            _logger.warning(
                "embedding intent stage unavailable — falling back to keyword "
                "classification; browser intent will be less accurate",
                exc_info=True,
            )
            return False, 0.5, "default", None

        decisive_margin: float | None = None
        decisive_dims: int | None = None

        for dims in _MRL_DIMS:
            p = self._truncate(full_emb, dims)
            br = self._truncate(browser_full, dims)
            ch = self._truncate(chat_full, dims)
            margin = float(np.mean(br @ p)) - float(np.mean(ch @ p))
            actual = dims if (dims is not None and dims < full_emb.shape[0]) else int(full_emb.shape[0])
            if abs(margin) >= _MARGIN_THRESHOLD:
                decisive_margin = margin
                decisive_dims = actual
                break

        if decisive_margin is not None:
            is_browser = decisive_margin > 0
            confidence = min(0.95, 0.5 + abs(decisive_margin) * 3.0)
            _logger.debug(
                "browser intent: mrl_embedding",
                extra={
                    "is_browser": is_browser,
                    "margin": round(decisive_margin, 4),
                    "dims": decisive_dims,
                    "confidence": round(confidence, 4),
                },
            )
            if confidence >= _EMBEDDING_CONFIDENCE_THRESHOLD:
                return is_browser, round(confidence, 4), "mrl_embedding", round(decisive_margin, 6)

        # Ambiguous at all dims — return best available margin for diagnostics
        last_margin: float | None = None
        if decisive_margin is None:
            p = full_emb
            browser_sim = float(np.mean(browser_full @ p))
            chat_sim = float(np.mean(chat_full @ p))
            last_margin = round(browser_sim - chat_sim, 6)

        return False, 0.5, "default", last_margin or (round(decisive_margin, 6) if decisive_margin is not None else None)

_TERMINAL_CUES: list[str] = [
    "run tests", "run the tests", "run command", "run a command", "pytest",
    "npm build", "npm run build", "git status", "inspect repo",
    "inspect the repo", "fix failing test", "fix the failing test",
    "run pytest", "git diff",
]

# Module-level detector singleton — shares the embedded prototype cache across calls.
# Tests can swap this out via monkeypatch.
_browser_intent_detector: BrowserIntentDetector | None = None


def _get_browser_intent_detector() -> BrowserIntentDetector:
    global _browser_intent_detector
    if _browser_intent_detector is None:
        _browser_intent_detector = BrowserIntentDetector()
    return _browser_intent_detector


@dataclass
class ChatIntentResult:
    intent: ChatIntent
    browser_signals: list[str]
    terminal_signals: list[str]
    capability: dict  # CapabilityDetector result (diagnostics / UI)
    # Observability fields (populated by classify_chat_intent)
    decision_source: str = "keyword"          # keyword | mrl_embedding | capability | default
    browser_confidence: float = 0.0
    semantic_margin: float | None = None      # cosine(browser) − cosine(chat); None for keyword hits

    def to_dict(self) -> dict:
        return {
            "intent": self.intent,
            "browser_signals": self.browser_signals,
            "terminal_signals": self.terminal_signals,
            "capability": self.capability,
            "decision_source": self.decision_source,
            "browser_confidence": self.browser_confidence,
            "semantic_margin": self.semantic_margin,
        }


def classify_chat_intent(
    message: str,
    *,
    embedding_backend: EmbeddingBackend | None = None,
) -> ChatIntentResult:
    """
    Classify a chat message as browser_agent / terminal_agent / normal_chat.

    Pipeline (browser side):
      1. Keyword guardrail — substring match against _BROWSER_CUES (fast, no deps).
         Confident hits (≥ 0.92) are final.
      2. MRL embedding — cosine similarity to browser/chat prototype corpora.
         Decisive margin (≥ 0.08) at confidence ≥ 0.75 is final.
      3. CapabilityDetector — legacy rule signal (requires_browser_navigation, etc.).
      4. Default → normal_chat.

    Args:
        message:           Raw user message.
        embedding_backend: Override the embedding model (e.g. MockEmbeddingBackend
                           in tests). If None, uses the module-level singleton.
    """
    low = (message or "").lower()
    terminal_signals = [c for c in _TERMINAL_CUES if c in low]

    # Resolve detector (use injected backend or shared singleton)
    if embedding_backend is not None:
        detector = BrowserIntentDetector(embedding_backend)
    else:
        detector = _get_browser_intent_detector()

    is_browser, confidence, decision_source, semantic_margin = detector.is_browser_intent(message)

    # Explicit terminal cues take precedence over a browser classification that
    # comes only from semantic prototypes. Multi-tool intent still applies when
    # an actual browser cue (for example "go to") is present. This avoids
    # hash-backed test embeddings turning "git status" into a browser task.
    explicit_browser_signals = [c for c in _BROWSER_CUES if c in low]
    if terminal_signals and not explicit_browser_signals and decision_source == "mrl_embedding":
        is_browser = False
        decision_source = "keyword"
        confidence = 0.92

    # CapabilityDetector is always run for its diagnostic dict and as a fallback
    cap = CapabilityDetector().detect(message or "")
    cap_dict = cap.to_dict()
    caps = cap.capabilities

    # Step 3: CapabilityDetector override when embedding was ambiguous
    if not is_browser and decision_source == "default":
        if caps.get("requires_browser_navigation") or caps.get("requires_current_web"):
            is_browser = True
            decision_source = "capability"
            confidence = 0.70

    # Terminal signals (keyword-only; embedding not used for terminal yet)
    if not terminal_signals and (
        caps.get("requires_local_command") or caps.get("requires_code_execution")
    ):
        terminal_signals.append("capability:local_command")

    # Derive final intent
    browser_signals: list[str] = []
    if is_browser:
        browser_signals = explicit_browser_signals or [decision_source]

    if browser_signals and terminal_signals:
        intent: ChatIntent = "multi_tool_agent"
    elif browser_signals:
        intent = "browser_agent"
    elif terminal_signals:
        intent = "terminal_agent"
    else:
        intent = "normal_chat"

    return ChatIntentResult(
        intent=intent,
        browser_signals=browser_signals,
        terminal_signals=terminal_signals,
        capability=cap_dict,
        decision_source=decision_source,
        browser_confidence=confidence,
        semantic_margin=semantic_margin,
    )


# ── Terminal command proposals (Stage 1: deterministic, proposals only) ───────

_COMMAND_TEMPLATES: list[tuple[str, str]] = [
    # (cue in message, proposed command)
    ("pytest", "python -m pytest"),
    ("run tests", "python -m pytest"),
    ("run the tests", "python -m pytest"),
    ("npm build", "npm run build"),
    ("npm run build", "npm run build"),
    ("git status", "git status"),
    ("git diff", "git diff --stat"),
    ("inspect repo", "git status"),
]


def propose_terminal_commands(message: str) -> list[str]:
    """Deterministic prompt → proposed commands (deduped, order preserved)."""
    low = (message or "").lower()
    proposed: list[str] = []
    for cue, command in _COMMAND_TEMPLATES:
        if cue in low and command not in proposed:
            proposed.append(command)
    return proposed


# ── Chat agent run registry ───────────────────────────────────────────────────

# UI phase labels (Part F item 1). Derived, not stored, except for terminal.
PHASE_PLANNING = "planning"
PHASE_RUNNING_BROWSER = "running_browser"
PHASE_RUNNING_COMMAND = "running_command"
PHASE_WAITING_FOR_APPROVAL = "waiting_for_approval"
PHASE_FAILED = "failed"
PHASE_COMPLETED = "completed"

_PHASE_LABELS = {
    PHASE_PLANNING: "Planning",
    PHASE_RUNNING_BROWSER: "Running browser",
    PHASE_RUNNING_COMMAND: "Running command",
    PHASE_WAITING_FOR_APPROVAL: "Waiting for approval",
    PHASE_FAILED: "Failed",
    PHASE_COMPLETED: "Completed",
}


@dataclass
class ChatAgentRun:
    """One chat-initiated agent run (browser, terminal, or both)."""

    message: str
    intent: ChatIntent
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    # Chat run ids are canonical durable TaskEngine ids. Kept separate only
    # so older callers that construct ChatAgentRun directly remain valid.
    task_id: Optional[str] = None
    conversation_id: Optional[str] = None
    owner_user_id: Optional[str] = None
    owner_organization_id: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    # Browser leg — the run reuses a full BrowserAgentSession; detailed trace
    # stays available on the Agent page via this id.
    browser_session_id: Optional[str] = None
    # Terminal leg — Stage 1 command proposals + broker results.
    terminal_commands: list[dict] = field(default_factory=list)
    terminal_status: Optional[str] = None  # planning|completed|failed|waiting_for_approval
    error: Optional[str] = None
    # API-first web searches complete directly from SearXNG/Brave without
    # opening a search-engine page in the browser agent.
    search_results: list[dict] = field(default_factory=list)
    search_answer: Optional[str] = None
    # Execution ledger for direct search/fetch runs. Browser sessions expose
    # the equivalent fields through BrowserAgentStatus.
    search_trace: dict = field(default_factory=dict)
    assistant_message_saved: bool = False  # set True after final answer is persisted

    def phase(self, browser_status: Optional[str]) -> str:
        """Resolve the compact UI phase from the underlying leg states."""
        if self.error:
            return PHASE_FAILED
        if self.search_answer is not None and self.terminal_status in (None, "completed"):
            return PHASE_COMPLETED
        if browser_status in ("waiting_for_approval", "needs_approval") or self.terminal_status == "waiting_for_approval":
            return PHASE_WAITING_FOR_APPROVAL
        if browser_status in ("running", "idle"):
            return PHASE_RUNNING_BROWSER
        if browser_status in ("failed", "stopped", "stopped_budget_exceeded"):
            return PHASE_FAILED
        if self.terminal_status == "running":
            return PHASE_RUNNING_COMMAND
        if browser_status == "completed" or (
            browser_status is None and self.terminal_status == "completed"
        ):
            return PHASE_COMPLETED
        if browser_status is None and self.terminal_status is None:
            return PHASE_PLANNING
        return PHASE_PLANNING

    def to_status(self, browser_status_payload: Optional[dict] = None) -> dict:
        """
        Compact, chat-UI-facing status (Part F): phase + expandable trace
        (steps, tools, URLs, commands, errors). The full browser diagnostics
        remain on the Agent page.
        """
        bs = browser_status_payload or {}
        direct_trace = self.search_trace if browser_status_payload is None else {}
        trace_source = direct_trace or bs
        browser_status = bs.get("status")
        phase = self.phase(browser_status)
        final_answer = self.search_answer or bs.get("final_answer")
        if not final_answer and phase == PHASE_FAILED:
            failure = self.error or bs.get("error") or "the agent could not finish the task"
            final_answer = f"I couldn’t complete the task: {failure}."
        elif not final_answer and phase == PHASE_COMPLETED and self.terminal_commands:
            executed = [
                item for item in self.terminal_commands if item.get("executed")
            ]
            final_answer = (
                f"The task completed. {len(executed)} approved command"
                f"{'s were' if len(executed) != 1 else ' was'} executed."
            )

        urls_visited: list[str] = []
        for entry in bs.get("action_log", []) or []:
            u = entry.get("url")
            if u and u != "about:blank" and u not in urls_visited:
                urls_visited.append(u)
        if direct_trace:
            for result in direct_trace.get("sources", []) or self.search_results:
                u = result.get("url") if isinstance(result, dict) else None
                if u and u not in urls_visited:
                    urls_visited.append(u)

        queries = list(trace_source.get("queries", []) or [])
        direct_actions = [
            {
                "step": index,
                "action": "search",
                "result": "completed",
                "summary": query,
            }
            for index, query in enumerate(queries, start=1)
        ]
        action_entries = bs.get("action_log") or []
        actions = (
            [
                {"step": e.get("step"), "action": e.get("action_type"),
                 "result": e.get("result"), "summary": e.get("summary")}
                for e in action_entries
            ]
            if action_entries
            else direct_actions
        )

        elapsed_seconds = trace_source.get("elapsed_seconds")
        if elapsed_seconds is None and browser_status_payload is None:
            elapsed_seconds = 0.0
        estimated_cost = trace_source.get("estimated_cost_usd")
        if estimated_cost is None and not trace_source:
            estimated_cost = 0.0
        status_value = browser_status or self.terminal_status
        if status_value is None:
            status_value = (
                "completed" if phase == PHASE_COMPLETED
                else "failed" if phase == PHASE_FAILED
                else "planning"
            )

        return {
            "agent_run_id": self.run_id,
            "task_id": self.task_id or self.run_id,
            "intent": self.intent,
            "phase": phase,
            "phase_label": _PHASE_LABELS.get(phase, phase),
            "status": status_value,
            "error": self.error or bs.get("error"),
            "browser_session_id": self.browser_session_id,
            "conversation_id": self.conversation_id,
            "trace": {
                "execution_kind": trace_source.get(
                    "execution_kind",
                    "browser_agent" if browser_status_payload is not None else "terminal_agent",
                ),
                "steps_taken": trace_source.get(
                    "steps_taken", trace_source.get("queries_used", 0)
                ),
                "elapsed_seconds": elapsed_seconds,
                "estimated_cost_usd": estimated_cost,
                "cost_known": trace_source.get("cost_known", True),
                "model_calls": trace_source.get("model_calls", bs.get("model_calls", 0)),
                "models_used": trace_source.get("models_used", bs.get("models_used", [])),
                "queries": queries,
                "queries_used": trace_source.get("queries_used", len(queries)),
                "source_count": trace_source.get(
                    "source_count", len(self.search_results)
                ),
                "sources": trace_source.get("sources", self.search_results),
                "retrieved_at": trace_source.get("retrieved_at"),
                "evidence_quality": trace_source.get("evidence_quality"),
                "evidence_status": trace_source.get("evidence_status"),
                "synthesis_status": trace_source.get("synthesis_status"),
                "synthesis_finish_reason": trace_source.get("synthesis_finish_reason"),
                "synthesis_error_type": trace_source.get("synthesis_error_type"),
                "free_only": trace_source.get("free_only", False),
                "current_url": bs.get("current_url"),
                "page_title": bs.get("page_title"),
                "urls_visited": urls_visited,
                "actions": actions,
                "commands": self.terminal_commands,
                "pending_approval": bs.get("pending_approval"),
                "final_result": bs.get("final_result"),
                "final_answer": final_answer,
                "captcha_detected": bs.get("captcha_detected", False),
                "captcha_reason": bs.get("captcha_reason"),
                "screenshot_base64": bs.get("screenshot_base64"),
                "search_results": self.search_results or bs.get("search_results", []),
                "observation_backend": bs.get("observation_backend"),
                "action_backend": bs.get("action_backend"),
                "progress_message": bs.get("progress_message"),
            },
        }


class ChatAgentRunStore:
    """Compatibility cache backed by durable TaskEngine metadata."""

    def __init__(self) -> None:
        self._runs: dict[str, ChatAgentRun] = {}

    def add(self, run: ChatAgentRun) -> None:
        self._runs[run.run_id] = run

    def get(
        self, run_id: str, *, owner_user_id: str | None = None
    ) -> Optional[ChatAgentRun]:
        cached = self._runs.get(run_id)
        if cached is not None:
            if cached.owner_user_id != owner_user_id and (
                cached.owner_user_id is not None or owner_user_id is not None
            ):
                return None
            return cached
        try:
            from .task_engine import get_task_engine

            task = get_task_engine().get(run_id)
        except Exception:
            return None
        raw = task.metadata.get("chat_run")
        if not isinstance(raw, dict):
            return None
        if task.owner_user_id != owner_user_id and (
            task.owner_user_id is not None or owner_user_id is not None
        ):
            return None
        try:
            run = ChatAgentRun(
                message=task.goal,
                intent=raw["intent"],
                run_id=task.task_id,
                task_id=task.task_id,
                conversation_id=raw.get("conversation_id"),
                owner_user_id=task.owner_user_id,
                owner_organization_id=task.owner_organization_id,
                created_at=task.created_at,
                browser_session_id=raw.get("browser_session_id"),
                terminal_commands=list(raw.get("terminal_commands", [])),
                terminal_status=raw.get("terminal_status"),
                error=raw.get("error"),
                search_results=list(raw.get("search_results", [])),
                search_answer=raw.get("search_answer") or task.final_answer,
                search_trace=dict(raw.get("search_trace", {})),
                assistant_message_saved=bool(raw.get("assistant_message_saved", False)),
            )
        except (KeyError, TypeError, ValueError):
            return None
        self.add(run)
        return run


_store = ChatAgentRunStore()


def get_chat_agent_run_store() -> ChatAgentRunStore:
    return _store


def checkpoint_chat_run(run: ChatAgentRun) -> None:
    """Persist the compatibility projection on its canonical task snapshot."""
    from .task_engine import get_task_engine

    task_id = run.task_id or run.run_id
    payload = {
        "intent": run.intent,
        "conversation_id": run.conversation_id,
        "browser_session_id": run.browser_session_id,
        "terminal_commands": run.terminal_commands,
        "terminal_status": run.terminal_status,
        "error": run.error,
        "search_results": run.search_results,
        "search_answer": run.search_answer,
        "search_trace": run.search_trace,
        "assistant_message_saved": run.assistant_message_saved,
    }
    get_task_engine().checkpoint(task_id, metadata={"chat_run": payload})
