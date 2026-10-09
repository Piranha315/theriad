"""
Layer 0 deterministic routing rules.

Design decisions:
- Each rule is a pure function wrapped in a class for polymorphism.
- Rules never call external services; cost of a rule is O(len(prompt)).
- Rules emit TIER LABELS, not model slugs (ADR D-016). The policy resolver
  maps tiers to concrete models at execution time.
- Confidence values are hand-tuned constants; deterministic signals = high confidence.
- Alternatives considered: a DSL config-driven rule system — rejected (Principle 7).
"""
from __future__ import annotations

import re

from theriad.core.logging.logger import get_logger
from theriad.core.routing.interfaces import RoutingDecision, RoutingRule
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_CODE_STRONG,
    TIER_STRONG_REASONING,
)

logger = get_logger("layer0.rules")

# ── helpers ──────────────────────────────────────────────────────────────────
_CODE_FENCE_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_JSON_RE = re.compile(
    r"^\s*[\[{]",  # starts with [ or {
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_MARKDOWN_LIST_RE = re.compile(r"^[\-\*\+]\s", re.MULTILINE)
_LANG_CODE_KEYWORDS = re.compile(
    r"\b(def |class |import |from |function |const |let |var |return |if \(|for \(|public |private |void )\b"
)
_CODE_INTENT_TERMS = re.compile(
    r"\b(python|javascript|typescript|java|rust|go|c\+\+|sql|html|css|"
    r"function|script|code|api|asyncio|http|url|urls|endpoint|regex|"
    r"status\s+codes?|json|database|query|pytest|npm|git)\b",
    re.IGNORECASE,
)

# Imperative coding verbs that indicate a generation/task request, not a question.
# Combined with even a single intent term this is a strong coding signal.
_CODING_VERBS = re.compile(
    r"\b(write|implement|debug|fix|refactor|optimize|parse|serialize)\b",
    re.IGNORECASE,
)


# ─────────────────────────────────────────────────────────────────────────────
class RegexRule(RoutingRule):
    """
    User-defined regex routing rule.

    Matches the prompt against a compiled regex and routes to the specified tier.
    Useful for routing by topic keyword, domain-specific syntax, etc.
    """

    def __init__(self, name: str, pattern: str, tier: str, confidence: float = 0.95):
        self._name = name
        self._pattern = re.compile(pattern, re.IGNORECASE | re.DOTALL)
        self._tier = tier
        self._confidence = confidence

    @property
    def name(self) -> str:
        return self._name

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        if self._pattern.search(prompt):
            decision = RoutingDecision(
                tier=self._tier,
                confidence=self._confidence,
                reason=f"RegexRule({self._name}): pattern matched",
                signals={"rule": self._name, "tier": self._tier},
            )
            logger.debug("RegexRule fired", extra={"rule": self._name, "tier": self._tier})
            return decision
        return None


# ─────────────────────────────────────────────────────────────────────────────
class CodeDetectionRule(RoutingRule):
    """
    Fires when the prompt contains code fences or dense keyword patterns.

    Decision: cheap_code tier — cost-efficient models that rank well on code
    benchmarks handle most coding tasks without escalating to premium.
    Tradeoff: very complex algorithmic design prompts may underperform; those
    should escalate to code_strong via Layer 1 or manual override.
    """

    _CONFIDENCE_FENCE = 0.92
    _CONFIDENCE_KEYWORD = 0.80

    @property
    def name(self) -> str:
        return "code_detection"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        if _CODE_FENCE_RE.search(prompt):
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=self._CONFIDENCE_FENCE,
                reason="code_detection: triple-backtick fence detected",
                signals={"code_fence": True},
            )
        # Keyword density check
        keywords = _LANG_CODE_KEYWORDS.findall(prompt)
        density = len(keywords) / max(len(prompt.split()), 1)
        if density > 0.06:
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=self._CONFIDENCE_KEYWORD,
                reason=f"code_detection: keyword density {density:.2f}",
                signals={"keyword_density": round(density, 3), "keyword_count": len(keywords)},
            )
        intent_terms = _CODE_INTENT_TERMS.findall(prompt)
        if len(intent_terms) >= 2:
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=self._CONFIDENCE_KEYWORD,
                reason=f"code_detection: coding intent terms ({len(intent_terms)})",
                signals={"code_intent_count": len(intent_terms)},
            )
        # A single domain term combined with an imperative coding verb is a
        # reliable coding signal even when the prompt is short and lacks fences.
        # e.g. "Write a function to sort a list" → verb="write" + term="function"
        if len(intent_terms) >= 1 and _CODING_VERBS.search(prompt):
            return RoutingDecision(
                tier=TIER_CHEAP_CODE,
                confidence=self._CONFIDENCE_KEYWORD,
                reason=f"code_detection: coding verb + intent term ({intent_terms[0]})",
                signals={"code_intent_count": len(intent_terms), "coding_verb": True},
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
class JsonDetectionRule(RoutingRule):
    """
    Fires when the prompt is predominantly JSON (schema/data tasks).

    Decision: cheap_bulk tier — JSON manipulation is mechanical; no reasoning
    model required.
    """

    @property
    def name(self) -> str:
        return "json_detection"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        stripped = prompt.strip()
        if not _JSON_RE.match(stripped):
            return None
        try:
            import json
            json.loads(stripped)
            return RoutingDecision(
                tier=TIER_CHEAP_BULK,
                confidence=0.90,
                reason="json_detection: prompt is valid JSON",
                signals={"valid_json": True},
            )
        except (ValueError, Exception):
            # Partial JSON or JSON-like — lower confidence
            brace_ratio = (stripped.count("{") + stripped.count("[")) / max(len(stripped), 1)
            if brace_ratio > 0.05:
                return RoutingDecision(
                    tier=TIER_CHEAP_BULK,
                    confidence=0.70,
                    reason=f"json_detection: high brace ratio {brace_ratio:.2f}",
                    signals={"brace_ratio": round(brace_ratio, 3)},
                )
        return None


# ─────────────────────────────────────────────────────────────────────────────
class MarkdownDetectionRule(RoutingRule):
    """
    Fires when the prompt contains significant Markdown structure.

    Use case: document editing, README generation — cheap_bulk sufficient.
    """

    @property
    def name(self) -> str:
        return "markdown_detection"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        headings = len(_MARKDOWN_HEADING_RE.findall(prompt))
        lists = len(_MARKDOWN_LIST_RE.findall(prompt))
        total = headings + lists
        if total >= 3:
            return RoutingDecision(
                tier=TIER_CHEAP_BULK,
                confidence=0.82,
                reason=f"markdown_detection: {headings} headings + {lists} list items",
                signals={"markdown_headings": headings, "markdown_lists": lists},
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
class BuildProjectRule(RoutingRule):
    """
    Fires when a prompt describes building, implementing, or developing a
    full software artifact (app, backend, API, auth system, test suite, etc.).

    Decision: code_strong tier.

    Rationale: These tasks require multi-file reasoning, architecture decisions,
    and code generation that exceeds what cheap_bulk or cheap_code handles well.
    The build signal is semantic intent, so it remains decisive even for a
    short prompt such as "Build a React app with auth and tests".

    Not fired when the prompt starts with a clarification verb (explain, describe,
    what, how) — those are educational requests, not build tasks.
    """

    # Imperative verbs that indicate a build/create intent
    _BUILD_VERBS: frozenset[str] = frozenset({
        "build", "create", "implement", "develop", "scaffold",
        "generate", "initialize", "init", "bootstrap",
    })

    # Tech signals that indicate a non-trivial software artifact
    _APP_SIGNALS: frozenset[str] = frozenset({
        "app", "application", "frontend", "backend", "database", "db",
        "auth", "authentication", "tests", "test", "api", "service",
        "server", "endpoint", "microservice", "fullstack", "full-stack",
    })

    # Prompts starting with these words are explanations, not build tasks
    _SKIP_VERBS: frozenset[str] = frozenset({
        "explain", "describe", "what", "how", "why", "when", "list",
        "summarize", "summarise", "show", "tell", "define",
    })

    _CONFIDENCE_FULL = 0.88   # ≥2 app signals
    _CONFIDENCE_SINGLE = 0.85  # exactly 1 app signal

    @property
    def name(self) -> str:
        return "build_project"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        import re
        tokens = re.sub(r"[^\w\s]", " ", prompt.lower()).split()
        if not tokens:
            return None

        # Skip if the prompt is an explanation/question, not a build command
        if tokens[0] in self._SKIP_VERBS:
            return None

        token_set = set(tokens)
        build_count = len(self._BUILD_VERBS & token_set)
        if build_count == 0:
            return None

        app_signals = self._APP_SIGNALS & token_set
        if not app_signals:
            return None

        confidence = self._CONFIDENCE_FULL if len(app_signals) >= 2 else self._CONFIDENCE_SINGLE
        return RoutingDecision(
            tier=TIER_CODE_STRONG,
            confidence=confidence,
            reason=(
                f"build_project: '{tokens[0]}' + app signals "
                f"{sorted(app_signals)[:4]}"
            ),
            signals={
                "build_verb_count": build_count,
                "app_signal_count": len(app_signals),
                "app_signals": sorted(app_signals)[:6],
            },
        )


# ─────────────────────────────────────────────────────────────────────────────
class PromptLengthRule(RoutingRule):
    """
    Routes very long prompts to a strong reasoning tier and otherwise abstains.

    Length remains useful as a hard context signal, but it is not an adequate
    success proxy: terse mathematics, debugging, and specialist questions are
    frequently hard. Shortness is therefore a feature for Router V2, never a
    terminal routing decision.

    Thresholds (tokens approximated as words * 1.3):
      > 1500 words → strong_reasoning (moderate confidence, other signals may override)
    """

    LONG_WORDS = 1500

    @property
    def name(self) -> str:
        return "prompt_length"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        word_count = len(prompt.split())
        if word_count > self.LONG_WORDS:
            return RoutingDecision(
                tier=TIER_STRONG_REASONING,
                confidence=0.72,
                reason=f"prompt_length: long prompt ({word_count} words)",
                signals={"word_count": word_count},
            )
        return None


# ─────────────────────────────────────────────────────────────────────────────
class LanguageDetectionRule(RoutingRule):
    """
    Detects non-English prompts and routes to cheap_bulk.

    Decision: cheap_bulk tier — most cheap models handle 50+ languages adequately.
    Fallback: if detection fails, return None (don't block routing).
    Tradeoff: langdetect can misclassify very short texts; only fires on ≥30 chars.
    """

    _ENGLISH_CODES = {"en"}

    @property
    def name(self) -> str:
        return "language_detection"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        if len(prompt.strip()) < 30:
            return None  # too short for reliable detection
        try:
            from langdetect import detect
            lang = detect(prompt)
            if lang not in self._ENGLISH_CODES:
                return RoutingDecision(
                    tier=TIER_CHEAP_BULK,
                    confidence=0.80,
                    reason=f"language_detection: detected language '{lang}'",
                    signals={"detected_lang": lang},
                )
        except Exception:
            pass  # detection failed; abstain
        return None
