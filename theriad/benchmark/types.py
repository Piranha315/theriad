"""
Benchmark data types.

BenchmarkPrompt  — one item from benchmark_prompts.jsonl
BenchmarkResult  — one provider×model×prompt execution record
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from decimal import Decimal, DecimalException, InvalidOperation
from typing import Any, Literal

_PREVIEW_CHARS = 500


def _make_output_preview(content: str) -> tuple[str, str]:
    """Return (preview, sha256_hex) for a model output string."""
    preview = content[:_PREVIEW_CHARS]
    digest = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return preview, digest


# Version every scoring change so historical grades remain identifiable.
GRADER_VERSION = "benchmark-v2"


@dataclass(frozen=True)
class GradeResult:
    """A missing rubric is unknown quality, not a correct or incorrect answer."""

    status: Literal["correct", "incorrect", "ungradable"]
    reason: str
    grader_version: str = GRADER_VERSION

    @property
    def passed(self) -> bool | None:
        if self.status == "ungradable":
            return None
        return self.status == "correct"


# Only explicit answer syntax or a single standalone choice counts. In particular,
# an article in prose ("A careful analysis...") is never an answer.
_L = "A-J"
_MC_MARKER = re.compile(r"\bANSWER\s*[:\-]\s*", re.IGNORECASE)
_MC_PHRASE = re.compile(r"\b(?:the\s+)?answer\s+is\s*[:\-]?\s*", re.IGNORECASE)
_MC_BOXED = re.compile(rf"\\boxed\{{\s*\**([{_L}])\**\s*\}}", re.IGNORECASE)
_MC_CHOICE = re.compile(
    rf"^\s*\**\(?([{_L}])\)?\**(?=$|[.):,;!]|\s+(?:because|is the answer)\b)",
    re.IGNORECASE,
)
_MC_STANDALONE = re.compile(
    rf"^\s*(?:\**([{_L}])\**(?:[.):](?:\s.*)?|\s*)|\(([{_L}])\)(?:\s+is the answer\.?)?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_MC_AMBIGUOUS = re.compile(r"^\s*(?:or|and|maybe|possibly|/|\|)\b", re.IGNORECASE)


def _parse_choice_answer(text: str) -> str | None:
    text = text.strip()
    match = _MC_CHOICE.match(text)
    if match is None:
        decorated = text.removesuffix(".").strip()
        for left, right in (("$", "$"), (r"\(", r"\)"), (r"\[", r"\]")):
            if decorated.startswith(left) and decorated.endswith(right):
                decorated = decorated[len(left):-len(right)].strip()
        boxed = _MC_BOXED.fullmatch(decorated)
        return boxed.group(1).upper() if boxed else None
    if _MC_AMBIGUOUS.match(text[match.end():].lstrip(" .):,;!")):
        return None
    return match.group(1).upper()


def _extract_mc_letter(output: str) -> str | None:
    """Extract an explicit final choice, without guessing from ordinary prose."""
    for pattern in (_MC_MARKER, _MC_PHRASE):
        matches = list(pattern.finditer(output))
        if matches:
            # An invalid final marker must not fall back to an earlier answer.
            answer = _parse_choice_answer(output[matches[-1].end():].strip())
            if answer is not None or pattern is _MC_MARKER:
                return answer
    boxed = _MC_BOXED.findall(output)
    if boxed:
        return boxed[-1].upper()
    standalone = list(_MC_STANDALONE.finditer(output))
    if len(standalone) == 1:
        match = standalone[0]
        return (match.group(1) or match.group(2)).upper()
    return None


_EXACT_MARKER = re.compile(r"\bANSWER\s*[:\-]\s*([^\n]*)", re.IGNORECASE)
_EXACT_BOXED = re.compile(r"\\boxed\{\s*([^{}]+?)\s*\}")
_EXACT_NOISE = re.compile(r"[$]|(?<=\d),(?=\d{3}(?:\D|$))")


def _extract_exact_answer(output: str) -> str:
    """Extract the last declared answer, preserving its complete value."""
    matches = _EXACT_MARKER.findall(output)
    if matches:
        selected = matches[-1].strip().strip("*$ ")
        boxed = _EXACT_BOXED.fullmatch(selected)
        return boxed.group(1) if boxed else selected
    boxed = _EXACT_BOXED.findall(output)
    return boxed[-1] if boxed else output


def _normalize_exact(s: str) -> str:
    """Normalize case, whitespace, currency and valid thousands separators."""
    s = _EXACT_NOISE.sub("", s.strip()).removesuffix(".")
    return " ".join(s.lower().split())


def _as_number(s: str) -> Decimal | None:
    """Parse a complete finite decimal, without float rounding false positives."""
    try:
        value = Decimal(s)
        return value if value.is_finite() else None
    except (InvalidOperation, TypeError, ValueError):
        return None


_NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
_NUMBER_WITH_UNIT = re.compile(
    rf"^({_NUMBER})(?:\s*/\s*({_NUMBER}))?\s*([a-zA-Z°%][a-zA-Z°% \-]*)?$"
)
# Unit suffixes can contain nouns ("apples", "meters", "years ago"), but cannot
# contain a second claim, alternative, or calculation. Never discard arbitrary
# trailing text as the previous numeric-prefix grader did.
_NON_UNIT_WORDS = {
    "and", "or", "but", "not", "no", "is", "are", "was", "were", "equals",
    "plus", "minus", "times", "divided", "over", "maybe", "possibly", "approximately",
    "wrong", "incorrect", "correct", "because", "rather", "actually", "versus",
}


def _numeric_value_with_unit(s: str) -> Decimal | None:
    match = _NUMBER_WITH_UNIT.fullmatch(s.strip())
    if match is None:
        return None
    unit = match.group(3)
    if unit and (len(unit.split()) > 4 or _NON_UNIT_WORDS.intersection(unit.lower().split())):
        return None
    numerator = _as_number(match.group(1))
    if match.group(2) is None:
        return numerator
    denominator = _as_number(match.group(2))
    if denominator in (None, 0):
        return None
    try:
        return numerator / denominator if numerator is not None else None
    except DecimalException:
        return None


def _exact_matches(output: str, answer: str) -> bool:
    """Compare a complete extracted value; tolerate units only for numeric keys."""
    got = _normalize_exact(_extract_exact_answer(output))
    want = _normalize_exact(answer)
    if got == want:
        return True
    want_n = _as_number(want)
    if want_n is not None:
        return _numeric_value_with_unit(got) == want_n
    return False


@dataclass
class BenchmarkPrompt:
    id: str
    category: str
    prompt: str
    max_output_tokens: int | None = None
    expected_properties: list[str] = field(default_factory=list)
    # HLE / structured-eval fields
    answer: str | None = None          # ground-truth answer string
    answer_type: str | None = None     # "multiple_choice" | "exact_match"

    def grade(self, output: str) -> GradeResult:
        """Return versioned correctness, or an explicit ungradable rubric error."""
        if self.answer_type is not None or self.answer is not None:
            if not isinstance(self.answer, str) or not self.answer.strip():
                return GradeResult("ungradable", "missing_answer_key")
            if self.answer_type == "multiple_choice":
                key = self.answer.strip().upper()
                if re.fullmatch(r"[A-J]", key) is None:
                    return GradeResult("ungradable", "invalid_choice_key")
                passed = _extract_mc_letter(output) == key
            elif self.answer_type == "exact_match":
                passed = _exact_matches(output, self.answer)
            else:
                return GradeResult("ungradable", "unsupported_answer_type")
        elif not self.expected_properties:
            return GradeResult("ungradable", "missing_rubric")
        elif any(not isinstance(prop, str) or not prop.strip() for prop in self.expected_properties):
            return GradeResult("ungradable", "invalid_expected_properties")
        else:
            passed = all(prop in output for prop in self.expected_properties)
        return GradeResult("correct" if passed else "incorrect", "answer_matches" if passed else "answer_does_not_match")

    def quality_check(self, output: str) -> bool | None:
        """Compatibility result: True/False for scored items; None without a rubric."""
        return self.grade(output).passed


@dataclass
class BenchmarkResult:
    timestamp: str
    provider: str
    model: str
    tier: str | None
    prompt_id: str
    category: str
    input_tokens: int
    output_tokens: int
    actual_cost_usd: float | None
    provider_latency_ms: float
    total_latency_ms: float
    tokens_per_second: float | None
    finish_reason: str | None
    status: str          # "success" | "error" | "timeout" | "skipped" | "budget_exceeded"
    error: str | None
    quality_pass: bool | None = None    # automated check result
    quality_score: float | None = None  # manual score, added later
    # Pricing metadata — populated by lanes that use PricingTable.lookup()
    cost_known: bool = False
    pricing_source: str | None = None   # "exact" | "alias" | "missing"
    pricing_model_slug: str | None = None   # table key matched
    requested_model_slug: str | None = None  # model slug sent to provider
    # Output capture — first 500 chars + hash (full output not stored by default)
    output_preview: str | None = None   # first 500 chars of model output
    output_sha256: str | None = None    # SHA-256 hex of full output (for dedup)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "provider": self.provider,
            "model": self.model,
            "tier": self.tier,
            "prompt_id": self.prompt_id,
            "category": self.category,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "actual_cost_usd": (
                round(self.actual_cost_usd, 8)
                if self.actual_cost_usd is not None else None
            ),
            "provider_latency_ms": round(self.provider_latency_ms, 1),
            "total_latency_ms": round(self.total_latency_ms, 1),
            "tokens_per_second": (
                round(self.tokens_per_second, 1)
                if self.tokens_per_second is not None else None
            ),
            "finish_reason": self.finish_reason,
            "status": self.status,
            "error": self.error,
            "quality_pass": self.quality_pass,
            "quality_score": self.quality_score,
            "cost_known": self.cost_known,
            "pricing_source": self.pricing_source,
            "pricing_model_slug": self.pricing_model_slug,
            "requested_model_slug": self.requested_model_slug,
            "output_preview": self.output_preview,
            "output_sha256": self.output_sha256,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> BenchmarkResult:
        return cls(
            timestamp=d["timestamp"],
            provider=d["provider"],
            model=d["model"],
            tier=d.get("tier"),
            prompt_id=d["prompt_id"],
            category=d["category"],
            input_tokens=d.get("input_tokens", 0),
            output_tokens=d.get("output_tokens", 0),
            actual_cost_usd=d.get("actual_cost_usd"),
            provider_latency_ms=d.get("provider_latency_ms", 0.0),
            total_latency_ms=d.get("total_latency_ms", 0.0),
            tokens_per_second=d.get("tokens_per_second"),
            finish_reason=d.get("finish_reason"),
            status=d.get("status", "unknown"),
            error=d.get("error"),
            quality_pass=d.get("quality_pass"),
            quality_score=d.get("quality_score"),
            cost_known=d.get("cost_known", False),
            pricing_source=d.get("pricing_source"),
            pricing_model_slug=d.get("pricing_model_slug"),
            requested_model_slug=d.get("requested_model_slug"),
            output_preview=d.get("output_preview"),
            output_sha256=d.get("output_sha256"),
        )
