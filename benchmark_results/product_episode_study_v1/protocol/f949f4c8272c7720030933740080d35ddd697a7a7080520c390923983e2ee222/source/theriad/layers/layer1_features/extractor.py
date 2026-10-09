"""
Layer 1: Lightweight numerical feature extraction.

Design decisions:
- All features are computed in a single pass over the prompt text where possible.
- Features are plain dataclass fields (floats/ints) — no tensors, no models.
- tiktoken used for token counting because it matches OpenAI tokenization exactly.
  Alternative considered: word-count heuristic (words * 1.3). Rejected because
  tiktoken is already a dep and gives exact counts for cost estimation.
- JSON probability is heuristic (brace/bracket density + json.loads attempt)
  rather than a neural classifier. Sufficient for Layer 1 purposes.
- This module has ZERO side effects; FeatureExtractor is stateless.
"""
from __future__ import annotations

import json
import re
import string
from dataclasses import asdict, dataclass

from theriad.core.logging.logger import get_logger

logger = get_logger("layer1_features")

# ── regexes (compiled once at import) ────────────────────────────────────────
_CODE_FENCE_RE = re.compile(r"```[\w]*\n.*?```", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,6}\s", re.MULTILINE)
_MARKDOWN_LIST_RE = re.compile(r"^[\-\*\+]\s", re.MULTILINE)
_MARKDOWN_BOLD_RE = re.compile(r"\*\*[^*]+\*\*|__[^_]+__")
_LANG_KEYWORDS_RE = re.compile(
    r"\b(def |class |import |from |function |const |let |var |return |"
    r"if \(|for \(|public |private |void |async |await |lambda |yield )\b"
)
_CODE_INTENT_RE = re.compile(
    r"\b(python|javascript|typescript|java|rust|go|c\+\+|sql|html|css|"
    r"function|script|code|api|asyncio|http|url|urls|endpoint|regex|"
    r"status\s+codes?|json|database|query|pytest|npm|git)\b",
    re.IGNORECASE,
)


@dataclass
class PromptFeatures:
    """All numerical features extracted from a prompt."""

    # Token / length
    token_count: int        # tiktoken approximation
    word_count: int
    line_count: int
    char_count: int

    # Punctuation
    punctuation_ratio: float   # punct chars / total chars

    # Code signals
    code_fence_count: int      # number of ``` blocks
    inline_code_count: int     # number of `inline` spans
    code_keyword_count: int    # language keyword matches
    code_intent_count: int      # coding/domain terms in a natural-language request
    code_ratio: float          # chars inside code fences / total chars

    # Markdown signals
    markdown_heading_count: int
    markdown_list_count: int
    markdown_bold_count: int
    markdown_ratio: float       # (headings + lists + bold) / line_count

    # JSON signals
    json_brace_ratio: float     # ('{' + '}' + '[' + ']') / char_count
    json_probability: float     # 0.0–1.0 heuristic

    def to_dict(self) -> dict:
        return asdict(self)


class FeatureExtractor:
    """Extracts PromptFeatures from a raw prompt string."""

    # tiktoken encoding — loaded lazily to avoid import-time cost
    _enc = None

    @classmethod
    def _token_count(cls, text: str) -> int:
        try:
            if cls._enc is None:
                import tiktoken
                cls._enc = tiktoken.get_encoding("cl100k_base")
            return len(cls._enc.encode(text))
        except Exception:
            # Fallback: word count * 1.3 heuristic
            return int(len(text.split()) * 1.3)

    def extract(self, prompt: str) -> PromptFeatures:
        """Extract all features in a single sweep."""
        char_count = len(prompt)
        word_count = len(prompt.split())
        line_count = max(prompt.count("\n") + 1, 1)
        token_count = self._token_count(prompt)

        # Punctuation
        punct_chars = sum(1 for c in prompt if c in string.punctuation)
        punctuation_ratio = punct_chars / max(char_count, 1)

        # Code
        code_fences = _CODE_FENCE_RE.findall(prompt)
        code_fence_count = len(code_fences)
        code_chars = sum(len(f) for f in code_fences)
        code_ratio = code_chars / max(char_count, 1)
        inline_code_count = len(_INLINE_CODE_RE.findall(prompt))
        code_keyword_count = len(_LANG_KEYWORDS_RE.findall(prompt))
        code_intent_count = len(_CODE_INTENT_RE.findall(prompt))

        # Markdown
        heading_count = len(_MARKDOWN_HEADING_RE.findall(prompt))
        list_count = len(_MARKDOWN_LIST_RE.findall(prompt))
        bold_count = len(_MARKDOWN_BOLD_RE.findall(prompt))
        md_elements = heading_count + list_count + bold_count
        markdown_ratio = md_elements / max(line_count, 1)

        # JSON
        brace_chars = sum(1 for c in prompt if c in "{}[]")
        json_brace_ratio = brace_chars / max(char_count, 1)
        json_probability = self._json_probability(prompt, json_brace_ratio)

        features = PromptFeatures(
            token_count=token_count,
            word_count=word_count,
            line_count=line_count,
            char_count=char_count,
            punctuation_ratio=punctuation_ratio,
            code_fence_count=code_fence_count,
            inline_code_count=inline_code_count,
            code_keyword_count=code_keyword_count,
            code_intent_count=code_intent_count,
            code_ratio=code_ratio,
            markdown_heading_count=heading_count,
            markdown_list_count=list_count,
            markdown_bold_count=bold_count,
            markdown_ratio=markdown_ratio,
            json_brace_ratio=json_brace_ratio,
            json_probability=json_probability,
        )
        logger.debug("features extracted", extra={"tokens": token_count, "lines": line_count})
        return features

    @staticmethod
    def _json_probability(text: str, brace_ratio: float) -> float:
        """
        Heuristic JSON probability in [0, 1].

        Levels:
          1.0 → parses as valid JSON
          0.7 → starts with { or [ and high brace ratio
          0.3 → moderate brace ratio
          0.0 → no signal
        """
        stripped = text.strip()
        if stripped and stripped[0] in "{[":
            try:
                json.loads(stripped)
                return 1.0
            except (ValueError, Exception):
                pass
            if brace_ratio > 0.08:
                return 0.7
            return 0.4
        if brace_ratio > 0.05:
            return 0.3
        return 0.0
