"""
Context-window requirement estimator and bucket routing.

Design decisions:
- Token estimation uses a chars/4 heuristic — the well-established approximation
  for English text. Exact tokenization would require a model-specific tokenizer
  and add latency and runtime dependencies on the hot path.
- Context buckets map to minimum context_window requirements; the router selects
  candidates whose context_window >= total_context_required.
- "huge" prompts receive a chunking/compression recommendation in diagnostics.
  The router does not refuse them — it lets the provider or caller decide.
- No active model caching or warm pools; model availability is provider-determined.
- All public types are plain dataclasses / stdlib to avoid coupling to FastAPI.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# ── Token estimation constants ────────────────────────────────────────────────

CHARS_PER_TOKEN: int = 4
"""Standard chars-per-token approximation for English text."""

DEFAULT_OUTPUT_ESTIMATE_TOKENS: int = 512
"""Conservative output budget used when the caller does not supply one.
Keeps routing decisions cheap-biased by default."""

# ── Context buckets ───────────────────────────────────────────────────────────


class ContextBucket(str, Enum):
    """Ordered context-size buckets.  Smallest bucket → smallest viable model."""

    SHORT = "short_context"    # ≤  8 K tokens
    MEDIUM = "medium_context"  # ≤ 32 K tokens
    LONG = "long_context"      # ≤ 128 K tokens
    HUGE = "huge_context"      # >  128 K tokens


BUCKET_MAX_TOKENS: dict[str, int] = {
    ContextBucket.SHORT:  8_000,
    ContextBucket.MEDIUM: 32_000,
    ContextBucket.LONG:   128_000,
    # HUGE has no finite upper bound; callers check bucket == HUGE
}
"""Maximum total_context_required that fits within each bucket."""


# ── Estimate ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ContextEstimate:
    """
    Per-call context breakdown produced by ContextRequirementEstimator.

    All token counts are approximations.  total_context_required is the value
    used for candidate filtering and bucket assignment.
    """

    input_tokens: int
    estimated_output_tokens: int
    attached_file_tokens: int
    retrieved_context_tokens: int
    total_context_required: int
    bucket: ContextBucket
    compression_or_chunking_recommended: bool

    # Route diagnostics — emitted in API responses
    @property
    def context_fit_reason(self) -> str:
        if self.compression_or_chunking_recommended:
            return (
                f"prompt exceeds 128K tokens ({self.total_context_required:,} estimated); "
                "chunking or RAG recommended"
            )
        return (
            f"{self.total_context_required:,} estimated tokens → {self.bucket.value}"
        )


# ── Estimator ────────────────────────────────────────────────────────────────


class ContextRequirementEstimator:
    """
    Estimates total context tokens required for a call and assigns a bucket.

    Usage::

        est = ContextRequirementEstimator().estimate(prompt)
        # est.bucket, est.total_context_required, est.context_fit_reason
    """

    def estimate(
        self,
        prompt: str,
        *,
        attached_file_chars: int = 0,
        retrieved_context_chars: int = 0,
        output_tokens_estimate: int = DEFAULT_OUTPUT_ESTIMATE_TOKENS,
    ) -> ContextEstimate:
        """
        Estimate context requirements from a prompt string.

        Args:
            prompt:                  The full prompt text (history + current message).
            attached_file_chars:     Character count of attached files (if any).
            retrieved_context_chars: Character count of RAG-retrieved context (if any).
            output_tokens_estimate:  Expected output size in tokens.  Defaults to a
                                     conservative 512 tokens to avoid over-routing.

        Returns:
            ContextEstimate with bucket and diagnostic fields populated.
        """
        input_tokens = max(0, len(prompt) // CHARS_PER_TOKEN)
        attached_file_tokens = max(0, attached_file_chars // CHARS_PER_TOKEN)
        retrieved_context_tokens = max(0, retrieved_context_chars // CHARS_PER_TOKEN)
        total = (
            input_tokens
            + attached_file_tokens
            + retrieved_context_tokens
            + output_tokens_estimate
        )

        bucket = _assign_bucket(total)
        chunking_recommended = bucket == ContextBucket.HUGE

        return ContextEstimate(
            input_tokens=input_tokens,
            estimated_output_tokens=output_tokens_estimate,
            attached_file_tokens=attached_file_tokens,
            retrieved_context_tokens=retrieved_context_tokens,
            total_context_required=total,
            bucket=bucket,
            compression_or_chunking_recommended=chunking_recommended,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────


def _assign_bucket(total_tokens: int) -> ContextBucket:
    """Map a token count to the smallest sufficient ContextBucket."""
    for bucket in (ContextBucket.SHORT, ContextBucket.MEDIUM, ContextBucket.LONG):
        if total_tokens <= BUCKET_MAX_TOKENS[bucket]:
            return bucket
    return ContextBucket.HUGE


def filter_candidates_by_context(
    candidates: list,
    required_tokens: int,
) -> list:
    """
    Return candidates whose context_window is sufficient for required_tokens.

    A candidate with context_window=None is treated as "unknown — assume
    sufficient" so that models without documented context limits are not
    inadvertently excluded.

    If no candidates survive the filter, returns the original list so the
    caller always has at least one option (the provider will surface any
    actual context error).
    """
    viable = [
        c for c in candidates
        if c.context_window is None or c.context_window >= required_tokens
    ]
    return viable if viable else candidates
