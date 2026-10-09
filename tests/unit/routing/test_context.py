"""
Tests for theriad/routing/context.py

Covers:
- Bucket assignment for token counts in each bucket range
- Boundary values (exactly at bucket limits)
- ContextRequirementEstimator with attached files and retrieved context
- filter_candidates_by_context: all fit, none fit (fallback), partial fit
- compression_or_chunking_recommended is True only for HUGE bucket
- context_fit_reason diagnostic strings
"""
from __future__ import annotations

import pytest

from theriad.routing.context import (
    BUCKET_MAX_TOKENS,
    CHARS_PER_TOKEN,
    DEFAULT_OUTPUT_ESTIMATE_TOKENS,
    ContextBucket,
    ContextEstimate,
    ContextRequirementEstimator,
    _assign_bucket,
    filter_candidates_by_context,
)
from theriad.models.tiers import ModelCandidate, TIER_CHEAP_CODE


# ── Helpers ───────────────────────────────────────────────────────────────────


def _candidate(context_window: int | None, model: str = "m") -> ModelCandidate:
    return ModelCandidate(
        tier=TIER_CHEAP_CODE,
        provider="openrouter",
        model=model,
        context_window=context_window,
    )


# ── ContextBucket + _assign_bucket ───────────────────────────────────────────


class TestBucketAssignment:
    def test_short_lower_bound(self):
        assert _assign_bucket(1) == ContextBucket.SHORT

    def test_short_at_limit(self):
        assert _assign_bucket(8_000) == ContextBucket.SHORT

    def test_medium_just_over_short(self):
        assert _assign_bucket(8_001) == ContextBucket.MEDIUM

    def test_medium_at_limit(self):
        assert _assign_bucket(32_000) == ContextBucket.MEDIUM

    def test_long_just_over_medium(self):
        assert _assign_bucket(32_001) == ContextBucket.LONG

    def test_long_at_limit(self):
        assert _assign_bucket(128_000) == ContextBucket.LONG

    def test_huge_just_over_long(self):
        assert _assign_bucket(128_001) == ContextBucket.HUGE

    def test_huge_very_large(self):
        assert _assign_bucket(1_000_000) == ContextBucket.HUGE

    def test_zero_tokens_is_short(self):
        assert _assign_bucket(0) == ContextBucket.SHORT

    def test_bucket_max_tokens_consistent(self):
        """BUCKET_MAX_TOKENS covers SHORT, MEDIUM, LONG (not HUGE)."""
        assert ContextBucket.SHORT in BUCKET_MAX_TOKENS
        assert ContextBucket.MEDIUM in BUCKET_MAX_TOKENS
        assert ContextBucket.LONG in BUCKET_MAX_TOKENS
        assert ContextBucket.HUGE not in BUCKET_MAX_TOKENS


# ── ContextRequirementEstimator ───────────────────────────────────────────────


class TestContextRequirementEstimator:
    def setup_method(self):
        self.est = ContextRequirementEstimator()

    def test_empty_prompt_uses_output_estimate(self):
        result = self.est.estimate("")
        assert result.input_tokens == 0
        assert result.estimated_output_tokens == DEFAULT_OUTPUT_ESTIMATE_TOKENS
        assert result.total_context_required == DEFAULT_OUTPUT_ESTIMATE_TOKENS

    def test_short_prompt_routes_to_short_bucket(self):
        # 100 chars → 25 tokens + 512 output = 537 total → SHORT
        result = self.est.estimate("x" * 100)
        assert result.bucket == ContextBucket.SHORT
        assert result.input_tokens == 25
        assert result.total_context_required == 25 + DEFAULT_OUTPUT_ESTIMATE_TOKENS

    def test_large_prompt_routes_to_medium_bucket(self):
        # Need > 8000 tokens.  (8000 - 512) * 4 = 29,952 chars
        result = self.est.estimate("x" * 30_000)
        assert result.bucket == ContextBucket.MEDIUM

    def test_huge_prompt_routes_to_huge_bucket(self):
        # > 128K tokens → need > (128000 - 512) * 4 = 510,048 chars
        result = self.est.estimate("x" * 520_000)
        assert result.bucket == ContextBucket.HUGE

    def test_attached_file_chars_included(self):
        result = self.est.estimate("hi", attached_file_chars=CHARS_PER_TOKEN * 1000)
        assert result.attached_file_tokens == 1000
        assert result.total_context_required == 0 + 1000 + 0 + DEFAULT_OUTPUT_ESTIMATE_TOKENS

    def test_retrieved_context_chars_included(self):
        result = self.est.estimate("hi", retrieved_context_chars=CHARS_PER_TOKEN * 500)
        assert result.retrieved_context_tokens == 500

    def test_custom_output_estimate(self):
        result = self.est.estimate("test", output_tokens_estimate=100)
        assert result.estimated_output_tokens == 100

    def test_compression_recommended_only_for_huge(self):
        short_result = self.est.estimate("x" * 100)
        huge_result = self.est.estimate("x" * 600_000)
        assert not short_result.compression_or_chunking_recommended
        assert huge_result.compression_or_chunking_recommended

    def test_context_fit_reason_short(self):
        result = self.est.estimate("x" * 100)
        assert "short_context" in result.context_fit_reason
        assert "estimated tokens" in result.context_fit_reason

    def test_context_fit_reason_huge_mentions_chunking(self):
        result = self.est.estimate("x" * 600_000)
        assert "chunking" in result.context_fit_reason or "RAG" in result.context_fit_reason

    def test_negative_chars_clamped_to_zero(self):
        """Negative character counts should not produce negative token counts."""
        result = self.est.estimate("", attached_file_chars=-100)
        assert result.attached_file_tokens == 0


# ── filter_candidates_by_context ─────────────────────────────────────────────


class TestFilterCandidatesByContext:
    def test_all_candidates_fit(self):
        candidates = [_candidate(8_000, "a"), _candidate(32_000, "b")]
        result = filter_candidates_by_context(candidates, 4_000)
        assert result == candidates

    def test_candidates_with_insufficient_window_excluded(self):
        small = _candidate(4_000, "small")
        large = _candidate(32_000, "large")
        result = filter_candidates_by_context([small, large], 8_000)
        assert large in result
        assert small not in result

    def test_no_candidates_fit_returns_all(self):
        """If no candidate has enough context, return all (don't leave caller with empty list)."""
        candidates = [_candidate(1_000, "tiny"), _candidate(2_000, "small")]
        result = filter_candidates_by_context(candidates, 128_000)
        assert result == candidates

    def test_unknown_context_window_passes_filter(self):
        """context_window=None means unknown — treat as sufficient."""
        unknown = _candidate(None, "unknown")
        small = _candidate(100, "small")
        result = filter_candidates_by_context([unknown, small], 50_000)
        assert unknown in result
        assert small not in result

    def test_exact_context_window_match_passes(self):
        """Candidate whose window == required_tokens should pass."""
        candidate = _candidate(8_000, "exact")
        result = filter_candidates_by_context([candidate], 8_000)
        assert result == [candidate]

    def test_one_below_limit_excluded(self):
        candidate = _candidate(7_999, "just_under")
        result = filter_candidates_by_context([candidate], 8_000)
        # Falls back to all since no candidate fits
        assert result == [candidate]

    def test_empty_candidates_returns_empty(self):
        result = filter_candidates_by_context([], 1_000)
        assert result == []

    def test_short_prompt_does_not_route_to_long_context_models(self):
        """Short prompts should not force selection of long-context models.

        This tests the spec requirement: 'short prompts do not route to
        long-context models'.  A short prompt fits in any window, so the
        filter doesn't exclude anything — the policy scorer picks cheapest.
        The filter itself should not add anything here; it should not REQUIRE
        a large window for a small prompt.
        """
        short_window = _candidate(8_000, "small_ctx")
        large_window = _candidate(128_000, "huge_ctx")
        # 500 tokens required — both pass
        result = filter_candidates_by_context([short_window, large_window], 500)
        assert short_window in result
        assert large_window in result

    def test_large_prompt_excludes_small_context_models(self):
        """Large prompts must exclude models whose windows are too small."""
        small_ctx = _candidate(8_000, "small")
        large_ctx = _candidate(128_000, "large")
        result = filter_candidates_by_context([small_ctx, large_ctx], 50_000)
        assert small_ctx not in result
        assert large_ctx in result
