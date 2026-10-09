"""Plain-language preflight estimates for unusually large chat requests."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

LARGE_SOURCE_TOKENS = 8_000
LARGE_EXPECTED_TOTAL_TOKENS = 12_000
NOTICEABLE_ESTIMATED_COST_USD = 0.05


def estimate_tokens(text: str) -> int:
    return max(1, math.ceil(len((text or "").split()) * 1.33))


@dataclass(frozen=True)
class TokenRiskAssessment:
    source_tokens: int
    sent_tokens: int
    expected_output_tokens: int
    estimated_cost_usd: float
    requires_confirmation: bool
    reason: str | None
    approximate_pages: int
    selected_model: str
    free_model: str

    def to_dict(self) -> dict:
        return asdict(self)


def assess_token_risk(
    *,
    source_tokens: int,
    sent_tokens: int,
    expected_output_tokens: int,
    estimated_cost_usd: float,
    selected_model: str,
    free_model: str,
    cost_notice_threshold_usd: float | None = None,
) -> TokenRiskAssessment:
    total = sent_tokens + expected_output_tokens
    cost_threshold = (
        cost_notice_threshold_usd
        if cost_notice_threshold_usd is not None and cost_notice_threshold_usd > 0
        else NOTICEABLE_ESTIMATED_COST_USD
    )
    reason = None
    if source_tokens >= LARGE_SOURCE_TOKENS:
        reason = "This request includes a lot of conversation or file context."
    elif total >= LARGE_EXPECTED_TOTAL_TOKENS:
        reason = "This request may produce a long response."
    elif estimated_cost_usd >= cost_threshold:
        reason = "This request is likely to cost more than a typical message."
    return TokenRiskAssessment(
        source_tokens=source_tokens,
        sent_tokens=sent_tokens,
        expected_output_tokens=expected_output_tokens,
        estimated_cost_usd=round(max(0.0, estimated_cost_usd), 6),
        requires_confirmation=reason is not None,
        reason=reason,
        approximate_pages=max(1, math.ceil(source_tokens / 500)),
        selected_model=selected_model,
        free_model=free_model,
    )
