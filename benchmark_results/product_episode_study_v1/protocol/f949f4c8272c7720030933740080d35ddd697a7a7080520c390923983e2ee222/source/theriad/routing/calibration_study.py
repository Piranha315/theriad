"""Statistical sizing for a small Theriad-specific calibration study."""
from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from .public_anchors import PublicAnchor

Z_BY_CONFIDENCE = {
    0.90: 1.6448536269514722,
    0.95: 1.959963984540054,
    0.99: 2.5758293035489004,
}


def _z_value(confidence: float) -> float:
    confidence = float(confidence)
    nearest = min(Z_BY_CONFIDENCE, key=lambda value: abs(value - confidence))
    if abs(nearest - confidence) > 1e-9:
        raise ValueError("confidence must be one of 0.90, 0.95, or 0.99")
    return Z_BY_CONFIDENCE[nearest]


@dataclass(frozen=True)
class CalibrationSizingReport:
    status: str
    public_prompt_count: int
    usable_outcome_count: int
    public_model_count: int
    domain_count: int
    domains: tuple[str, ...]
    confidence: float
    target_half_width: float
    assumed_transfer_correlation: float
    design_effect: float
    worst_case_single_rate_rows: int
    worst_case_paired_transfer_rows: int
    minimum_rows_per_domain: int
    recommended_rows: int
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "public_prompt_count": self.public_prompt_count,
            "usable_outcome_count": self.usable_outcome_count,
            "public_model_count": self.public_model_count,
            "domain_count": self.domain_count,
            "domains": list(self.domains),
            "confidence": self.confidence,
            "target_half_width": self.target_half_width,
            "assumed_transfer_correlation": self.assumed_transfer_correlation,
            "design_effect": self.design_effect,
            "worst_case_single_rate_rows": self.worst_case_single_rate_rows,
            "worst_case_paired_transfer_rows": self.worst_case_paired_transfer_rows,
            "minimum_rows_per_domain": self.minimum_rows_per_domain,
            "recommended_rows": self.recommended_rows,
            "rationale": self.rationale,
        }


def estimate_calibration_size(
    anchors: Iterable[PublicAnchor],
    *,
    confidence: float = 0.95,
    target_half_width: float = 0.05,
    assumed_transfer_correlation: float = 0.0,
    design_effect: float = 1.0,
    minimum_rows_per_domain: int = 20,
) -> CalibrationSizingReport:
    """Estimate rows from an explicit precision target, not a fixed 2,000.

    The calculation is conservative: it uses Bernoulli variance 0.25 for a
    single model rate and a paired-difference variance bound for transfer from
    public anchors to Theriad.  Domain minimums preserve coverage; they are
    visible in the report and are not hidden tuning constants.
    """

    rows = tuple(anchors)
    domains = tuple(sorted({anchor.domain for anchor in rows if anchor.domain}))
    usable = sum(
        1
        for anchor in rows
        for observation in anchor.observations
        if observation.outcome is not None
    )
    models = {
        observation.model_id
        for anchor in rows
        for observation in anchor.observations
        if observation.outcome is not None
    }
    if not 0.0 < target_half_width < 1.0:
        raise ValueError("target_half_width must be between zero and one")
    if not 0.0 <= assumed_transfer_correlation < 1.0:
        raise ValueError("assumed_transfer_correlation must be in [0,1)")
    if design_effect < 1.0:
        raise ValueError("design_effect must be at least one")
    if minimum_rows_per_domain < 1:
        raise ValueError("minimum_rows_per_domain must be positive")
    z = _z_value(confidence)
    single_variance = 0.25
    paired_variance = 2.0 * single_variance * (1.0 - assumed_transfer_correlation)
    single_rows = math.ceil(z * z * single_variance * design_effect / target_half_width**2)
    paired_rows = math.ceil(z * z * paired_variance * design_effect / target_half_width**2)
    coverage_rows = len(domains) * minimum_rows_per_domain
    recommended = max(single_rows, paired_rows, coverage_rows)
    status = "insufficient_public_outcomes" if not usable else "sized"
    return CalibrationSizingReport(
        status=status,
        public_prompt_count=len(rows),
        usable_outcome_count=usable,
        public_model_count=len(models),
        domain_count=len(domains),
        domains=domains,
        confidence=float(confidence),
        target_half_width=float(target_half_width),
        assumed_transfer_correlation=float(assumed_transfer_correlation),
        design_effect=float(design_effect),
        worst_case_single_rate_rows=single_rows,
        worst_case_paired_transfer_rows=paired_rows,
        minimum_rows_per_domain=minimum_rows_per_domain,
        recommended_rows=recommended,
        rationale=(
            "Recommended rows are max(single-model Bernoulli precision, paired "
            "public-to-Theriad transfer precision, and explicit domain coverage); "
            "they are not a fixed replacement for the old 2,000-row grid."
        ),
    )
