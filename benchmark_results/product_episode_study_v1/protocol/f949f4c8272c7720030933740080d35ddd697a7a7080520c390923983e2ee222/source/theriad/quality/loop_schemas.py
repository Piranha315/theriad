"""
Pydantic schemas for quality-loop requests, results, and per-pass diagnostics.
"""
from __future__ import annotations

import time
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ── Stage names ───────────────────────────────────────────────────────────────

class LoopStage(str, Enum):
    DRAFT    = "draft"
    CRITIQUE = "critique"
    REPAIR   = "repair"
    VALIDATE = "validate"
    FINAL    = "final"


# ── Validation signal ─────────────────────────────────────────────────────────

class ValidationSignal(BaseModel):
    """
    Objective validation result fed back into the loop decision.

    Source examples: pytest output, JSON schema check, eslint, HTML preview QA,
    policy gate, source verification.  The loop prefers these over model-only
    critique to avoid useless churn.
    """
    source: str                        # "pytest" | "json_schema" | "lint" | "model_critique" | …
    passed: bool
    score: Optional[float] = None      # 0.0–1.0 quality score if available
    details: str = ""
    is_objective: bool = True          # False for model-only critique signals


# ── Stop reasons ─────────────────────────────────────────────────────────────

class LoopStopReason(str, Enum):
    VALIDATION_PASSED     = "validation_passed"
    MAX_PASSES_REACHED    = "max_passes_reached"
    COST_CAP_REACHED      = "cost_cap_reached"
    LATENCY_CAP_REACHED   = "latency_cap_reached"
    POLICY_DISABLED       = "policy_disabled"
    NO_VALIDATION_SIGNAL  = "no_validation_signal"
    LOOP_NOT_ENABLED      = "loop_not_enabled"
    ERROR                 = "error"


# ── Per-pass diagnostic ───────────────────────────────────────────────────────

class LoopPassDiagnostic(BaseModel):
    pass_number: int
    stage: LoopStage
    tier_used: str
    model_used: Optional[str] = None
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    validation_result: Optional[ValidationSignal] = None
    reason_for_continue: str = ""
    reason_for_stop: str = ""


# ── Request / result ──────────────────────────────────────────────────────────

class QualityLoopRequest(BaseModel):
    """
    Input to the quality loop engine.

    Callers supply the draft output from the initial model call plus the
    original prompt and task metadata.  The engine decides whether to loop
    based on policy and validation signals.
    """
    prompt: str
    draft_output: str
    policy: str = "balanced"            # aggressive_savings | balanced | quality_first | fast
    task_type: str = "general"          # code | json | html | prose | research | general
    # Override: force the loop on/off regardless of policy.
    quality_loop_enabled: Optional[bool] = None
    # Validation signal already available from the draft stage (e.g. test run).
    initial_validation: Optional[ValidationSignal] = None
    # Cost of the draft call (used to compute max_extra_cost_multiplier cap).
    draft_cost_usd: float = 0.0
    # Caller-supplied config overrides (merged over TheriadConfig.quality_loop).
    max_passes_override: Optional[int] = None
    extra: dict[str, Any] = Field(default_factory=dict)


class QualityLoopResult(BaseModel):
    """
    Output from the quality loop engine.

    final_output is the best output produced (draft if loop was disabled/skipped).
    """
    final_output: str
    loop_enabled: bool
    passes_run: int
    stop_reason: LoopStopReason
    final_validation: Optional[ValidationSignal] = None
    total_extra_cost_usd: float = 0.0
    total_extra_latency_ms: float = 0.0
    pass_diagnostics: list[LoopPassDiagnostic] = Field(default_factory=list)

    # Compact summary for the UI diagnostics panel.
    def summary(self) -> dict[str, Any]:
        return {
            "loop_enabled": self.loop_enabled,
            "passes_run": self.passes_run,
            "stop_reason": self.stop_reason.value,
            "validation_passed": self.final_validation.passed if self.final_validation else None,
            "extra_cost_usd": round(self.total_extra_cost_usd, 8),
            "extra_latency_ms": round(self.total_extra_latency_ms, 1),
        }
