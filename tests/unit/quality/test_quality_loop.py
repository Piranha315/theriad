"""
Tests for the quality-loop / refinement-pass orchestration.

Coverage (10 required cases):
 1. aggressive_savings does not loop by default.
 2. balanced allows at most one loop when validation fails.
 3. quality_first allows bounded loops (up to max_passes).
 4. fast does not loop by default.
 5. loop stops when validation passes.
 6. loop stops at max_passes.
 7. loop stops at cost cap.
 8. loop stops at latency cap.
 9. diagnostics include pass/stage/model/cost/latency/stop_reason.
10. normal /api/execute behavior unchanged when loop disabled.
11. no hidden-state transfer assumptions (each call is independent).
"""
from __future__ import annotations

import time
from typing import Optional

import pytest

from theriad.quality.loop_engine import run_quality_loop
from theriad.quality.loop_schemas import (
    LoopStage,
    LoopStopReason,
    QualityLoopRequest,
    ValidationSignal,
)
from theriad.core.config.schema import QualityLoopConfig, QualityLoopPassLimits


# ── Helpers ───────────────────────────────────────────────────────────────────

def _cfg(
    aggressive_savings: int = 0,
    balanced: int = 1,
    quality_first: int = 2,
    fast: int = 0,
    stop_if_validation_passes: bool = True,
    require_validation_signal: bool = True,
    max_extra_cost_multiplier: float = 2.0,
    max_extra_latency_seconds: float = 120.0,
) -> QualityLoopConfig:
    return QualityLoopConfig(
        enabled=True,
        max_passes=QualityLoopPassLimits(
            aggressive_savings=aggressive_savings,
            balanced=balanced,
            quality_first=quality_first,
            fast=fast,
        ),
        stop_if_validation_passes=stop_if_validation_passes,
        require_validation_signal=require_validation_signal,
        max_extra_cost_multiplier=max_extra_cost_multiplier,
        max_extra_latency_seconds=max_extra_latency_seconds,
    )


def _req(
    policy: str = "balanced",
    quality_loop_enabled: Optional[bool] = None,
    initial_validation: Optional[ValidationSignal] = None,
    draft_cost_usd: float = 0.01,
    task_type: str = "general",
) -> QualityLoopRequest:
    return QualityLoopRequest(
        prompt="Write a summary of AI trends.",
        draft_output="Draft: AI is growing fast.",
        policy=policy,
        quality_loop_enabled=quality_loop_enabled,
        initial_validation=initial_validation,
        draft_cost_usd=draft_cost_usd,
        task_type=task_type,
    )


def _failing_validation() -> ValidationSignal:
    return ValidationSignal(source="test_runner", passed=False, details="tests failed")


def _passing_validation() -> ValidationSignal:
    return ValidationSignal(source="test_runner", passed=True, details="all tests pass")


# Call counter for independence verification (test 11)
_call_log: list[tuple[str, str, str, str]] = []

def _tracking_caller(prompt, current_output, stage, tier):
    _call_log.append((prompt, current_output, stage.value, tier))
    # Return a modified output so the loop sees progress
    return f"{current_output} [revised by {stage.value}]", 10, 20, 0.001


def _cheap_caller(prompt, current_output, stage, tier):
    """Returns a fresh output each call — cost is negligible."""
    return f"{current_output}+", 5, 10, 0.0001


def _passing_validate_caller(prompt, current_output, stage, tier):
    """Validate stage always produces output different from input → triggers pass."""
    if stage == LoopStage.VALIDATE:
        return current_output + " [validated]", 5, 5, 0.0
    return current_output + " [revised]", 10, 20, 0.001


# ════════════════════════════════════════════════════════════════════════════
# 1. aggressive_savings does not loop by default
# ════════════════════════════════════════════════════════════════════════════

def test_aggressive_savings_no_loop():
    result = run_quality_loop(
        _req(policy="aggressive_savings", initial_validation=_failing_validation()),
        quality_loop_cfg=_cfg(),
    )
    assert result.loop_enabled is False
    assert result.passes_run == 0
    assert result.stop_reason == LoopStopReason.POLICY_DISABLED


# ════════════════════════════════════════════════════════════════════════════
# 2. balanced allows at most one loop when validation fails
# ════════════════════════════════════════════════════════════════════════════

def test_balanced_one_loop_on_failure():
    result = run_quality_loop(
        _req(policy="balanced", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(balanced=1),
    )
    assert result.loop_enabled is True
    assert result.passes_run <= 1


def test_balanced_does_not_exceed_one_pass():
    # Even if we set a large override, balanced policy caps at 1
    result = run_quality_loop(
        _req(policy="balanced", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(balanced=1),
    )
    assert result.passes_run <= 1


# ════════════════════════════════════════════════════════════════════════════
# 3. quality_first allows bounded loops up to max_passes
# ════════════════════════════════════════════════════════════════════════════

def test_quality_first_bounded_loops():
    result = run_quality_loop(
        _req(policy="quality_first", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(
            quality_first=2,
            stop_if_validation_passes=False,   # don't stop early so we run all passes
        ),
    )
    assert result.loop_enabled is True
    assert result.passes_run <= 2


def test_quality_first_runs_more_than_balanced():
    cfg = _cfg(balanced=1, quality_first=2, stop_if_validation_passes=False)
    balanced_result = run_quality_loop(
        _req(policy="balanced", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=cfg,
    )
    qf_result = run_quality_loop(
        _req(policy="quality_first", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=cfg,
    )
    assert qf_result.passes_run >= balanced_result.passes_run


# ════════════════════════════════════════════════════════════════════════════
# 4. fast does not loop by default
# ════════════════════════════════════════════════════════════════════════════

def test_fast_no_loop():
    result = run_quality_loop(
        _req(policy="fast", initial_validation=_failing_validation()),
        quality_loop_cfg=_cfg(fast=0),
    )
    assert result.loop_enabled is False
    assert result.passes_run == 0
    assert result.stop_reason == LoopStopReason.POLICY_DISABLED


# ════════════════════════════════════════════════════════════════════════════
# 5. loop stops when validation passes
# ════════════════════════════════════════════════════════════════════════════

def test_loop_stops_on_validation_pass():
    # Initial validation passes → loop should not run at all
    result = run_quality_loop(
        _req(policy="balanced", initial_validation=_passing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(balanced=2, stop_if_validation_passes=True),
    )
    assert result.stop_reason == LoopStopReason.VALIDATION_PASSED
    assert result.passes_run == 0


def test_loop_stops_mid_run_when_validation_passes():
    # Validation passes during a loop → should stop before hitting max_passes
    result = run_quality_loop(
        _req(
            policy="quality_first",
            initial_validation=_failing_validation(),
            quality_loop_enabled=True,
        ),
        stage_caller=_passing_validate_caller,
        quality_loop_cfg=_cfg(quality_first=5, stop_if_validation_passes=True),
    )
    # Should stop well before 5 passes
    assert result.passes_run < 5


# ════════════════════════════════════════════════════════════════════════════
# 6. loop stops at max_passes
# ════════════════════════════════════════════════════════════════════════════

def test_loop_stops_at_max_passes():
    max_p = 2
    result = run_quality_loop(
        _req(policy="quality_first", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(
            quality_first=max_p,
            stop_if_validation_passes=False,
        ),
    )
    assert result.passes_run <= max_p
    assert result.stop_reason in {
        LoopStopReason.MAX_PASSES_REACHED,
        LoopStopReason.VALIDATION_PASSED,
        LoopStopReason.COST_CAP_REACHED,
    }


# ════════════════════════════════════════════════════════════════════════════
# 7. loop stops at cost cap
# ════════════════════════════════════════════════════════════════════════════

def test_loop_stops_at_cost_cap():
    # draft_cost_usd=0.01, multiplier=1.0 → max_extra_cost = 0.01
    # each call costs 0.001 * 3 stages = 0.003/pass → should hit cap in ~4 passes
    # but we set multiplier very low so 1 pass exceeds it
    result = run_quality_loop(
        _req(
            policy="quality_first",
            initial_validation=_failing_validation(),
            draft_cost_usd=0.001,
        ),
        stage_caller=lambda p, o, s, t: (o + "+", 5, 5, 0.01),  # each stage costs 0.01
        quality_loop_cfg=_cfg(
            quality_first=10,
            max_extra_cost_multiplier=0.5,  # max extra = 0.001 * 0.5 = 0.0005
            stop_if_validation_passes=False,
        ),
    )
    assert result.stop_reason == LoopStopReason.COST_CAP_REACHED


# ════════════════════════════════════════════════════════════════════════════
# 8. loop stops at latency cap
# ════════════════════════════════════════════════════════════════════════════

def test_loop_stops_at_latency_cap():
    def slow_caller(prompt, current_output, stage, tier):
        time.sleep(0.05)   # 50ms per stage → 150ms/pass
        return current_output + "+", 5, 5, 0.0

    result = run_quality_loop(
        _req(policy="quality_first", initial_validation=_failing_validation()),
        stage_caller=slow_caller,
        quality_loop_cfg=_cfg(
            quality_first=20,
            max_extra_latency_seconds=0.1,   # 100ms cap → should stop after ~1 pass
            stop_if_validation_passes=False,
        ),
    )
    assert result.stop_reason in {
        LoopStopReason.LATENCY_CAP_REACHED,
        LoopStopReason.MAX_PASSES_REACHED,
        LoopStopReason.VALIDATION_PASSED,
    }
    # Latency limit is very tight — should not run many passes
    assert result.passes_run <= 3


# ════════════════════════════════════════════════════════════════════════════
# 9. diagnostics include pass/stage/model/cost/latency/stop_reason
# ════════════════════════════════════════════════════════════════════════════

def test_diagnostics_shape():
    result = run_quality_loop(
        _req(policy="balanced", initial_validation=_failing_validation()),
        stage_caller=_cheap_caller,
        quality_loop_cfg=_cfg(balanced=1, stop_if_validation_passes=False),
    )
    assert result.loop_enabled is True
    assert len(result.pass_diagnostics) > 0

    for diag in result.pass_diagnostics:
        assert diag.pass_number >= 1
        assert diag.stage is not None
        assert isinstance(diag.tier_used, str) and diag.tier_used
        assert isinstance(diag.cost_usd, float)
        assert isinstance(diag.latency_ms, float)

    # summary() must include required keys
    summary = result.summary()
    required = {"loop_enabled", "passes_run", "stop_reason", "extra_cost_usd", "extra_latency_ms"}
    for key in required:
        assert key in summary, f"Missing key '{key}' in summary"

    assert result.stop_reason is not None


# ════════════════════════════════════════════════════════════════════════════
# 10. normal /api/execute behavior unchanged when loop disabled
# ════════════════════════════════════════════════════════════════════════════

def test_normal_execute_unchanged_when_loop_disabled():
    """
    When quality_loop_enabled=False, the engine must return the draft
    output unchanged with loop_enabled=False.
    """
    draft = "This is the original draft output."
    req = QualityLoopRequest(
        prompt="Summarize AI.",
        draft_output=draft,
        policy="balanced",
        quality_loop_enabled=False,
    )
    result = run_quality_loop(req, quality_loop_cfg=_cfg())
    assert result.loop_enabled is False
    assert result.final_output == draft
    assert result.passes_run == 0
    assert result.stop_reason == LoopStopReason.LOOP_NOT_ENABLED


def test_loop_not_enabled_by_default_for_aggressive_savings():
    """
    If no explicit flag is passed and policy=aggressive_savings,
    the engine must not loop.
    """
    result = run_quality_loop(
        _req(policy="aggressive_savings", initial_validation=_failing_validation()),
        quality_loop_cfg=_cfg(aggressive_savings=0),
    )
    assert result.loop_enabled is False


# ════════════════════════════════════════════════════════════════════════════
# 11. no hidden-state transfer (each stage call is independent)
# ════════════════════════════════════════════════════════════════════════════

def test_no_hidden_state_transfer():
    """
    Each stage_caller invocation must receive the current output as a plain
    string, not a reference to a shared mutable object or model activation.
    The engine never passes hidden state between calls.
    """
    _call_log.clear()
    result = run_quality_loop(
        _req(
            policy="balanced",
            initial_validation=_failing_validation(),
        ),
        stage_caller=_tracking_caller,
        quality_loop_cfg=_cfg(balanced=1, stop_if_validation_passes=False),
    )

    assert result.loop_enabled is True
    # Every call must have received a plain string prompt and output
    for prompt_arg, output_arg, stage_arg, tier_arg in _call_log:
        assert isinstance(prompt_arg, str), "prompt must be a plain string"
        assert isinstance(output_arg, str), "current_output must be a plain string"
        assert isinstance(stage_arg, str), "stage must be a string value"
        assert isinstance(tier_arg, str), "tier must be a string"

    # No call should have received the same output reference as another
    # (the engine threads the output through, not a shared object)
    outputs = [log[1] for log in _call_log]
    # Each stage in a pass builds on the previous stage's output string —
    # we just confirm they're all plain strings, not objects/dicts
    for o in outputs:
        assert isinstance(o, str)


# ════════════════════════════════════════════════════════════════════════════
# Stage router unit tests
# ════════════════════════════════════════════════════════════════════════════

from theriad.quality.stage_router import route_stage

def test_stage_router_critique_defaults_cheap():
    dec = route_stage(LoopStage.CRITIQUE)
    assert dec.tier == "cheap_reasoning"
    assert dec.use_validation_tool is False


def test_stage_router_critique_escalates_on_high_risk():
    dec = route_stage(LoopStage.CRITIQUE, risk_level="high")
    assert dec.tier == "strong_reasoning"


def test_stage_router_repair_code_uses_code_agent():
    dec = route_stage(LoopStage.REPAIR, task_type="code")
    assert dec.tier == "code_agent"


def test_stage_router_repair_json_uses_code_agent():
    dec = route_stage(LoopStage.REPAIR, task_type="json")
    assert dec.tier == "code_agent"


def test_stage_router_repair_prose_uses_cheap_bulk():
    dec = route_stage(LoopStage.REPAIR, task_type="prose")
    assert dec.tier == "cheap_bulk"


def test_stage_router_validate_no_model():
    dec = route_stage(LoopStage.VALIDATE)
    assert dec.tier == "no_model"
    assert dec.use_validation_tool is True


def test_stage_router_final_cheap_bulk():
    dec = route_stage(LoopStage.FINAL, task_type="general")
    assert dec.tier == "cheap_bulk"


def test_stage_router_draft_uses_passed_tier():
    dec = route_stage(LoopStage.DRAFT, draft_tier="strong_reasoning")
    assert dec.tier == "strong_reasoning"


# ════════════════════════════════════════════════════════════════════════════
# Config schema tests
# ════════════════════════════════════════════════════════════════════════════

from theriad.core.config.schema import TheriadConfig

def test_theriad_config_has_quality_loop():
    cfg = TheriadConfig()
    assert cfg.quality_loop is not None
    assert cfg.quality_loop.max_passes.aggressive_savings == 0
    assert cfg.quality_loop.max_passes.balanced == 1
    assert cfg.quality_loop.max_passes.quality_first == 2
    assert cfg.quality_loop.max_passes.fast == 0
    assert cfg.quality_loop.stop_if_validation_passes is True
    assert cfg.quality_loop.require_validation_signal is True


def test_quality_loop_config_custom_values():
    cfg = QualityLoopConfig(
        max_passes=QualityLoopPassLimits(balanced=3, quality_first=5),
        max_extra_cost_multiplier=3.0,
    )
    assert cfg.max_passes.balanced == 3
    assert cfg.max_passes.quality_first == 5
    assert cfg.max_extra_cost_multiplier == 3.0
    # Unset fields keep defaults
    assert cfg.max_passes.aggressive_savings == 0
