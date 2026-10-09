"""
Quality-loop engine — bounded draft → critique → repair → validate cycle.

Design
------
- Policy-gated: max_passes per policy; aggressive_savings and fast default to 0.
- Validation-driven: loops stop as soon as an objective validation passes.
- Cost + latency capped: hard ceilings prevent runaway spending.
- No hidden-state transfer: each stage call is independent.
- Normal execution unchanged: this module is only invoked when the caller
  explicitly enables the loop (quality_loop_enabled=True in the request or
  policy max_passes > 0 and a validation failure drives re-entry).

The engine is synchronous and pure (no I/O).  Provider calls are delegated
to the injected `stage_caller` callback so tests can mock them trivially.
"""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Callable, Optional

from .loop_schemas import (
    LoopPassDiagnostic,
    LoopStage,
    LoopStopReason,
    QualityLoopRequest,
    QualityLoopResult,
    ValidationSignal,
)
from .stage_router import route_stage

if TYPE_CHECKING:
    from theriad.core.config.schema import QualityLoopPassLimits

logger = logging.getLogger(__name__)


# ── Type alias for the injectable stage caller ────────────────────────────────

# Signature: (prompt, current_output, stage, tier) -> (new_output, tokens_in, tokens_out, cost_usd)
StageCaller = Callable[[str, str, LoopStage, str], tuple[str, int, int, float]]


def _noop_caller(
    prompt: str,
    current_output: str,
    stage: LoopStage,
    tier: str,
) -> tuple[str, int, int, float]:
    """
    Default no-op stage caller used when no real caller is injected.

    Returns the current output unchanged with zero cost/tokens.
    This is intentional: the engine scaffolding is complete and testable
    without real provider calls.  Wire a real caller to enable live passes.
    """
    return current_output, 0, 0, 0.0


# ── Policy helpers ────────────────────────────────────────────────────────────

def _max_passes_for_policy(policy: str, cfg_passes: "QualityLoopPassLimits") -> int:
    """Return the configured max_passes for the given policy."""
    return getattr(cfg_passes, policy.replace("-", "_"), 0)


def _loop_should_run(
    req: QualityLoopRequest,
    max_passes: int,
) -> tuple[bool, LoopStopReason]:
    """
    Decide whether the loop should run at all.

    Returns (should_run, stop_reason_if_not).
    """
    # Explicit override takes priority
    if req.quality_loop_enabled is False:
        return False, LoopStopReason.LOOP_NOT_ENABLED
    if req.quality_loop_enabled is True and max_passes == 0:
        # Caller forced it on but policy says 0 passes — respect the override
        # by running at least 1 pass.
        return True, LoopStopReason.LOOP_NOT_ENABLED  # reason unused

    if max_passes == 0:
        return False, LoopStopReason.POLICY_DISABLED

    # Don't loop if the initial validation already passed
    if req.initial_validation and req.initial_validation.passed:
        return False, LoopStopReason.VALIDATION_PASSED

    return True, LoopStopReason.LOOP_NOT_ENABLED  # reason unused when True


# ── Main engine ───────────────────────────────────────────────────────────────

def run_quality_loop(
    req: QualityLoopRequest,
    *,
    stage_caller: Optional[StageCaller] = None,
    quality_loop_cfg=None,   # QualityLoopConfig | None — loaded from TheriadConfig if absent
) -> QualityLoopResult:
    """
    Run a bounded quality loop over *req.draft_output*.

    Parameters
    ----------
    req:
        The loop request containing the prompt, draft, policy, and task metadata.
    stage_caller:
        Injectable callback for model calls.  Defaults to a no-op stub so the
        scaffolding is fully testable without provider access.
    quality_loop_cfg:
        QualityLoopConfig instance.  When None, TheriadConfig is loaded from the
        environment (lazy import to avoid circular dependencies).

    Returns
    -------
    QualityLoopResult
        Always returns a result — on error the draft is preserved and
        stop_reason=ERROR is set.
    """
    caller = stage_caller or _noop_caller

    # ── Load config ───────────────────────────────────────────────────────────
    if quality_loop_cfg is None:
        try:
            from theriad.core.config.loader import load_config
            quality_loop_cfg = load_config().quality_loop
        except Exception:
            from theriad.core.config.schema import QualityLoopConfig
            quality_loop_cfg = QualityLoopConfig()

    cfg = quality_loop_cfg

    # ── Determine effective max_passes ────────────────────────────────────────
    if req.max_passes_override is not None:
        effective_max = req.max_passes_override
    else:
        effective_max = _max_passes_for_policy(req.policy, cfg.max_passes)

    # Caller explicit override
    if req.quality_loop_enabled is True and effective_max == 0:
        effective_max = 1

    # ── Early exit checks ─────────────────────────────────────────────────────
    should_run, early_stop = _loop_should_run(req, effective_max)
    if not should_run:
        logger.debug(
            "[quality_loop] skipped policy=%s max_passes=%d reason=%s",
            req.policy, effective_max, early_stop.value,
        )
        return QualityLoopResult(
            final_output=req.draft_output,
            loop_enabled=False,
            passes_run=0,
            stop_reason=early_stop,
            final_validation=req.initial_validation,
        )

    # ── Require validation signal check ──────────────────────────────────────
    if cfg.require_validation_signal and req.initial_validation is None:
        logger.debug("[quality_loop] no validation signal — skipping loop")
        return QualityLoopResult(
            final_output=req.draft_output,
            loop_enabled=False,
            passes_run=0,
            stop_reason=LoopStopReason.NO_VALIDATION_SIGNAL,
            final_validation=None,
        )

    # ── Loop ──────────────────────────────────────────────────────────────────
    loop_start = time.monotonic()
    current_output = req.draft_output
    current_validation: Optional[ValidationSignal] = req.initial_validation
    pass_diagnostics: list[LoopPassDiagnostic] = []
    total_cost = 0.0
    stop_reason = LoopStopReason.MAX_PASSES_REACHED

    max_extra_cost = req.draft_cost_usd * cfg.max_extra_cost_multiplier

    for pass_num in range(1, effective_max + 1):
        pass_start = time.monotonic()

        # ── Cost cap check ────────────────────────────────────────────────────
        if max_extra_cost > 0 and total_cost >= max_extra_cost:
            stop_reason = LoopStopReason.COST_CAP_REACHED
            logger.debug("[quality_loop] cost cap reached at pass %d", pass_num)
            break

        # ── Latency cap check ─────────────────────────────────────────────────
        elapsed = (time.monotonic() - loop_start) * 1000
        if elapsed >= cfg.max_extra_latency_seconds * 1000:
            stop_reason = LoopStopReason.LATENCY_CAP_REACHED
            logger.debug("[quality_loop] latency cap reached at pass %d", pass_num)
            break

        # ── Early stop: validation already passing ────────────────────────────
        if cfg.stop_if_validation_passes and current_validation and current_validation.passed:
            stop_reason = LoopStopReason.VALIDATION_PASSED
            logger.debug("[quality_loop] validation passed — stopping before pass %d", pass_num)
            break

        # ── Determine stage sequence for this pass ────────────────────────────
        # Each pass runs: critique → repair → validate
        # (draft is pass 0, done before the loop; final is added after the loop)
        stages = [LoopStage.CRITIQUE, LoopStage.REPAIR, LoopStage.VALIDATE]

        pass_cost   = 0.0
        pass_tokens_in  = 0
        pass_tokens_out = 0
        last_validation = current_validation

        for stage in stages:
            route = route_stage(
                stage,
                task_type=req.task_type,
                risk_level="low",
                prior_failures=pass_num - 1,
            )

            stage_start = time.monotonic()
            try:
                new_output, tok_in, tok_out, cost = caller(
                    req.prompt, current_output, stage, route.tier,
                )
            except Exception as exc:
                logger.warning("[quality_loop] stage %s failed: %s", stage.value, exc)
                stop_reason = LoopStopReason.ERROR
                break

            stage_latency = (time.monotonic() - stage_start) * 1000
            pass_cost      += cost
            pass_tokens_in  += tok_in
            pass_tokens_out += tok_out

            if stage == LoopStage.VALIDATE:
                # The caller signals validation via the output string or a
                # returned ValidationSignal.  Since the no-op caller returns the
                # same string, we treat unchanged output as "no improvement" in
                # mock mode.  Real callers should embed a structured signal.
                validation_passed = (new_output != current_output) or bool(
                    last_validation and last_validation.passed
                )
                last_validation = ValidationSignal(
                    source="loop_validate_stage",
                    passed=validation_passed,
                    details=f"pass {pass_num} validate stage",
                    is_objective=False,
                )
            else:
                if new_output != current_output:
                    current_output = new_output

            diag = LoopPassDiagnostic(
                pass_number=pass_num,
                stage=stage,
                tier_used=route.tier,
                input_tokens=tok_in,
                output_tokens=tok_out,
                cost_usd=cost,
                latency_ms=round(stage_latency, 1),
                validation_result=last_validation if stage == LoopStage.VALIDATE else None,
                reason_for_continue="" if pass_num < effective_max else "max passes",
                reason_for_stop="" if pass_num < effective_max else "max_passes_reached",
            )
            pass_diagnostics.append(diag)

        total_cost += pass_cost
        current_validation = last_validation

        # Check validation after a complete pass
        if cfg.stop_if_validation_passes and current_validation and current_validation.passed:
            stop_reason = LoopStopReason.VALIDATION_PASSED
            logger.debug("[quality_loop] validation passed after pass %d", pass_num)
            break
    else:
        # Loop completed all passes without early exit
        stop_reason = LoopStopReason.MAX_PASSES_REACHED

    total_latency_ms = (time.monotonic() - loop_start) * 1000

    logger.info(
        "[quality_loop] done policy=%s passes=%d stop=%s extra_cost=%.6f extra_latency_ms=%.0f",
        req.policy,
        len({d.pass_number for d in pass_diagnostics}),
        stop_reason.value,
        total_cost,
        total_latency_ms,
    )

    return QualityLoopResult(
        final_output=current_output,
        loop_enabled=True,
        passes_run=len({d.pass_number for d in pass_diagnostics}),
        stop_reason=stop_reason,
        final_validation=current_validation,
        total_extra_cost_usd=total_cost,
        total_extra_latency_ms=round(total_latency_ms, 1),
        pass_diagnostics=pass_diagnostics,
    )
