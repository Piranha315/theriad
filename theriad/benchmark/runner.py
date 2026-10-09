"""
Benchmark runner.

Iterates provider lanes → models → prompts, enforces a budget ceiling,
and writes BenchmarkResult records to a JSONL file incrementally (so a
crash or keyboard interrupt doesn't lose completed results).

Key features:
- Resume mode: skip already-completed (provider, model, prompt_id, category) keys
- Per-row fsync: lose at most one row on crash, not the whole run
- Local model allowlist: blocks large models unless --allow-large-local-models
- Per-category output caps: bound token usage and cost
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from theriad.benchmark.budget import BudgetExceeded, BudgetGuard
from theriad.benchmark.formatting import format_usd
from theriad.benchmark.pricing import estimate
from theriad.benchmark.prompts import load_prompts, stratified_sample
from theriad.benchmark.providers.base import BenchmarkLane
from theriad.benchmark.resume import ResumeState
from theriad.benchmark.types import BenchmarkPrompt, BenchmarkResult
from theriad.core.logging.logger import get_logger

logger = get_logger("benchmark.runner")


def _format_cost_inline(cost: float | None) -> str:
    """Compact cost string for the per-call progress log line."""
    return format_usd(cost)


_RESULTS_DIR = Path("benchmark_results")


def _make_output_path(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / f"benchmark_{ts}.jsonl"


def _write_result(fh, result: BenchmarkResult) -> None:
    """Write one result row, flush, and fsync — lose at most one row on crash."""
    fh.write(json.dumps(result.to_dict()) + "\n")
    fh.flush()
    try:
        os.fsync(fh.fileno())
    except (AttributeError, OSError):
        pass  # not all file objects support fsync; best-effort


def _apply_output_cap(
    bp: BenchmarkPrompt, output_caps: dict[str, int] | None,
) -> BenchmarkPrompt:
    """Return a (possibly modified) BenchmarkPrompt with max_output_tokens capped."""
    if not output_caps:
        return bp
    cap = output_caps.get(bp.category)
    if cap is None or bp.max_output_tokens is None or bp.max_output_tokens <= cap:
        return bp
    # Return a new instance with the cap applied
    return BenchmarkPrompt(
        id=bp.id,
        category=bp.category,
        prompt=bp.prompt,
        max_output_tokens=cap,
        expected_properties=list(bp.expected_properties),
    )


def _estimate_lane_cost(lane: BenchmarkLane, prompts: list[BenchmarkPrompt]) -> float | None:
    """
    Rough pre-run cost estimate for a provider lane.
    Uses 50 input tokens + max_output_tokens as a conservative proxy.

    Returns None when pricing is unknown for the lane (e.g. openrouter/auto,
    where the model is chosen dynamically per prompt).
    """
    from theriad.benchmark.pricing import (
        GROQ_PRICES, CEREBRAS_PRICES, OLLAMA_PRICE,
    )
    pname = lane.provider_name
    total = 0.0
    any_priced = False
    for model in lane.models():
        price = None
        if pname == "groq":
            price = GROQ_PRICES.get(model)
        elif pname == "cerebras":
            price = CEREBRAS_PRICES.get(model)
        elif pname == "ollama":
            price = OLLAMA_PRICE
        for p in prompts:
            c = estimate(price, 50, p.max_output_tokens or 512)
            if c:
                total += c
                any_priced = True
    return total if any_priced else None


async def run_benchmark(
    lanes: list[BenchmarkLane],
    prompts: list[BenchmarkPrompt],
    budget_usd: float = 10.0,
    output_path: Path | None = None,
    on_result: Callable[[BenchmarkResult], None] | None = None,
    resume_state: ResumeState | None = None,
    output_caps: dict[str, int] | None = None,
) -> list[BenchmarkResult]:
    """
    Run the full benchmark matrix and return all results.

    Args:
        lanes:         Provider lanes to benchmark.
        prompts:       Prompts to run (already sampled/filtered).
        budget_usd:    Hard spend ceiling. Calls are skipped once reached.
        output_path:   JSONL file to write results to (incrementally).
        on_result:     Optional callback invoked after each result.
        resume_state:  If provided, skip already-completed (provider, model,
                       prompt_id, category) keys.
        output_caps:   Per-category max_output_tokens override dict.

    Returns:
        All BenchmarkResult records produced in this run (not including
        previously-completed rows from resume_state).
    """
    guard = BudgetGuard(limit_usd=budget_usd)
    results: list[BenchmarkResult] = []

    if output_path is None:
        output_path = _make_output_path(_RESULTS_DIR)

    resume = resume_state or ResumeState(completed_keys=frozenset(), source_path=None)
    logger.info("Writing results to %s", output_path)

    # Open in append mode so resume adds to existing file
    open_mode = "a" if (output_path.exists() and resume.completed > 0) else "w"

    with output_path.open(open_mode) as fh:
        for lane in lanes:
            models = lane.models()
            if not models:
                logger.info("Lane %r has no available models — skipping", lane.provider_name)
                continue

            for model in models:
                # Filter prompts by resume state
                remaining = [
                    bp for bp in prompts
                    if not resume.is_done(lane.provider_name, model, bp.id, bp.category)
                ]
                skipped_count = len(prompts) - len(remaining)

                if skipped_count:
                    logger.info(
                        "Resume: skipping %d/%d already-complete rows for %s/%s",
                        skipped_count, len(prompts), lane.provider_name, model,
                    )

                if not remaining:
                    logger.info(
                        "Lane=%s model=%s fully complete — nothing to run",
                        lane.provider_name, model,
                    )
                    continue

                logger.info(
                    "Starting lane=%s model=%s  (%d/%d prompts remaining)",
                    lane.provider_name, model, len(remaining), len(prompts),
                )

                for i, bp in enumerate(remaining, 1):
                    # Apply per-category output cap
                    bp_capped = _apply_output_cap(bp, output_caps)

                    if guard.is_exhausted:
                        result = _budget_skip(bp_capped, lane.provider_name, model)
                        results.append(result)
                        _write_result(fh, result)
                        if on_result:
                            on_result(result)
                        continue

                    # Pre-check budget with a rough cost estimate
                    try:
                        from theriad.benchmark.pricing import (
                            GROQ_PRICES, CEREBRAS_PRICES, OLLAMA_PRICE,
                        )
                        pname = lane.provider_name
                        price = None
                        if pname == "groq":
                            price = GROQ_PRICES.get(model)
                        elif pname == "cerebras":
                            price = CEREBRAS_PRICES.get(model)
                        elif pname == "ollama":
                            price = OLLAMA_PRICE
                        estimated = estimate(price, 50, bp_capped.max_output_tokens)
                        guard.check_before_call(estimated)
                    except BudgetExceeded:
                        result = _budget_skip(bp_capped, lane.provider_name, model)
                        results.append(result)
                        _write_result(fh, result)
                        if on_result:
                            on_result(result)
                        logger.warning("Budget ceiling reached — skipping remaining calls")
                        continue

                    # Execute
                    result = await lane.run_one(model, bp_capped)
                    guard.record(result.actual_cost_usd)

                    results.append(result)
                    _write_result(fh, result)
                    if on_result:
                        on_result(result)

                    status_icon = "✓" if result.status == "success" else "✗"
                    logger.info(
                        "  [%d/%d] %s %s %s  %.0fms  %s",
                        i, len(remaining),
                        status_icon,
                        result.prompt_id,
                        result.model,
                        result.provider_latency_ms,
                        _format_cost_inline(result.actual_cost_usd),
                    )

    logger.info(guard.summary_line())
    return results


def _budget_skip(bp: BenchmarkPrompt, provider: str, model: str) -> BenchmarkResult:
    return BenchmarkResult(
        timestamp=datetime.now(timezone.utc).isoformat(),
        provider=provider,
        model=model,
        tier=None,
        prompt_id=bp.id,
        category=bp.category,
        input_tokens=0,
        output_tokens=0,
        actual_cost_usd=None,
        provider_latency_ms=0.0,
        total_latency_ms=0.0,
        tokens_per_second=None,
        finish_reason=None,
        status="budget_exceeded",
        error="Budget ceiling reached before this call",
        quality_pass=None,
    )


def build_lanes(
    provider_names: list[str],
    *,
    allow_large_local_models: bool = False,
) -> list[BenchmarkLane]:
    """
    Instantiate lane objects from a list of provider name strings.

    Supported names: openrouter, groq, cerebras, ollama

    Args:
        provider_names:           List of provider name strings.
        allow_large_local_models: If False (default), Ollama lane filters
                                  large models via the overnight-safe allowlist.
    """
    from theriad.benchmark.providers.cerebras_lane import CerebrasLane
    from theriad.benchmark.providers.groq_lane import GroqLane
    from theriad.benchmark.providers.ollama_lane import OllamaLane
    from theriad.benchmark.providers.openrouter_lane import OpenRouterLane

    _LANE_MAP = {
        "openrouter": OpenRouterLane,
        "groq":       GroqLane,
        "cerebras":   CerebrasLane,
        "ollama":     lambda: OllamaLane(allow_large=allow_large_local_models),
    }
    lanes: list[BenchmarkLane] = []
    for name in provider_names:
        factory = _LANE_MAP.get(name.lower())
        if factory is None:
            logger.warning("Unknown provider %r — skipping", name)
        else:
            lanes.append(factory())
    return lanes
