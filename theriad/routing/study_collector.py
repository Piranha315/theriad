"""Dry-run-first, resumable Router V2 study collection.

The collector is deliberately sequential.  Parallel requests make a $25 hard
cap harder to reason about, increase rate-limit bursts, and provide little value
for a study that must be resumable and auditable.  Provider calls are impossible
unless the caller explicitly supplies an execution acknowledgement tied to a
previously written plan hash.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from theriad.accounting.actuals import provider_reported_cost, realized_cost
from theriad.accounting.pricing import PricingTable
from theriad.core.providers.interfaces import (
    CompletionRequest,
    Message,
    ProviderError,
    ProviderProtocol,
)
from theriad.runtime.circuit_breaker import CircuitOpenError

from .study import (
    CORE_MODELS,
    FORMAT_TOKEN_BUDGETS,
    GROK_MODEL,
    GROK_TRAIN_COUNT,
    StudyPrompt,
    corpus_checksum,
    grok_selection_metadata,
    select_informative_training,
    study_call_scope,
)
from .study_budget import StudyBudgetExceeded, StudyBudgetLedger
from .study_grading import EVALUATOR_VERSION, grade_response

HARD_CAP_USD = 25.0
TEST_RESERVE_USD = 10.0
DEFAULT_MAX_OUTPUT_TOKENS = 512  # legacy API default; frozen rows override it
DEFAULT_TAU_EXPECTED_TURNS = 3
DEFAULT_TAU_MAX_TURNS = 8
PLAN_SCHEMA_VERSION = 2
STATE_SCHEMA_VERSION = 1
PHASE_A = "A_core_train"
PHASE_B = "B_grok_train"
PHASE_C = "C_validation"
PHASE_D = "D_test"
STUDY_PHASES = (PHASE_A, PHASE_B, PHASE_C, PHASE_D)


class StudyCollectorError(RuntimeError):
    """Raised before an unsafe or ambiguous provider call is made."""


_RETRYABLE_FREE_ENDPOINT_STATUS_CODES = frozenset({408, 409, 425, 429, 500, 502, 503, 504})


class _FreeEndpointRetryProvider:
    """Retry only the exact zero-cost model slug; never substitute a model."""

    def __init__(self, provider: ProviderProtocol, *, max_attempts: int):
        self._provider = provider
        self._max_attempts = max(1, int(max_attempts))
        self.api_invocations = 0

    @property
    def name(self) -> str:
        return self._provider.name

    @property
    def supports_tool_calls(self) -> bool:
        return bool(getattr(self._provider, "supports_tool_calls", False))

    @property
    def supports_agent_tools(self) -> bool:
        return bool(getattr(self._provider, "supports_agent_tools", False))

    async def health_check(self) -> bool:
        return await self._provider.health_check()

    async def _invoke(self, method: Any, request: Any) -> Any:
        model = str(getattr(request, "model", ""))
        attempt_limit = self._max_attempts if model.endswith(":free") else 1
        attempt = 0
        while attempt < attempt_limit:
            self.api_invocations += 1
            try:
                return await method(request)
            except CircuitOpenError as exc:
                # The breaker rejected the request before an API invocation.
                self.api_invocations -= 1
                if not model.endswith(":free"):
                    raise
                await asyncio.sleep(max(0.01, float(exc.retry_after_seconds)))
                continue
            except ProviderError as exc:
                attempt += 1
                retryable = (
                    exc.status_code is None
                    or exc.status_code in _RETRYABLE_FREE_ENDPOINT_STATUS_CODES
                )
                if attempt >= attempt_limit or not retryable:
                    raise
                await asyncio.sleep(float(2 ** (attempt - 1)))
        raise AssertionError("free-endpoint retry loop exhausted without returning")

    async def complete(self, request: CompletionRequest) -> Any:
        return await self._invoke(self._provider.complete, request)

    async def complete_with_tools(self, request: Any) -> Any:
        method = getattr(self._provider, "complete_with_tools", None)
        if not callable(method):
            raise AttributeError("wrapped provider has no structured tool interface")
        return await self._invoke(method, request)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)


@dataclass(frozen=True)
class PlannedStudyCall:
    prompt_id: str
    model: str
    split: str
    max_output_tokens: int
    maximum_cost_usd: float
    idempotency_key: str
    input_tokens: int = 0
    expected_output_tokens: int = 0
    expected_cost_usd: float = 0.0
    grading_cost_usd: float = 0.0
    task_format: str = ""
    adapter_id: str = ""
    grading_status: str = ""
    expected_turns: int = 1
    max_turns: int = 1
    potential_api_calls: int = 1
    expected_trajectory_tokens: int = 0
    max_trajectory_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_id": self.prompt_id,
            "model": self.model,
            "split": self.split,
            "max_output_tokens": self.max_output_tokens,
            "maximum_cost_usd": self.maximum_cost_usd,
            "idempotency_key": self.idempotency_key,
            "input_tokens": self.input_tokens,
            "expected_output_tokens": self.expected_output_tokens,
            "expected_cost_usd": self.expected_cost_usd,
            "grading_cost_usd": self.grading_cost_usd,
            "task_format": self.task_format,
            "adapter_id": self.adapter_id,
            "grading_status": self.grading_status,
            "expected_turns": self.expected_turns,
            "max_turns": self.max_turns,
            "potential_api_calls": self.potential_api_calls,
            "expected_trajectory_tokens": self.expected_trajectory_tokens,
            "max_trajectory_tokens": self.max_trajectory_tokens,
        }


@dataclass(frozen=True)
class StudyPlan:
    calls: tuple[PlannedStudyCall, ...]
    price_snapshot: dict[str, dict[str, Any]]
    corpus_hash: str
    max_attempts: int = 1
    generated_at: str = ""
    plan_hash: str = ""
    grok_selection: dict[str, Any] = field(default_factory=dict)
    cost_scenarios: dict[str, dict[str, Any]] = field(default_factory=dict)
    phase: str = "legacy_all"
    status: str = "ready"
    preconditions: tuple[str, ...] = ()
    potential_call_count: int | None = None
    ineligible_rows: tuple[dict[str, Any], ...] = ()
    grading_cost_status: str = "local_deterministic_only"
    planned_call_count: int = 0
    planned_potential_api_call_count: int = 0
    planned_maximum_cost_usd: float = 0.0
    planned_expected_cost_usd: float = 0.0
    # Locked phase plans retain the exact call schedule without making those
    # calls executable. The schedule is included in the review hash; only
    # ``calls`` are materialized into collector state.
    planned_calls: tuple[PlannedStudyCall, ...] = ()

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def task_model_runs(self) -> int:
        """Number of task/model assignments, excluding tau2 turn calls."""
        return len(self.calls)

    @property
    def potential_api_call_count(self) -> int:
        return sum(max(1, call.potential_api_calls) for call in self.calls)

    @property
    def maximum_cost_usd(self) -> float:
        return sum(call.maximum_cost_usd for call in self.calls)

    @property
    def expected_cost_usd(self) -> float:
        return sum(call.expected_cost_usd for call in self.calls)

    @property
    def grading_cost_usd(self) -> float:
        return sum(call.grading_cost_usd for call in self.calls)

    def by_model(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for call in self.calls:
            row = result.setdefault(call.model, {"calls": 0, "maximum_cost_usd": 0.0})
            row["calls"] += 1
            row["maximum_cost_usd"] += call.maximum_cost_usd
        return result

    def by_phase(self) -> dict[str, dict[str, float]]:
        result: dict[str, dict[str, float]] = {}
        for call in self.calls:
            row = result.setdefault(
                call.split,
                {
                    "calls": 0,
                    "task_model_runs": 0,
                    "potential_api_calls": 0,
                    "maximum_cost_usd": 0.0,
                },
            )
            row["calls"] += 1
            row["task_model_runs"] += 1
            row["potential_api_calls"] += max(1, call.potential_api_calls)
            row["maximum_cost_usd"] += call.maximum_cost_usd
        return result

    def _payload(self) -> dict[str, Any]:
        schedule = self.planned_calls or self.calls
        planned_call_count = self.planned_call_count or len(schedule)
        planned_potential_api_call_count = self.planned_potential_api_call_count or sum(
            max(1, call.potential_api_calls) for call in schedule
        )
        planned_maximum_cost_usd = self.planned_maximum_cost_usd or sum(
            call.maximum_cost_usd for call in schedule
        )
        planned_expected_cost_usd = self.planned_expected_cost_usd or sum(
            call.expected_cost_usd for call in schedule
        )
        return {
            "schema_version": PLAN_SCHEMA_VERSION,
            "calls": [call.to_dict() for call in schedule],
            "planned_calls": [call.to_dict() for call in schedule],
            "price_snapshot": self.price_snapshot,
            "corpus_hash": self.corpus_hash,
            "max_attempts": self.max_attempts,
            "generated_at": self.generated_at,
            "grok_selection": self.grok_selection,
            "cost_scenarios": self.cost_scenarios,
            "phase": self.phase,
            "status": self.status,
            "preconditions": list(self.preconditions),
            "potential_call_count": self.potential_call_count,
            "task_model_runs": self.task_model_runs,
            "potential_api_call_count": self.potential_api_call_count,
            "ineligible_rows": list(self.ineligible_rows),
            "grading_cost_status": self.grading_cost_status,
            "planned_call_count": planned_call_count,
            "planned_potential_api_call_count": planned_potential_api_call_count,
            "planned_maximum_cost_usd": planned_maximum_cost_usd,
            "planned_expected_cost_usd": planned_expected_cost_usd,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        schedule = self.planned_calls or self.calls
        payload.update(
            {
                "calls": [call.to_dict() for call in self.calls],
                "planned_calls": [call.to_dict() for call in schedule],
                "plan_hash": self.plan_hash,
                "call_count": self.call_count,
                "maximum_cost_usd": self.maximum_cost_usd,
                "expected_cost_usd": self.expected_cost_usd,
                "grading_cost_usd": self.grading_cost_usd,
                "worst_case_total_cost_usd": self.maximum_cost_usd + self.grading_cost_usd,
                "by_model": self.by_model(),
                "by_phase": self.by_phase(),
                "cost_scenarios": self.cost_scenarios,
                "phase": self.phase,
                "status": self.status,
                "potential_call_count": self.potential_call_count,
                "task_model_runs": self.task_model_runs,
                "potential_api_call_count": self.potential_api_call_count,
                "ineligible_row_count": len(self.ineligible_rows),
                "grading_cost_status": self.grading_cost_status,
                "planned_call_count": self.planned_call_count or len(schedule),
                "planned_potential_api_call_count": self.planned_potential_api_call_count
                or sum(max(1, call.potential_api_calls) for call in schedule),
                "planned_maximum_cost_usd": self.planned_maximum_cost_usd
                or sum(call.maximum_cost_usd for call in schedule),
                "planned_expected_cost_usd": self.planned_expected_cost_usd
                or sum(call.expected_cost_usd for call in schedule),
            }
        )
        return payload


def _canonical_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def corpus_fingerprint(prompts: tuple[StudyPrompt, ...]) -> str:
    return corpus_checksum(prompts)


def conservative_input_tokens(prompt: str) -> int:
    """Upper-bound-ish reservation estimate, independent of tokenizer packages."""

    # UTF-8 bytes are a conservative bound for ordinary tokenizers; the fixed
    # allowance covers chat-message framing and provider-side system wrappers.
    return max(1, len(prompt.encode("utf-8")) + 256)


def _price_snapshot(model: str, pricing: PricingTable) -> dict[str, Any]:
    lookup = pricing.lookup(model)
    if lookup is None:
        raise StudyCollectorError(
            f"no offline price for {model}; refusing to treat an unknown model as free"
        )
    return {
        "input_per_1m": float(lookup.price.input_per_1m),
        "output_per_1m": float(lookup.price.output_per_1m),
        "source": lookup.source,
        "pricing_model_slug": lookup.model_slug,
    }


def _cost_scenario(
    calls: tuple[PlannedStudyCall, ...],
    price_snapshot: dict[str, dict[str, Any]],
    *,
    expected_output_tokens: int | None,
    max_attempts: int,
    grading_cost_usd: float,
    name: str,
    quote_source: str,
) -> dict[str, Any]:
    by_phase: dict[str, dict[str, float]] = {}
    expected_model_cost = 0.0
    worst_model_cost = 0.0
    for call in calls:
        unit = price_snapshot[call.model]
        # The study's bounded retry policy applies only to the exact free
        # endpoint. Paid endpoints receive one attempt, and the paid-Nemotron
        # sensitivity scenario prices one successful task rather than free-tier
        # availability retries.
        attempts = (
            max_attempts
            if name == "free_nemotron" and call.model.endswith(":free")
            else 1
        )
        expected_output = min(
            call.max_output_tokens,
            call.expected_output_tokens or int(expected_output_tokens or 0),
        )
        expected_turns = max(1, call.expected_turns)
        max_turns = max(expected_turns, call.max_turns)
        expected_output_total = call.expected_trajectory_tokens or expected_turns * expected_output
        max_output_total = call.max_trajectory_tokens or max_turns * call.max_output_tokens
        expected = attempts * (
            expected_turns * call.input_tokens * float(unit["input_per_1m"])
            + expected_output_total * float(unit["output_per_1m"])
        ) / 1_000_000.0
        priced_worst = attempts * (
            max_turns * call.input_tokens * float(unit["input_per_1m"])
            + max_output_total * float(unit["output_per_1m"])
        ) / 1_000_000.0
        # A reviewed reservation envelope may intentionally exceed the
        # tokenizer-based estimate when provider-reported cost includes hidden
        # or reasoning tokens. Cost gates must honor that larger hard cap.
        worst = max(priced_worst, call.maximum_cost_usd)
        expected_model_cost += expected
        worst_model_cost += worst
        phase = by_phase.setdefault(
            call.split,
            {
                "task_model_runs": 0,
                "potential_api_calls": 0,
                "expected_model_cost_usd": 0.0,
                "worst_model_cost_usd": 0.0,
            },
        )
        phase["calls"] = phase.get("calls", 0) + 1
        phase["task_model_runs"] += 1
        phase["potential_api_calls"] += max(1, call.potential_api_calls)
        phase["expected_model_cost_usd"] += expected
        phase["worst_model_cost_usd"] += worst
    non_test_worst = sum(
        values["worst_model_cost_usd"]
        for split, values in by_phase.items()
        if split != "test"
    )
    test_worst = by_phase.get("test", {}).get("worst_model_cost_usd", 0.0)
    expected_total = expected_model_cost + grading_cost_usd
    worst_total = worst_model_cost + grading_cost_usd
    return {
        "name": name,
        "status": "priced",
        "quote_source": quote_source,
        "expected_output_tokens_policy": "row task-format budget",
        "format_token_budgets": FORMAT_TOKEN_BUDGETS,
        "expected_model_cost_usd": expected_model_cost,
        "worst_case_model_cost_usd": worst_model_cost,
        "grading_cost_usd": grading_cost_usd,
        "expected_total_cost_usd": expected_total,
        "worst_case_total_cost_usd": worst_total,
        "hard_cap_usd": HARD_CAP_USD,
        "test_reserve_usd": TEST_RESERVE_USD,
        "worst_case_non_test_cost_usd": non_test_worst,
        "worst_case_test_cost_usd": test_worst,
        "within_hard_cap": worst_total <= HARD_CAP_USD + 1e-12,
        "untouched_test_reserve_preserved": (
            non_test_worst + grading_cost_usd
            <= HARD_CAP_USD - TEST_RESERVE_USD + 1e-12
        ),
        "by_phase": by_phase,
        "grading_cost_policy": "local deterministic/trusted-result grading only; external judge calls are not authorized",
    }


def build_study_plan(
    prompts: tuple[StudyPrompt, ...],
    *,
    pricing: PricingTable | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_attempts: int = 1,
    existing_call_keys: set[str] | None = None,
    outcome_evidence: dict[str, dict[str, Any]] | None = None,
    expected_output_tokens: int = 256,
    grading_cost_usd: float = 0.0,
    paid_nemotron_quote: dict[str, Any] | None = None,
) -> StudyPlan:
    """Build the exact task/model coverage plan without contacting providers."""

    if max_output_tokens < 1:
        raise ValueError("max_output_tokens must be positive")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    if expected_output_tokens < 1:
        raise ValueError("expected_output_tokens must be positive")
    if grading_cost_usd < 0:
        raise ValueError("grading_cost_usd cannot be negative")
    pricing = pricing or PricingTable(use_live=False)
    existing = existing_call_keys or set()
    scope = study_call_scope(prompts, outcome_evidence=outcome_evidence)
    grok_selection = grok_selection_metadata(
        prompts, outcome_evidence=outcome_evidence
    )
    snapshot = {model: _price_snapshot(model, pricing) for model in scope}
    calls: list[PlannedStudyCall] = []
    for model in (*CORE_MODELS, *[name for name in scope if name not in CORE_MODELS]):
        for row in scope[model]:
            key = f"{row.prompt_id}:{model}"
            if key in existing:
                continue
            cap = row.max_output_tokens or max_output_tokens
            input_tokens = conservative_input_tokens(row.prompt)
            unit = snapshot[model]
            is_tau = row.task_format == "agent_state" or str(row.grader.get("kind") or "").startswith("tau")
            expected_turns = (
                max(1, int(row.grader.get("expected_turns") or DEFAULT_TAU_EXPECTED_TURNS))
                if is_tau
                else 1
            )
            max_turns = (
                max(expected_turns, int(row.grader.get("max_turns") or DEFAULT_TAU_MAX_TURNS))
                if is_tau
                else 1
            )
            maximum = max_attempts * (
                max_turns * input_tokens * unit["input_per_1m"]
                + max_turns * cap * unit["output_per_1m"]
            ) / 1_000_000.0
            expected_tokens = min(
                cap,
                row.expected_output_tokens or expected_output_tokens,
            )
            expected = (
                expected_turns * input_tokens * unit["input_per_1m"]
                + expected_turns * expected_tokens * unit["output_per_1m"]
            ) / 1_000_000.0
            calls.append(
                PlannedStudyCall(
                    prompt_id=row.prompt_id,
                    model=model,
                    split=row.split,
                    max_output_tokens=cap,
                    maximum_cost_usd=maximum,
                    idempotency_key=key,
                    input_tokens=input_tokens,
                    expected_output_tokens=expected_tokens,
                    expected_cost_usd=expected,
                    grading_cost_usd=0.0,
                    expected_turns=expected_turns,
                    max_turns=max_turns,
                    potential_api_calls=max_turns,
                    expected_trajectory_tokens=expected_turns * expected_tokens,
                    max_trajectory_tokens=max_turns * cap,
                )
            )
    grading_per_call = grading_cost_usd / max(len(calls), 1)
    calls_tuple = tuple(
        replace(call, grading_cost_usd=grading_per_call) for call in calls
    )
    free_scenario = _cost_scenario(
        calls_tuple,
        snapshot,
        expected_output_tokens=expected_output_tokens,
        max_attempts=max_attempts,
        grading_cost_usd=grading_cost_usd,
        name="free_nemotron",
        quote_source="offline PricingTable free Nemotron seed plus pinned local prices",
    )
    scenarios = {"free_nemotron": free_scenario}
    if paid_nemotron_quote is None:
        scenarios["paid_nemotron"] = {
            "name": "paid_nemotron",
            "status": "missing_pinned_quote",
            "quote_source": "no paid Nemotron quote supplied",
            "hard_cap_usd": HARD_CAP_USD,
            "test_reserve_usd": TEST_RESERVE_USD,
            "within_hard_cap": None,
            "untouched_test_reserve_preserved": None,
        }
    else:
        paid_snapshot = {model: dict(values) for model, values in snapshot.items()}
        paid_snapshot[CORE_MODELS[0]] = {
            **paid_snapshot[CORE_MODELS[0]],
            "input_per_1m": float(paid_nemotron_quote["input_per_1m"]),
            "output_per_1m": float(paid_nemotron_quote["output_per_1m"]),
            "source": str(paid_nemotron_quote.get("source") or "pinned_local_quote"),
            "pricing_model_slug": str(
                paid_nemotron_quote.get("pricing_model_slug")
                or "nvidia/nemotron-3-super-120b-a12b"
            ),
        }
        scenarios["paid_nemotron"] = _cost_scenario(
            calls_tuple,
            paid_snapshot,
            expected_output_tokens=expected_output_tokens,
            max_attempts=max_attempts,
            grading_cost_usd=grading_cost_usd,
            name="paid_nemotron",
            quote_source=str(paid_nemotron_quote.get("source") or "pinned_local_quote"),
        )
    generated_at = datetime.now(timezone.utc).isoformat()
    draft = StudyPlan(
        calls=calls_tuple,
        price_snapshot=snapshot,
        corpus_hash=corpus_fingerprint(prompts),
        max_attempts=max_attempts,
        generated_at=generated_at,
        grok_selection=grok_selection,
        cost_scenarios=scenarios,
        planned_calls=calls_tuple,
    )
    return StudyPlan(
        calls=draft.calls,
        price_snapshot=draft.price_snapshot,
        corpus_hash=draft.corpus_hash,
        max_attempts=draft.max_attempts,
        generated_at=draft.generated_at,
        plan_hash=_canonical_hash(draft._payload()),
        grok_selection=draft.grok_selection,
        cost_scenarios=draft.cost_scenarios,
    )


def build_calibration_study_plan(
    prompts: tuple[StudyPrompt, ...],
    *,
    models: tuple[str, ...],
    pricing: PricingTable | None = None,
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS,
    max_attempts: int = 1,
    existing_call_keys: set[str] | None = None,
) -> StudyPlan:
    """Plan a small Theriad calibration grid after public-anchor sizing.

    Unlike :func:`build_study_plan`, this function intentionally does not
    require the historical 1,200/400/400 corpus contract.  The caller supplies
    a statistically sized, traffic-representative calibration set and an
    explicit model list.  The same offline pricing, idempotency, hard-cap, and
    untouched-test protections apply when the plan is executed.
    """

    if not prompts:
        raise ValueError("calibration prompts cannot be empty")
    if not models:
        raise ValueError("calibration models cannot be empty")
    if max_output_tokens < 1 or max_attempts < 1:
        raise ValueError("calibration token and attempt limits must be positive")
    prompt_ids = [row.prompt_id for row in prompts]
    if any(not prompt_id for prompt_id in prompt_ids) or len(set(prompt_ids)) != len(prompt_ids):
        raise ValueError("calibration prompts require unique prompt_id values")
    pricing = pricing or PricingTable(use_live=False)
    existing = existing_call_keys or set()
    unique_models = tuple(dict.fromkeys(str(model) for model in models if str(model)))
    snapshot = {model: _price_snapshot(model, pricing) for model in unique_models}
    calls: list[PlannedStudyCall] = []
    for model in unique_models:
        for row in prompts:
            key = f"{row.prompt_id}:{model}"
            if key in existing:
                continue
            cap = row.max_output_tokens or max_output_tokens
            input_tokens = conservative_input_tokens(row.prompt)
            unit = snapshot[model]
            maximum = max_attempts * (
                input_tokens * unit["input_per_1m"] + cap * unit["output_per_1m"]
            ) / 1_000_000.0
            calls.append(
                PlannedStudyCall(
                    prompt_id=row.prompt_id,
                    model=model,
                    split=row.split,
                    max_output_tokens=cap,
                    maximum_cost_usd=maximum,
                    idempotency_key=key,
                )
            )
    generated_at = datetime.now(timezone.utc).isoformat()
    draft = StudyPlan(
        calls=tuple(calls),
        price_snapshot=snapshot,
        corpus_hash=corpus_fingerprint(prompts),
        max_attempts=max_attempts,
        generated_at=generated_at,
        grok_selection={"mode": "not_applicable"},
        planned_calls=tuple(calls),
    )
    return StudyPlan(
        calls=draft.calls,
        price_snapshot=draft.price_snapshot,
        corpus_hash=draft.corpus_hash,
        max_attempts=draft.max_attempts,
        generated_at=draft.generated_at,
        plan_hash=_canonical_hash(draft._payload()),
        grok_selection=draft.grok_selection,
    )


def write_plan(plan: StudyPlan, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _planned_call_from_dict(row: Mapping[str, Any]) -> PlannedStudyCall:
    return PlannedStudyCall(
        prompt_id=str(row["prompt_id"]),
        model=str(row["model"]),
        split=str(row["split"]),
        max_output_tokens=int(row["max_output_tokens"]),
        maximum_cost_usd=float(row["maximum_cost_usd"]),
        idempotency_key=str(row["idempotency_key"]),
        input_tokens=int(row.get("input_tokens", 0)),
        expected_output_tokens=int(row.get("expected_output_tokens", 0)),
        expected_cost_usd=float(row.get("expected_cost_usd", 0.0)),
        grading_cost_usd=float(row.get("grading_cost_usd", 0.0)),
        task_format=str(row.get("task_format") or ""),
        adapter_id=str(row.get("adapter_id") or ""),
        grading_status=str(row.get("grading_status") or ""),
        expected_turns=int(row.get("expected_turns", 1)),
        max_turns=int(row.get("max_turns", 1)),
        potential_api_calls=int(row.get("potential_api_calls", 1)),
        expected_trajectory_tokens=int(row.get("expected_trajectory_tokens", 0)),
        max_trajectory_tokens=int(row.get("max_trajectory_tokens", 0)),
    )


def load_plan(path: str | Path) -> StudyPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise StudyCollectorError("unsupported study plan schema")
    calls = tuple(_planned_call_from_dict(row) for row in raw.get("calls", []))
    planned_calls = tuple(
        _planned_call_from_dict(row)
        for row in raw.get("planned_calls", raw.get("calls", []))
    )
    plan = StudyPlan(
        calls=calls,
        price_snapshot=dict(raw.get("price_snapshot") or {}),
        corpus_hash=str(raw.get("corpus_hash") or ""),
        max_attempts=int(raw.get("max_attempts", 1)),
        generated_at=str(raw.get("generated_at") or ""),
        plan_hash=str(raw.get("plan_hash") or ""),
        grok_selection=dict(raw.get("grok_selection") or {}),
        cost_scenarios=dict(raw.get("cost_scenarios") or {}),
        phase=str(raw.get("phase") or "legacy_all"),
        status=str(raw.get("status") or "ready"),
        preconditions=tuple(str(value) for value in raw.get("preconditions", [])),
        potential_call_count=(
            int(raw["potential_call_count"])
            if raw.get("potential_call_count") is not None
            else None
        ),
        ineligible_rows=tuple(raw.get("ineligible_rows") or ()),
        grading_cost_status=str(
            raw.get("grading_cost_status") or "local_deterministic_only"
        ),
        planned_call_count=int(raw.get("planned_call_count") or len(planned_calls)),
        planned_potential_api_call_count=int(
            raw.get("planned_potential_api_call_count")
            or sum(max(1, call.potential_api_calls) for call in planned_calls)
        ),
        planned_maximum_cost_usd=float(
            raw.get("planned_maximum_cost_usd")
            or sum(call.maximum_cost_usd for call in planned_calls)
        ),
        planned_expected_cost_usd=float(
            raw.get("planned_expected_cost_usd")
            or sum(call.expected_cost_usd for call in planned_calls)
        ),
        planned_calls=planned_calls,
    )
    expected = _canonical_hash(plan._payload())
    if not plan.plan_hash or plan.plan_hash != expected:
        raise StudyCollectorError("study plan hash does not match its contents")
    return plan


PHASE_A_MIN_USABLE_GRADED_COVERAGE = 0.95


def phase_a_usable_graded_coverage(
    prompts: tuple[StudyPrompt, ...], state: Mapping[str, Any]
) -> float:
    train = [
        row
        for row in prompts
        if row.split == "train"
        and not (
            row.label_type == "pairwise"
            and row.grader.get("judging_cost_status") == "postponed_unpriced"
        )
    ]
    if not train:
        return 0.0
    state_rows = state.get("rows") or {}
    usable = 0
    for row in train:
        outcomes = (state_rows.get(row.prompt_id) or {}).get("outcomes") or {}
        if all(model in outcomes for model in CORE_MODELS):
            usable += 1
    return usable / len(train)


def assert_phase_a_coverage(
    prompts: tuple[StudyPrompt, ...],
    state: Mapping[str, Any],
    *,
    threshold: float = PHASE_A_MIN_USABLE_GRADED_COVERAGE,
) -> None:
    coverage = phase_a_usable_graded_coverage(prompts, state)
    if coverage + 1e-12 < threshold:
        raise StudyCollectorError(
            f"Phase A usable graded coverage {coverage:.4f} is below frozen threshold {threshold:.4f}"
        )


def _phase_preconditions(phase: str) -> tuple[str, ...]:
    if phase == PHASE_A:
        return ("corpus_validated", "no_provider_calls_in_prior_phase")
    if phase == PHASE_B:
        return (
            "phase_A_complete",
            "all_core_train_outcomes_explicitly_graded",
            "frozen_Grok_selection_algorithm_applied_to_core_outcomes",
        )
    if phase == PHASE_C:
        return (
            "phase_A_complete",
            "phase_B_complete",
            "router_training_uses_train_only",
            "validation_calls_are_not_used_for_training",
        )
    if phase == PHASE_D:
        return (
            "phase_C_complete",
            "final_pre_test_configuration_freeze_manifest_verified",
            "untouched_test_is_unlocked_exactly_once",
        )
    raise StudyCollectorError(f"unknown Router V2 phase {phase!r}")


def _phase_rows(
    prompts: tuple[StudyPrompt, ...],
    phase: str,
    *,
    outcome_evidence: dict[str, dict[str, Any]] | None,
) -> tuple[tuple[StudyPrompt, ...], dict[str, Any]]:
    """Return the phase's eligible rows and its frozen model scope."""

    from .study_adapters import collection_eligibility

    rows = tuple(prompts)
    if phase == PHASE_A:
        candidates = tuple(row for row in rows if row.split == "train")
        models = CORE_MODELS
    elif phase == PHASE_B:
        if not outcome_evidence:
            # Use the frozen pre-outcome ordering only as a bounded cost-shape
            # estimate.  The selected IDs and idempotency keys are not
            # materialized until complete Phase A outcomes exist.
            candidates = select_informative_training(rows, GROK_TRAIN_COUNT)
            models = (GROK_MODEL,)
        else:
            selected = select_informative_training(
                rows,
                GROK_TRAIN_COUNT,
                outcome_evidence=outcome_evidence,
            )
            candidates = selected
            models = (GROK_MODEL,)
    elif phase == PHASE_C:
        candidates = tuple(row for row in rows if row.split == "validation")
        models = (*CORE_MODELS, GROK_MODEL)
    elif phase == PHASE_D:
        candidates = tuple(row for row in rows if row.split == "test")
        models = (*CORE_MODELS, GROK_MODEL)
    else:
        raise StudyCollectorError(f"unknown Router V2 phase {phase!r}")
    eligible: list[StudyPrompt] = []
    ineligible: list[dict[str, Any]] = []
    for row in candidates:
        allowed, reason = collection_eligibility(row)
        if allowed:
            eligible.append(row)
        else:
            ineligible.append(
                {
                    "prompt_id": row.prompt_id,
                    "split": row.split,
                    "source": row.source,
                    "task_format": row.task_format,
                    "reason": reason,
                }
            )
    return tuple(eligible), {
        "models": models,
        "candidate_row_count": len(candidates),
        "eligible_row_count": len(eligible),
        "ineligible_rows": ineligible,
    }


def _phase_calls(
    rows: tuple[StudyPrompt, ...],
    models: tuple[str, ...],
    *,
    pricing: PricingTable,
    max_attempts: int,
    existing_call_keys: set[str],
    phase: str,
    reservation_multipliers: Mapping[str, float] | None = None,
) -> tuple[tuple[PlannedStudyCall, ...], dict[str, dict[str, Any]]]:
    from .study_adapters import adapter_for_prompt

    snapshot = {model: _price_snapshot(model, pricing) for model in models}
    multipliers = dict(reservation_multipliers or {})
    for target, value in multipliers.items():
        model, separator, task_format = target.rpartition("@")
        if not separator:
            model, task_format = target, ""
        if model not in models:
            raise StudyCollectorError(
                f"reservation multiplier references model outside {phase}: {model}"
            )
        if task_format and task_format not in FORMAT_TOKEN_BUDGETS:
            raise StudyCollectorError(
                f"reservation multiplier references unknown task format: {task_format}"
            )
        if not math.isfinite(float(value)) or float(value) < 1.0:
            raise StudyCollectorError(
                f"reservation multiplier for {target} must be finite and at least 1"
            )
    calls: list[PlannedStudyCall] = []
    for model in models:
        for row in rows:
            key = f"{phase}:{row.prompt_id}:{model}"
            if key in existing_call_keys:
                continue
            input_tokens = conservative_input_tokens(row.prompt)
            cap = int(row.max_output_tokens or FORMAT_TOKEN_BUDGETS[row.task_format]["max_output_tokens"])
            expected_tokens = min(
                cap,
                int(
                    row.expected_output_tokens
                    or FORMAT_TOKEN_BUDGETS[row.task_format]["expected_output_tokens"]
                ),
            )
            is_tau = row.task_format == "agent_state" or str(row.grader.get("kind") or "").startswith("tau")
            expected_turns = (
                max(1, int(row.grader.get("expected_turns") or DEFAULT_TAU_EXPECTED_TURNS))
                if is_tau
                else 1
            )
            max_turns = (
                max(expected_turns, int(row.grader.get("max_turns") or DEFAULT_TAU_MAX_TURNS))
                if is_tau
                else 1
            )
            attempts = max_attempts if model.endswith(":free") else 1
            potential_api_calls = max_turns * attempts
            expected_trajectory_tokens = expected_turns * expected_tokens
            max_trajectory_tokens = max_turns * cap
            unit = snapshot[model]
            maximum = attempts * (
                max_turns * input_tokens * float(unit["input_per_1m"])
                + max_trajectory_tokens * float(unit["output_per_1m"])
            ) / 1_000_000.0
            multiplier = multipliers.get(
                f"{model}@{row.task_format}",
                multipliers.get(model, 1.0),
            )
            maximum *= float(multiplier)
            expected = (
                expected_turns * input_tokens * float(unit["input_per_1m"])
                + expected_trajectory_tokens * float(unit["output_per_1m"])
            ) / 1_000_000.0
            calls.append(
                PlannedStudyCall(
                    prompt_id=row.prompt_id,
                    model=model,
                    split=row.split,
                    max_output_tokens=cap,
                    maximum_cost_usd=maximum,
                    idempotency_key=key,
                    input_tokens=input_tokens,
                    expected_output_tokens=expected_tokens,
                    expected_cost_usd=expected,
                    task_format=row.task_format,
                    adapter_id=adapter_for_prompt(row).adapter_id,
                    grading_status=(
                        "pending_pairwise"
                        if row.label_type == "pairwise"
                        else "executable_grader"
                    ),
                    expected_turns=expected_turns,
                    max_turns=max_turns,
                    potential_api_calls=potential_api_calls,
                    expected_trajectory_tokens=expected_trajectory_tokens,
                    max_trajectory_tokens=max_trajectory_tokens,
                )
            )
    return tuple(calls), snapshot


def _apply_paid_nemotron_quote(
    snapshot: dict[str, dict[str, Any]], paid_quote: dict[str, Any] | None
) -> dict[str, dict[str, Any]]:
    result = {model: dict(value) for model, value in snapshot.items()}
    if paid_quote is not None and CORE_MODELS[0] in result:
        result[CORE_MODELS[0]] = {
            **result[CORE_MODELS[0]],
            "input_per_1m": float(paid_quote["input_per_1m"]),
            "output_per_1m": float(paid_quote["output_per_1m"]),
            "source": str(paid_quote.get("source") or "pinned_local_quote"),
            "pricing_model_slug": str(
                paid_quote.get("pricing_model_slug") or CORE_MODELS[0].removesuffix(":free")
            ),
        }
    return result


def _phase_cost_scenarios(
    calls: tuple[PlannedStudyCall, ...],
    snapshot: dict[str, dict[str, Any]],
    *,
    max_attempts: int,
    paid_quote: dict[str, Any] | None,
    grading_cost_status: str,
) -> dict[str, dict[str, Any]]:
    scenarios: dict[str, dict[str, Any]] = {}
    for name, quote_snapshot, quote_source in (
        ("free_nemotron", snapshot, "offline PricingTable free Nemotron seed plus pinned local prices"),
        (
            "paid_nemotron",
            _apply_paid_nemotron_quote(snapshot, paid_quote),
            str((paid_quote or {}).get("source") or "missing_pinned_paid_quote"),
        ),
    ):
        if name == "paid_nemotron" and paid_quote is None:
            scenarios[name] = {
                "name": name,
                "status": "missing_pinned_quote",
                "hard_cap_usd": HARD_CAP_USD,
                "test_reserve_usd": TEST_RESERVE_USD,
                "grading_cost_status": grading_cost_status,
            }
            continue
        scenario = _cost_scenario(
            calls,
            quote_snapshot,
            expected_output_tokens=None,
            max_attempts=max_attempts,
            grading_cost_usd=0.0,
            name=name,
            quote_source=quote_source,
        )
        scenario["grading_cost_status"] = grading_cost_status
        scenario["expected_judging_cost_usd"] = None
        scenario["worst_case_judging_cost_usd"] = None
        if grading_cost_status == "unpriced_pairwise_pending":
            scenario["status"] = "blocked_unpriced_grading"
            scenario["within_hard_cap"] = None
            scenario["untouched_test_reserve_preserved"] = None
        scenario["cap_check_status"] = (
            "model_calls_only_pairwise_cost_unpriced"
            if grading_cost_status == "unpriced_pairwise_pending"
            else "model_and_local_grading_priced"
        )
        scenarios[name] = scenario
    return scenarios


def build_phase_plan(
    prompts: tuple[StudyPrompt, ...],
    *,
    phase: str,
    pricing: PricingTable | None = None,
    max_attempts: int = 1,
    existing_call_keys: set[str] | None = None,
    outcome_evidence: dict[str, dict[str, Any]] | None = None,
    paid_nemotron_quote: dict[str, Any] | None = None,
    state: Mapping[str, Any] | None = None,
    final_freeze_manifest: Mapping[str, Any] | None = None,
    materialize_calls: bool = True,
    reservation_multipliers: Mapping[str, float] | None = None,
) -> StudyPlan:
    """Build one phase only; locked phases contain no executable calls."""

    if phase not in STUDY_PHASES:
        raise StudyCollectorError(f"unknown Router V2 phase {phase!r}")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    pricing = pricing or PricingTable(use_live=False)
    existing = existing_call_keys or set()
    rows, metadata = _phase_rows(prompts, phase, outcome_evidence=outcome_evidence)
    status = "ready"
    gate_state = dict((state or {}).get("phase_status") or {})
    if phase == PHASE_B and not outcome_evidence:
        status = "locked_pending_phase_A_outcomes"
    elif phase == PHASE_C and not all(
        gate_state.get(required) == "complete" for required in (PHASE_A, PHASE_B)
    ):
        status = "locked_pending_phase_A_and_B"
    elif phase == PHASE_D and not (
        gate_state.get(PHASE_C) == "complete"
        and isinstance(final_freeze_manifest, Mapping)
        and final_freeze_manifest.get("status") == "ready_for_test"
        and str(final_freeze_manifest.get("manifest_hash") or "")
    ):
        status = "locked_pending_final_pretest_freeze"
    if phase == PHASE_A and not materialize_calls:
        status = "locked_pending_provider_approval"
    all_calls, snapshot = _phase_calls(
        rows,
        tuple(metadata["models"]),
        pricing=pricing,
        max_attempts=max_attempts,
        existing_call_keys=existing,
        phase=phase,
        reservation_multipliers=reservation_multipliers,
    )
    calls = all_calls if status == "ready" and materialize_calls else ()
    grading_cost_status = (
        "unpriced_pairwise_pending"
        if any(
            row.label_type == "pairwise"
            and row.grader.get("judging_cost_status") == "postponed_unpriced"
            for row in rows
        )
        else "local_deterministic_only"
    )
    scenarios = _phase_cost_scenarios(
        all_calls,
        snapshot,
        max_attempts=max_attempts,
        paid_quote=paid_nemotron_quote,
        grading_cost_status=grading_cost_status,
    )
    selection = (
        grok_selection_metadata(prompts, outcome_evidence=outcome_evidence)
        if phase in {PHASE_A, PHASE_B}
        else {"mode": "not_used_in_this_phase"}
    )
    if phase == PHASE_A:
        selection = {
            **selection,
            "prompt_ids": [],
            "selected_ids": [],
            "selected_metrics": {},
            "selected_ids_status": "unresolved_until_complete_phase_A_outcomes",
            "selection_evidence_status": "algorithm_frozen_only_before_phase_A",
        }
    if phase == PHASE_B and not outcome_evidence:
        selection = {
            **selection,
            "prompt_ids": [],
            "selected_ids": [],
            "selected_metrics": {},
            "selected_ids_status": "unresolved_until_complete_phase_A_outcomes",
            "cost_estimate_basis": "frozen_pre_outcome_proxy_ordering_only",
        }
    generated_at = datetime.now(timezone.utc).isoformat()
    draft = StudyPlan(
        calls=calls,
        price_snapshot=snapshot,
        corpus_hash=corpus_fingerprint(prompts),
        max_attempts=max_attempts,
        generated_at=generated_at,
        grok_selection=selection,
        cost_scenarios=scenarios,
        phase=phase,
        status=status,
        preconditions=_phase_preconditions(phase),
        potential_call_count=sum(call.potential_api_calls for call in all_calls),
        ineligible_rows=tuple(metadata["ineligible_rows"]),
        grading_cost_status=grading_cost_status,
        planned_call_count=len(all_calls),
        planned_potential_api_call_count=sum(call.potential_api_calls for call in all_calls),
        planned_maximum_cost_usd=sum(call.maximum_cost_usd for call in all_calls),
        planned_expected_cost_usd=sum(call.expected_cost_usd for call in all_calls),
        planned_calls=all_calls,
    )
    return StudyPlan(
        **{**draft.__dict__, "plan_hash": _canonical_hash(draft._payload())}
    )


@dataclass(frozen=True)
class PhasedStudyPlan:
    phases: dict[str, StudyPlan]
    corpus_hash: str
    generated_at: str
    grok_algorithm: dict[str, Any]
    plan_hash: str = ""

    def _payload(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "corpus_hash": self.corpus_hash,
            "generated_at": self.generated_at,
            "grok_algorithm": self.grok_algorithm,
            "phases": {name: plan.to_dict() for name, plan in self.phases.items()},
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self._payload()
        payload["plan_hash"] = self.plan_hash
        payload["provider_calls_materialized"] = sum(
            plan.call_count for plan in self.phases.values()
        )
        payload["planned_provider_call_count"] = sum(
            plan.planned_call_count for plan in self.phases.values()
        )
        payload["task_model_runs"] = sum(
            plan.task_model_runs for plan in self.phases.values()
        )
        payload["planned_task_model_runs"] = sum(
            plan.planned_call_count for plan in self.phases.values()
        )
        payload["potential_api_call_count"] = sum(
            plan.potential_api_call_count for plan in self.phases.values()
        )
        payload["potential_call_count"] = sum(
            int(plan.potential_call_count or plan.potential_api_call_count)
            for plan in self.phases.values()
        )
        payload["planned_potential_api_call_count"] = sum(
            plan.planned_potential_api_call_count for plan in self.phases.values()
        )
        return payload


def build_phased_study_plan(
    prompts: tuple[StudyPrompt, ...],
    *,
    pricing: PricingTable | None = None,
    max_attempts: int = 1,
    paid_nemotron_quote: dict[str, Any] | None = None,
    outcome_evidence: dict[str, dict[str, Any]] | None = None,
    state: Mapping[str, Any] | None = None,
    final_freeze_manifest: Mapping[str, Any] | None = None,
    materialize_phase_a: bool = False,
    phase_reservation_multipliers: Mapping[str, Mapping[str, float]] | None = None,
) -> PhasedStudyPlan:
    """Create four separately gated plans, never one all-split call list."""

    multiplier_by_phase = dict(phase_reservation_multipliers or {})
    unknown_phases = set(multiplier_by_phase) - set(STUDY_PHASES)
    if unknown_phases:
        raise StudyCollectorError(
            "reservation multipliers reference unknown phases: "
            + ", ".join(sorted(unknown_phases))
        )
    phases = {
        phase: build_phase_plan(
            prompts,
            phase=phase,
            pricing=pricing,
            max_attempts=max_attempts,
            paid_nemotron_quote=paid_nemotron_quote,
            outcome_evidence=outcome_evidence if phase == PHASE_B else None,
            state=state,
            final_freeze_manifest=final_freeze_manifest,
            materialize_calls=(materialize_phase_a or phase != PHASE_A),
            reservation_multipliers=multiplier_by_phase.get(phase),
        )
        for phase in STUDY_PHASES
    }
    grok_algorithm = grok_selection_metadata(prompts)
    if grok_algorithm.get("mode") == "pre_outcome_proxy":
        grok_algorithm = {
            **grok_algorithm,
            "prompt_ids": [],
            "selected_ids": [],
            "selected_metrics": {},
            "selected_ids_status": "unresolved_until_complete_phase_A_outcomes",
        }
    draft = PhasedStudyPlan(
        phases=phases,
        corpus_hash=corpus_fingerprint(prompts),
        generated_at=datetime.now(timezone.utc).isoformat(),
        grok_algorithm=grok_algorithm,
    )
    return PhasedStudyPlan(
        **{**draft.__dict__, "plan_hash": _canonical_hash(draft._payload())}
    )


def write_phased_plan(plan: PhasedStudyPlan, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(plan.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_phased_plan(path: str | Path) -> PhasedStudyPlan:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if raw.get("schema_version") != 1 or not isinstance(raw.get("phases"), Mapping):
        raise StudyCollectorError("unsupported phased study plan schema")
    phase_plans: dict[str, StudyPlan] = {}
    for name, value in raw["phases"].items():
        if not isinstance(value, Mapping):
            raise StudyCollectorError(f"phased plan {name} is not an object")
        calls = tuple(_planned_call_from_dict(item) for item in value.get("calls", []))
        planned_calls = tuple(
            _planned_call_from_dict(item)
            for item in value.get("planned_calls", value.get("calls", []))
        )
        plan = StudyPlan(
            calls=calls,
            price_snapshot=dict(value.get("price_snapshot") or {}),
            corpus_hash=str(value.get("corpus_hash") or ""),
            max_attempts=int(value.get("max_attempts", 1)),
            generated_at=str(value.get("generated_at") or ""),
            plan_hash=str(value.get("plan_hash") or ""),
            grok_selection=dict(value.get("grok_selection") or {}),
            cost_scenarios=dict(value.get("cost_scenarios") or {}),
            phase=str(value.get("phase") or name),
            status=str(value.get("status") or "ready"),
            preconditions=tuple(str(item) for item in value.get("preconditions", [])),
            potential_call_count=(
                int(value["potential_call_count"])
                if value.get("potential_call_count") is not None
                else None
            ),
            ineligible_rows=tuple(value.get("ineligible_rows") or ()),
            grading_cost_status=str(
                value.get("grading_cost_status") or "local_deterministic_only"
            ),
            planned_call_count=int(value.get("planned_call_count") or len(planned_calls)),
            planned_potential_api_call_count=int(
                value.get("planned_potential_api_call_count")
                or sum(max(1, call.potential_api_calls) for call in planned_calls)
            ),
            planned_maximum_cost_usd=float(
                value.get("planned_maximum_cost_usd")
                or sum(call.maximum_cost_usd for call in planned_calls)
            ),
            planned_expected_cost_usd=float(
                value.get("planned_expected_cost_usd")
                or sum(call.expected_cost_usd for call in planned_calls)
            ),
            planned_calls=planned_calls,
        )
        if plan.plan_hash != _canonical_hash(plan._payload()):
            raise StudyCollectorError(f"phase {name} plan hash does not match its contents")
        phase_plans[str(name)] = plan
    result = PhasedStudyPlan(
        phases=phase_plans,
        corpus_hash=str(raw.get("corpus_hash") or ""),
        generated_at=str(raw.get("generated_at") or ""),
        grok_algorithm=dict(raw.get("grok_algorithm") or {}),
        plan_hash=str(raw.get("plan_hash") or ""),
    )
    if result.plan_hash != _canonical_hash(result._payload()):
        raise StudyCollectorError("phased study plan hash does not match its contents")
    return result


def assert_phase_gate(
    plan: StudyPlan,
    state: Mapping[str, Any],
    *,
    final_freeze_manifest: Mapping[str, Any] | None = None,
    prompts: tuple[StudyPrompt, ...] | None = None,
) -> None:
    """Hard gate provider execution by phase and persisted prior results."""

    if plan.phase not in STUDY_PHASES:
        raise StudyCollectorError("legacy all-split plans cannot be executed")
    if plan.status != "ready":
        raise StudyCollectorError(f"{plan.phase} is locked: {plan.status}")
    phase_status = state.get("phase_status") or {}
    if plan.phase == PHASE_B and phase_status.get(PHASE_A) != "complete":
        raise StudyCollectorError("Phase B requires completed Phase A outcomes")
    if plan.phase == PHASE_B and prompts is not None:
        assert_phase_a_coverage(prompts, state)
    if plan.phase == PHASE_C and any(
        phase_status.get(name) != "complete" for name in (PHASE_A, PHASE_B)
    ):
        raise StudyCollectorError("Phase C requires completed Phase A and Phase B")
    if plan.phase == PHASE_D:
        if phase_status.get(PHASE_C) != "complete":
            raise StudyCollectorError("Phase D requires completed Phase C validation")
        if not isinstance(final_freeze_manifest, Mapping) or final_freeze_manifest.get("status") != "ready_for_test":
            raise StudyCollectorError("Phase D requires a verified final pre-test freeze manifest")
    allowed_split = {
        PHASE_A: "train",
        PHASE_B: "train",
        PHASE_C: "validation",
        PHASE_D: "test",
    }[plan.phase]
    if any(call.split != allowed_split for call in plan.calls):
        raise StudyCollectorError(f"{plan.phase} contains a call outside {allowed_split}")



def _safe_error(exc: Exception) -> str:
    text = str(exc)
    for marker in ("Bearer ", "OPENROUTER_API_KEY=", "OPENAI_API_KEY="):
        if marker in text:
            text = text.split(marker, 1)[0] + "[REDACTED_SECRET]"
    return f"{type(exc).__name__}: {text[:500]}"


def _atomic_write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _output_preview(output: str) -> str:
    return output[:500]


def _output_hash(output: str) -> str:
    return hashlib.sha256(output.encode("utf-8")).hexdigest()


def _run_is_completed(run: Mapping[str, Any]) -> bool:
    """Return true for any settled, persisted terminal collection result."""

    return (
        run.get("status") not in {"reserved", "error"}
        and run.get("completed_at") is not None
        and run.get("realized_cost_usd") is not None
    )


@dataclass
class StudyCollector:
    prompts: tuple[StudyPrompt, ...]
    plan: StudyPlan
    ledger: StudyBudgetLedger
    state_path: Path
    provider_factory: Callable[[], ProviderProtocol] | None = None
    pricing: PricingTable = field(default_factory=lambda: PricingTable(use_live=False))
    adapter_context: dict[str, Any] = field(default_factory=dict)
    pre_provider_verifier: Callable[[], None] | None = None

    def _load_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "schema_version": STATE_SCHEMA_VERSION,
                "plan_hash": self.plan.plan_hash,
                "corpus_hash": self.plan.corpus_hash,
                "evaluator_version": EVALUATOR_VERSION,
                "phase_status": {},
                "rows": {},
            }
        raw = json.loads(self.state_path.read_text(encoding="utf-8"))
        if (
            raw.get("schema_version") != STATE_SCHEMA_VERSION
            or raw.get("plan_hash") != self.plan.plan_hash
            or raw.get("corpus_hash") != self.plan.corpus_hash
        ):
            raise StudyCollectorError("existing study state belongs to a different plan or corpus")
        if not isinstance(raw.get("rows"), dict):
            raise StudyCollectorError("study state rows must be an object")
        if not isinstance(raw.get("phase_status", {}), dict):
            raise StudyCollectorError("study state phase_status must be an object")
        return raw

    def _save_state(self, state: dict[str, Any]) -> None:
        _atomic_write_json(self.state_path, state)

    def _provider(self) -> ProviderProtocol:
        if self.provider_factory is not None:
            return self.provider_factory()
        from theriad.core.providers.factory import build_provider

        return build_provider("openrouter")

    def _prompt_by_id(self) -> dict[str, StudyPrompt]:
        return {row.prompt_id: row for row in self.prompts}

    def _assert_plan_matches_corpus(self) -> None:
        expected = corpus_fingerprint(self.prompts)
        if expected != self.plan.corpus_hash:
            raise StudyCollectorError("study plan was generated from a different corpus")

    def _assert_price_snapshot(self) -> None:
        """Reject execution when the reviewed offline quote has drifted."""

        if self.adapter_context.get("enforce_price_snapshot") is not True:
            return
        for model, snapshot in self.plan.price_snapshot.items():
            current = _price_snapshot(model, self.pricing)
            for key in ("input_per_1m", "output_per_1m"):
                if abs(float(current[key]) - float(snapshot[key])) > 1e-12:
                    raise StudyCollectorError(
                        f"price snapshot drift for {model} {key}: "
                        f"reviewed={snapshot[key]} current={current[key]}"
                    )

    def summary(self, state: dict[str, Any] | None = None) -> dict[str, Any]:
        state = state or self._load_state()
        runs = [
            run
            for row in state["rows"].values()
            for run in (row.get("runs") or {}).values()
        ]
        outcomes = [
            outcome
            for row in state["rows"].values()
            for outcome in (row.get("outcomes") or {}).values()
        ]
        settled_runs = [
            run
            for run in runs
            if run.get("realized_cost_usd") is not None
        ]
        provider_api_invocations = sum(
            max(0, int(run.get("api_call_count") or 0)) for run in runs
        )
        trajectory_turns = sum(
            max(0, int(run.get("api_call_count") or 0))
            for run in runs
            if run.get("adapter_id") == "router-v2-tau-bench"
        )
        accounting = self.ledger.accounting()
        return {
            "plan_hash": self.plan.plan_hash,
            "planned_calls": self.plan.call_count,
            "maximum_cost_usd": self.plan.maximum_cost_usd,
            "runs": len(runs),
            "task_model_runs": len(runs),
            "provider_api_invocations": provider_api_invocations,
            "trajectory_turns": trajectory_turns,
            "completed_runs": sum(
                _run_is_completed(run) for run in runs
            ),
            "graded_outcomes": len(outcomes),
            "completed_graded_outcomes": len(outcomes),
            "actual_cost_usd": sum(
                float(run.get("realized_cost_usd") or 0.0) for run in settled_runs
            ),
            "reserved_cost_usd": accounting["reserved_cost_usd"],
            "outstanding_reserved_cost_usd": accounting["outstanding_reserved_cost_usd"],
            "realized_provider_cost_usd": accounting["realized_provider_cost_usd"],
            "ledger": self.ledger.status(),
        }

    def preflight_all_calls(
        self,
        provider: ProviderProtocol,
        *,
        state: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Check every task/model adapter before the first reservation.

        This is intentionally a separate pass.  A later missing BFCL checker,
        tau environment, code runtime, or provider capability must not be
        discovered after an earlier call has already reserved budget.
        """

        self._assert_plan_matches_corpus()
        self._assert_price_snapshot()
        current_state = state if state is not None else self._load_state()
        assert_phase_gate(
            self.plan,
            current_state,
            final_freeze_manifest=current_state.get("final_freeze_manifest"),
            prompts=self.prompts,
        )
        from .study_adapters import runtime_collection_eligibility

        prompts = self._prompt_by_id()
        errors: list[dict[str, str]] = []
        seen_keys: set[str] = set()
        allowed_split = {
            PHASE_A: "train",
            PHASE_B: "train",
            PHASE_C: "validation",
            PHASE_D: "test",
        }.get(self.plan.phase)
        for call in self.plan.calls:
            prompt = prompts.get(call.prompt_id)
            if prompt is None:
                errors.append({"prompt_id": call.prompt_id, "reason": "missing prompt"})
                continue
            if allowed_split is not None and call.split != allowed_split:
                errors.append(
                    {
                        "prompt_id": call.prompt_id,
                        "reason": f"call split {call.split!r} is outside {allowed_split!r}",
                    }
                )
            if call.idempotency_key in seen_keys:
                errors.append(
                    {
                        "prompt_id": call.prompt_id,
                        "reason": f"duplicate idempotency key {call.idempotency_key!r}",
                    }
                )
            seen_keys.add(call.idempotency_key)
            eligible, reason = runtime_collection_eligibility(
                provider,
                prompt,
                adapter_context=self.adapter_context,
            )
            if not eligible:
                errors.append({"prompt_id": call.prompt_id, "reason": reason})
        if errors:
            preview = "; ".join(
                f"{item['prompt_id']}: {item['reason']}" for item in errors[:8]
            )
            raise StudyCollectorError(
                "all-call preflight failed before any reservation/provider call: "
                f"{len(errors)} issue(s); {preview}"
            )
        return {
            "status": "passed",
            "call_count": len(self.plan.calls),
            "idempotency_key_count": len(seen_keys),
            "provider_calls": 0,
            "reservation_count": 0,
        }

    def execute(
        self,
        *,
        reviewed_plan_hash: str,
        allow_provider_calls: bool,
        allow_paid_calls: bool,
        yes: bool,
        retry_errors: bool = False,
        retry_error_prompt_ids: frozenset[str] | None = None,
        selected_models: frozenset[str] | None = None,
    ) -> dict[str, Any]:
        """Execute only after an explicit, hash-bound plan acknowledgement."""

        retry_allowlist = frozenset(retry_error_prompt_ids or ())
        model_allowlist = frozenset(selected_models or ())
        if retry_errors and not retry_allowlist:
            raise StudyCollectorError(
                "error retries require an explicit prompt-id allowlist"
            )
        planned_models = {call.model for call in self.plan.calls}
        unknown_models = model_allowlist - planned_models
        if unknown_models:
            raise StudyCollectorError(
                "selected models are absent from the reviewed plan: "
                + ", ".join(sorted(unknown_models))
            )

        self._assert_plan_matches_corpus()
        if not allow_provider_calls or not yes:
            raise StudyCollectorError(
                "provider calls require --execute plus --allow-provider-calls and --yes"
            )
        if reviewed_plan_hash != self.plan.plan_hash:
            raise StudyCollectorError("reviewed plan hash does not match the plan on disk")
        if self.plan.maximum_cost_usd > HARD_CAP_USD + 1e-12:
            raise StudyCollectorError(
                f"planned worst-case cost ${self.plan.maximum_cost_usd:.6f} exceeds the $25 cap"
            )
        for name, scenario in self.plan.cost_scenarios.items():
            if scenario.get("status") != "priced":
                continue
            if not scenario.get("within_hard_cap"):
                raise StudyCollectorError(
                    f"{name} worst-case cost exceeds the $25 cap"
                )
            if not scenario.get("untouched_test_reserve_preserved"):
                raise StudyCollectorError(
                    f"{name} would consume the $10 untouched-test reserve"
                )
        paid_maximum = sum(
            call.maximum_cost_usd
            for call in self.plan.calls
            if any(
                float(self.plan.price_snapshot[call.model][key]) > 0
                for key in ("input_per_1m", "output_per_1m")
            )
        )
        if paid_maximum > 0 and not allow_paid_calls:
            raise StudyCollectorError("paid calls require --allow-paid-calls after plan review")
        if self.plan.grading_cost_status == "unpriced_pairwise_pending":
            raise StudyCollectorError(
                "pairwise judging cost is unpriced; pending rows cannot be represented as zero-cost execution"
            )

        state = self._load_state()
        assert_phase_gate(
            self.plan,
            state,
            final_freeze_manifest=state.get("final_freeze_manifest"),
            prompts=self.prompts,
        )
        prompts = self._prompt_by_id()
        if self.pre_provider_verifier is not None:
            self.pre_provider_verifier()
        provider = _FreeEndpointRetryProvider(
            self._provider(), max_attempts=self.plan.max_attempts
        )
        self.preflight_all_calls(provider, state=state)
        for call in self.plan.calls:
            if model_allowlist and call.model not in model_allowlist:
                continue
            prompt = prompts.get(call.prompt_id)
            if prompt is None:
                raise StudyCollectorError(f"plan references missing prompt {call.prompt_id}")
            from .study_adapters import (
                BFCLAdapter,
                adapter_for_prompt,
                runtime_collection_eligibility,
            )

            eligible, reason = runtime_collection_eligibility(
                provider, prompt, adapter_context=self.adapter_context
            )
            if not eligible:
                row = state["rows"].setdefault(
                    call.prompt_id,
                    {**prompt.to_dict(), "outcomes": {}, "runs": {}},
                )
                row.setdefault("runs", {})[call.model] = {
                    "status": "ineligible_before_reservation",
                    "requested_model": call.model,
                    "provider": str(getattr(provider, "name", "unknown")),
                    "reservation_id": None,
                    "reason": reason,
                    "provider_call_blocked": True,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                self._save_state(state)
                raise StudyCollectorError(
                    f"{prompt.prompt_id} is not executable by its task-family adapter before reservation: {reason}"
                )
            row = state["rows"].setdefault(
                call.prompt_id,
                {**prompt.to_dict(), "outcomes": {}, "runs": {}},
            )
            runs = row.setdefault("runs", {})
            existing = runs.get(call.model)
            retry_count = 0
            attempt_history: list[dict[str, Any]] = []
            reservation_key = call.idempotency_key
            if existing and _run_is_completed(existing):
                continue
            if existing and existing.get("status") == "reserved":
                raise StudyCollectorError(
                    f"unresolved reservation for {call.prompt_id}/{call.model}; refusing a duplicate call"
                )
            if existing and existing.get("status") == "error":
                reservation_id = existing.get("reservation_id")
                reservation = (
                    self.ledger.reservation(str(reservation_id)) if reservation_id else None
                )
                if reservation is not None and reservation.get("actual_cost_usd") is None:
                    raise StudyCollectorError(
                        f"ambiguous provider error for {call.prompt_id}/{call.model}; "
                        "reconcile the reservation before retrying"
                    )
                if not retry_errors or call.prompt_id not in retry_allowlist:
                    continue
                retry_count = int(existing.get("retry_count") or 0) + 1
                attempt_history = list(existing.get("attempt_history") or ())
                attempt_history.append(
                    {
                        key: existing.get(key)
                        for key in (
                            "status",
                            "error",
                            "reservation_id",
                            "api_call_count",
                            "completed_at",
                        )
                    }
                )
                reservation_key = f"{call.idempotency_key}:retry:{retry_count}"

            reservation_id = self.ledger.reserve(
                phase=call.split,
                model=call.model,
                maximum_cost_usd=call.maximum_cost_usd,
                idempotency_key=reservation_key,
                potential_api_calls=call.potential_api_calls,
                expected_turns=call.expected_turns,
                max_turns=call.max_turns,
                expected_trajectory_tokens=call.expected_trajectory_tokens,
                max_trajectory_tokens=call.max_trajectory_tokens,
            )
            reservation = self.ledger.reservation(reservation_id)
            if reservation is None:
                raise StudyCollectorError("budget ledger lost a new reservation")
            if reservation.get("actual_cost_usd") is not None:
                raise StudyCollectorError(
                    f"settled reservation {reservation_id} has no persisted result; manual reconciliation required"
                )
            runs[call.model] = {
                "status": "reserved",
                "reservation_id": reservation_id,
                "requested_model": call.model,
                "provider": str(getattr(provider, "name", "unknown")),
                "retry_fallback_status": "not_started",
                "reserved_maximum_cost_usd": call.maximum_cost_usd,
                "expected_turns": call.expected_turns,
                "max_turns": call.max_turns,
                "potential_api_calls": call.potential_api_calls,
                "expected_trajectory_tokens": call.expected_trajectory_tokens,
                "max_trajectory_tokens": call.max_trajectory_tokens,
                "retry_count": retry_count,
                "attempt_history": attempt_history,
                "started_at": datetime.now(timezone.utc).isoformat(),
            }
            self._save_state(state)
            started = time.monotonic()
            api_invocations_before = provider.api_invocations
            observed_realized_cost_usd: float | None = None
            try:
                from .study_adapters import (
                    AdapterResult,
                    PlainTextAdapter,
                    TauBenchAdapter,
                    adapter_for_prompt,
                )

                adapter = adapter_for_prompt(prompt)
                if isinstance(adapter, BFCLAdapter) and "official_checker" in self.adapter_context:
                    adapter = BFCLAdapter(
                        official_checker=self.adapter_context["official_checker"]
                    )
                if isinstance(adapter, PlainTextAdapter):
                    response = asyncio.run(
                        provider.complete(
                            CompletionRequest(
                                model=call.model,
                                messages=[Message(role="user", content=prompt.prompt)],
                                max_tokens=call.max_output_tokens,
                                temperature=0.0,
                            )
                        )
                    )
                    adapter_result: AdapterResult | None = None
                else:
                    if isinstance(adapter, TauBenchAdapter):
                        context = {
                            key: value
                            for key, value in self.adapter_context.items()
                            if key in {"environment_factory", "resume", "persist"}
                        }
                        adapter_result = adapter.collect(
                            provider, prompt, model=call.model, **context
                        )
                    else:
                        # BFCL and executable-code adapters have no tau
                        # environment hooks.  Do not leak unrelated context
                        # kwargs into their strict collection interfaces.
                        adapter_result = adapter.collect(
                            provider, prompt, model=call.model
                        )
                    if not adapter_result.eligible:
                        raise StudyCollectorError(adapter_result.details)
                    response = adapter_result
                latency_ms = float(getattr(response, "latency_ms", 0.0) or 0.0)
                if latency_ms <= 0:
                    latency_ms = (time.monotonic() - started) * 1000.0
                cost_response = response
                if isinstance(response, AdapterResult):
                    from theriad.core.providers.interfaces import CompletionResponse

                    cost_response = CompletionResponse(
                        model=call.model,
                        content=response.output,
                        prompt_tokens=response.prompt_tokens,
                        completion_tokens=response.completion_tokens,
                        latency_ms=latency_ms,
                        provider=response.provider or "task-adapter",
                        raw=response.raw,
                        cost_usd=response.cost_usd,
                    )
                actual = provider_reported_cost(cost_response)
                if actual is None:
                    actual = realized_cost(cost_response, self.pricing)
                if actual is None or not math.isfinite(actual) or actual < 0:
                    raise StudyCollectorError(
                        f"realized cost is unavailable for {call.model}; refusing to continue"
                    )
                observed_realized_cost_usd = float(actual)
                if actual > call.maximum_cost_usd + 1e-12:
                    raise StudyBudgetExceeded(
                        f"realized cost for {call.prompt_id}/{call.model} exceeded its reservation"
                    )
                finish_reason = getattr(response, "finish_reason", None)
                output = str(
                    getattr(response, "content", None)
                    or getattr(response, "output", "")
                    or ""
                )
                if finish_reason in {"length", "max_tokens", "MAX_TOKENS"}:
                    grade = None
                    status = "truncated"
                else:
                    if adapter_result is not None and adapter_result.grade is not None:
                        from .study_grading import GradeResult

                        status = adapter_result.status
                        grade = GradeResult(
                            bool(adapter_result.grade),
                            "graded",
                            adapter_result.adapter_id,
                            details=adapter_result.details,
                            deterministic_label=bool(adapter_result.grade),
                        )
                    elif adapter_result is not None:
                        grade = None
                        status = adapter_result.status
                    else:
                        grade = grade_response(prompt, output)
                        status = grade.status
                if grade is None or grade.label is None:
                    run_status = status
                elif adapter_result is not None:
                    run_status = "success" if grade.label else "failure"
                else:
                    run_status = "success"
                self.ledger.settle(reservation_id, actual)
                run = {
                    "status": run_status,
                    "grade_status": status,
                    "reservation_id": reservation_id,
                    "requested_model": call.model,
                    "provider_model": str(getattr(response, "model", call.model)),
                    "provider": str(getattr(response, "provider", "openrouter")),
                    "prompt_tokens": max(0, int(getattr(response, "prompt_tokens", 0) or 0)),
                    "completion_tokens": max(0, int(getattr(response, "completion_tokens", 0) or 0)),
                    "realized_cost_usd": float(actual),
                    "latency_ms": latency_ms,
                    "finish_reason": finish_reason,
                    "evaluator_version": grade.evaluator_version if grade else EVALUATOR_VERSION,
                    "retry_fallback_status": (
                        "same_free_endpoint_retried"
                        if provider.api_invocations - api_invocations_before
                        > (adapter_result.api_call_count if adapter_result else 1)
                        else "not_attempted"
                    ),
                    "grade": grade.to_dict() if grade else None,
                    "adapter_id": adapter_result.adapter_id if adapter_result else "plain-text",
                    "adapter_status": adapter_result.status if adapter_result else None,
                    "adapter_evidence": adapter_result.evidence if adapter_result else {},
                    "environment_state": adapter_result.environment_state if adapter_result else None,
                    "tool_calls": (
                        [call_value.to_dict() for call_value in adapter_result.tool_calls]
                        if adapter_result
                        else []
                    ),
                    "api_call_count": provider.api_invocations - api_invocations_before,
                    "output_preview": _output_preview(output),
                    "output_sha256": _output_hash(output),
                    "retry_count": retry_count,
                    "attempt_history": attempt_history,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                runs[call.model] = run
                if grade is not None and grade.label is not None:
                    row.setdefault("outcomes", {})[call.model] = {
                        "success": float(grade.label),
                        "cost_usd": float(actual),
                        "output_tokens": run["completion_tokens"],
                        "latency_ms": latency_ms,
                    }
                self._save_state(state)
            except Exception as exc:
                # Keep the reservation unsettled on an ambiguous provider error.
                # Releasing it would make a potentially billed request disappear
                # from the hard-cap accounting.  The persisted status makes the
                # next run stop rather than duplicate it silently.
                error_run = {
                    **runs[call.model],
                    "status": "error",
                    "error": _safe_error(exc),
                    "api_call_count": provider.api_invocations - api_invocations_before,
                    "retry_fallback_status": "not_retried",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                if observed_realized_cost_usd is not None:
                    # Preserve the provider-reported amount even when the hard
                    # reservation gate correctly refuses to settle it.  This
                    # makes a later, explicitly reviewed envelope migration
                    # exact instead of forcing a guessed accounting hold.
                    error_run["observed_realized_cost_usd"] = (
                        observed_realized_cost_usd
                    )
                runs[call.model] = error_run
                self._save_state(state)
                raise
        if model_allowlist:
            summary = self.summary(state)
            summary["execution_scope"] = "selected_models"
            summary["selected_models"] = sorted(model_allowlist)
            summary["phase_completion_deferred"] = True
            return summary
        if self.plan.phase == PHASE_A:
            assert_phase_a_coverage(self.prompts, state)
        if self.plan.phase in STUDY_PHASES:
            state.setdefault("phase_status", {})[self.plan.phase] = "complete"
            self._save_state(state)
        return self.summary(state)
