"""Pure, offline reservation proposal derived from frozen episode study inputs.

These are enforceable worst-case reservations, not forecasts of realized spend.
Producing a proposal never authorizes provider calls or changes a study ledger.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict
from decimal import ROUND_CEILING, Decimal
from typing import Any

from .episode_study import Episode, digest, validate_episodes

STAGE_SPLITS = {
    "pilot": ("pilot",),
    "development": ("train", "calibration", "threshold_selection", "validation"),
    "test": ("test",),
    "smoke": ("smoke",),
}
DETERMINISTIC_GRADERS = frozenset({"numeric", "exact", "json", "code", "workspace", "state", "contains", "evidence"})


def _amount(value: Any, label: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    result = Decimal(str(value))
    if not result.is_finite() or result < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return result


def _cents(value: Decimal) -> float:
    return float(value.quantize(Decimal("0.01"), rounding=ROUND_CEILING))


def propose_episode_budget(
    rows: Sequence[Episode],
    catalog: Mapping[str, Mapping[str, Any]],
    baseline_plans: Mapping[str, Mapping[str, Any]],
    eligibility_by_episode: Mapping[str, Mapping[str, Any]],
    *,
    alternatives: Sequence[str] | None = None,
    contingency_fraction: float = 0.20,
) -> dict[str, Any]:
    """Price each actual configuration using its frozen per-call token caps.

    Pilot evaluates baseline plus the first eligible alternative. Development
    evaluates baseline plus every eligible alternative. Test and smoke reserve
    two configurations, using the highest eligible cost for the as-yet-unlocked
    router. Baseline fallback chains reserve the most expensive possible model
    on every bounded call, matching the runtime's maximum-cost calculation.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("budget proposal needs actual episode rows")
    manifest = validate_episodes(rows, require_all_families=False)
    contingency = _amount(contingency_fraction, "contingency_fraction")
    candidates = sorted(catalog if alternatives is None else alternatives)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("alternatives must be nonempty and unique")
    if any(model not in catalog for model in candidates):
        raise ValueError("alternative absent from frozen catalog")

    def model_bound(row: Episode, model: str) -> Decimal:
        if model not in catalog:
            raise ValueError(f"missing frozen price: {model}")
        quote = catalog[model]
        input_price = _amount(quote.get("reservation_input_per_1m", quote.get("input_per_1m")), f"input price for {model}")
        output_price = _amount(quote.get("reservation_output_per_1m", quote.get("output_per_1m")), f"output price for {model}")
        if (_amount(quote.get("request_cost_usd", 0), "request charge")
                or _amount(quote.get("image_cost_usd", 0), "image charge")):
            raise ValueError("per-request/image charges require a bounded reservation adapter")
        return row.max_calls * (row.max_input_tokens * input_price + row.max_output_tokens * output_price) / Decimal(1_000_000)

    stages: dict[str, Any] = {}
    for stage, splits in STAGE_SPLITS.items():
        members = [row for row in rows if row.split in splits]
        configurations: list[dict[str, Any]] = []
        stage_cost = Decimal(0)
        for row in members:
            if row.grader.get("kind") not in DETERMINISTIC_GRADERS:
                raise ValueError("unsupported grading costs; only frozen deterministic graders are budgeted")
            plan = baseline_plans.get(row.episode_id)
            if not plan:
                raise ValueError(f"missing baseline plan: {row.episode_id}")
            baseline_models = ([str(item[1]) for item in plan["chain"]] if plan.get("chain")
                               else list(plan.get("models", {}).values()))
            if not baseline_models:
                raise ValueError(f"empty baseline plan: {row.episode_id}")
            eligibility = eligibility_by_episode.get(row.episode_id)
            if eligibility is None or any(model not in eligibility for model in candidates):
                raise ValueError(f"incomplete frozen eligibility: {row.episode_id}")
            eligible = []
            for model in candidates:
                record = eligibility[model]
                value = record.get("eligible") if isinstance(record, Mapping) else record
                if type(value) is not bool:
                    raise ValueError("eligibility must explicitly be true or false")
                if value:
                    eligible.append(model)
            if not eligible:
                raise ValueError(f"no eligible alternative: {row.episode_id}")
            options = [("baseline", sorted(set(baseline_models)))]
            if stage == "development":
                options += [(model, [model]) for model in eligible]
            else:
                options.append(("challenger", eligible[:1] if stage == "pilot" else eligible))
            for configuration, models in options:
                bound = max(model_bound(row, model) for model in models)
                stage_cost += bound
                configurations.append({
                    "episode_id": row.episode_id, "group_id": row.group_id,
                    "family": row.family, "split": row.split,
                    "configuration": configuration, "possible_models": models,
                    "max_model_calls": row.max_calls,
                    "max_input_tokens_per_call": row.max_input_tokens,
                    "max_output_tokens_per_call": row.max_output_tokens,
                    "max_input_tokens": row.max_calls * row.max_input_tokens,
                    "max_output_tokens": row.max_calls * row.max_output_tokens,
                    "reservation_usd": float(bound),
                })
        stages[stage] = {
            "episodes": len(members),
            "independent_groups": len({row.group_id for row in members}),
            "configuration_count": len(configurations),
            "max_model_calls": sum(item["max_model_calls"] for item in configurations),
            "max_input_tokens": sum(item["max_input_tokens"] for item in configurations),
            "max_output_tokens": sum(item["max_output_tokens"] for item in configurations),
            "model_reservation_usd": float(stage_cost),
            "grading_cost_usd": 0.0,
            "contingency_usd": float(stage_cost * contingency),
            "stage_cap_usd": _cents(stage_cost * (1 + contingency)),
            "configurations": configurations,
        }
    stage_caps = {stage: detail["stage_cap_usd"] for stage, detail in stages.items()}
    hard_cap = sum(Decimal(str(cap)) for cap in stage_caps.values())
    return {
        "version": "episode-budget-proposal-v1",
        "basis": "worst_case_frozen_call_and_token_bounds_not_expected_spend",
        "paid_calls_authorized": False,
        "episodes": len(rows), "independent_groups": manifest["independent_groups"],
        "provenance": {
            "corpus_hash": digest([asdict(row) for row in rows]),
            "catalog_hash": digest(catalog), "baseline_hash": digest(baseline_plans),
            "eligibility_hash": digest(eligibility_by_episode),
            "alternatives": candidates,
        },
        "stages": stages,
        "grading": {"kind": "versioned_deterministic_checks", "paid_judge_calls": 0,
                    "cost_usd": 0.0, "grader_versions": sorted({row.grader_version for row in rows})},
        "contingency_fraction": float(contingency),
        "budget_config": {"hard_cap_usd": float(hard_cap), "stage_caps_usd": stage_caps},
        "limitations": [
            "A reservation ceiling does not establish statistical power.",
            "Test and smoke reserve the most expensive eligible challenger before selection.",
            "Repeated trials increase spend and calls, never independent group counts.",
            "Local grading compute is unpriced; provider grading cost is explicitly zero.",
        ],
    }


def render_budget_proposal(proposal: Mapping[str, Any]) -> str:
    """Render the offline proposal without presenting a cap as expected spend."""
    config = proposal["budget_config"]
    lines = [
        "# Episode study budget proposal", "",
        f"The proposed maximum is **${config['hard_cap_usd']:,.2f}** for "
        f"{proposal['episodes']} episode rows in {proposal['independent_groups']} template groups. "
        "This is a worst-case reservation ceiling, not a forecast of realized charges.", "",
        "Generating this proposal does not authorize paid calls. "
        "Corpus size and this budget do not establish statistical power.", "",
        "| Stage | Episodes | Groups | Configurations | Max model calls | Model reservation | Contingency | Stage cap |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, detail in proposal["stages"].items():
        lines.append(f"| {name} | {detail['episodes']} | {detail['independent_groups']} | "
                     f"{detail['configuration_count']} | {detail['max_model_calls']} | "
                     f"${detail['model_reservation_usd']:,.6f} | ${detail['contingency_usd']:,.6f} | "
                     f"${detail['stage_cap_usd']:,.2f} |")
    lines += ["", "Pilot reserves baseline plus its first eligible candidate. Development reserves "
              "baseline plus every eligible candidate. Test and smoke reserve baseline plus the "
              "most expensive eligible challenger until the router is locked. Every configuration "
              "uses the actual row's maximum calls and input/output tokens per call.", "",
              f"Contingency is {proposal['contingency_fraction']:.0%}; stage caps round upward to cents. "
              "Grading uses versioned deterministic checks with zero paid judge calls. "
              "Local grading compute is unpriced.", "",
              "The JSON proposal contains per-configuration bounds and hashes of the corpus, "
              "catalog, baseline plans, and eligibility records.", ""]
    lines += [f"- {note}" for note in proposal["limitations"]]
    return "\n".join(lines) + "\n"
