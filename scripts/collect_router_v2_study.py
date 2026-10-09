#!/usr/bin/env python3
"""Build a provider-free, explicitly gated Router V2 study phase plan.

The command never executes a provider call. It writes four separately gated
phase plans while leaving Phase A call records unmaterialized until an explicit
approval and the persisted phase prerequisites are satisfied.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from theriad.routing.study import StudyCorpusError, load_jsonl, validate_corpus  # noqa: E402
from theriad.routing.study_budget import StudyBudgetConfig, StudyBudgetLedger  # noqa: E402
from theriad.routing.study_collector import (  # noqa: E402
    StudyCollectorError,
    build_phased_study_plan,
    write_phased_plan,
)


def _budget_summary(phased: object) -> dict[str, object]:
    plans = phased.phases
    scenarios: dict[str, dict[str, object]] = {}
    for name in ("free_nemotron", "paid_nemotron"):
        phase_values = {
            phase: plan.cost_scenarios[name]
            for phase, plan in plans.items()
        }
        total_expected = sum(
            float(value.get("expected_total_cost_usd") or 0.0)
            for value in phase_values.values()
        )
        total_worst = sum(
            float(value.get("worst_case_total_cost_usd") or 0.0)
            for value in phase_values.values()
        )
        test_worst = float(
            phase_values["D_test"].get("worst_case_total_cost_usd") or 0.0
        )
        pre_test_worst = total_worst - test_worst
        scenarios[name] = {
            "phases": phase_values,
            "expected_total_cost_usd": total_expected,
            "worst_case_total_cost_usd": total_worst,
            "untouched_test_worst_case_usd": test_worst,
            "pre_test_worst_case_usd": pre_test_worst,
            "hard_cap_usd": 25.0,
            "test_reserve_usd": 10.0,
            "pre_test_cap_usd": 15.0,
            "within_hard_cap": total_worst <= 25.0,
            "within_test_reserve": test_worst <= 10.0,
            "within_pre_test_cap": pre_test_worst <= 15.0,
        }
    return scenarios


def _load_paid_nemotron_quote(path: Path) -> dict[str, object]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    models = raw.get("models") if isinstance(raw, dict) else None
    if not isinstance(models, dict):
        raise ValueError("paid Nemotron pricing snapshot needs a models object")
    slug = "nvidia/nemotron-3-super-120b-a12b"
    quote = models.get(slug)
    if not isinstance(quote, dict):
        raise ValueError(f"pricing snapshot has no paid quote for {slug}")
    if quote.get("input_per_1m") is None or quote.get("output_per_1m") is None:
        raise ValueError("paid Nemotron quote is missing input/output rates")
    return {
        "input_per_1m": float(quote["input_per_1m"]),
        "output_per_1m": float(quote["output_per_1m"]),
        "pricing_model_slug": slug,
        "source": f"{path}:{raw.get('refreshed_at', 'unknown')}",
    }


def _reservation_multipliers(values: list[str]) -> dict[str, dict[str, float]]:
    """Parse reviewed Phase-A MODEL[@TASK_FORMAT]=FACTOR overrides."""

    result: dict[str, float] = {}
    for value in values:
        model, separator, factor_text = value.rpartition("=")
        if not separator or not model.strip():
            raise ValueError(
                "--phase-a-reservation-multiplier requires MODEL[@TASK_FORMAT]=FACTOR"
            )
        factor = float(factor_text)
        if factor < 1.0:
            raise ValueError("reservation multiplier must be at least 1")
        result[model.strip()] = factor
    return {"A_core_train": result} if result else {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--budget-ledger", type=Path, required=True)
    parser.add_argument(
        "--paid-nemotron-pricing",
        type=Path,
        default=ROOT / "theriad/accounting/openrouter_prices.json",
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--phase-a-reservation-multiplier",
        action="append",
        default=[],
        metavar="MODEL[@TASK_FORMAT]=FACTOR",
        help=(
            "Increase Phase A's per-call hard-cap reservation for one model, "
            "optionally restricted to a task format. "
            "This changes reservation accounting, not expected-cost economics."
        ),
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--allow-provider-calls", action="store_true")
    parser.add_argument("--allow-paid-calls", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    try:
        if args.execute:
            raise StudyCollectorError(
                "provider execution is disabled in this audit pass; execute only a reviewed, "
                "phase-specific plan through the gated StudyCollector workflow"
            )
        prompts = load_jsonl(args.corpus)
        audit = validate_corpus(prompts)
        paid_quote = _load_paid_nemotron_quote(args.paid_nemotron_pricing)
        phase_reservation_multipliers = _reservation_multipliers(
            args.phase_a_reservation_multiplier
        )
        phased = build_phased_study_plan(
            prompts,
            max_attempts=args.max_attempts,
            paid_nemotron_quote=paid_quote,
            phase_reservation_multipliers=phase_reservation_multipliers,
        )
        write_phased_plan(phased, args.plan)
        ledger = StudyBudgetLedger(
            args.budget_ledger,
            StudyBudgetConfig(hard_cap_usd=25.0, test_reserve_usd=10.0),
        )
        report = {
            "mode": "dry_run",
            "provider_calls": 0,
            "provider_spend_usd": 0.0,
            "corpus": audit.to_dict(),
            "plan": phased.to_dict(),
            "budget_ledger": {"path": str(args.budget_ledger), "status": ledger.status()},
            "next_step": "review the phase gates and executor evidence; no execution is unlocked by this command",
            "phase_a_materialized": phased.to_dict()["provider_calls_materialized"] > 0,
            "phase_reservation_multipliers": phase_reservation_multipliers,
            "budget_summary": _budget_summary(phased),
        }
        if args.report is not None:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            args.report.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    except (
        StudyCorpusError,
        StudyCollectorError,
        ValueError,
        OSError,
        json.JSONDecodeError,
    ) as exc:
        print(f"router-v2 study stopped safely: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
