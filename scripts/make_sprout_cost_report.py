#!/usr/bin/env python3
"""Build the portable DIU cost/quality report input from cost_analysis.json."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ANALYSIS = ROOT / "artifacts" / "gist-sprout-router" / "cost_analysis.json"
DEFAULT_OUTPUT = ROOT / "reports" / "sprout-cost-economics" / "artifact.json"


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def percent_points(value: float) -> float:
    return 100.0 * value


def frontier_row(
    label: str,
    policy_type: str,
    metrics: dict[str, Any],
    gpt_4o: dict[str, Any],
) -> dict[str, Any]:
    cost = float(metrics["cost_per_1000_requests_usd"])
    score = float(metrics["mean_score"])
    return {
        "policy": label,
        "policy_type": policy_type,
        "mean_score": score,
        "score_percent": percent_points(score),
        "cost_per_1000_requests_usd": cost,
        "successful_equivalent_tasks_per_dollar": float(
            metrics["successful_equivalent_tasks_per_dollar"]
        ),
        "savings_vs_gpt_4o": 1.0 - cost / float(gpt_4o["cost_per_1000_requests_usd"]),
        "score_difference_vs_gpt_4o_points": percent_points(score - float(gpt_4o["mean_score"])),
        "test_prompts": int(metrics["prompt_count"]),
    }


def build_report(analysis: dict[str, Any], generated_at: str) -> dict[str, Any]:
    test = analysis["test"]
    fixed = test["fixed_models"]
    quality = test["quality_first_router"]
    balanced = test["cost_aware_routers"]["within_0.5_points"]["test"]
    within_one = test["cost_aware_routers"]["within_1_points"]["test"]
    within_two = test["cost_aware_routers"]["within_2_points"]["test"]
    uniform = test["uniform"]
    gpt_4o = fixed["gpt_4o"]
    claude = fixed["claude_3_5_sonnet"]

    frontier = [
        frontier_row("Claude 3.5 Sonnet", "Fixed model", claude, gpt_4o),
        frontier_row("GPT-4o", "Fixed model", gpt_4o, gpt_4o),
        frontier_row("Quality-first router", "Router", quality, gpt_4o),
        frontier_row("Balanced router", "Router", balanced, gpt_4o),
        frontier_row("Cost-aware (1-point guardrail)", "Router", within_one, gpt_4o),
        frontier_row("Cost-aware (2-point guardrail)", "Router", within_two, gpt_4o),
        frontier_row("GPT-4o mini", "Fixed model", fixed["gpt_4o_mini"], gpt_4o),
        frontier_row("Uniform selection", "Baseline", uniform, gpt_4o),
        frontier_row(
            "Cheapest fixed model",
            "Fixed model",
            fixed["cheapest_fixed_selected_on_train_cost"],
            gpt_4o,
        ),
    ]

    summary = [
        {
            "balanced_cost_per_1000": balanced["cost_per_1000_requests_usd"],
            "gpt_4o_cost_per_1000": gpt_4o["cost_per_1000_requests_usd"],
            "balanced_mean_score": balanced["mean_score"],
            "gpt_4o_mean_score": gpt_4o["mean_score"],
            "balanced_savings_vs_gpt_4o": balanced["comparisons"]["gpt_4o"][
                "cost_savings_fraction"
            ],
            "quality_first_cost_per_1000": quality["cost_per_1000_requests_usd"],
            "quality_first_mean_score": quality["mean_score"],
            "quality_first_savings_vs_gpt_4o": quality["comparisons"]["gpt_4o"][
                "cost_savings_fraction"
            ],
            "test_prompts": balanced["prompt_count"],
        }
    ]

    comparison = [
        {
            "policy": row["policy"],
            "mean_score": row["mean_score"],
            "cost_per_1000_requests_usd": row["cost_per_1000_requests_usd"],
            "savings_vs_gpt_4o": row["savings_vs_gpt_4o"],
            "score_difference_vs_gpt_4o_points": row["score_difference_vs_gpt_4o_points"],
        }
        for row in frontier
        if row["policy"]
        in {
            "GPT-4o",
            "Quality-first router",
            "Balanced router",
            "GPT-4o mini",
            "Uniform selection",
        }
    ]

    volumes = []
    for requests in (1_000_000, 10_000_000, 100_000_000):
        balanced_cost = balanced["cost_per_1000_requests_usd"] * requests / 1000.0
        gpt_cost = gpt_4o["cost_per_1000_requests_usd"] * requests / 1000.0
        claude_cost = claude["cost_per_1000_requests_usd"] * requests / 1000.0
        volumes.append(
            {
                "requests": requests,
                "balanced_cost_usd": balanced_cost,
                "savings_vs_gpt_4o_usd": gpt_cost - balanced_cost,
                "savings_vs_claude_usd": claude_cost - balanced_cost,
            }
        )

    source_id = "sprout_cost_analysis"
    price_paper_id = "carrot_paper_table_2"
    price_hf_id = "carrot_hf_pricing"
    price_repo_id = "carrot_repository_constants"
    sources = [
        {
            "id": source_id,
            "label": "SPROUT cost analysis",
            "path": "artifacts/gist-sprout-router/cost_analysis.json",
            "query": {
                "engine": "duckdb",
                "sql": "SELECT * FROM read_json_auto('artifacts/gist-sprout-router/cost_analysis.json')",
                "description": (
                    "Loads the reviewed JSON produced by scripts/analyze_sprout_costs.py. That "
                    "analysis re-infers the saved embedding router, joins official SPROUT token "
                    "counts, selects cost-aware tolerances on grouped validation, and evaluates "
                    "the frozen policies on the official test split."
                ),
                "language": "sql",
                "executed_at": generated_at,
                "tables_used": [
                    "SPROUT official train split",
                    "SPROUT official validation split",
                    "SPROUT official test split",
                    "artifacts/gist-sprout-router/performance_head.safetensors",
                ],
                "filters": [
                    "Official test split only for reported outcomes",
                    "All 13 model outcomes must be present",
                    "Cost policy selected on grouped validation before test evaluation",
                ],
                "metric_definitions": [
                    "Mean score: average observed SPROUT score of the model selected for each test prompt.",
                    "Cost per 1,000 requests: mean realized token cost multiplied by 1,000.",
                    "Savings versus GPT-4o: 1 minus policy cost divided by always-GPT-4o cost.",
                    "Successful-equivalent tasks per dollar: mean score divided by mean cost per request; this is descriptive, not a calibrated completion count.",
                ],
            },
        },
        {
            "id": price_paper_id,
            "label": "CARROT paper, Table 2",
            "href": "https://arxiv.org/html/2502.03261v2",
            "query": {
                "engine": "document",
                "url": "https://arxiv.org/html/2502.03261v2",
                "description": "Benchmark-contemporaneous input and output prices used for SPROUT models.",
                "executed_at": generated_at,
            },
        },
        {
            "id": price_hf_id,
            "label": "CARROT Hugging Face pricing table",
            "href": "https://huggingface.co/CARROT-LLM-Routing",
            "query": {
                "engine": "document",
                "url": "https://huggingface.co/CARROT-LLM-Routing",
                "description": "Independent publication of the same 13-model SPROUT pricing table.",
                "executed_at": generated_at,
            },
        },
        {
            "id": price_repo_id,
            "label": "CARROT repository pricing constants",
            "href": "https://github.com/somerstep/CARROT/blob/main/carrot/constants.py",
            "query": {
                "engine": "source-code",
                "url": "https://github.com/somerstep/CARROT/blob/main/carrot/constants.py",
                "description": "Current repository constants used for the Titan price sensitivity check.",
                "executed_at": generated_at,
            },
        },
    ]

    title = "Theriad router economics: quality per dollar"
    manifest = {
        "version": 1,
        "surface": "report",
        "title": title,
        "description": "Official SPROUT test-set economics using benchmark-contemporaneous model prices.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "balanced_cost",
                "dataset": "summary",
                "sourceId": source_id,
                "description": "Estimated token cost for the validation-selected balanced policy.",
                "metrics": [
                    {
                        "label": "Balanced cost per 1k",
                        "field": "balanced_cost_per_1000",
                        "format": "currency",
                    },
                    {
                        "label": "GPT-4o cost per 1k",
                        "field": "gpt_4o_cost_per_1000",
                        "format": "currency",
                    },
                ],
            },
            {
                "id": "balanced_savings",
                "dataset": "summary",
                "sourceId": source_id,
                "description": "Cost reduction versus routing every request to GPT-4o.",
                "metrics": [
                    {
                        "label": "Savings vs GPT-4o",
                        "field": "balanced_savings_vs_gpt_4o",
                        "format": "percent",
                    },
                ],
            },
            {
                "id": "balanced_quality",
                "dataset": "summary",
                "sourceId": source_id,
                "description": "Observed mean SPROUT score on 6,633 untouched test prompts.",
                "metrics": [
                    {
                        "label": "Balanced test score",
                        "field": "balanced_mean_score",
                        "format": "percent",
                    },
                    {
                        "label": "GPT-4o test score",
                        "field": "gpt_4o_mean_score",
                        "format": "percent",
                    },
                ],
            },
        ],
        "charts": [
            {
                "id": "cost_quality_frontier",
                "title": "Cost and quality across routing policies",
                "subtitle": "Lower cost and higher score is better; results use 6,633 official SPROUT test prompts.",
                "type": "scatter",
                "intent": "relationship",
                "question": "Which tested policies offer the strongest cost-quality trade-off?",
                "rationale": "A scatter plot reveals policies that improve quality, reduce cost, or are dominated on both dimensions.",
                "comparisonContext": {
                    "baseline": "Always GPT-4o",
                    "grain": "One point per evaluated routing policy",
                    "unit": "USD per 1,000 requests and mean score",
                },
                "dataset": "frontier",
                "sourceId": source_id,
                "encodings": {
                    "x": {
                        "field": "cost_per_1000_requests_usd",
                        "type": "quantitative",
                        "label": "Cost per 1,000 requests",
                        "format": "currency",
                    },
                    "y": {
                        "field": "mean_score",
                        "type": "quantitative",
                        "label": "Mean SPROUT score",
                        "format": "percent",
                    },
                    "color": {
                        "field": "policy_type",
                        "type": "nominal",
                        "label": "Policy type",
                    },
                    "tooltip": [
                        {"field": "policy", "type": "text", "label": "Policy"},
                        {
                            "field": "cost_per_1000_requests_usd",
                            "type": "quantitative",
                            "format": "currency",
                            "label": "Cost per 1,000",
                        },
                        {
                            "field": "mean_score",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Mean score",
                        },
                        {
                            "field": "savings_vs_gpt_4o",
                            "type": "quantitative",
                            "format": "percent",
                            "label": "Savings vs GPT-4o",
                        },
                    ],
                },
                "layout": "full",
                "maxRows": 20,
                "surface": {
                    "legend": {"show": True, "position": "bottom", "sort": "spec"},
                    "valueLabels": {"mode": "none"},
                },
            },
        ],
        "tables": [
            {
                "id": "policy_comparison",
                "title": "Decision-relevant policy comparison",
                "subtitle": "The quality-first and balanced routers answer different operating objectives.",
                "dataset": "comparison",
                "sourceId": source_id,
                "defaultSort": {"field": "mean_score", "direction": "desc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "policy", "label": "Policy", "type": "text"},
                    {"field": "mean_score", "label": "Mean score", "format": "percent"},
                    {
                        "field": "cost_per_1000_requests_usd",
                        "label": "Cost / 1k",
                        "format": "currency",
                    },
                    {
                        "field": "savings_vs_gpt_4o",
                        "label": "Savings vs GPT-4o",
                        "format": "percent",
                    },
                    {
                        "field": "score_difference_vs_gpt_4o_points",
                        "label": "Score delta, points",
                        "format": "number",
                    },
                ],
            },
            {
                "id": "volume_savings",
                "title": "Balanced-policy savings at scale",
                "subtitle": "Linear projection at the benchmark's observed token mix; excludes router and platform overhead.",
                "dataset": "volumes",
                "sourceId": source_id,
                "defaultSort": {"field": "requests", "direction": "asc"},
                "density": "dense",
                "layout": "full",
                "columns": [
                    {"field": "requests", "label": "Requests", "format": "compact"},
                    {
                        "field": "balanced_cost_usd",
                        "label": "Balanced cost",
                        "format": "currency",
                    },
                    {
                        "field": "savings_vs_gpt_4o_usd",
                        "label": "Savings vs GPT-4o",
                        "format": "currency",
                    },
                    {
                        "field": "savings_vs_claude_usd",
                        "label": "Savings vs Claude",
                        "format": "currency",
                    },
                ],
            },
        ],
        "sources": sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {title}"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "sourceId": source_id,
                "body": (
                    "## Executive Summary\n\n"
                    "The **quality-first router** is the cleanest technical win: it scored **86.2%**, "
                    "beating always-GPT-4o by **1.4 points** while costing **26% less**.\n\n"
                    "For a cost-constrained deployment, the validation-selected **balanced router** "
                    "scored **83.7%** at **$0.63 per 1,000 requests**—**87% cheaper than GPT-4o** "
                    "and **92% cheaper than Claude 3.5 Sonnet** at this benchmark's token mix. "
                    "That is an estimated **$4.4k saved per 1 million requests** versus GPT-4o.\n\n"
                    "The balanced policy is promising, not production-proven: its validation quality "
                    "guardrail transferred imperfectly to the untouched test split."
                ),
            },
            {
                "id": "headline_metrics",
                "type": "metric-strip",
                "cardIds": ["balanced_cost", "balanced_savings", "balanced_quality"],
            },
            {
                "id": "key_findings",
                "type": "markdown",
                "sourceId": source_id,
                "body": (
                    "## Key findings\n\n"
                    "The previous **58% regret reduction was a quality result, not a savings result**. "
                    "The quality-first router costs **$3.69 per 1,000 requests**, or **2.7x** uniform "
                    "model selection, because it routes most prompts to GPT-4o. It still dominates "
                    "always-GPT-4o and always-Claude on both measured quality and cost.\n\n"
                    "The balanced router moves most prompts to GPT-4o mini and selectively escalates. "
                    "It saves **$4,365 per 1 million requests** versus GPT-4o, with a measured "
                    "**1.15-point** score reduction."
                ),
            },
            {"id": "frontier_chart", "type": "chart", "chartId": "cost_quality_frontier"},
            {"id": "comparison_table", "type": "table", "tableId": "policy_comparison"},
            {"id": "volume_table", "type": "table", "tableId": "volume_savings"},
            {
                "id": "recommended_next_steps",
                "type": "markdown",
                "body": (
                    "## Recommended next steps\n\n"
                    "1. Present the quality-first result as the validated router result; present the "
                    "balanced policy as an economic design point.\n"
                    "2. Retrain with a prompt-conditioned cost and output-token objective instead of "
                    "a post-hoc average-cost threshold.\n"
                    "3. Reprice against Theriad's current deployable catalog and add router, retry, "
                    "network, and observability overhead.\n"
                    "4. Validate on a frozen defense-relevant workload, multiple seeds, calibration, "
                    "and a hard cost ceiling before making a production savings claim."
                ),
            },
            {
                "id": "further_questions",
                "type": "markdown",
                "body": (
                    "## Further questions\n\n"
                    "- What minimum quality floor and maximum cost per task should Theriad enforce?\n"
                    "- Which current hosted or on-prem models are actually eligible for a DIU deployment?\n"
                    "- How much do retries, tool calls, long-context prompts, and security controls change the token mix?"
                ),
            },
            {
                "id": "caveats",
                "type": "markdown",
                "body": (
                    "## Caveats and assumptions\n\n"
                    "- Prices reproduce the SPROUT benchmark; they are not current replacement-model quotes.\n"
                    "- The cost-aware policy was selected on grouped validation, but the saved quality head "
                    "was not trained with a cost objective.\n"
                    "- The 0.5-point validation guardrail became a 2.5-point loss versus the quality-first "
                    "router on test, indicating generalization uncertainty.\n"
                    "- Costs exclude router compute, network, retries, observability, and enterprise pricing.\n"
                    "- SPROUT has no declared dataset license and is not a defense-domain operational benchmark.\n"
                    "- The CARROT paper and Hugging Face table list Titan at $0.50/$1.50 per million input/output "
                    "tokens; the repository currently lists $0.80/$3.20. Neither reported router selects Titan, "
                    "so the discrepancy does not change the router results."
                ),
            },
        ],
    }

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "summary": summary,
                "frontier": frontier,
                "comparison": comparison,
                "volumes": volumes,
            },
        },
        "sources": sources,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--analysis", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    generated_at = (
        datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    )
    report = build_report(load_json(args.analysis), generated_at)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(args.output)


if __name__ == "__main__":
    main()
