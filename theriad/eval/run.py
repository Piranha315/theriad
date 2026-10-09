"""
Router evaluation harness — offline by default (D-018).

Subcommands:
    plan       Show the grid cells needed and what they would cost. No calls.
    report     Score a router against the oracle from the cached grid. No calls.
    calibrate  Turn the grid into the router's own parameters. No calls.
    populate   Execute planned cells. Requires --allow-paid-calls and --yes.

The grid is the only step that spends money, and it is cached and additive, so
``plan`` → inspect → ``populate`` → ``report`` → ``calibrate`` is the intended
loop; everything after ``populate`` can be re-run at any V without further spend.

``calibrate --write`` is the step that closes it: it converts measured success
rates into the ``quality_profile`` and per-tier value-of-quality the router
actually consults, so re-running the eval changes routing without anyone
hand-editing ``theriad/models/registry.py``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from theriad.accounting.pricing import PricingTable
from theriad.models.calibration import CALIBRATION_PATH
from theriad.models.registry import DEFAULT_REGISTRY
from theriad.models.unified import TIER_VALUE_OF_QUALITY

# Load the project-root .env before anything reads a key, matching
# theriad/api/server.py. Resolved from __file__ rather than cwd so the harness
# behaves the same whether invoked from the repo root or elsewhere. Absent .env
# is fine — real environment variables take precedence either way.
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:  # pragma: no cover - dotenv is optional at runtime
    pass

from .corpus import EvalItem, load_corpus, vintages
from .executors import CallResult
from .grid import ModelCoverage, OutcomeGrid, PlannedCall, derive_strata
from .oracle import Action, ExpectedValueOracle
from .regret import evaluate_router

# ── Token profile ─────────────────────────────────────────────────────────────

DEFAULT_INPUT_TOKENS = 600
DEFAULT_OUTPUT_TOKENS = 1200
"""Per-call token profile used **only to estimate cost before spending**.

This is not a generation cap and must never be passed as max_tokens. The two are
different numbers, and conflating them silently corrupts the grid: imposing a
short assumed length as a cap truncates any model that reasons before answering,
hardest on the difficult items where reasoning runs longest, so the harness
records "bad at hard questions" when the truth is "cut off mid-sentence".

The generation cap lives in theriad.eval.executors and defaults to None (uncapped).
1200 is a deliberately generous costing assumption for exam-style items so the
budget guard errs toward over-estimating spend rather than under-estimating it."""

DEFAULT_GENERATION_CAP: int | None = None
"""max_tokens sent to providers. None = uncapped, which is correct for
measurement. Set an integer only to bound a runaway; truncation counters will
report whether it bit."""


# A versioned, reviewable model set for the current-and-candidate replacement
# study. Keep this in code rather than a shell snippet: a copied command silently goes stale
# as the registry changes, which is how the original pre-merge instructions
# ended up benchmarking a different router from the one in production.
REPLACEMENT_STUDY_MODELS: dict[str, tuple[str, ...]] = {
    "cheap": (
        "nvidia/nemotron-3-nano-30b-a3b:free",
        "nvidia/nemotron-3-super-120b-a12b:free",
        "poolside/laguna-xs-2.1:free",
        "cohere/north-mini-code:free",
        "xiaomi/mimo-v2.5",
        "deepseek/deepseek-v4-flash-0731",
        "qwen/qwen3.6-plus",
        "openai/gpt-oss-20b",
        "openai/gpt-oss-120b",
        "google/gemini-2.5-flash",
    ),
    "premium": (
        "gpt-5.4-mini",
        "z-ai/glm-5.2",
        "openai/gpt-5.6-luna",
    ),
    "frontier": (
        "x-ai/grok-4.5",
        "z-ai/glm-5.3-flash",
        "meta/muse-spark-1.2",
        "anthropic/claude-opus-5",
        "openai/gpt-4o",
    ),
}


def model_lists(args) -> tuple[list[str], list[str], list[str]]:
    """Resolve explicit CLI lists or the named replacement-study preset."""
    explicit = (args.cheap, args.premium, args.frontier)
    if args.model_set == "replacement-study":
        if any(explicit):
            raise ValueError(
                "--model-set replacement-study cannot be combined with "
                "--cheap, --premium, or --frontier"
            )
        return (
            list(REPLACEMENT_STUDY_MODELS["cheap"]),
            list(REPLACEMENT_STUDY_MODELS["premium"]),
            list(REPLACEMENT_STUDY_MODELS["frontier"]),
        )

    def split(value: str) -> list[str]:
        return [model.strip() for model in value.split(",") if model.strip()]

    return split(args.cheap), split(args.premium), split(args.frontier)


def call_costs(
    models: list[str],
    input_tokens: int = DEFAULT_INPUT_TOKENS,
    output_tokens: int = DEFAULT_OUTPUT_TOKENS,
    table: PricingTable | None = None,
) -> dict[str, float]:
    """Return {model: USD per call} from the pricing table.

    Models with no price resolve to 0.0, matching PricingTable's documented
    fallback. A zero here makes a model look free to the oracle, so ``plan``
    reports unpriced models explicitly rather than letting them slip through.
    """
    pricing = table or PricingTable(use_live=False)
    costs: dict[str, float] = {}
    for model in models:
        price = pricing.get(model)
        costs[model] = (
            0.0
            if price is None
            else (input_tokens * price.input_per_1m + output_tokens * price.output_per_1m)
            / 1_000_000
        )
    return costs


def unpriced(models: list[str], table: PricingTable | None = None) -> list[str]:
    """Models the pricing table cannot resolve — they would be scored as free."""
    pricing = table or PricingTable(use_live=False)
    return [m for m in models if pricing.get(m) is None]


# ── Coverage presets ──────────────────────────────────────────────────────────
#
# Coverage is asymmetric on purpose: cost per cell varies ~20x across tiers while
# the number of cells needed does not. Cheap candidates carry most production
# traffic and are cheap to measure, so they get full coverage. Expensive
# candidates exist in the grid only to price a small number of escalation
# boundaries, so a stratified subsample is sufficient.
#
# The ceiling tier is subsampled rather than skipped. Skipping it entirely is
# tempting — if policy always escalates to the ceiling when everything below
# fails, the action is fixed and the measurement buys no decision. But under an
# expected-value oracle, whether to escalate at all depends on the *gap* between
# the ceiling and the tier below it, so with no ceiling data that boundary cannot
# be priced: you can score a policy that always escalates, but not one that
# decides whether to. 20% coverage keeps it priceable at a fifth of the cost,
# and stratum-pooled estimates are what the router needs anyway.


def default_coverage(
    cheap: list[str],
    premium: list[str],
    frontier: list[str] | None = None,
    samples: int = 5,
) -> list[ModelCoverage]:
    """Standard asymmetric coverage plan."""
    plan = [
        ModelCoverage(model=m, tier="cheap", item_fraction=1.0, samples_per_item=samples)
        for m in cheap
    ]
    plan += [
        ModelCoverage(
            model=m, tier="premium", item_fraction=0.5, samples_per_item=samples
        )
        for m in premium
    ]
    plan += [
        ModelCoverage(
            model=m, tier="frontier", item_fraction=0.2, samples_per_item=samples
        )
        for m in (frontier or [])
    ]
    return plan


# ── Planning ──────────────────────────────────────────────────────────────────


@dataclass
class GridPlan:
    calls: list[PlannedCall]
    costs: dict[str, float]

    @property
    def n_calls(self) -> int:
        return sum(c.n_samples for c in self.calls)

    @property
    def estimated_cost_usd(self) -> float:
        return sum(self.costs.get(c.model, 0.0) * c.n_samples for c in self.calls)

    def by_model(self) -> dict[str, tuple[int, float]]:
        """{model: (n_calls, estimated_usd)}."""
        out: dict[str, tuple[int, float]] = {}
        for call in self.calls:
            n, usd = out.get(call.model, (0, 0.0))
            out[call.model] = (
                n + call.n_samples,
                usd + self.costs.get(call.model, 0.0) * call.n_samples,
            )
        return dict(sorted(out.items(), key=lambda kv: -kv[1][1]))


def plan_grid(
    grid: OutcomeGrid,
    items: list[EvalItem],
    coverage: list[ModelCoverage],
    costs: dict[str, float],
    retry_truncated: bool = False,
) -> GridPlan:
    return GridPlan(
        calls=grid.plan(items, coverage, retry_truncated=retry_truncated),
        costs=costs,
    )


# ── Population ────────────────────────────────────────────────────────────────

# Executes one call for (model, item) and returns a CallResult.
Executor = Callable[[str, EvalItem], "CallResult"]


class PopulateAborted(RuntimeError):
    """Raised when population stops early because errors became sustained.

    Distinct from a normal failure so callers can tell "the run was cut short"
    from "the models did badly" — the whole point of not swallowing errors.
    """


def populate_grid(
    grid: OutcomeGrid,
    items: list[EvalItem],
    plan: GridPlan,
    executor: Executor,
    on_progress: Callable[[int, int], None] | None = None,
    on_error: Callable[[str, str, Exception], None] | None = None,
    max_consecutive_errors: int = 10,
    transcript_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
) -> OutcomeGrid:
    """Execute planned calls, grading each with the item's answer key.

    ``executor`` is injected so the harness is testable without a provider and so
    callers choose their own lane.

    Three failure modes are handled differently on purpose:

    - **Exceptions** are *not* recorded. An errored call is missing data, not
      evidence about the model, and recording it as an unsuccessful trial makes
      an outage indistinguishable from incompetence — a throttled run then reads
      as "this model answers nothing correctly". Leaving the cell absent also
      means a later resume retries it instead of skipping it forever.
    - **Truncation** is recorded as a discarded attempt, not a failure. A
      response cut off by an output cap is missing data; scoring it as wrong
      would bias the grid against models that reason at length.
    - **Sustained errors** trip a circuit breaker. ``max_consecutive_errors``
      consecutive failures abort the run, because the realistic cause is
      throttling or an outage affecting every remaining call, and spinning
      through the rest of the plan produces nothing but noise.

    ``transcript_path`` appends one JSON line per executed call, including the
    raw response text. The grid stores only counts, so without this a grading
    bug is unrecoverable: on 2026-08-15 a \\boxed{} formatting mismatch was found
    to be scoring correct answers as failures, and because no output text had
    been kept, re-grading the affected cells meant re-paying for all of them.
    Keeping transcripts makes any future grader fix a free offline re-run.

    ``checkpoint_path`` saves the aggregate grid after every recorded call, so
    a long paid run can resume even if the process disappears between lanes.

    Returns the grid; inspect ``errors`` via ``on_error`` for what failed.
    """
    by_id = {item.id: item for item in items}
    invalid = [item.id for item in items if item.to_prompt().grade("").status == "ungradable"]
    if invalid:
        raise ValueError(f"Ungradable corpus items must be repaired before collection: {invalid[:10]}")
    transcript = None
    if transcript_path is not None:
        # Line-buffered append: a crashed or budget-killed run keeps everything
        # written so far, which is the case that matters most.
        transcript = open(transcript_path, "a", encoding="utf-8", buffering=1)
    done = 0
    total = plan.n_calls
    consecutive = 0
    for call in plan.calls:
        item = by_id.get(call.item_id)
        if item is None:
            continue
        for _ in range(call.n_samples):
            try:
                result = executor(call.model, item)
                graded = None
                if result.truncated:
                    grid.record(
                        call.model, item.id, False, result.cost_usd, truncated=True
                    )
                else:
                    graded = item.grade(result.output)
                    if graded is not None:
                        grid.record(call.model, item.id, graded, result.cost_usd)
                if transcript is not None:
                    transcript.write(
                        json.dumps(
                            {
                                "model": call.model,
                                "item_id": item.id,
                                "graded_correct": graded,
                                "grader_version": item.to_prompt().grade("").grader_version,
                                "grade_status": "ungradable" if graded is None else ("correct" if graded else "incorrect"),
                                "truncated": result.truncated,
                                "finish_reason": result.finish_reason,
                                "cost_usd": result.cost_usd,
                                "output": result.output,
                            }
                        )
                        + "\n"
                    )
                    transcript.flush()
                    os.fsync(transcript.fileno())
                if checkpoint_path is not None:
                    grid.save(checkpoint_path)
                consecutive = 0
            except Exception as exc:  # noqa: BLE001 - surfaced, not swallowed
                consecutive += 1
                if on_error:
                    on_error(call.model, item.id, exc)
                if consecutive >= max_consecutive_errors:
                    raise PopulateAborted(
                        f"aborted after {consecutive} consecutive errors on "
                        f"{call.model}; last error: {type(exc).__name__}: {exc}"
                    ) from exc
            done += 1
            if on_progress:
                on_progress(done, total)
    if transcript is not None:
        transcript.close()
    return grid


# ── Reporting ─────────────────────────────────────────────────────────────────


def tier_router(tier_of: dict[str, str], choice: dict[str, str]) -> Callable:
    """Build a simple router that maps an item's category to a model.

    Useful as a baseline in reports; the real router is wired in by the caller.
    """

    def _route(item: EvalItem) -> Action:
        model = choice.get(item.category) or next(iter(tier_of))
        return Action(model=model, n_samples=1)

    return _route


def build_report(
    grid: OutcomeGrid,
    items: list[EvalItem],
    router,
    value_correct: float,
    models: list[str] | None = None,
    max_samples: int = 1,
    sample_correlation: float | None = None,
) -> dict:
    """Score a router at one V and return a serialisable summary."""
    strata = derive_strata(grid, items)
    model_list = models or grid.models()
    costs = call_costs(model_list)
    if sample_correlation is None:
        from .oracle import DEFAULT_SAMPLE_CORRELATION, estimate_sample_correlation

        measured = [
            rho
            for m in model_list
            if (rho := estimate_sample_correlation(grid, m)) is not None
        ]
        sample_correlation = (
            sum(measured) / len(measured) if measured else DEFAULT_SAMPLE_CORRELATION
        )
    oracle = ExpectedValueOracle(
        grid=grid,
        costs=costs,
        value_correct=value_correct,
        max_samples=max_samples,
        strata=strata,
        sample_correlation=sample_correlation,
    )
    report = evaluate_router(router, items, oracle, costs, strata, model_list)
    return {
        "summary": report.summary(),
        "strata_counts": strata.counts(),
        "by_stratum": {k: v.summary() for k, v in report.by_stratum().items()},
        "by_vintage": {str(k): v.summary() for k, v in report.by_vintage().items()},
    }


def sweep_value(
    grid: OutcomeGrid,
    items: list[EvalItem],
    router,
    values: list[float],
    models: list[str] | None = None,
) -> list[dict]:
    """Report across a range of V.

    Always prefer a sweep to a single V: the oracle's action changes as the value
    of a correct answer rises, so a result at one arbitrary V says little about
    whether the router's thresholds are set sensibly.
    """
    return [
        {"value_correct": v, **build_report(grid, items, router, v, models)}
        for v in values
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────


def _cmd_plan(args) -> int:
    items = load_corpus(args.corpus)
    grid = OutcomeGrid.load(args.grid)
    cheap, premium, frontier = model_lists(args)
    coverage = default_coverage(cheap, premium, frontier, samples=args.samples)
    all_models = cheap + premium + frontier
    costs = call_costs(all_models, args.input_tokens, args.output_tokens)
    plan = plan_grid(
        grid, items, coverage, costs, retry_truncated=args.retry_truncated
    )

    print(f"corpus:   {len(items)} items   vintages={vintages(items)}")
    print(f"grid:     {len(grid)} cells already recorded, {grid.total_trials} trials")
    print(f"planned:  {plan.n_calls} calls   est. ${plan.estimated_cost_usd:.2f}")
    print()
    print(f"{'model':<40}{'calls':>8}{'est. USD':>12}")
    for model, (n, usd) in plan.by_model().items():
        print(f"{model:<40}{n:>8}{usd:>12.3f}")

    missing = unpriced(all_models)
    if missing:
        print()
        print("WARNING: no price found for these models; they score as free:")
        for m in missing:
            print(f"  - {m}")
    return 0


def _cmd_verify_sources(args) -> int:
    """Probe each benchmark source and confirm its field mapping still works.

    One HTTP request per source, no provider spend. Worth running before every
    corpus build: a drifted field name does not error, it produces well-formed
    nonsense that grades as universal failure, and the bill arrives first.
    """
    from .benchmarks import DEFAULT_SOURCES, HLE_SOURCE, verify_sources

    sources = list(DEFAULT_SOURCES) + ([HLE_SOURCE] if args.include_hle else [])
    checks = verify_sources(sources, hf_token=args.hf_token)
    for check in checks:
        print(check.line)
        if not check.ok and check.available_fields:
            print(f"       declared: question={check.source.question_field!r} "
                  f"answer={check.source.answer_field!r} "
                  f"choices={check.source.choices_field!r}")
    failed = [c for c in checks if not c.ok]
    print()
    print(f"{len(checks) - len(failed)}/{len(checks)} sources OK")
    if failed:
        print("Fix the field mappings in theriad/eval/benchmarks.py before building "
              "a corpus. Gated sources (GPQA, HLE) need HF_TOKEN and accepted terms.")
    return 1 if failed else 0


def _cmd_build_corpus(args) -> int:
    """Fetch and convert benchmark items into a vintage-tagged corpus."""
    from .benchmarks import DEFAULT_SOURCES, build_corpus, verify_sources
    from .corpus import write_corpus

    wanted = {n.strip() for n in args.sources.split(",")} if args.sources else None
    sources = [s for s in DEFAULT_SOURCES if wanted is None or s.name in wanted]
    if not sources:
        print(f"No sources matched {args.sources!r}.", file=sys.stderr)
        return 2

    # Verify before fetching in bulk: a drifted schema costs one request to
    # detect and a whole corpus of nonsense to miss.
    checks = verify_sources(sources, hf_token=args.hf_token)
    usable = [c.source for c in checks if c.ok]
    for check in checks:
        print(check.line)
    if not usable:
        print("\nNo usable sources.", file=sys.stderr)
        return 1
    if len(usable) < len(sources) and not args.skip_failed:
        print(
            "\nSome sources failed. Re-run with --skip-failed to build from the "
            "rest, or fix the mappings.",
            file=sys.stderr,
        )
        return 1

    print()
    items = build_corpus(usable, args.per_source, hf_token=args.hf_token, seed=args.seed)
    write_corpus(items, args.corpus)
    print(f"wrote {len(items)} items to {args.corpus}")
    print(f"vintages: {vintages(items)}")
    by_source: dict[str, int] = {}
    for item in items:
        by_source[item.source] = by_source.get(item.source, 0) + 1
    print(f"sources:  {by_source}")
    return 0


def _cmd_report(args) -> int:
    items = load_corpus(args.corpus)
    grid = OutcomeGrid.load(args.grid)
    if len(grid) == 0:
        print(f"Grid at {args.grid} is empty — run 'populate' first.", file=sys.stderr)
        return 1
    models = grid.models()
    costs = call_costs(models, args.input_tokens, args.output_tokens)
    # Brackets are drawn from the registry, not from the whole grid. The grid
    # accumulates models the router can no longer select — slugs that were
    # measured before a registry change, or measured speculatively and never
    # adopted. Bracketing against one of those compares the router to a
    # strategy it could not have played: the Aug 2026 report picked
    # always-dearest[gpt-5.4-mini], a model absent from every tier, and the
    # resulting "we beat the expensive baseline" was beating nothing.
    from theriad.models.registry import ModelRegistry

    _registry = ModelRegistry()
    _selectable = {
        cand.model
        for tier in _registry.all_tiers()
        for cand in _registry.candidates_for_tier(tier)
    }
    bracket_models = [m for m in models if m in _selectable]
    if not bracket_models:
        print(
            "WARNING: no grid model is in the registry; bracketing against the "
            "raw grid instead. The baselines below may be unselectable models.",
            file=sys.stderr,
        )
        bracket_models = list(models)
    cheapest = min(bracket_models, key=lambda m: costs.get(m, 0.0))
    dearest = max(bracket_models, key=lambda m: costs.get(m, 0.0))
    values = [float(v) for v in args.values.split(",")]

    from .routers import fixed_router, theriad_router, unmeasured_models

    router = theriad_router()
    missing = unmeasured_models(router, items, grid)
    if missing:
        print(
            "WARNING: the router selects models the grid never measured: "
            f"{sorted(missing)}.\nThey score on the uniform prior (p=0.5) and, if "
            "unpriced, at zero cost, so the numbers below are partly fiction.\n"
            "Extend coverage or restrict the registry before trusting this.\n",
            file=sys.stderr,
        )

    candidates = {
        "theriad(unified)": router,
        f"always-cheapest[{cheapest}]": fixed_router(cheapest),
        f"always-dearest[{dearest}]": fixed_router(dearest),
    }

    print(f"{'V (USD)':>9}  {'router':<34}{'regret/1k':>11}{'match':>8}"
          f"{'cost/1k':>10}{'accuracy':>10}")
    for v in values:
        for name, fn in candidates.items():
            r = build_report(
                grid, items, fn, v, models,
                max_samples=args.max_samples,
                sample_correlation=args.sample_correlation,
            )["summary"]
            print(
                f"{v:>9.3f}  {name:<34}"
                f"{r['regret_per_1k_requests_usd']:>11.3f}"
                f"{r['match_rate']:>8.2f}"
                f"{r['router_cost_usd'] / max(len(items), 1) * 1000:>10.3f}"
                f"{r['router_accuracy']:>10.3f}"
            )
        print()

    if args.json:
        import json

        payload = {
            name: sweep_value(grid, items, fn, values, models)
            for name, fn in candidates.items()
        }
        Path(args.json).write_text(json.dumps(payload, indent=2))
        print(f"full report written to {args.json}")
    return 0


def _cmd_calibrate(args) -> int:
    """Compute router parameters from the grid, and optionally install them.

    Prints before it writes, always. The numbers this produces move production
    traffic and money, so they get read by a human once before they take effect
    — ``--write`` is the second step, not the default.
    """
    from .calibrate import ScaleFitError, build_calibration
    from .routers import theriad_router, unmeasured_models

    items = load_corpus(args.corpus)
    grid = OutcomeGrid.load(args.grid)
    if len(grid) == 0:
        print(f"Grid at {args.grid} is empty — run 'populate' first.", file=sys.stderr)
        return 1

    router = theriad_router()
    missing = unmeasured_models(router, items, grid)
    if missing:
        print(
            "WARNING: the router selects models the grid never measured: "
            f"{sorted(missing)}.\nThe V sweep below scores them on the uniform "
            "prior (p=0.5), so their tiers' values are not evidence.\n",
            file=sys.stderr,
        )

    values = [float(v) for v in args.values.split(",")]
    tokens_per_call = args.input_tokens + args.output_tokens
    try:
        calibration, scale, skipped = build_calibration(
            grid, items, router, values, tokens_per_call=tokens_per_call
        )
    except ScaleFitError as exc:
        print(f"cannot calibrate: {exc}", file=sys.stderr)
        return 1

    print(
        f"scale:    quality = {scale.slope:.4f} x success_rate "
        f"{scale.intercept:+.4f}   ({len(scale.anchors)} anchors: "
        f"{', '.join(scale.anchors)})"
    )
    print(f"grid:     {grid.total_trials} trials over {len(grid)} cells")
    print(f"tokens:   {tokens_per_call}/call assumed for the V conversion")
    print()
    print(f"{'model':<40}{'axis':<12}{'quality':>9}")
    for model, axes in calibration.quality_profiles.items():
        for axis, score in axes.items():
            print(f"{model:<40}{axis:<12}{score:>9.4f}")
    print()
    print(f"{'tier':<22}{'V (measured)':>14}{'V (hand-set)':>14}")
    for tier, v in calibration.value_of_quality.items():
        print(f"{tier:<22}{v:>14.3f}{TIER_VALUE_OF_QUALITY.get(tier, 0.0):>14.3f}")

    if skipped:
        print("\nnot installed, keeping the hand-set V:")
        for tier, reason in sorted(skipped.items()):
            print(f"  {tier:<22}{reason}")

    if not args.write:
        print("\nRe-run with --write to install this calibration.")
        return 0

    calibration.save(args.out)
    print(f"\nwritten to {args.out} — the router picks it up on next import.")
    return 0


def _cmd_populate(args) -> int:
    if not args.allow_paid_calls:
        print(
            "populate makes paid provider calls. Re-run with --allow-paid-calls.",
            file=sys.stderr,
        )
        return 2
    items = load_corpus(args.corpus)
    grid = OutcomeGrid.load(args.grid)
    cheap, premium, frontier = model_lists(args)
    coverage = default_coverage(cheap, premium, frontier, samples=args.samples)
    costs = call_costs(cheap + premium + frontier, args.input_tokens, args.output_tokens)
    plan = plan_grid(
        grid, items, coverage, costs, retry_truncated=args.retry_truncated
    )

    print(f"{plan.n_calls} calls, estimated ${plan.estimated_cost_usd:.2f}")
    if not args.yes:
        print("Re-run with --yes to execute.", file=sys.stderr)
        return 2
    if plan.estimated_cost_usd > args.budget:
        print(
            f"Estimate ${plan.estimated_cost_usd:.2f} exceeds --budget "
            f"${args.budget:.2f}; refusing.",
            file=sys.stderr,
        )
        return 2

    from .executors import openrouter_executor

    errors: list[tuple[str, str, str]] = []

    def _record_error(model: str, item_id: str, exc: Exception) -> None:
        errors.append((model, item_id, f"{type(exc).__name__}: {exc}"))

    aborted: PopulateAborted | None = None
    try:
        populate_grid(
            grid,
            items,
            plan,
            openrouter_executor(args.generation_cap),
            on_progress=lambda d, t: print(f"\r{d}/{t}", end="", flush=True),
            on_error=_record_error,
            # Sidecar next to the grid rather than a new CLI flag, so every
            # populate keeps its transcripts whether or not the caller asked.
            transcript_path=Path(args.grid).with_suffix(".transcripts.jsonl"),
            checkpoint_path=args.grid,
        )
    except PopulateAborted as exc:
        aborted = exc
    print()
    # Save whatever completed before an abort — the calls were already paid for.
    grid.save(args.grid)
    print(f"grid saved to {args.grid}: {len(grid)} cells, {grid.total_trials} trials")

    if errors:
        by_model: dict[str, int] = {}
        for model, _item_id, _msg in errors:
            by_model[model] = by_model.get(model, 0) + 1
        print(
            f"\nERROR: {len(errors)} of {plan.n_calls} calls failed and were NOT "
            f"recorded. Those cells are absent from the grid, so re-running this "
            f"command retries them."
        )
        for model, count in sorted(by_model.items(), key=lambda kv: -kv[1]):
            print(f"  {model:<40} {count} failed")
        seen: set[str] = set()
        print("\n  distinct errors:")
        for _model, _item_id, msg in errors:
            if msg not in seen:
                seen.add(msg)
                print(f"    {msg[:200]}")
            if len(seen) >= 5:
                break

    if aborted is not None:
        print(f"\nABORTED: {aborted}", file=sys.stderr)
        return 1

    truncated = grid.truncated_cells()
    if truncated:
        print(
            f"\nWARNING: {grid.total_truncated} calls were truncated by the output "
            f"cap across {len(truncated)} cells and were EXCLUDED from the "
            f"success-rate estimates.\nThose cells are measured on a biased "
            f"subsample. Raise --generation-cap (or leave it unset for no cap) "
            f"and re-run before trusting the grid."
        )
        for cell in truncated[:10]:
            print(
                f"  {cell.model:<36} {cell.item_id:<16} "
                f"{cell.truncated}/{cell.attempts} truncated"
            )
    return 1 if errors else 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="theriad.eval.run", description="Router evaluation harness"
    )
    p.add_argument("--corpus", default="eval_corpus.jsonl")
    p.add_argument("--grid", default="eval_grid.json")
    p.add_argument("--cheap", default="", help="comma-separated cheap-tier models")
    p.add_argument("--premium", default="", help="comma-separated premium-tier models")
    p.add_argument("--frontier", default="", help="comma-separated frontier-tier models")
    p.add_argument(
        "--model-set",
        choices=("manual", "replacement-study"),
        default="manual",
        help=(
            "named model list; replacement-study includes current routed models "
            "and the priceable replacement candidates"
        ),
    )
    p.add_argument("--samples", type=int, default=5, help="k samples per cell")
    p.add_argument("--input-tokens", type=int, default=DEFAULT_INPUT_TOKENS)
    p.add_argument(
        "--output-tokens",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help="assumed output length for COST ESTIMATION only; not a cap",
    )
    p.add_argument(
        "--generation-cap",
        type=int,
        default=DEFAULT_GENERATION_CAP,
        help="max_tokens sent to providers; omit for uncapped (recommended)",
    )
    p.add_argument(
        "--retry-truncated",
        action="store_true",
        help=(
            "explicitly retry cells with only truncated attempts; use with an "
            "uncapped or higher generation limit"
        ),
    )

    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("plan").set_defaults(func=_cmd_plan)

    ver = sub.add_parser(
        "verify-sources", help="check benchmark field mappings (no provider spend)"
    )
    ver.add_argument("--hf-token", default=None)
    ver.add_argument("--include-hle", action="store_true")
    ver.set_defaults(func=_cmd_verify_sources)

    bc = sub.add_parser("build-corpus", help="fetch and convert benchmark items")
    bc.add_argument("--sources", default="", help="comma-separated source names")
    bc.add_argument("--per-source", type=int, default=50)
    bc.add_argument("--seed", type=int, default=42)
    bc.add_argument("--hf-token", default=None)
    bc.add_argument("--skip-failed", action="store_true",
                    help="build from usable sources even if some fail")
    bc.set_defaults(func=_cmd_build_corpus)

    rep = sub.add_parser("report")
    rep.add_argument("--values", default="0.01,0.05,0.10,0.50", help="V sweep, USD")
    rep.add_argument("--json", default=None, help="write the full sweep to this path")
    rep.add_argument(
        "--max-samples", type=int, default=1,
        help="largest n for best-of-n actions; 1 (default) makes the oracle a "
             "pure model-choice oracle. Raise only if a verifier exists at "
             "inference time.",
    )
    rep.add_argument(
        "--sample-correlation", type=float, default=None,
        help="redundancy between repeated samples [0-1]; default measures it "
             "from the grid's over-dispersion",
    )
    rep.set_defaults(func=_cmd_report)

    cal = sub.add_parser(
        "calibrate",
        help="turn the grid into router parameters (theriad/models/calibration.json)",
    )
    cal.add_argument("--values", default="0.01,0.05,0.10,0.50", help="V sweep, USD")
    cal.add_argument(
        "--write", action="store_true",
        help="write the calibration file; without it the result is printed only",
    )
    cal.add_argument(
        "--out", default=str(CALIBRATION_PATH),
        help="destination path for the calibration file",
    )
    cal.set_defaults(func=_cmd_calibrate)

    pop = sub.add_parser("populate")
    pop.add_argument("--allow-paid-calls", action="store_true")
    pop.add_argument("--yes", action="store_true")
    pop.add_argument("--budget", type=float, default=5.0)
    pop.set_defaults(func=_cmd_populate)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
