"""
Benchmark runner CLI entry point.

    python3 -m theriad.benchmark.run [options]

Options:
    --providers   Comma-separated list: openrouter,groq,cerebras,ollama
                  Default: all four
    --limit       Max number of prompts to run (stratified across categories)
                  Default: 30
    --budget      Hard spend ceiling in USD. Default: 10.0
    --output      Output directory for JSONL results. Default: ./benchmark_results
    --prompts     Path to benchmark_prompts.jsonl. Default: project root
    --yes, -y     Skip the paid-provider confirmation prompt
    --resume      Resume from an existing JSONL file, skipping completed rows
    --resume-file Path to existing JSONL to resume from (default: output/latest.jsonl)
    --preset      Named preset: smoke | overnight_safe | full
    --allow-large-local-models  Allow large (>14B) Ollama models (default: off)
    --skip-preflight            Skip disk/memory preflight checks (for CI/testing)
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m theriad.benchmark.run",
        description="Run Theriad benchmark across multiple providers.",
    )
    p.add_argument(
        "--providers", "-p",
        default="openrouter,groq,cerebras,ollama",
        help="Comma-separated provider list (default: all)",
    )
    p.add_argument(
        "--limit", "-n",
        type=int,
        default=30,
        help="Max prompts per run, stratified across categories (default: 30)",
    )
    p.add_argument(
        "--budget",
        type=float,
        default=10.0,
        help="Hard spend ceiling in USD (default: $10.00)",
    )
    p.add_argument(
        "--output", "-o",
        type=Path,
        default=Path("benchmark_results"),
        help="Output directory for JSONL results (default: ./benchmark_results)",
    )
    p.add_argument(
        "--prompts",
        type=Path,
        default=None,
        help="Path to benchmark_prompts.jsonl (default: auto-locate in project root)",
    )
    p.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip the paid-provider confirmation prompt",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing JSONL file, skipping completed rows",
    )
    p.add_argument(
        "--resume-file",
        type=Path,
        default=None,
        dest="resume_file",
        help="Existing JSONL to resume from (default: <output>/latest.jsonl)",
    )
    p.add_argument(
        "--preset",
        choices=["smoke", "overnight_safe", "full", "code_free_comparison"],
        default=None,
        help="Named preset that sets default providers/budget/caps "
             "(smoke|overnight_safe|full|code_free_comparison)",
    )
    p.add_argument(
        "--allow-large-local-models",
        action="store_true",
        dest="allow_large_local_models",
        help="Allow Ollama models > 14B (qwen3.6:27b, deepseek-r1, etc.)",
    )
    p.add_argument(
        "--skip-preflight",
        action="store_true",
        dest="skip_preflight",
        help="Skip disk/memory preflight checks (for CI or testing only)",
    )

    args = p.parse_args(argv)

    # Track which args were left at defaults so presets can override them.
    args._providers_default = (args.providers == "openrouter,groq,cerebras,ollama")
    args._limit_default     = (args.limit == 30)
    args._budget_default    = (args.budget == 10.0)

    return args


def _apply_preset(args: argparse.Namespace) -> dict | None:
    """Apply preset to args and return the preset's output_caps, or None."""
    if args.preset is None:
        return None
    from theriad.benchmark.presets import get_preset
    preset = get_preset(args.preset)
    preset.apply_to_args(args)
    print(f"\n  Preset: {preset.name}")
    print(f"  {preset.description}")
    return preset.output_caps


def _run_preflight(args: argparse.Namespace, provider_names: list[str]) -> bool:
    """Run preflight checks. Returns True if safe to proceed."""
    if args.skip_preflight:
        print("  [preflight skipped]")
        return True

    from theriad.benchmark.preflight import run_preflight
    result = run_preflight(
        providers=provider_names,
        budget_usd=args.budget,
        check_ollama="ollama" in provider_names,
    )
    result.print_report()

    if not result.ok_to_run:
        print("ERROR: Preflight checks failed. Aborting.", file=sys.stderr)
        return False

    # If disk is warn-level, block local benchmark automatically
    if not result.local_allowed and "ollama" in provider_names:
        print(
            "  ⚠  Low disk: Ollama lane will be disabled by preflight.\n"
            "     (remove --providers ollama or free disk space to enable)",
        )
        provider_names[:] = [p for p in provider_names if p != "ollama"]

    return True


def _confirm_paid(lanes, prompts, budget) -> bool:
    """Warn the user and get consent before calling paid providers."""
    from theriad.benchmark.formatting import format_usd
    from theriad.benchmark.runner import _estimate_lane_cost

    paid = [l for l in lanes if l.provider_name != "ollama"]
    if not paid:
        return True

    print("\n⚠️  PAID PROVIDER WARNING")
    print(f"   Budget ceiling: ${budget:.2f}")
    print(f"   Prompts per model: {len(prompts)}")
    print()
    for lane in paid:
        est = _estimate_lane_cost(lane, prompts)
        models = lane.models()
        est_str = format_usd(est) if est is not None else "pricing unavailable"
        print(f"   {lane.provider_name:12s}  {len(models)} model(s)  "
              f"~{est_str} estimated")
    print()
    answer = input("Proceed? [y/N] ").strip().lower()
    return answer == "y"


def _print_result(result) -> None:
    from theriad.benchmark.formatting import format_usd
    icon = "✓" if result.status == "success" else ("⏱" if result.status == "timeout" else "✗")
    cost = format_usd(result.actual_cost_usd)
    tps = f"{result.tokens_per_second:.0f} tok/s" if result.tokens_per_second else "     "
    qp = "Q✓" if result.quality_pass else ("Q✗" if result.quality_pass is False else "  ")
    print(
        f"  {icon} {result.provider:12s} {result.model[:30]:30s} "
        f"{result.prompt_id:12s} {result.provider_latency_ms:7.0f}ms "
        f"{tps:14s} {cost:12s} {qp}"
    )


async def _main(args: argparse.Namespace) -> int:
    from theriad.benchmark.prompts import load_prompts, stratified_sample
    from theriad.benchmark.resume import load_completed_keys
    from theriad.benchmark.runner import build_lanes, run_benchmark
    from theriad.benchmark.presets import get_preset

    # ── Apply preset (before reading other flags) ─────────────────────────────
    output_caps = _apply_preset(args)

    # ── Load prompts ──────────────────────────────────────────────────────────
    try:
        all_prompts = load_prompts(args.prompts)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    prompts = stratified_sample(all_prompts, args.limit)
    print(f"\nLoaded {len(prompts)} prompts "
          f"({', '.join(sorted({p.category for p in prompts}))})")

    # ── Build lanes (before preflight so we pass only active providers) ──────
    provider_names = [n.strip() for n in args.providers.split(",") if n.strip()]
    lanes = build_lanes(provider_names, allow_large_local_models=args.allow_large_local_models)
    active = [l for l in lanes if l.is_available()]

    if not active:
        print(
            "\nNo provider lanes available. Check API keys:\n"
            "  OPENROUTER_API_KEY, GROQ_API_KEY, CEREBRAS_API_KEY\n"
            "Or start Ollama locally.",
            file=sys.stderr,
        )
        return 1

    # ── Preflight (uses active provider names so absent Ollama doesn't block) ─
    active_provider_names = [l.provider_name for l in active]
    if not _run_preflight(args, active_provider_names):
        return 1

    print(f"\nActive providers:")
    for lane in active:
        print(f"  {lane.provider_name:12s}  {', '.join(lane.models())}")

    total_calls = sum(len(l.models()) for l in active) * len(prompts)
    print(f"\nTotal calls planned: {total_calls}")
    print(f"Budget ceiling:      ${args.budget:.2f}\n")

    # ── Resume ────────────────────────────────────────────────────────────────
    resume_path: Path | None = None
    if args.resume:
        resume_path = args.resume_file or (args.output / "latest.jsonl")

    from theriad.benchmark.resume import load_completed_keys, ResumeState
    resume_state = load_completed_keys(resume_path)
    resume_state.print_summary()

    if resume_state.completed:
        remaining_estimate = sum(
            1 for lane in active
            for model in lane.models()
            for bp in prompts
            if not resume_state.is_done(lane.provider_name, model, bp.id, bp.category)
        )
        print(f"  Remaining calls after resume filter: {remaining_estimate}")

    # ── Confirm paid calls ────────────────────────────────────────────────────
    if not args.yes:
        if not _confirm_paid(active, prompts, args.budget):
            print("Aborted.")
            return 0

    # ── Run ───────────────────────────────────────────────────────────────────
    output_path = args.output / "latest.jsonl"
    args.output.mkdir(parents=True, exist_ok=True)

    print("\n  ✓/✗  Provider       Model                          Prompt       "
          "Latency    Tok/s          Cost         QA")
    print("  " + "-" * 110)

    results = await run_benchmark(
        lanes=active,
        prompts=prompts,
        budget_usd=args.budget,
        output_path=output_path,
        on_result=_print_result,
        resume_state=resume_state,
        output_caps=output_caps,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    success = sum(1 for r in results if r.status == "success")
    errors = sum(1 for r in results if r.status in ("error", "timeout"))
    skipped = sum(1 for r in results if r.status in ("skipped", "budget_exceeded"))
    total_cost = sum(r.actual_cost_usd for r in results if r.actual_cost_usd is not None)

    print(f"\n{'─' * 60}")
    if not results:
        print(f"  Results:  0 new rows — benchmark already complete (resume)")
    else:
        print(f"  Results:  {success} success  {errors} error  {skipped} skipped")
        print(f"  Total cost (this run):  ${total_cost:.4f}")
    if resume_state.completed:
        print(f"  Previously completed:   {resume_state.completed} rows (skipped)")
    print(f"  Saved to:    {output_path.resolve()}")
    print(f"\n  Run summary:  python3 -m theriad.benchmark.summary {output_path}")

    return 0 if errors == 0 else 1


def main(argv=None):
    args = _parse_args(argv)
    sys.exit(asyncio.run(_main(args)))


if __name__ == "__main__":
    main()
