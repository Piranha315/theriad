"""
Theriad CLI — cost-first AI inference optimizer.

Commands:
  theriad route     — classify a prompt, show tier + selected model
  theriad explain   — full tier/model/cost explanation of all layer decisions
  theriad simulate  — offline cost simulation against a baseline model (no paid calls)
  theriad stats     — show aggregate metrics from local JSONL store
  theriad config    — show/validate active configuration
  theriad benchmark — benchmark routing latency across all layers

Design:
- Click for command parsing (industry standard, clean API).
- Rich for output (already a dep; much better than print for tables/JSON).
- Each command is self-contained. No shared state between commands.
- 'theriad simulate' is ALWAYS offline; no provider calls.
- Commands that call providers (future 'route --execute') must require explicit flags.

ADR D-018: simulate is free by default; --allow-paid-calls is not yet implemented
           because dry-run mode is sufficient for the current use case.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from rich import print as rprint
from rich.console import Console
from rich.table import Table
from theriad.models.registry import DEFAULT_BASELINE_MODEL

console = Console()


def _build_layer0():
    from theriad.layers.layer0.layer import build_default_layer
    return build_default_layer()


def _build_layer1():
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    return FeatureRoutingLayer()


def _load_config(config_path: str | None):
    from theriad.core.config.loader import load_config
    return load_config(config_path)


def _default_registry():
    from theriad.models.registry import DEFAULT_REGISTRY
    return DEFAULT_REGISTRY


def _default_resolver():
    from theriad.models.policy import PolicyResolver
    return PolicyResolver()


def _estimate_prompt_tokens(prompt: str) -> int:
    """Rough token count: words × 1.33."""
    return max(1, int(len(prompt.split()) * 1.33))


@click.group()
@click.version_option(package_name="theriad")
def cli():
    """Theriad — cost-first AI inference optimization engine."""


# ── theriad route ──────────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt", required=False)
@click.option("--file", "-f", "prompt_file", help="Read prompt from file")
@click.option("--config", "-c", help="Path to theriad.yaml config file")
@click.option(
    "--policy",
    default=None,
    help="Policy override (balanced|aggressive_savings|quality_first|free_moe_experimental)",
)
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def route(
    prompt: str | None,
    prompt_file: str | None,
    config: str | None,
    policy: str | None,
    as_json: bool,
):
    """Classify a prompt, resolve tier to model, and show the routing decision."""
    if prompt_file:
        prompt = Path(prompt_file).read_text()
    elif prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            click.echo("Provide a prompt as argument, --file, or via stdin.", err=True)
            sys.exit(1)

    layer0 = _build_layer0()
    layer1 = _build_layer1()

    ctx: dict = {}
    result = layer0.route(prompt, ctx) or layer1.route(prompt, ctx)

    if result is None:
        cfg = _load_config(config)
        from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
        decision = RoutingDecision(
            tier=cfg.router.fallback_tier,
            confidence=0.0,
            reason="all layers abstained — using config fallback_tier",
        )
        result = RoutingResult(decision=decision, layer_name="fallback")

    # Resolve tier → model
    resolver = _default_resolver()
    cfg = _load_config(config)
    active_policy = policy or cfg.router.policy
    resolution = resolver.resolve(
        result.decision.tier,
        active_policy,
        result.decision.confidence,
        message=prompt,
    )

    if as_json:
        output = {
            "tier": resolution.tier,
            "model": resolution.model,
            "provider": resolution.provider,
            "policy": active_policy,
            "confidence": result.decision.confidence,
            "reason": result.decision.reason,
            "layer": result.layer_name,
            "signals": result.decision.signals,
            "fallback_chain": resolution.fallback_tuples(),
        }
        click.echo(json.dumps(output, indent=2))
    else:
        rprint(f"\n[bold green]Tier:[/bold green]     {resolution.tier}")
        rprint(f"[bold green]Model:[/bold green]    {resolution.model}")
        rprint(f"[bold]Provider:[/bold]  {resolution.provider}")
        rprint(f"[bold]Policy:[/bold]    {active_policy}")
        rprint(f"[bold]Confidence:[/bold] {result.decision.confidence:.1%}")
        rprint(f"[bold]Reason:[/bold]    {result.decision.reason}")
        rprint(f"[dim]Layer:[/dim]     {result.layer_name}\n")


# ── theriad explain ────────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt", required=False)
@click.option("--file", "-f", "prompt_file", help="Read prompt from file")
@click.option("--policy", default="balanced", help="Policy to use for model resolution")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def explain(
    prompt: str | None,
    prompt_file: str | None,
    policy: str,
    as_json: bool,
):
    """Show full routing path, tier, selected model, cost estimate, and signals."""
    if prompt_file:
        prompt = Path(prompt_file).read_text()
    elif prompt is None:
        if not sys.stdin.isatty():
            prompt = sys.stdin.read()
        else:
            click.echo("Provide a prompt as argument or --file.", err=True)
            sys.exit(1)

    from theriad.accounting.pricing import PricingTable, estimate_cost
    from theriad.layers.layer0.rules import (
        CodeDetectionRule,
        JsonDetectionRule,
        LanguageDetectionRule,
        MarkdownDetectionRule,
        PromptLengthRule,
    )
    from theriad.layers.layer1_features.extractor import FeatureExtractor

    # ── Run all Layer 0 rules ────────────────────────────────────────────────
    rules = [
        JsonDetectionRule(), CodeDetectionRule(), MarkdownDetectionRule(),
        LanguageDetectionRule(), PromptLengthRule(),
    ]
    layer0_decisions = []
    winning_decision = None
    for rule in rules:
        d = rule.evaluate(prompt)
        layer0_decisions.append((rule.name, d))
        if d is not None and d.confidence >= 0.75 and winning_decision is None:
            winning_decision = d

    # ── Layer 1 features ─────────────────────────────────────────────────────
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer
    extractor = FeatureExtractor()
    features = extractor.extract(prompt)
    layer1_result = FeatureRoutingLayer().route(prompt, {})
    if winning_decision is None and layer1_result is not None:
        winning_decision = layer1_result.decision

    # ── Resolve tier → model ─────────────────────────────────────────────────
    if winning_decision is None:
        from theriad.core.routing.interfaces import RoutingDecision
        winning_decision = RoutingDecision(tier="cheap_bulk", confidence=0.0,
                                           reason="all layers abstained")

    resolver = _default_resolver()
    resolution = resolver.resolve(
        winning_decision.tier,
        policy,
        winning_decision.confidence,
        message=prompt,
    )

    # ── Cost estimate ─────────────────────────────────────────────────────────
    pt = PricingTable()
    prompt_tokens = _estimate_prompt_tokens(prompt)
    est_output = 200  # conservative default
    cost = estimate_cost(resolution.model, prompt_tokens, est_output, pt)

    if as_json:
        click.echo(json.dumps({
            "routing_path": [
                {"layer": f"L0:{name}", "fired": d is not None,
                 "tier": d.tier if d else None, "confidence": d.confidence if d else None,
                 "reason": d.reason if d else "abstained", "signals": d.signals if d else {}}
                for name, d in layer0_decisions
            ] + ([{
                "layer": "L1:features",
                "fired": layer1_result is not None,
                "tier": layer1_result.decision.tier if layer1_result else None,
                "confidence": layer1_result.decision.confidence if layer1_result else None,
                "reason": layer1_result.decision.reason if layer1_result else "abstained",
            }]),
            "winning_layer": "layer0" if any(d for _, d in layer0_decisions if d and d.confidence >= 0.75) else (
                "layer1_features" if layer1_result else "fallback"
            ),
            "tier": resolution.tier,
            "model": resolution.model,
            "provider": resolution.provider,
            "policy": policy,
            "confidence": winning_decision.confidence,
            "reason": winning_decision.reason,
            "signals": winning_decision.signals,
            "estimated_prompt_tokens": prompt_tokens,
            "estimated_output_tokens": est_output,
            "estimated_cost_usd": round(cost, 8),
            "fallback_chain": resolution.fallback_tuples(),
            "features": {
                "token_count": features.token_count,
                "code_ratio": round(features.code_ratio, 3),
                "json_probability": round(features.json_probability, 3),
                "markdown_ratio": round(features.markdown_ratio, 3),
            },
        }, indent=2))
        return

    # ── Rich table output ─────────────────────────────────────────────────────
    table = Table(title="Routing Layer Decisions", show_header=True)
    table.add_column("Layer / Rule", style="cyan")
    table.add_column("Fired?", style="green")
    table.add_column("Tier", style="yellow")
    table.add_column("Confidence")
    table.add_column("Reason")

    for name, d in layer0_decisions:
        table.add_row(
            f"L0:{name}",
            "✓" if d else "—",
            d.tier if d else "",
            f"{d.confidence:.0%}" if d else "",
            d.reason if d else "abstained",
        )

    table.add_row(
        "L1:features",
        "✓" if layer1_result else "—",
        layer1_result.decision.tier if layer1_result else "",
        f"{layer1_result.decision.confidence:.0%}" if layer1_result else "",
        (layer1_result.decision.reason if layer1_result else
         f"tokens={features.token_count} code_ratio={features.code_ratio:.2f}"),
    )

    console.print(table)

    rprint("\n[bold]── Resolution ──[/bold]")
    rprint(f"  [bold green]Tier:[/bold green]           {resolution.tier}")
    rprint(f"  [bold green]Model:[/bold green]          {resolution.model}")
    rprint(f"  [bold]Provider:[/bold]       {resolution.provider}")
    rprint(f"  [bold]Policy:[/bold]         {policy}")
    rprint(f"  [bold]Confidence:[/bold]     {winning_decision.confidence:.1%}")
    rprint(f"  [bold]Reason:[/bold]         {winning_decision.reason}")
    if winning_decision.signals:
        rprint(f"  [bold]Signals:[/bold]        {winning_decision.signals}")
    rprint("\n[bold]── Cost Estimate ──[/bold]")
    rprint(f"  Prompt tokens:  ~{prompt_tokens}")
    rprint(f"  Output tokens:  ~{est_output} (default estimate)")
    rprint(f"  Estimated cost: ${cost:.6f}" if cost else "  Estimated cost: unknown (model not in pricing table)")
    rprint("\n[bold]── Fallback Chain ──[/bold]")
    for i, (fb_prov, fb_model) in enumerate(resolution.fallback_tuples(), 1):
        rprint(f"  {i}. {fb_prov}/{fb_model}")
    if not resolution.fallback_tuples():
        rprint("  (none)")
    rprint(f"\n[dim]Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}[/dim]\n")


# ── theriad simulate ───────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt_file")
@click.option("--baseline", default=DEFAULT_BASELINE_MODEL, show_default=True,
              help="Baseline model to compare against")
@click.option("--policy", default="balanced", show_default=True,
              help="Routing policy (balanced|aggressive_savings|quality_first|free_moe_experimental)")
@click.option("--default-output-tokens", default=200, show_default=True,
              help="Default output token estimate when not specified in JSONL")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def simulate(
    prompt_file: str,
    baseline: str,
    policy: str,
    default_output_tokens: int,
    as_json: bool,
):
    """
    Dry-run cost simulation — NO provider calls are made.

    Routes each prompt in PROMPT_FILE (JSONL) through Theriad's routing layers,
    resolves to a concrete model via the chosen policy, estimates token costs,
    and compares against a baseline model.

    PROMPT_FILE format (one JSON object per line):
      {"prompt": "Summarize this text..."}

    Optional fields per line:
      {"id": "task_001", "prompt": "...", "expected_output_tokens": 600, "task_type": "code"}
    """
    from theriad.accounting.pricing import PricingTable, estimate_cost

    path = Path(prompt_file)
    if not path.exists():
        click.echo(f"Error: file not found: {prompt_file}", err=True)
        sys.exit(1)

    lines = [ln.strip() for ln in path.read_text().splitlines() if ln.strip()]
    if not lines:
        click.echo("Error: prompt file is empty.", err=True)
        sys.exit(1)

    layer0 = _build_layer0()
    layer1 = _build_layer1()
    resolver = _default_resolver()
    pt = PricingTable()

    tier_counts: dict[str, int] = {}
    theriad_total_cost = 0.0
    baseline_total_cost = 0.0
    premium_count = 0
    unknown_cost_count = 0

    results = []

    for raw in lines:
        try:
            obj = json.loads(raw)
        except json.JSONDecodeError:
            continue
        prompt_text = obj.get("prompt", "")
        if not prompt_text:
            continue

        out_tokens = int(obj.get("expected_output_tokens", default_output_tokens))
        prompt_tokens = _estimate_prompt_tokens(prompt_text)

        # Route offline (no provider calls)
        ctx: dict = {}
        result = layer0.route(prompt_text, ctx) or layer1.route(prompt_text, ctx)
        if result is None:
            from theriad.core.routing.interfaces import RoutingDecision, RoutingResult
            result = RoutingResult(
                decision=RoutingDecision(tier="cheap_bulk", confidence=0.0,
                                         reason="all layers abstained"),
                layer_name="fallback",
            )

        tier = result.decision.tier

        # Resolve tier → model
        resolution = resolver.resolve(
            tier,
            policy,
            result.decision.confidence,
            message=prompt_text,
        )
        tier = resolution.tier
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if tier == "premium":
            premium_count += 1
        theriad_model = resolution.model

        # Cost estimate
        theriad_cost = estimate_cost(theriad_model, prompt_tokens, out_tokens, pt)
        baseline_cost = estimate_cost(baseline, prompt_tokens, out_tokens, pt)

        if theriad_cost == 0.0 and theriad_model not in ("llama3", "mistral", "gemma", "phi3"):
            # Distinguish "free" from "unknown"
            if pt.get(theriad_model) is None:
                unknown_cost_count += 1

        theriad_total_cost += theriad_cost
        baseline_total_cost += baseline_cost

        results.append({
            "id": obj.get("id"),
            "tier": tier,
            "theriad_model": theriad_model,
            "prompt_tokens": prompt_tokens,
            "output_tokens": out_tokens,
            "theriad_cost_usd": theriad_cost,
            "baseline_cost_usd": baseline_cost,
        })

    n = len(results)
    if n == 0:
        click.echo("No valid prompts found in file.", err=True)
        sys.exit(1)

    savings_pct = (
        (baseline_total_cost - theriad_total_cost) / baseline_total_cost * 100
        if baseline_total_cost > 0
        else 0.0
    )
    premium_pct = premium_count / n * 100
    top_tiers = sorted(tier_counts.items(), key=lambda x: -x[1])

    if as_json:
        click.echo(json.dumps({
            "prompts": n,
            "baseline_model": baseline,
            "policy": policy,
            "baseline_estimated_cost_usd": round(baseline_total_cost, 6),
            "theriad_estimated_cost_usd": round(theriad_total_cost, 6),
            "estimated_savings_pct": round(savings_pct, 2),
            "premium_usage_pct": round(premium_pct, 2),
            "tier_counts": dict(top_tiers),
            "unknown_cost_models": unknown_cost_count,
            "note": "Dry-run only. No provider calls were made.",
        }, indent=2))
        return

    rprint("\n[bold]Theriad Simulate — Dry Run[/bold]  [dim](no provider calls made)[/dim]\n")
    rprint(f"  Prompts:                 {n}")
    rprint(f"  Baseline model:          {baseline}")
    rprint(f"  Policy:                  {policy}")
    rprint(f"  Baseline estimated cost: [red]${baseline_total_cost:.4f}[/red]")
    rprint(f"  Theriad estimated cost:   [green]${theriad_total_cost:.4f}[/green]")
    rprint(f"  Estimated savings:       [bold green]{savings_pct:.1f}%[/bold green]")
    rprint(f"  Premium usage:           {premium_pct:.1f}%")
    if unknown_cost_count:
        rprint(f"  [yellow]Unknown-price models:    {unknown_cost_count} prompts (shown as $0.00)[/yellow]")

    t = Table(title="\nTop Selected Tiers")
    t.add_column("Tier", style="cyan")
    t.add_column("Count", justify="right")
    t.add_column("Share", justify="right")
    for tier_name, count in top_tiers:
        t.add_row(tier_name, str(count), f"{count/n:.1%}")
    console.print(t)
    rprint()


# ── theriad stats ──────────────────────────────────────────────────────────────

@cli.command()
@click.option("--config", "-c", help="Path to theriad.yaml")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def stats(config: str | None, as_json: bool):
    """Show aggregate cost and usage statistics from local metrics store."""
    cfg = _load_config(config)
    from theriad.accounting.tracker import CostTracker

    records = CostTracker.load_from_disk(cfg.metrics_path)
    if not records:
        rprint("[yellow]No metrics recorded yet. Run 'theriad route' first.[/yellow]")
        return

    total_requests = len(records)
    total_tokens = sum(r.get("total_tokens", 0) for r in records)
    total_cost = sum(r.get("estimated_cost_usd", 0.0) for r in records)
    avg_latency = sum(r.get("latency_ms", 0) for r in records) / total_requests
    fallbacks = sum(1 for r in records if r.get("used_fallback"))
    models: dict[str, int] = {}
    for r in records:
        m = r.get("model", "unknown")
        models[m] = models.get(m, 0) + 1

    if as_json:
        click.echo(json.dumps({
            "total_requests": total_requests,
            "total_tokens": total_tokens,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(avg_latency, 2),
            "fallback_rate": fallbacks / total_requests,
            "models_used": models,
        }, indent=2))
        return

    rprint(f"\n[bold]Theriad Stats[/bold] ({total_requests} requests)\n")
    rprint(f"  Total tokens:    {total_tokens:,}")
    rprint(f"  Total cost:      ${total_cost:.4f}")
    rprint(f"  Avg latency:     {avg_latency:.0f}ms")
    rprint(f"  Fallback rate:   {fallbacks / total_requests:.1%}")

    t = Table(title="\nModels Used")
    t.add_column("Model")
    t.add_column("Requests", justify="right")
    t.add_column("Share", justify="right")
    for model, count in sorted(models.items(), key=lambda x: -x[1]):
        t.add_row(model, str(count), f"{count/total_requests:.1%}")
    console.print(t)


# ── theriad config ─────────────────────────────────────────────────────────────

@cli.command("config")
@click.option("--config", "-c", help="Path to theriad.yaml")
@click.option("--json", "as_json", is_flag=True, help="Output as JSON")
def show_config(config: str | None, as_json: bool):
    """Show the active configuration."""
    cfg = _load_config(config)
    if as_json:
        click.echo(cfg.model_dump_json(indent=2))
    else:
        rprint("\n[bold]Active Theriad Configuration[/bold]\n")
        rprint(f"  Fallback tier:     {cfg.router.fallback_tier}")
        rprint(f"  Active policy:     {cfg.router.policy}")
        rprint(f"  Min confidence:    {cfg.router.min_confidence}")
        rprint(f"  Active layers:     {', '.join(cfg.router.layers)}")
        rprint(f"  Metrics path:      {cfg.metrics_path}")
        rprint(f"  Log level:         {cfg.logging.level}")
        providers = list(cfg.providers.keys()) or ["(none configured)"]
        rprint(f"  Providers:         {', '.join(providers)}\n")


# ── theriad benchmark ──────────────────────────────────────────────────────────

@cli.command()
@click.argument("prompt", required=False)
@click.option("--file", "-f", "prompt_file", help="Read prompt from file")
@click.option("--iterations", "-n", default=10, show_default=True, help="Number of iterations")
def benchmark(prompt: str | None, prompt_file: str | None, iterations: int):
    """
    Benchmark routing latency for a prompt across all layers.

    NOTE: This command measures routing speed only and does NOT call providers.
    To run quality/accuracy benchmarks that call providers, use a separate eval
    harness (see evals/MODEL_EVALS.md) with explicit credentials.
    """
    if prompt_file:
        prompt = Path(prompt_file).read_text()
    elif prompt is None:
        prompt = "What is the capital of France?"

    from theriad.layers.layer0.layer import build_default_layer
    from theriad.layers.layer1_features.layer import FeatureRoutingLayer

    layers = [("layer0", build_default_layer()), ("layer1", FeatureRoutingLayer())]

    rprint(f"\n[bold]Benchmark[/bold] — {iterations} iterations\n")
    rprint(f"Prompt: {prompt[:60]}{'...' if len(prompt) > 60 else ''}\n")

    t = Table()
    t.add_column("Layer")
    t.add_column("Avg (ms)", justify="right")
    t.add_column("Min (ms)", justify="right")
    t.add_column("Max (ms)", justify="right")

    for layer_name, layer in layers:
        times = []
        for _ in range(iterations):
            t0 = time.perf_counter()
            layer.route(prompt, {})
            times.append((time.perf_counter() - t0) * 1000)
        avg = sum(times) / len(times)
        t.add_row(layer_name, f"{avg:.2f}", f"{min(times):.2f}", f"{max(times):.2f}")

    console.print(t)
    rprint()
