"""
Benchmark presets.

Presets bundle a set of benchmark options so common configurations can be
selected with a single --preset flag rather than specifying every flag by hand.

Available presets:
  smoke                — 6 prompts, MiMo + llama3.1:8b, very low output caps
  overnight_safe       — resume-safe, OpenRouter MiMo + small Ollama only, $2 budget
  full                 — resume-safe, all providers, $10 budget
  code_free_comparison — cohere/north-mini-code:free vs llama3.1:8b / MiMo; code+instruct+extract+reason

Usage:
    from theriad.benchmark.presets import get_preset, BenchmarkPreset
    preset = get_preset("overnight_safe")
    # apply preset values as defaults, CLI flags override
"""
from __future__ import annotations

from dataclasses import dataclass, field


# Per-category hard output token caps (prevents giant outputs consuming disk/time).
# Values chosen to be sufficient for quality assessment while bounding cost.
DEFAULT_OUTPUT_CAPS: dict[str, int] = {
    "summarize":  512,
    "extract":    512,
    "code":      1200,
    "instruct":   800,
    "creative":   800,
}

SMOKE_OUTPUT_CAPS: dict[str, int] = {
    "summarize":  256,
    "extract":    256,
    "code":       512,
    "instruct":   256,
    "creative":   256,
}


@dataclass
class BenchmarkPreset:
    name: str
    providers: list[str]
    prompt_limit: int
    budget_usd: float
    allow_large_local_models: bool
    output_caps: dict[str, int]
    resume: bool
    description: str = ""

    def apply_to_args(self, args) -> None:
        """
        Apply preset defaults to an argparse Namespace.
        Only overwrites values that the user left at their argparse defaults
        (i.e. this is a soft merge — explicit CLI flags win).
        """
        # Providers: only override if user left the default ("all")
        if getattr(args, "_providers_default", True):
            args.providers = ",".join(self.providers)
        # Prompt limit
        if getattr(args, "_limit_default", True):
            args.limit = self.prompt_limit
        # Budget
        if getattr(args, "_budget_default", True):
            args.budget = self.budget_usd
        # Resume
        if not args.resume:
            args.resume = self.resume
        # Large models
        if not args.allow_large_local_models:
            args.allow_large_local_models = self.allow_large_local_models


# ── Preset definitions ────────────────────────────────────────────────────────

SMOKE = BenchmarkPreset(
    name="smoke",
    providers=["openrouter", "ollama"],
    prompt_limit=6,
    budget_usd=0.50,
    allow_large_local_models=False,
    output_caps=SMOKE_OUTPUT_CAPS,
    resume=False,
    description=(
        "6 prompts, OpenRouter MiMo + Ollama llama3.1:8b, low output caps. "
        "Fast sanity check."
    ),
)

OVERNIGHT_SAFE = BenchmarkPreset(
    name="overnight_safe",
    providers=["openrouter", "ollama"],
    prompt_limit=30,
    budget_usd=2.00,
    allow_large_local_models=False,
    output_caps=DEFAULT_OUTPUT_CAPS,
    resume=True,
    description=(
        "Resume-safe, small Ollama models only (llama3.1:8b, llama3.2:latest), "
        "OpenRouter MiMo. Skips qwen3.6:27b and deepseek-r1. Budget $2."
    ),
)

FULL = BenchmarkPreset(
    name="full",
    providers=["openrouter", "groq", "cerebras", "ollama"],
    prompt_limit=30,
    budget_usd=10.00,
    allow_large_local_models=False,
    output_caps=DEFAULT_OUTPUT_CAPS,
    resume=True,
    description=(
        "All providers, resume-safe, preflight-protected, $10 budget. "
        "Still skips large local models unless --allow-large-local-models."
    ),
)

# Output caps for the code_free_comparison preset — strict to keep latency/cost bounded.
CODE_FREE_COMPARISON_CAPS: dict[str, int] = {
    "code":      1200,
    "instruct":   800,
    "extract":    512,
    "summarize":  512,
    "creative":   512,
}

# Compares cohere/north-mini-code:free against free/cheap baselines.
# Focuses on code_01–05, instruct_01–05, extract_01–05, reason_01–05.
# qwen/qwen3.6-plus is included as a paid baseline only if budget allows.
# Do NOT promote cohere/north-mini-code:free to production tiers until
# this preset has been run and the overnight report reviewed.
CODE_FREE_COMPARISON = BenchmarkPreset(
    name="code_free_comparison",
    providers=["openrouter", "ollama"],
    prompt_limit=20,          # 5 per category × 4 categories (code/instruct/extract/reason)
    budget_usd=0.50,          # mostly free models; small allowance for qwen baseline
    allow_large_local_models=False,
    output_caps=CODE_FREE_COMPARISON_CAPS,
    resume=True,
    description=(
        "Evaluate cohere/north-mini-code:free as a free code candidate. "
        "Compares against ollama/llama3.1:8b, xiaomi/mimo-v2.5-20260422, "
        "and qwen/qwen3.6-plus-04-02 (budget permitting). "
        "Categories: code, instruct, extract, reason (control). "
        "No production routing until benchmarked. Budget $0.50."
    ),
)

_PRESETS: dict[str, BenchmarkPreset] = {
    "smoke":                SMOKE,
    "overnight_safe":       OVERNIGHT_SAFE,
    "full":                 FULL,
    "code_free_comparison": CODE_FREE_COMPARISON,
}


def get_preset(name: str) -> BenchmarkPreset:
    preset = _PRESETS.get(name)
    if preset is None:
        available = ", ".join(_PRESETS)
        raise ValueError(f"Unknown preset {name!r}. Available: {available}")
    return preset


def list_presets() -> list[BenchmarkPreset]:
    return list(_PRESETS.values())
