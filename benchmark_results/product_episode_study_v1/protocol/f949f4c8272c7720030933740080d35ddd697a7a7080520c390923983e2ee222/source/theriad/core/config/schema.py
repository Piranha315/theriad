"""Pydantic config schemas for Theriad."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ProviderConfig(BaseModel):
    """Configuration for a single inference provider."""
    name: str
    base_url: str
    api_key: str = ""
    timeout_seconds: float = 30.0
    max_retries: int = 3
    extra: dict[str, Any] = Field(default_factory=dict)


class ModelCandidateConfig(BaseModel):
    """Per-candidate override in model_tiers config."""
    provider: str
    model: str
    role: str = "primary"
    input_cost_per_million: float | None = None
    output_cost_per_million: float | None = None
    context_window: int | None = None
    quality_prior: float | None = None
    notes: str | None = None


class TierConfig(BaseModel):
    """Config block for a single tier in model_tiers."""
    description: str = ""
    candidates: list[ModelCandidateConfig] = Field(default_factory=list)


class RouterConfig(BaseModel):
    """Routing policy configuration."""
    # Confidence threshold below which next routing layer is consulted
    min_confidence: float = 0.75
    # Tier to use when routing confidence is still low after all layers
    fallback_tier: str = "cheap_bulk"
    # Ordered list of active routing layer names
    layers: list[str] = Field(default_factory=lambda: ["layer0", "layer1_features"])
    # Active optimization policy: aggressive_savings | balanced | quality_first
    # | free_moe_experimental (explicit opt-in)
    policy: str = "balanced"
    # When True, and Theriad's routing confidence is below min_confidence, the
    # last-resort fallback is openrouter/auto (OpenRouter's own auto-router).
    # Disabled by default: openrouter/auto is NOT the default Theriad router.
    enable_openrouter_auto_fallback: bool = False


class LoggingConfig(BaseModel):
    level: str = "INFO"
    format: str = "json"   # "json" | "text"
    file: str | None = None


class QualityLoopPassLimits(BaseModel):
    """Per-policy maximum refinement passes (0 = loop disabled for that policy)."""
    aggressive_savings: int = 0
    balanced: int = 1
    quality_first: int = 2
    fast: int = 0


class QualityLoopConfig(BaseModel):
    """
    Configuration for the bounded quality-loop / refinement-pass orchestration.

    Quality loops run bounded draft → critique → repair → validate cycles to
    improve output quality.  They are cost-first: loops only fire when policy
    and validation signals justify the extra spend.

    Normal /api/execute is unchanged unless quality_loop_enabled=True is
    explicitly passed in the request OR the policy enables it here.
    """
    enabled: bool = True
    max_passes: QualityLoopPassLimits = Field(default_factory=QualityLoopPassLimits)
    # Loop stops early when accumulated cost exceeds this multiple of the
    # original draft cost (e.g. 2.0 = allow up to 2× the draft cost in loops).
    max_extra_cost_multiplier: float = 2.0
    # Hard ceiling on total extra wall-clock seconds across all loop passes.
    max_extra_latency_seconds: float = 120.0
    # When True, the loop stops as soon as a validation pass succeeds —
    # even if max_passes has not been reached.
    stop_if_validation_passes: bool = True
    # When True, a loop pass is only started if a concrete validation signal
    # is available (tests, schema check, lint, etc.).  When False, the loop
    # may run on model-only critique (higher chance of useless churn).
    require_validation_signal: bool = True


class BrowserConfig(BaseModel):
    """Configuration for the browser agent feature."""
    enabled: bool = False
    max_steps: int = 20
    headless: bool = True
    # "strict" requires approval for all sensitive actions; "minimal" only for
    # the hard-stop set defined in theriad.runtime.profile.HARD_STOP_ACTIONS.
    approval_mode: str = "strict"
    timeout: float = 30.0
    # When non-empty, navigation is restricted to these domains.
    allowed_domains: list[str] = Field(default_factory=list)


class TheriadConfig(BaseModel):
    """Root configuration object."""
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    router: RouterConfig = Field(default_factory=RouterConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    metrics_path: str = "~/.theriad/metrics.jsonl"
    # Optional per-tier overrides. When provided, these replace the defaults in
    # ModelRegistry for the named tiers only. Omitted tiers use built-in defaults.
    model_tiers: dict[str, TierConfig] = Field(default_factory=dict)
    browser: BrowserConfig = Field(default_factory=BrowserConfig)
    quality_loop: QualityLoopConfig = Field(default_factory=QualityLoopConfig)
