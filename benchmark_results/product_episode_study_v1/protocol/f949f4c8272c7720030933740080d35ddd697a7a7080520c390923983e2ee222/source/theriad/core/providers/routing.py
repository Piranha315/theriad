"""Provider-endpoint routing preferences.

Model routing decides *which model* should answer.  This module independently
decides how an aggregator such as OpenRouter should choose the serving endpoint.
Keeping those decisions separate lets future providers implement the same
preference contract without changing the model registry.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from theriad.models.preferences import clamp_preference, estimated_price_multiplier

_FAST_PROVIDER_ORDER: dict[str, list[str]] = {
    "xiaomi/mimo-v2.5": ["novita", "xiaomi", "parasail"],
    "deepseek/deepseek-v4-flash-0731": ["deepseek", "novita", "siliconflow", "deepinfra"],
    "z-ai/glm-5.2": ["streamlake", "novita", "inceptron"],
    # Legacy explicit-request compatibility; Kimi is no longer in the default
    # registry or agent route.
    "moonshotai/kimi-k2.7-code": ["deepinfra", "inceptron", "ambient", "parasail"],
}


@dataclass(frozen=True)
class ProviderRoutingPreferences:
    """Provider-neutral serving intent plus an OpenRouter serialization helper."""

    speed: int
    quality: int
    strategy: str
    order: list[str] = field(default_factory=list)
    allow_fallbacks: bool = True
    require_parameters: bool = True
    preferred_min_throughput: int | None = None
    preferred_max_latency: float | None = None

    @property
    def estimated_price_multiplier(self) -> float:
        return estimated_price_multiplier(self.speed, self.quality)

    def to_openrouter(self) -> dict:
        payload: dict[str, object] = {
            "allow_fallbacks": self.allow_fallbacks,
            "require_parameters": self.require_parameters,
        }
        if self.order:
            payload["order"] = self.order
        else:
            payload["sort"] = self.strategy
        if self.preferred_min_throughput is not None:
            payload["preferred_min_throughput"] = self.preferred_min_throughput
        if self.preferred_max_latency is not None:
            payload["preferred_max_latency"] = self.preferred_max_latency
        return payload


def build_provider_routing_preferences(
    model: str,
    *,
    speed: int | None,
    quality: int | None,
) -> ProviderRoutingPreferences:
    speed_value = clamp_preference(speed)
    quality_value = clamp_preference(quality)

    if speed_value >= 70:
        return ProviderRoutingPreferences(
            speed=speed_value,
            quality=quality_value,
            strategy="throughput",
            order=list(_FAST_PROVIDER_ORDER.get(model, ())),
            preferred_min_throughput=70 + round((speed_value - 70) * 1.5),
            preferred_max_latency=max(0.35, round(1.8 - (speed_value - 70) * 0.035, 2)),
        )
    if speed_value <= 30:
        return ProviderRoutingPreferences(
            speed=speed_value,
            quality=quality_value,
            strategy="price",
        )
    return ProviderRoutingPreferences(
        speed=speed_value,
        quality=quality_value,
        strategy="price",
        preferred_min_throughput=35 + round(speed_value * 0.5),
    )
