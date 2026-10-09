"""Continuous user routing preferences shared by the API and model resolver.

The UI exposes two 0-100 controls instead of asking users to understand named
policies.  Named policies remain a compatibility layer for CLI and older API
clients.
"""

from __future__ import annotations

from dataclasses import dataclass

STANDARD_PREFERENCE = 50


def clamp_preference(value: int | float | None, default: int = STANDARD_PREFERENCE) -> int:
    if value is None:
        return default
    return max(0, min(100, int(round(value))))


def preferences_for_legacy_policy(policy: str) -> tuple[int, int]:
    """Map old policy names to equivalent speed/quality defaults."""
    return {
        "aggressive_savings": (25, 25),
        "balanced": (50, 50),
        "quality_first": (50, 90),
        "free_moe_experimental": (20, 35),
    }.get(policy, (STANDARD_PREFERENCE, STANDARD_PREFERENCE))


def legacy_policy_for_preferences(speed: int, quality: int) -> str:
    """Return a stable policy label for persistence and legacy consumers."""
    speed = clamp_preference(speed)
    quality = clamp_preference(quality)
    if quality >= 75:
        return "quality_first"
    if quality <= 30 and speed <= 55:
        return "aggressive_savings"
    return "balanced"


def estimated_price_multiplier(speed: int, quality: int) -> float:
    """Estimate spend relative to the standard 50/50 routing profile.

    This is intentionally an estimate, not a quote.  The curve reflects two
    observed effects: low-latency endpoints usually carry a serving premium,
    while model-quality premiums accelerate near the top of the quality range.
    The function is normalized so the standard profile is exactly 1.0x.
    """
    s = clamp_preference(speed) / 100
    q = clamp_preference(quality) / 100
    speed_factor = 0.72 + 0.72 * (s**1.7)
    quality_factor = 0.58 + 1.68 * (q**2.2)
    standard_speed = 0.72 + 0.72 * (0.5**1.7)
    standard_quality = 0.58 + 1.68 * (0.5**2.2)
    return round((speed_factor * quality_factor) / (standard_speed * standard_quality), 2)


@dataclass(frozen=True)
class RoutingPreferences:
    speed: int = STANDARD_PREFERENCE
    quality: int = STANDARD_PREFERENCE

    @classmethod
    def resolve(
        cls,
        *,
        speed: int | None,
        quality: int | None,
        legacy_policy: str = "balanced",
    ) -> RoutingPreferences:
        legacy_speed, legacy_quality = preferences_for_legacy_policy(legacy_policy)
        return cls(
            speed=clamp_preference(speed, legacy_speed),
            quality=clamp_preference(quality, legacy_quality),
        )

    @property
    def legacy_policy(self) -> str:
        return legacy_policy_for_preferences(self.speed, self.quality)

    @property
    def price_multiplier(self) -> float:
        return estimated_price_multiplier(self.speed, self.quality)
