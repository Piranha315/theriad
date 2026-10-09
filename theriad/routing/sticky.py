"""
Conversation/session sticky route cache.

Stores the model selected for a conversation so that follow-up messages
use the same model when the task type has not changed significantly.

Design decisions:
- In-memory only: restarts clear the cache, which is acceptable — a cold
  restart is a natural opportunity to re-route. A persistent sticky cache
  would require careful invalidation on model deprecation.
- TTL-based expiry: old conversations auto-expire so stale entries don't
  accumulate. Default TTL is 30 minutes.
- Tier-change detection: "major" = 2+ tier ranks apart (e.g., cheap_bulk →
  code_strong). Minor drift (cheap_bulk → cheap_code) reuses sticky model.
- Thread safety: a threading.Lock protects the dict. FastAPI is async but
  the TestClient is sync, and future worker-thread setups need this too.

Usage:
    cache = StickyRouteCache()
    entry = cache.get("conv-123")
    if entry:
        model, tier = entry.selected_model, entry.selected_tier
    else:
        cache.set("conv-123", model="poolside/laguna-xs-2.1:free",
                  tier="cheap_code", provider="openrouter",
                  reason="first_call")
    cache.clear("conv-123", reason="provider_error")
"""
from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from theriad.core.logging.logger import get_logger

logger = get_logger("routing.sticky")

# Tier rank for change-detection — higher = more capable / expensive.
_TIER_RANK: dict[str, int] = {
    "free_or_floor":     0,
    "cheap_bulk":        1,
    "cheap_code":        2,
    "code_strong":       3,
    "strong_reasoning":  4,
    "premium":           5,
    # Non-production sentinels — treated as rank 0 (never trigger a major change)
    "benchmark_direct":  0,
    "fallback":          0,
}

# A "major" task-type change is a tier-rank delta of 2 or more, e.g.:
#   cheap_bulk → code_strong  (Δ=2) → break stickiness
#   cheap_bulk → cheap_code   (Δ=1) → keep stickiness
_MAJOR_TIER_DELTA = 2


@dataclass
class StickyRoute:
    """Cached routing decision for a single conversation."""
    selected_model: str
    selected_tier: str
    provider: str
    created_at: str          # ISO-8601 UTC
    expires_at: str          # ISO-8601 UTC
    reason: str              # Why this entry was created ("first_call", "sticky_reuse", …)


class StickyRouteCache:
    """
    Thread-safe in-memory sticky route cache keyed by conversation_id.

    Clear reasons (passed to clear()):
        "task_type_change"  — routing detected a major tier shift
        "provider_error"    — the cached model/provider failed
        "model_failure"     — explicit model error (e.g. 4xx from the cached model)
        "manual_override"   — caller explicitly requested a different model
        "expired"           — TTL exceeded (handled internally by get())
    """

    DEFAULT_TTL_SECONDS = 1800  # 30 minutes

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._cache: dict[str, StickyRoute] = {}
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def get(self, conversation_id: str) -> StickyRoute | None:
        """
        Return the cached StickyRoute, or None if missing / expired.

        Expired entries are evicted on access so the dict doesn't grow
        indefinitely across long-running server processes.
        """
        with self._lock:
            entry = self._cache.get(conversation_id)
            if entry is None:
                return None
            if self._is_expired(entry):
                del self._cache[conversation_id]
                logger.debug(
                    "sticky route expired",
                    extra={"conversation_id": conversation_id,
                           "model": entry.selected_model},
                )
                return None
            return entry

    def set(
        self,
        conversation_id: str,
        selected_model: str,
        selected_tier: str,
        provider: str,
        reason: str,
    ) -> StickyRoute:
        """Store or update a sticky route. Returns the stored entry."""
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=self._ttl)
        entry = StickyRoute(
            selected_model=selected_model,
            selected_tier=selected_tier,
            provider=provider,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            reason=reason,
        )
        with self._lock:
            self._cache[conversation_id] = entry
        logger.debug(
            "sticky route set",
            extra={
                "conversation_id": conversation_id,
                "model": selected_model,
                "tier": selected_tier,
                "reason": reason,
            },
        )
        return entry

    def clear(self, conversation_id: str, reason: str = "manual_override") -> bool:
        """
        Remove the sticky route for a conversation.

        Returns True if an entry existed and was removed, False otherwise.
        """
        with self._lock:
            existed = conversation_id in self._cache
            if existed:
                model = self._cache[conversation_id].selected_model
                del self._cache[conversation_id]
                logger.info(
                    "sticky route cleared",
                    extra={
                        "conversation_id": conversation_id,
                        "model": model,
                        "reason": reason,
                    },
                )
            return existed

    def is_major_tier_change(self, sticky_tier: str, new_tier: str) -> bool:
        """
        Return True if the routing tier has changed significantly enough to
        invalidate the sticky route.

        Examples:
            cheap_bulk → cheap_code   → False  (Δ=1, tolerated)
            cheap_bulk → code_strong  → True   (Δ=2, major change)
            code_strong → premium     → True   (Δ=2, major change)
        """
        old_rank = _TIER_RANK.get(sticky_tier, 0)
        new_rank = _TIER_RANK.get(new_tier, 0)
        return abs(old_rank - new_rank) >= _MAJOR_TIER_DELTA

    def get_continuity_signal(
        self,
        conversation_id: str,
        new_tier: str,
    ):
        """
        Return a ContinuitySignal for the conversation, or None if no cached
        entry exists.

        ``same_task_family`` is True when the tier change is minor (Δ < 2),
        i.e. the conversation has not shifted work category.  This mirrors
        the check that was previously used to decide whether to clear the
        sticky cache — now repurposed as a soft routing hint.

        If the tier change is major, the sticky entry is cleared (same as
        before) and None is returned so the resolver gets no continuity bonus.
        """
        from theriad.routing.continuity import ContinuitySignal

        entry = self.get(conversation_id)
        if entry is None:
            return None

        if self.is_major_tier_change(entry.selected_tier, new_tier):
            self.clear(conversation_id, reason="task_type_change")
            return None

        return ContinuitySignal(
            previous_model=entry.selected_model,
            previous_tier=entry.selected_tier,
            previous_task_type=entry.selected_tier,   # tier as task-type proxy
            current_task_type=new_tier,
            conversation_summary_available=False,
            same_task_family=True,
        )

    def size(self) -> int:
        """Return number of live (non-expired) entries. Primarily for tests."""
        with self._lock:
            now_iso = datetime.now(timezone.utc).isoformat()
            return sum(
                1 for e in self._cache.values() if e.expires_at > now_iso
            )

    # ── Internals ──────────────────────────────────────────────────────────────

    @staticmethod
    def _is_expired(entry: StickyRoute) -> bool:
        now_iso = datetime.now(timezone.utc).isoformat()
        return entry.expires_at <= now_iso
