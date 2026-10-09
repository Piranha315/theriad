"""
Live model prices resolved from the OpenRouter catalog.

Design rules (these are load-bearing — read before changing anything here):

1. **OpenRouter is the source of truth.** Prices come from the public catalog at
   ``CATALOG_URL`` and are refreshed on a TTL (:data:`REFRESH_TTL_SECONDS`).
   The prices baked into ``theriad.models.registry`` and ``pricing._DEFAULT_PRICES``
   are *seeds* — a floor for slugs the catalog does not cover, not a record.

2. **``openrouter_prices.json`` is a cache, not a snapshot of record.** Every
   successful fetch overwrites it atomically. It exists so a cold start has a
   warm floor, not so a human can hand-curate prices.

3. **Failure serves the last-known-good cache, marked stale.** We never report a
   price as unknown just because the catalog is unreachable, and we retry on the
   next cycle. This matters more than it looks: ``estimate_cost`` degrades an
   unknown price to ``0.0``, which would tell the router every model is free —
   a silent, expensive failure mode for a cost-optimisation engine. Serving a
   slightly stale price is always better than serving zero.

4. **Resolution never blocks routing.** Refreshes run on a daemon thread and no
   exception escapes :meth:`LivePriceSource.snapshot`. The one exception is a
   fully cold start with no cache on disk at all, which does a single blocking
   fetch capped at :data:`COLD_START_TIMEOUT_SECONDS`; there is genuinely
   nothing to serve in that case.

5. **Determinism is preserved for tests** via ``PricingTable(use_live=False)``,
   which bypasses this module entirely.
"""
from __future__ import annotations

import json
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from theriad.core.logging.logger import get_logger

logger = get_logger("accounting.live_prices")

CATALOG_URL = "https://openrouter.ai/api/v1/models"

REFRESH_TTL_SECONDS: float = 3600.0
"""How long a fetched catalog is considered fresh. One hour."""

COLD_START_TIMEOUT_SECONDS: float = 5.0
"""Cap on the *only* blocking fetch: a cold start with no cache on disk."""

FETCH_TIMEOUT_SECONDS: float = 20.0
"""Socket timeout for background refreshes, which block nobody."""

CACHE_PATH = Path(__file__).with_name("openrouter_prices.json")

_CACHE_PATH_ENV = "THERIAD_PRICE_CACHE_PATH"
_DISABLE_ENV = "THERIAD_DISABLE_LIVE_PRICING"

# slug -> (input_per_1m, output_per_1m), both USD per 1M tokens.
PriceMap = dict[str, tuple[float, float]]


def live_pricing_disabled() -> bool:
    """True when ``THERIAD_DISABLE_LIVE_PRICING`` opts the process out."""
    return os.environ.get(_DISABLE_ENV, "").strip().lower() in {"1", "true", "yes", "on"}


def _cache_path() -> Path:
    override = os.environ.get(_CACHE_PATH_ENV, "").strip()
    return Path(override).expanduser() if override else CACHE_PATH


@dataclass(frozen=True)
class PriceSnapshot:
    """An immutable view of the best prices currently known.

    ``stale`` is the honest signal callers should surface: it means these prices
    have not been confirmed against the live catalog within the TTL. They are
    still the best available and are always safe to price with.
    """

    prices: PriceMap = field(default_factory=dict)
    vision: frozenset[str] = frozenset()
    # Models whose catalog entry lists ``tools`` in supported_parameters.
    # Empty means unknown (a cache written before this was tracked).
    tool_models: frozenset[str] = frozenset()
    fetched_at: float | None = None
    stale: bool = True
    origin: str = "empty"      # "live" | "cache" | "empty"
    error: str | None = None

    @property
    def model_count(self) -> int:
        return len(self.prices)

    @property
    def age_seconds(self) -> float | None:
        return None if self.fetched_at is None else max(0.0, time.time() - self.fetched_at)

    def as_dict(self) -> dict:
        """Serialisable form for the API layer."""
        return {
            "stale": self.stale,
            "origin": self.origin,
            "model_count": self.model_count,
            "vision_model_count": len(self.vision),
            "fetched_at": (
                datetime.fromtimestamp(self.fetched_at, timezone.utc).isoformat()
                if self.fetched_at is not None else None
            ),
            "age_seconds": self.age_seconds,
            "source": CATALOG_URL,
            "error": self.error,
        }


def _parse_catalog(payload: dict) -> tuple[PriceMap, set[str], set[str]]:
    """Extract per-1M prices and vision capability from an OpenRouter catalog.

    Returns ``(prices, vision_slugs, tool_slugs)``. Entries without usable numeric pricing
    are skipped rather than defaulted, so a malformed row can never masquerade
    as a free model.

    Vision support is read from ``architecture.input_modalities`` rather than
    guessed from the slug name — the catalog is the only thing that actually
    knows, and a wrong guess here routes an image to a model that cannot see it.
    """
    prices: PriceMap = {}
    vision: set[str] = set()
    tools: set[str] = set()
    for entry in payload.get("data", []):
        slug = entry.get("id")
        if not slug:
            continue
        pricing = entry.get("pricing") or {}
        try:
            prices[slug] = (
                float(pricing["prompt"]) * 1_000_000,
                float(pricing["completion"]) * 1_000_000,
            )
        except (KeyError, TypeError, ValueError):
            continue
        architecture = entry.get("architecture") or {}
        modalities = architecture.get("input_modalities")
        if isinstance(modalities, list) and "image" in modalities:
            vision.add(slug)
        parameters = entry.get("supported_parameters")
        if isinstance(parameters, list) and "tools" in parameters:
            tools.add(slug)
    return prices, vision, tools


def _parse_refreshed_at(value: object) -> float | None:
    """Read a cache timestamp, tolerating the legacy date-only format."""
    if not isinstance(value, str) or not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class LivePriceSource:
    """Resolves model prices from OpenRouter, backed by a last-known-good cache."""

    def __init__(
        self,
        *,
        url: str = CATALOG_URL,
        cache_path: Path | None = None,
        ttl_seconds: float = REFRESH_TTL_SECONDS,
    ) -> None:
        self._url = url
        self._cache_path = cache_path
        self._ttl = ttl_seconds
        self._lock = threading.Lock()
        self._snapshot: PriceSnapshot | None = None
        self._refreshing = False

    # ── public API ────────────────────────────────────────────────────────────

    def snapshot(self) -> PriceSnapshot:
        """Return the best prices known right now, refreshing in the background.

        Never raises. Only a cold start with no cache on disk blocks, and then
        for at most :data:`COLD_START_TIMEOUT_SECONDS`.
        """
        try:
            return self._snapshot_inner()
        except Exception as exc:                     # pragma: no cover - defensive
            # Routing must never fail because pricing did.
            logger.warning("live price resolution failed", extra={"error": str(exc)})
            return self._snapshot or PriceSnapshot(error=str(exc))

    def force_refresh(self, *, timeout: float = FETCH_TIMEOUT_SECONDS) -> PriceSnapshot:
        """Fetch synchronously and adopt the result. For scripts and tests."""
        snapshot = self._fetch(timeout=timeout)
        with self._lock:
            if snapshot.origin != "live" and self._snapshot is None:
                # Nothing loaded yet — a failed refresh must fall back to the
                # cache on disk rather than adopting an empty snapshot.
                self._snapshot = self._load_cache()
            if snapshot.origin == "live":
                self._snapshot = snapshot
            elif self._snapshot is not None and self._snapshot.prices:
                # Keep serving last-known-good, but record why we are stale.
                self._snapshot = PriceSnapshot(
                    prices=self._snapshot.prices,
                    vision=self._snapshot.vision,
                    fetched_at=self._snapshot.fetched_at,
                    stale=True,
                    origin=self._snapshot.origin,
                    error=snapshot.error,
                )
            else:
                self._snapshot = snapshot
            return self._snapshot

    # ── internals ─────────────────────────────────────────────────────────────

    def _snapshot_inner(self) -> PriceSnapshot:
        with self._lock:
            if self._snapshot is None:
                self._snapshot = self._load_cache()

            current = self._snapshot
            fresh = (
                current.fetched_at is not None
                and (time.time() - current.fetched_at) < self._ttl
                and current.origin == "live"
            )
            if fresh or live_pricing_disabled():
                return current

            if current.prices:
                # We have something to serve: refresh off the hot path.
                self._start_background_refresh()
                return current

        # Fully cold: nothing cached, nothing to serve. This is the only
        # blocking path, and it is capped.
        if live_pricing_disabled():
            return PriceSnapshot(error="live pricing disabled")
        snapshot = self._fetch(timeout=COLD_START_TIMEOUT_SECONDS)
        with self._lock:
            if snapshot.origin == "live" or self._snapshot is None:
                self._snapshot = snapshot
            return self._snapshot

    def _start_background_refresh(self) -> None:
        """Kick off a daemon refresh. Caller must hold the lock."""
        if self._refreshing:
            return
        self._refreshing = True
        thread = threading.Thread(
            target=self._background_refresh,
            name="openrouter-price-refresh",
            daemon=True,
        )
        thread.start()

    def _background_refresh(self) -> None:
        try:
            snapshot = self._fetch(timeout=FETCH_TIMEOUT_SECONDS)
            with self._lock:
                if snapshot.origin == "live":
                    self._snapshot = snapshot
                elif self._snapshot is not None:
                    self._snapshot = PriceSnapshot(
                        prices=self._snapshot.prices,
                        vision=self._snapshot.vision,
                        fetched_at=self._snapshot.fetched_at,
                        stale=True,
                        origin=self._snapshot.origin,
                        error=snapshot.error,
                    )
        except Exception as exc:                     # pragma: no cover - defensive
            logger.warning("background price refresh failed", extra={"error": str(exc)})
        finally:
            with self._lock:
                self._refreshing = False

    def _fetch_payload(self, *, timeout: float) -> dict:
        """Perform the actual HTTP request. Seam for tests — never mock urllib."""
        request = urllib.request.Request(
            self._url, headers={"User-Agent": "Theriad/live-pricing"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)

    def _fetch(self, *, timeout: float) -> PriceSnapshot:
        """Fetch the catalog. Returns an errored snapshot instead of raising."""
        try:
            payload = self._fetch_payload(timeout=timeout)
            prices, vision, tools = _parse_catalog(payload)
            if not prices:
                raise ValueError("catalog returned no usable prices")
        except Exception as exc:
            logger.warning(
                "OpenRouter catalog unreachable; serving last-known-good prices",
                extra={"error": str(exc)},
            )
            return PriceSnapshot(origin="empty", error=str(exc))

        now = time.time()
        self._write_cache(prices, vision, now, tools)
        logger.info(
            "refreshed live prices",
            extra={"models": len(prices), "vision_models": len(vision)},
        )
        return PriceSnapshot(
            prices=prices,
            vision=frozenset(vision),
            tool_models=frozenset(tools),
            fetched_at=now,
            stale=False,
            origin="live",
        )

    def _load_cache(self) -> PriceSnapshot:
        path = self._cache_path or _cache_path()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            prices: PriceMap = {
                slug: (float(values["input_per_1m"]), float(values["output_per_1m"]))
                for slug, values in (raw.get("models") or {}).items()
            }
        except (OSError, ValueError, TypeError, KeyError, AttributeError):
            return PriceSnapshot(origin="empty", error="no usable price cache")
        if not prices:
            return PriceSnapshot(origin="empty", error="empty price cache")
        fetched_at = _parse_refreshed_at(raw.get("refreshed_at"))
        cached_vision = raw.get("vision_models")
        cached_tools = raw.get("tool_models")
        return PriceSnapshot(
            prices=prices,
            # A cache written before vision was tracked has no list at all —
            # frozenset() then means "unknown", and vision.py falls back to seeds
            # rather than concluding no model can see.
            vision=frozenset(cached_vision) if isinstance(cached_vision, list) else frozenset(),
            tool_models=frozenset(cached_tools) if isinstance(cached_tools, list) else frozenset(),
            fetched_at=fetched_at,
            stale=True,          # cached prices are unconfirmed until a live fetch
            origin="cache",
        )

    def _write_cache(
        self, prices: PriceMap, vision: set[str], fetched_at: float,
        tools: set[str] | None = None,
    ) -> None:
        """Persist last-known-good atomically. Cache failure must not be fatal."""
        path = self._cache_path or _cache_path()
        snapshot = {
            "refreshed_at": datetime.fromtimestamp(fetched_at, timezone.utc).isoformat(),
            "source": self._url,
            "models": {
                slug: {"input_per_1m": pin, "output_per_1m": pout}
                for slug, (pin, pout) in sorted(prices.items())
            },
            "vision_models": sorted(vision),
            "tool_models": sorted(tools or ()),
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            temporary.replace(path)
        except OSError as exc:
            logger.warning("could not persist price cache", extra={"error": str(exc)})


_SOURCE = LivePriceSource()


def get_live_snapshot() -> PriceSnapshot:
    """Module-level accessor for the shared price source."""
    return _SOURCE.snapshot()


def reset_for_tests() -> None:
    """Drop cached state so a test can start from a known-cold source."""
    global _SOURCE
    _SOURCE = LivePriceSource()
