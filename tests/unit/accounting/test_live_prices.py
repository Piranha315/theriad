"""
Unit tests for the live OpenRouter price source.

No real network calls are made — the catalog fetch is always mocked.

The failure paths carry most of the weight here. `estimate_cost` degrades an
unknown price to 0.0, so "catalog unreachable" must never become "everything is
free"; it must become "last-known-good, marked stale".
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from theriad.accounting.live_prices import (
    LivePriceSource,
    PriceSnapshot,
    _parse_catalog,
    _parse_refreshed_at,
)

CATALOG = {
    "data": [
        {"id": "a/one", "pricing": {"prompt": "0.00000014", "completion": "0.00000028"}},
        {"id": "b/two", "pricing": {"prompt": "0.000001", "completion": "0.000002"}},
    ]
}


def _write_cache(path, models, refreshed_at="2026-07-01T00:00:00+00:00"):
    path.write_text(
        json.dumps({"refreshed_at": refreshed_at, "source": "test", "models": models}),
        encoding="utf-8",
    )
    return path


@pytest.fixture(autouse=True)
def _live_pricing_enabled(monkeypatch):
    monkeypatch.delenv("THERIAD_DISABLE_LIVE_PRICING", raising=False)


# ── catalog parsing ───────────────────────────────────────────────────────────

class TestParseCatalog:

    def test_converts_per_token_to_per_million(self):
        prices, _vision, _tools = _parse_catalog(CATALOG)
        assert prices["a/one"] == pytest.approx((0.14, 0.28))
        assert prices["b/two"] == pytest.approx((1.0, 2.0))

    def test_entry_without_pricing_is_skipped_not_zeroed(self):
        # A malformed row must never masquerade as a free model.
        prices, _vision, _tools = _parse_catalog({"data": [{"id": "x/broken"}]})
        assert "x/broken" not in prices

    def test_non_numeric_pricing_is_skipped(self):
        prices, _vision, _tools = _parse_catalog(
            {"data": [{"id": "x/bad", "pricing": {"prompt": "n/a", "completion": "0"}}]}
        )
        assert "x/bad" not in prices

    def test_entry_without_id_is_skipped(self):
        prices, vision, _tools = _parse_catalog(
            {"data": [{"pricing": {"prompt": "1", "completion": "1"}}]}
        )
        assert prices == {} and vision == set()


class TestParseRefreshedAt:

    def test_full_timestamp(self):
        assert _parse_refreshed_at("2026-07-28T12:00:00+00:00") is not None

    def test_legacy_date_only_format_is_accepted(self):
        assert _parse_refreshed_at("2026-07-16") is not None

    def test_garbage_returns_none(self):
        assert _parse_refreshed_at("not-a-date") is None
        assert _parse_refreshed_at(None) is None


# ── success path ──────────────────────────────────────────────────────────────

class TestLiveFetch:

    def test_cold_start_fetches_and_marks_fresh(self, tmp_path):
        source = LivePriceSource(cache_path=tmp_path / "cache.json")
        with patch.object(LivePriceSource, "_fetch_payload", return_value=CATALOG):
            snapshot = source.snapshot()
        assert snapshot.origin == "live"
        assert snapshot.stale is False
        assert snapshot.prices["a/one"] == pytest.approx((0.14, 0.28))

    def test_successful_fetch_persists_cache_atomically(self, tmp_path):
        cache = tmp_path / "cache.json"
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", return_value=CATALOG):
            source.snapshot()
        assert cache.exists()
        written = json.loads(cache.read_text())
        assert written["models"]["a/one"]["input_per_1m"] == pytest.approx(0.14)
        # No temp file left behind.
        assert not list(tmp_path.glob("*.tmp"))

    def test_fresh_snapshot_is_not_refetched(self, tmp_path):
        source = LivePriceSource(cache_path=tmp_path / "cache.json")
        with patch.object(
            LivePriceSource, "_fetch_payload", return_value=CATALOG
        ) as fetch:
            source.snapshot()
            source.snapshot()
            source.snapshot()
        assert fetch.call_count == 1


# ── failure path: the important one ───────────────────────────────────────────

class TestFailureServesLastKnownGood:

    def test_unreachable_catalog_serves_cache_marked_stale(self, tmp_path):
        cache = _write_cache(
            tmp_path / "cache.json",
            {"a/one": {"input_per_1m": 0.14, "output_per_1m": 0.28}},
        )
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            snapshot = source.snapshot()
        assert snapshot.prices["a/one"] == pytest.approx((0.14, 0.28))
        assert snapshot.stale is True
        assert snapshot.origin == "cache"

    def test_unreachable_catalog_never_reports_zero_prices(self, tmp_path):
        """The failure mode that would silently tell the router everything is free."""
        cache = _write_cache(
            tmp_path / "cache.json",
            {"paid/model": {"input_per_1m": 3.0, "output_per_1m": 15.0}},
        )
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            snapshot = source.snapshot()
        assert snapshot.prices["paid/model"][0] > 0
        assert snapshot.prices["paid/model"][1] > 0

    def test_fully_cold_failure_returns_empty_without_raising(self, tmp_path):
        source = LivePriceSource(cache_path=tmp_path / "missing.json")
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            snapshot = source.snapshot()
        assert snapshot.prices == {}
        assert snapshot.stale is True
        assert snapshot.error is not None

    def test_empty_catalog_response_is_treated_as_failure(self, tmp_path):
        cache = _write_cache(
            tmp_path / "cache.json",
            {"a/one": {"input_per_1m": 0.14, "output_per_1m": 0.28}},
        )
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", return_value={"data": []}):
            snapshot = source.force_refresh()
        # Cache retained rather than being wiped by a useless response.
        assert snapshot.prices["a/one"] == pytest.approx((0.14, 0.28))
        assert snapshot.stale is True

    def test_failed_refresh_does_not_overwrite_cache_file(self, tmp_path):
        cache = _write_cache(
            tmp_path / "cache.json",
            {"a/one": {"input_per_1m": 0.14, "output_per_1m": 0.28}},
        )
        before = cache.read_text()
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            source.force_refresh()
        assert cache.read_text() == before

    def test_corrupt_cache_is_survivable(self, tmp_path):
        cache = tmp_path / "cache.json"
        cache.write_text("{not json", encoding="utf-8")
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", return_value=CATALOG):
            snapshot = source.snapshot()
        assert snapshot.origin == "live"

    def test_snapshot_never_raises(self, tmp_path):
        source = LivePriceSource(cache_path=tmp_path / "cache.json")
        with patch.object(
            LivePriceSource, "_snapshot_inner", side_effect=RuntimeError("boom")
        ):
            snapshot = source.snapshot()   # must not raise
        assert isinstance(snapshot, PriceSnapshot)


# ── opt-out ───────────────────────────────────────────────────────────────────

class TestDisableSwitch:

    def test_disabled_serves_cache_without_fetching(self, tmp_path, monkeypatch):
        monkeypatch.setenv("THERIAD_DISABLE_LIVE_PRICING", "1")
        cache = _write_cache(
            tmp_path / "cache.json",
            {"a/one": {"input_per_1m": 0.14, "output_per_1m": 0.28}},
        )
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload") as fetch:
            snapshot = source.snapshot()
        fetch.assert_not_called()
        assert snapshot.prices["a/one"] == pytest.approx((0.14, 0.28))

    def test_disabled_with_no_cache_does_not_fetch(self, tmp_path, monkeypatch):
        monkeypatch.setenv("THERIAD_DISABLE_LIVE_PRICING", "1")
        source = LivePriceSource(cache_path=tmp_path / "missing.json")
        with patch.object(LivePriceSource, "_fetch_payload") as fetch:
            snapshot = source.snapshot()
        fetch.assert_not_called()
        assert snapshot.prices == {}


# ── PricingTable integration ──────────────────────────────────────────────────

class TestPricingTableIntegration:

    def test_use_live_false_is_deterministic_and_offline(self):
        from theriad.accounting.pricing import PricingTable

        with patch.object(LivePriceSource, "_fetch_payload") as fetch:
            table = PricingTable(use_live=False)
        fetch.assert_not_called()
        assert table.live_stale is None
        assert table.get("xiaomi/mimo-v2.5").input_per_1m == pytest.approx(0.14)

    def test_live_overlay_wins_over_seed(self):
        from theriad.accounting.pricing import PricingTable

        overlay = PriceSnapshot(
            prices={"xiaomi/mimo-v2.5": (9.99, 19.99)},
            fetched_at=0.0,
            stale=False,
            origin="live",
        )
        with patch.object(PricingTable, "_live_overlay_snapshot", return_value=overlay):
            table = PricingTable()
        assert table.get("xiaomi/mimo-v2.5").input_per_1m == pytest.approx(9.99)
        assert table.live_stale is False

    def test_explicit_prices_win_over_live(self):
        from theriad.accounting.pricing import ModelPrice, PricingTable

        overlay = PriceSnapshot(
            prices={"xiaomi/mimo-v2.5": (9.99, 19.99)},
            fetched_at=0.0,
            stale=False,
            origin="live",
        )
        with patch.object(PricingTable, "_live_overlay_snapshot", return_value=overlay):
            table = PricingTable({"xiaomi/mimo-v2.5": ModelPrice(1.0, 2.0)})
        assert table.get("xiaomi/mimo-v2.5").input_per_1m == pytest.approx(1.0)

    def test_overlay_failure_falls_back_to_seeds(self):
        from theriad.accounting.pricing import PricingTable

        with patch.object(
            PricingTable, "_live_overlay_snapshot", side_effect=RuntimeError("boom")
        ):
            table = PricingTable()      # must not raise
        assert table.get("xiaomi/mimo-v2.5").input_per_1m == pytest.approx(0.14)

    def test_seeds_are_not_mutated_by_a_live_overlay(self):
        """Regression: the disk cache used to be merged into _DEFAULT_PRICES at
        import, which made PricingTable(use_live=False) non-deterministic."""
        from theriad.accounting.pricing import _DEFAULT_PRICES, PricingTable

        overlay = PriceSnapshot(
            prices={"xiaomi/mimo-v2.5": (9.99, 19.99)},
            fetched_at=0.0, stale=False, origin="live",
        )
        with patch.object(PricingTable, "_live_overlay_snapshot", return_value=overlay):
            PricingTable()
        assert _DEFAULT_PRICES["xiaomi/mimo-v2.5"].input_per_1m == pytest.approx(0.14)
        assert PricingTable(use_live=False).get(
            "xiaomi/mimo-v2.5"
        ).input_per_1m == pytest.approx(0.14)


# ── vision capability from the catalog ────────────────────────────────────────

class TestCatalogVision:

    VISION_CATALOG = {
        "data": [
            {
                "id": "sees/things",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                "architecture": {"input_modalities": ["text", "image"]},
            },
            {
                "id": "text/only",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
                "architecture": {"input_modalities": ["text"]},
            },
            {
                "id": "no/architecture",
                "pricing": {"prompt": "0.0000001", "completion": "0.0000002"},
            },
        ]
    }

    def test_image_modality_marks_vision(self):
        _prices, vision, _tools = _parse_catalog(self.VISION_CATALOG)
        assert "sees/things" in vision

    def test_text_only_is_not_vision(self):
        _prices, vision, _tools = _parse_catalog(self.VISION_CATALOG)
        assert "text/only" not in vision

    def test_missing_architecture_is_not_vision(self):
        """Absent metadata must not be read as capability."""
        _prices, vision, _tools = _parse_catalog(self.VISION_CATALOG)
        assert "no/architecture" not in vision

    def test_vision_persists_to_cache_and_reloads(self, tmp_path):
        cache = tmp_path / "cache.json"
        source = LivePriceSource(cache_path=cache)
        with patch.object(
            LivePriceSource, "_fetch_payload", return_value=self.VISION_CATALOG
        ):
            source.snapshot()
        assert json.loads(cache.read_text())["vision_models"] == ["sees/things"]

        reloaded = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            snapshot = reloaded.snapshot()
        assert snapshot.vision == frozenset({"sees/things"})

    def test_legacy_cache_without_vision_yields_empty_not_wrong(self, tmp_path):
        """A pre-vision cache means 'unknown', which must fall back to seeds."""
        cache = _write_cache(
            tmp_path / "cache.json",
            {"a/one": {"input_per_1m": 0.14, "output_per_1m": 0.28}},
        )
        source = LivePriceSource(cache_path=cache)
        with patch.object(LivePriceSource, "_fetch_payload", side_effect=OSError("down")):
            snapshot = source.snapshot()
        assert snapshot.vision == frozenset()
        assert snapshot.prices["a/one"] == pytest.approx((0.14, 0.28))
