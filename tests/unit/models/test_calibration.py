"""Calibration loading and application.

A calibration file moves production traffic, so the failure modes that matter
are the quiet ones: a corrupt file must not take the router down, and a partial
measurement must not delete quality data it did not measure.
"""
from __future__ import annotations

import json

from theriad.models.calibration import (
    SCHEMA_VERSION,
    Calibration,
    load_calibration,
    reset_calibration_cache,
)
from theriad.models.tiers import ModelCandidate


def _candidate(**kw) -> ModelCandidate:
    return ModelCandidate(
        **{
            "tier": "premium",
            "provider": "openrouter",
            "model": "vendor/model",
            "quality_prior": 0.80,
            **kw,
        }
    )


def _write(path, payload: dict):
    path.write_text(json.dumps(payload))
    return path


# ── Loading ───────────────────────────────────────────────────────────────────


def test_missing_file_is_the_supported_cold_start(tmp_path):
    cal = Calibration.load(tmp_path / "nope.json")
    assert cal.is_empty


def test_corrupt_file_degrades_to_empty_rather_than_raising(tmp_path):
    path = tmp_path / "calibration.json"
    path.write_text("{not json")
    assert Calibration.load(path).is_empty


def test_wrong_schema_version_is_ignored(tmp_path):
    path = _write(
        tmp_path / "c.json",
        {"schema_version": SCHEMA_VERSION + 1, "value_of_quality": {"premium": 3.0}},
    )
    assert Calibration.load(path).is_empty


def test_malformed_values_are_ignored(tmp_path):
    path = _write(
        tmp_path / "c.json",
        {"schema_version": SCHEMA_VERSION, "value_of_quality": {"premium": "cheap"}},
    )
    assert Calibration.load(path).is_empty


def test_round_trip(tmp_path):
    original = Calibration(
        quality_profiles={"vendor/model": {"code": 0.81}},
        value_of_quality={"premium": 9.5},
        grid_trials=1200,
        generated_at="2026-08-02",
        corpus_sources=["mmlu_pro", "gpqa"],
    )
    path = tmp_path / "c.json"
    original.save(path)
    loaded = Calibration.load(path)

    assert loaded == original


# ── Application ───────────────────────────────────────────────────────────────


def test_measured_axis_overrides_the_registry():
    cal = Calibration(
        quality_profiles={"vendor/model": {"code": 0.91}}, generated_at="2026-08-02"
    )
    candidate = _candidate(quality_profile={"code": 0.70}, quality_profile_source="hand")

    assert cal.apply_to(candidate) is True
    assert candidate.quality_for("code") == 0.91


def test_unmeasured_axes_survive_application():
    """Merge, not replace — a narrow corpus must not delete existing axes."""
    cal = Calibration(
        quality_profiles={"vendor/model": {"code": 0.91}}, generated_at="2026-08-02"
    )
    candidate = _candidate(
        quality_profile={"code": 0.70, "knowledge": 0.85}, quality_profile_source="hand"
    )

    cal.apply_to(candidate)

    assert candidate.quality_for("knowledge") == 0.85


def test_unmeasured_model_is_left_alone():
    cal = Calibration(quality_profiles={"other/model": {"code": 0.91}})
    candidate = _candidate()

    assert cal.apply_to(candidate) is False
    assert candidate.quality_profile is None
    assert candidate.quality_profile_source is None


def test_application_stamps_traceable_provenance():
    cal = Calibration(
        quality_profiles={"vendor/model": {"code": 0.91}},
        grid_trials=1200,
        generated_at="2026-08-02",
    )
    candidate = _candidate()

    cal.apply_to(candidate)

    assert "2026-08-02" in candidate.quality_profile_source
    assert "1200_trials" in candidate.quality_profile_source


def test_empty_calibration_touches_nothing():
    candidate = _candidate()
    tiers = {"premium": [candidate]}

    assert Calibration.empty().apply_to_tiers(tiers) == 0
    assert candidate.quality_profile is None


# ── Router integration ────────────────────────────────────────────────────────


def test_value_of_quality_prefers_a_measured_value(tmp_path, monkeypatch):
    from theriad.models import unified

    path = _write(
        tmp_path / "c.json",
        {
            "schema_version": SCHEMA_VERSION,
            "value_of_quality": {"premium": 3.25},
            "quality_profiles": {},
        },
    )
    reset_calibration_cache()
    monkeypatch.setattr(unified, "load_calibration", lambda: load_calibration(path))
    try:
        assert unified.value_of_quality("premium") == 3.25
        # A tier the grid did not measure keeps its hand-authored value.
        assert unified.value_of_quality("frontier") == (
            unified.TIER_VALUE_OF_QUALITY["frontier"]
        )
    finally:
        reset_calibration_cache()


def test_shipped_checkout_routes_on_the_hand_authored_table():
    """No calibration file is committed, so the defaults must still be live."""
    from theriad.models import unified

    reset_calibration_cache()
    assert unified.value_of_quality("premium") == unified.TIER_VALUE_OF_QUALITY[
        "premium"
    ]
