from __future__ import annotations

import pytest

from theriad.routing.study_freeze import (
    build_pretest_freeze_manifest,
    validate_pretest_freeze_manifest,
)

from .test_study import _valid_corpus


def _manifest():
    return build_pretest_freeze_manifest(
        _valid_corpus(),
        phase_status={},
        plan_hashes={},
        price_snapshot={"model": {"input_per_1m": 1.0, "output_per_1m": 2.0}},
        task_adapters={"plain-text": "v1"},
        sampling_settings={"temperature": 0.0, "top_p": 1.0},
        grader_versions={"deterministic": "v1"},
        feature_set=["prompt_length"],
        router_families=["surface_baseline"],
        encoders={"baseline": "tfidf@v1"},
        calibration_method={"name": "validation_only"},
        lower_confidence_bound_method={"name": "lcb@v1"},
        knee_selection_algorithm={"name": "knee@v1"},
        grok_selection={"algorithm_version": "grok@v1", "selected_ids": []},
        source_revisions=["source@v1"],
    )


def test_pretest_manifest_is_complete_but_locked_until_phases_finish():
    manifest = _manifest()
    validate_pretest_freeze_manifest(manifest)
    assert manifest["status"] == "blocked_pending_phased_study"
    assert manifest["test_unlock"] is False
    with pytest.raises(ValueError, match="does not unlock"):
        validate_pretest_freeze_manifest(manifest, require_test_unlock=True)


def test_pretest_manifest_tampering_fails_closed():
    manifest = _manifest()
    manifest["feature_set"] = ["tampered"]
    with pytest.raises(ValueError, match="hash"):
        validate_pretest_freeze_manifest(manifest)
