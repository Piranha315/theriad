from __future__ import annotations

import json
from pathlib import Path

import pytest

from theriad.models.tiers import ModelCandidate
from theriad.routing.v2 import (
    RouterV2,
    clear_router_cache,
    is_sensitive_prompt,
    observation_rows,
    resolve_from_environment,
)


def _candidate(model: str, cost: float | None, tier: str = "cheap_bulk") -> ModelCandidate:
    return ModelCandidate(
        tier=tier,
        provider="test",
        model=model,
        input_cost_per_million=cost,
        output_cost_per_million=cost,
        quality_prior=0.8,
    )


def _artifact(*, threshold: float = 0.75, ood: bool = False) -> dict:
    return {
        "schema_version": 1,
        "router_version": "test-v2",
        "feature_mean": {"log_words": 0.0} if ood else {},
        "feature_stdev": {"log_words": 1.0} if ood else {},
        "ood_max_abs_z": 0.1 if ood else 4.0,
        "lanes": {
            "general": {"knee_threshold": threshold, "fallback_model": "luna"},
            "code": {"knee_threshold": threshold, "fallback_model": "luna"},
        },
        "models": {
            "cheap": {
                "intercept": 2.0,
                "uncertainty": 0.0,
                "sample_count": 1_000_000,
                "expected_output_tokens": 100,
            },
            "luna": {
                "intercept": 3.0,
                "uncertainty": 0.0,
                "sample_count": 1_000_000,
                "expected_output_tokens": 100,
            },
        },
    }


def _write(path: Path, artifact: dict) -> Path:
    path.write_text(json.dumps(artifact), encoding="utf-8")
    return path


def test_selects_cheapest_candidate_whose_lower_bound_clears(tmp_path: Path):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact()))
    decision = router.predict(
        "A normal request", [_candidate("cheap", 0.1), _candidate("luna", 2.0)], lane="general"
    )
    assert decision.selected_model == "cheap"
    assert decision.reason == "cheapest_above_knee"


def test_when_none_clear_selects_highest_success(tmp_path: Path):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact(threshold=0.99)))
    decision = router.predict(
        "A normal request", [_candidate("cheap", 0.1), _candidate("luna", 2.0)], lane="general"
    )
    assert decision.selected_model == "luna"
    assert decision.reason == "highest_success_fallback"


def test_ood_uses_configured_fallback(tmp_path: Path):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact(ood=True)))
    decision = router.predict(
        "many words make this far from the training mean",
        [_candidate("cheap", 0.1), _candidate("luna", 2.0)],
        lane="general",
    )
    assert decision.ood_score > 1.0
    assert decision.selected_model == "luna"
    assert decision.reason == "ood_fallback"


def test_missing_price_and_unavailable_models_are_ineligible(tmp_path: Path):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact()))
    decision = router.predict(
        "request",
        [_candidate("cheap", None), _candidate("luna", 2.0)],
        lane="general",
    )
    assert next(row for row in decision.candidate_scores if row.model == "cheap").exclusion_reason == "missing_price"
    decision = router.predict(
        "request",
        [_candidate("cheap", 0.1), _candidate("luna", 2.0)],
        lane="general",
        unavailable_models=frozenset({"cheap"}),
    )
    assert next(row for row in decision.candidate_scores if row.model == "cheap").exclusion_reason == "provider_unavailable"


@pytest.mark.parametrize("price", [None, float("nan"), float("inf"), -1.0])
def test_unknown_or_invalid_price_is_json_null_and_never_free(tmp_path: Path, price):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact()))
    decision = router.predict(
        "request", [_candidate("cheap", price), _candidate("luna", 2.0)], lane="general"
    )
    cheap = next(row for row in decision.candidate_scores if row.model == "cheap")
    assert decision.selected_model == "luna"
    assert cheap.expected_cost_usd is None
    assert cheap.eligible is False
    json.dumps(decision.to_dict(), allow_nan=False)


def test_research_only_artifact_is_offline_evaluable_but_never_serves(monkeypatch, tmp_path: Path):
    artifact = {**_artifact(), "research_only": True}
    path = _write(tmp_path / "router.json", artifact)
    candidates = [_candidate("cheap", 0.1), _candidate("luna", 2.0)]
    assert RouterV2.from_path(path).predict("request", candidates, lane="general").selected_model == "cheap"
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    clear_router_cache()
    for mode in ("on", "canary", "shadow"):
        monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", mode)
        assert resolve_from_environment("request", candidates, tier="cheap_bulk", baseline_model="luna") is None


def test_fixed_model_policy_respects_runtime_eligibility(tmp_path: Path):
    artifact = _artifact()
    artifact["lanes"]["general"]["fixed_model"] = "luna"
    router = RouterV2.from_path(_write(tmp_path / "router.json", artifact))
    candidates = [_candidate("cheap", 0.1), _candidate("luna", 2.0)]
    decision = router.predict("request", candidates, lane="general")
    assert decision.selected_model == "luna"
    assert decision.reason == "fixed_model_policy"
    with pytest.raises(ValueError, match="fixed policy"):
        router.predict("request", candidates, lane="general", unavailable_models=frozenset({"luna"}))


def test_corrupt_or_missing_artifact_fails_closed(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(tmp_path / "missing.json"))
    clear_router_cache()
    assert resolve_from_environment(
        "prompt", [_candidate("cheap", 0.1)], tier="cheap_bulk", baseline_model="cheap"
    ) is None
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(broken))
    clear_router_cache()
    assert resolve_from_environment(
        "prompt", [_candidate("cheap", 0.1)], tier="cheap_bulk", baseline_model="cheap"
    ) is None


@pytest.mark.parametrize("mutation", [
    "missing_threshold",
    "missing_code_lane",
    "bad_lane_shape",
    "nan_feature",
    "unknown_lane_model",
])
def test_incomplete_or_malformed_artifact_fails_closed(monkeypatch, tmp_path: Path, mutation: str):
    artifact = _artifact()
    if mutation == "missing_threshold":
        del artifact["lanes"]["general"]["knee_threshold"]
    elif mutation == "missing_code_lane":
        del artifact["lanes"]["code"]
    elif mutation == "bad_lane_shape":
        artifact["lanes"]["general"] = []
    elif mutation == "nan_feature":
        artifact["feature_mean"] = {"log_words": float("nan")}
    elif mutation == "unknown_lane_model":
        artifact["lanes"]["general"]["models"] = ["not-measured"]
    else:
        raise AssertionError(mutation)
    path = _write(tmp_path / f"router-{mutation}.json", artifact)
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    clear_router_cache()
    assert resolve_from_environment(
        "prompt", [_candidate("cheap", 0.1)], tier="cheap_bulk", baseline_model="cheap"
    ) is None


def test_shadow_never_applies_and_sensitive_prompt_has_no_challenger(monkeypatch, tmp_path: Path):
    artifact = _artifact(threshold=0.99)
    path = _write(tmp_path / "router.json", artifact)
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "shadow")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    clear_router_cache()
    decision = resolve_from_environment(
        "My password is hunter2",
        [_candidate("cheap", 0.1), _candidate("luna", 2.0)],
        tier="cheap_bulk",
        baseline_model="cheap",
    )
    assert decision is not None
    assert decision.selected_model == "cheap"
    assert decision.shadow
    assert not decision.applied
    assert decision.shadow_challenger is None
    assert is_sensitive_prompt("api_key=secret")


def test_canary_zero_percent_keeps_baseline_with_correct_propensity(monkeypatch, tmp_path: Path):
    path = _write(tmp_path / "router.json", _artifact(threshold=0.99))
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "canary")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    monkeypatch.setenv("THERIAD_ROUTER_V2_CANARY_PCT", "0")
    clear_router_cache()
    decision = resolve_from_environment(
        "prompt",
        [_candidate("cheap", 0.1), _candidate("luna", 2.0)],
        tier="cheap_bulk",
        baseline_model="cheap",
    )
    assert decision is not None
    assert not decision.applied
    assert decision.selected_model == "cheap"
    assert decision.selected_propensity == 1.0


def test_observations_store_hash_not_raw_prompt(tmp_path: Path):
    router = RouterV2.from_path(_write(tmp_path / "router.json", _artifact()))
    prompt = "private but not stored verbatim"
    decision = router.predict(
        prompt, [_candidate("cheap", 0.1), _candidate("luna", 2.0)], lane="general"
    )
    rows = observation_rows(decision, prompt=prompt, actual_model="cheap")
    assert rows
    assert all(row["prompt_hash"] != prompt for row in rows)
    assert prompt not in json.dumps(rows)


def test_embedding_artifact_requires_backend(tmp_path: Path):
    artifact = _artifact()
    artifact["embedding_model"] = "example/encoder"
    with pytest.raises(ValueError, match="embedding backend"):
        RouterV2.from_path(_write(tmp_path / "router.json", artifact))


def test_policy_resolver_applies_v2_only_when_enabled(monkeypatch, tmp_path: Path):
    from theriad.models.policy import PolicyResolver
    from theriad.models.registry import ModelRegistry

    cheap = _candidate("cheap", 0.1, "cheap_bulk")
    luna = _candidate("luna", 2.0, "premium")
    registry = ModelRegistry({"cheap_bulk": [cheap], "premium": [luna]})
    path = _write(tmp_path / "router.json", _artifact(threshold=0.99))
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    clear_router_cache()

    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "off")
    baseline = PolicyResolver(registry).resolve(
        "cheap_bulk", message="hard terse prompt", escalation=False
    )
    assert baseline.model == "cheap"
    assert baseline.router_version is None

    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    routed = PolicyResolver(registry).resolve(
        "cheap_bulk", message="hard terse prompt", escalation=False
    )
    assert routed.model == "luna"
    assert routed.tier == "premium"
    assert routed.router_version == "test-v2"
    assert routed.candidate_scores


def test_policy_resolver_excludes_model_with_open_runtime_circuit(monkeypatch, tmp_path: Path):
    from theriad.models.policy import PolicyResolver
    from theriad.models.registry import ModelRegistry
    from theriad.runtime.circuit_breaker import get_circuit_registry, model_circuit_key

    cheap = _candidate("cheap", 0.1, "cheap_bulk")
    luna = _candidate("luna", 2.0, "premium")
    registry = ModelRegistry({"cheap_bulk": [cheap], "premium": [luna]})
    path = _write(tmp_path / "router.json", _artifact())
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(path))
    clear_router_cache()
    circuits = get_circuit_registry()
    circuits.reset()
    key = model_circuit_key("test", "cheap")
    try:
        for _ in range(circuits.failure_threshold):
            circuits.record_failure(key, RuntimeError("rate limited"))
        routed = PolicyResolver(registry).resolve(
            "cheap_bulk", message="normal request", escalation=False
        )
        assert routed.model == "luna"
        cheap_prediction = next(row for row in routed.candidate_scores if row["model"] == "cheap")
        assert cheap_prediction["exclusion_reason"] == "provider_unavailable"
    finally:
        circuits.reset()


def _learned_registry():
    from theriad.models.registry import ModelRegistry

    return ModelRegistry({
        "cheap_bulk": [_candidate("cheap", 0.1, "cheap_bulk")],
        "cheap_code": [_candidate("coder", 0.05, "cheap_code"), _candidate("cheap", 0.1, "cheap_code")],
        "premium": [_candidate("luna", 2.0, "premium")],
    })


def _learned_artifact() -> dict:
    artifact = _artifact()
    artifact["models"]["coder"] = dict(artifact["models"]["cheap"])
    return artifact


def test_resolve_learned_considers_both_lanes(monkeypatch, tmp_path: Path):
    from theriad.models.policy import PolicyResolver

    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(_write(tmp_path / "r.json", _learned_artifact())))
    clear_router_cache()
    resolution = PolicyResolver(_learned_registry()).resolve_learned("write a news story")
    assert resolution is not None
    # "coder" is only reachable up the code lane; a bulk-seeded pool would never offer it.
    assert resolution.model == "coder"
    assert resolution.policy == "router_v2"
    assert resolution.router_version == "test-v2"
    assert {c.model for c in resolution.fallback_chain} == {"cheap", "luna"}


def test_resolve_learned_keeps_largest_output_cap_label(monkeypatch, tmp_path: Path):
    from theriad.models.policy import PolicyResolver

    artifact = _learned_artifact()
    del artifact["models"]["coder"]  # leave "cheap" (listed in bulk and code) as the cheapest
    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(_write(tmp_path / "r.json", artifact)))
    clear_router_cache()
    resolution = PolicyResolver(_learned_registry()).resolve_learned("summarise this")
    assert resolution is not None and resolution.model == "cheap"
    assert resolution.tier == "cheap_code"  # 3500-token cap, not cheap_bulk's 1200


@pytest.mark.parametrize("mode", ["off", "shadow", "canary"])
def test_resolve_learned_only_in_on_mode(monkeypatch, tmp_path: Path, mode: str):
    from theriad.models.policy import PolicyResolver

    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", mode)
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(_write(tmp_path / "r.json", _learned_artifact())))
    clear_router_cache()
    assert PolicyResolver(_learned_registry()).resolve_learned("write a news story") is None


def test_resolve_learned_fails_closed_on_missing_artifact(monkeypatch, tmp_path: Path):
    from theriad.models.policy import PolicyResolver

    monkeypatch.setenv("THERIAD_ROUTER_V2_MODE", "on")
    monkeypatch.setenv("THERIAD_ROUTER_V2_ARTIFACT", str(tmp_path / "missing.json"))
    clear_router_cache()
    assert PolicyResolver(_learned_registry()).resolve_learned("write a news story") is None
