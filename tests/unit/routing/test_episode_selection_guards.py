"""Regression tests for the previously accepted fixed-model substitute."""
import importlib.util
from dataclasses import asdict
from pathlib import Path

import pytest

from theriad.routing.episode_study import Episode, EpisodeOutcome, FAMILIES, atomic_json, append_event, digest
from theriad.routing import v2_training

spec = importlib.util.spec_from_file_location("selection_cli", Path(__file__).resolve().parents[3] / "scripts/run_router_episode_study.py")
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def artifact():
    return {"schema_version": 1, "router_version": "guard-test", "feature_mean": {}, "feature_stdev": {},
            "lanes": {lane: {"knee_threshold": .5, "fallback_model": "pricey"} for lane in ("general", "code")},
            "models": {model: {"intercept": 2., "uncertainty": 0., "sample_count": 100, "expected_output_tokens": 20} for model in ("cheap", "pricey")}}


@pytest.mark.parametrize("kind", ["model", "by_family", "current"])
def test_cached_challenger_cannot_substitute_fixed_policy(kind):
    with pytest.raises(ValueError, match="actual Router V2"):
        cli.require_surface_challenger({"kind": kind})


def test_surface_wrapper_cannot_hide_a_fixed_model_lane():
    value = artifact()
    value["lanes"]["code"]["fixed_model"] = "cheap"
    with pytest.raises(ValueError, match="fixed-model lane"):
        cli.require_surface_challenger({"kind": "surface", "artifacts": {f: value for f in FAMILIES}})


@pytest.mark.parametrize("pricey_success,pricey_cost", [(True, .019), (False, .001)])
def test_infeasible_router_does_not_fall_back_to_cheaper_fixed_model(tmp_path, monkeypatch, pricey_success, pricey_cost):
    rows = [Episode(f"{f}-{split}", f"{f}-{split}", f, split, f"{f} {split}", {}, {"kind": "exact", "answer": "yes"}, source="newly_authored", template_id=f"{f}-{split}") for split in cli.DEVELOPMENT_SPLITS for f in FAMILIES]
    contract = {"alternatives": ["cheap", "pricey"], "created_at": "2026-09-10T00:00:00Z", "split_hashes": {split: digest([asdict(r) for r in rows if r.split == split]) for split in cli.DEVELOPMENT_SPLITS}}
    atomic_json(tmp_path / "contract.json", contract)
    for split in cli.DEVELOPMENT_SPLITS:
        atomic_json(tmp_path / "fixtures" / f"{split}.json", [asdict(r) for r in rows if r.split == split])
    for row in rows:
        for model, success, cost in (("baseline", True, .02), ("cheap", True, .001), ("pricey", pricey_success, pricey_cost)):
            value = EpisodeOutcome(row.episode_id, row.group_id, row.family, model, row.content_hash, success, "completed", cost, 10., 1, evidence={"output_tokens": 20})
            append_event(tmp_path / "development-outcomes.jsonl", {"event": "episode_completed", "outcome": asdict(value)})
    catalog = {m: {"capabilities": ["text"], "context_window": 100000, "input_per_1m": 1., "output_per_1m": 1.} for m in contract["alternatives"]}
    monkeypatch.setattr(cli, "validate_contract", lambda *args: (contract, catalog, {}))
    def train(rows, **kwargs):
        assert kwargs["allow_fixed_models"] is False
        assert set(kwargs["baseline_outcomes_by_prompt"]) == {r.prompt_id for r in rows if r.split == "threshold_selection"}
        # Baseline outcome must not overwrite this model's measured cell.
        assert all(r.outcomes["cheap"].cost_usd == .001 for r in rows)
        return artifact()
    monkeypatch.setattr(v2_training, "train_surface_artifact", train)
    class Runtime:
        def models_for(self, *args):
            return ["pricey"]
    with pytest.raises(RuntimeError, match="fixed-model substitution is forbidden"):
        cli.select_challenger(tmp_path, Runtime(), contract, catalog)
    assert not (tmp_path / "challenger.json").exists()
    summary = cli.read_json(tmp_path / "validation-selection.json")
    assert summary["feasible_count"] == 0
    assert any(c.get("model") == "cheap" for c in summary["candidates"])


def test_seed_fails_training_support_before_credentials_or_dispatch(tmp_path, monkeypatch):
    from theriad.routing.episode_fixtures import build_four_way_seed_corpus
    rows = build_four_way_seed_corpus()
    catalog = {"model": {"capabilities": ["text", "tools", "search", "vision"], "context_window": 100000, "input_per_1m": 1., "output_per_1m": 1.}}
    ready = cli.training_readiness(rows, ["model"], catalog)
    assert ready["support"]["model"]["train"] == 15
    assert ready["ready_for_development_collection"] is False


def test_dispatch_guard_requires_a_fresh_validated_router_decision():
    from theriad.routing.episode_hooks import EpisodeRoutingSession, EpisodeRoutingError
    from theriad.core.providers.interfaces import CompletionRequest, Message
    metadata = {m: {"capabilities": ["text"], "context_window": 100000} for m in ("baseline", "cheap")}
    session = EpisodeRoutingSession("guard-test", model_metadata=metadata,
        decision_resolver=lambda **kwargs: "cheap", assistant_provider="theriad")
    session.prepare("chat", {"completion_0": "baseline"}, "theriad")
    request = CompletionRequest("cheap", [Message("user", "answer")], max_tokens=20)
    with pytest.raises(EpisodeRoutingError, match="dispatch model differs"):
        session.guard_request(request)
    assert session.resolve_model("completion_0", "baseline", "theriad") == "cheap"
    assert session.guard_request(request).model == "cheap"
    with pytest.raises(EpisodeRoutingError, match="dispatch model differs"):
        session.guard_request(request)
    session.resolve_model("completion_0", "baseline", "theriad")
    vision_request = CompletionRequest("cheap", [Message("user", [{"type": "image_url", "image_url": {"url": "fixture"}}])], max_tokens=20)
    with pytest.raises(EpisodeRoutingError, match="actual request capabilities"):
        session.guard_request(vision_request)
