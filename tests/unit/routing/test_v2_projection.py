"""An embedding artifact may carry the projection its heads were fitted in."""
from __future__ import annotations

import pytest

from theriad.models.tiers import ModelCandidate
from theriad.routing.v2 import RouterV2, RouterV2Artifact

ENCODER = "test/encoder"


class FixedBackend:
    def __init__(self, vector):
        self.vector = vector

    def encode(self, text):
        return self.vector


def _candidate(model: str, cost: float) -> ModelCandidate:
    return ModelCandidate(
        tier="cheap_bulk", provider="test", model=model,
        input_cost_per_million=cost, output_cost_per_million=cost, quality_prior=0.8,
    )


CANDIDATES = [_candidate("cheap", 0.1), _candidate("luna", 2.0)]


def _artifact(cheap_weights: list[float], projection: dict | None = None) -> dict:
    raw = {
        "schema_version": 1,
        "router_version": "test-projection",
        "feature_mean": {},
        "feature_stdev": {},
        "embedding_model": ENCODER,
        "lanes": {
            "general": {"knee_threshold": 0.75, "fallback_model": "luna"},
            "code": {"knee_threshold": 0.75, "fallback_model": "luna"},
        },
        "models": {
            # cheap succeeds only where its embedding weights point; luna always.
            "cheap": {"intercept": 0.0, "embedding_weights": cheap_weights,
                      "uncertainty": 0.0, "sample_count": 1_000_000},
            "luna": {"intercept": 3.0, "embedding_weights": [0.0] * len(cheap_weights),
                     "uncertainty": 0.0, "sample_count": 1_000_000},
        },
    }
    if projection is not None:
        raw["projection"] = {"encoder": ENCODER, **projection}
    return raw


def _route(raw: dict, vector: list[float]):
    router = RouterV2(RouterV2Artifact.from_dict(raw), embedding_backend=FixedBackend(vector))
    return router.predict("any prompt", CANDIDATES, lane="general")


def test_artifact_without_projection_behaves_as_before():
    raw = _artifact([10.0, 0.0, 0.0])
    assert RouterV2Artifact.from_dict(raw).projection is None
    assert _route(raw, [1.0, 0.0, 0.0]).selected_model == "cheap"
    assert _route(raw, [0.0, 1.0, 0.0]).selected_model == "luna"


def test_identity_projection_matches_no_projection():
    plain = _artifact([10.0, 0.0, 0.0])
    identity = _artifact(
        [10.0, 0.0, 0.0],
        {"mean": [0.0, 0.0, 0.0], "basis": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
    )
    for vector in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-0.5, 0.2, 0.9]):
        a, b = _route(plain, vector), _route(identity, vector)
        assert a.selected_model == b.selected_model
        assert [p.p_success for p in a.candidate_scores] == pytest.approx(
            [p.p_success for p in b.candidate_scores]
        )


def test_projection_runs_before_the_heads():
    # Heads live in a 2-d space whose first axis is raw dimension 2.
    raw = _artifact([10.0, 0.0], {"mean": [0.0, 0.0, 0.0], "basis": [[0, 0, 1], [1, 0, 0]]})
    assert _route(raw, [0.0, 0.0, 1.0]).selected_model == "cheap"
    assert _route(raw, [0.0, 1.0, 0.0]).selected_model == "luna"


def test_mean_is_subtracted_before_projecting():
    raw = _artifact([10.0], {"mean": [0.0, 0.0, 1.0], "basis": [[0, 0, 1]]})
    # (v - mean) on axis 2 is 0 -> cheap is a coin flip, below the 0.75 knee.
    assert _route(raw, [0.0, 0.0, 1.0]).selected_model == "luna"
    assert _route(raw, [0.0, 0.0, 2.0]).selected_model == "cheap"


def test_reconstruction_error_is_the_ood_score():
    projection = {"mean": [0.0, 0.0, 0.0], "basis": [[1, 0, 0]], "ood_reconstruction_p99": 0.5}
    raw = _artifact([10.0], projection)
    inside = _route(raw, [1.0, 0.1, 0.0])  # residual 0.1 -> score 0.2
    outside = _route(raw, [1.0, 0.0, 2.0])  # residual 2.0 -> score 4.0
    assert inside.ood_score == pytest.approx(0.2)
    assert inside.selected_model == "cheap"
    assert outside.ood_score == pytest.approx(4.0)
    assert outside.reason == "ood_fallback"


@pytest.mark.parametrize(
    "projection, message",
    [
        ({"mean": [0.0, 0.0], "basis": [[1, 0, 0]]}, "K×D"),
        ({"mean": [0.0, 0.0, 0.0], "basis": [[1, 0, 0]], "encoder": "other"}, "must match"),
        ({"mean": [0.0, 0.0, 0.0], "basis": [[1, 0, 0]], "ood_reconstruction_p99": 0}, "positive"),
    ],
)
def test_malformed_projection_is_rejected(projection, message):
    raw = _artifact([1.0])
    raw["projection"] = {"encoder": ENCODER, **projection}
    with pytest.raises(ValueError, match=message):
        RouterV2Artifact.from_dict(raw)
