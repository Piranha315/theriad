"""Tests for routing interface contracts."""
from theriad.core.routing.interfaces import RoutingDecision, RoutingResult


def test_routing_decision_immutable():
    d = RoutingDecision(tier="cheap_bulk", confidence=0.9, reason="test")
    assert d.tier == "cheap_bulk"
    # frozen=True means attempts to mutate raise FrozenInstanceError
    try:
        d.tier = "other"
        raise AssertionError("Should have raised")
    except Exception:
        pass


def test_routing_decision_signals_default():
    d = RoutingDecision(tier="cheap_code", confidence=0.85, reason="code fence")
    assert d.signals == {}


def test_routing_decision_signals_set():
    d = RoutingDecision(
        tier="cheap_code", confidence=0.92, reason="fence",
        signals={"code_fence": True, "keyword_density": 0.12},
    )
    assert d.signals["code_fence"] is True
    assert d.signals["keyword_density"] == 0.12


def test_routing_result_defaults():
    d = RoutingDecision(tier="cheap_bulk", confidence=1.0, reason="r")
    result = RoutingResult(decision=d, layer_name="layer0")
    assert result.all_decisions == []
    assert result.features == {}
