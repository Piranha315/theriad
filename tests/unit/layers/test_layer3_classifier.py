"""Tests for Layer 3 optional local classifier."""
from __future__ import annotations

import pytest

pytest.importorskip("sklearn", reason="scikit-learn required for layer3 tests")

from theriad.layers.layer3_classifier.classifier import ClassifierResult, LocalClassifier
from theriad.layers.layer3_classifier.layer import ClassifierRoutingLayer
from theriad.models.tiers import TIER_CHEAP_BULK, TIER_CHEAP_CODE, TIER_STRONG_REASONING


@pytest.fixture
def clf():
    c = LocalClassifier()
    c.fit()  # uses synthetic data
    return c


# ── LocalClassifier ───────────────────────────────────────────────────────────

def test_classifier_predict_returns_result(clf):
    result = clf.predict("Write a Python function")
    assert isinstance(result, ClassifierResult)
    assert 0.0 <= result.confidence <= 1.0
    assert result.tier  # non-empty tier string
    assert result.task_type in {"code", "qa", "summarize", "translate", "analysis"}


def test_classifier_predicts_code(clf):
    result = clf.predict("Implement quicksort in Python")
    assert result.task_type == "code"
    assert result.tier == TIER_CHEAP_CODE


def test_classifier_predicts_qa(clf):
    result = clf.predict("What is the capital of France?")
    assert result.task_type == "qa"
    assert result.tier == TIER_CHEAP_BULK


def test_classifier_analysis_tier(clf):
    result = clf.predict("Compare PostgreSQL vs MySQL for high-traffic applications")
    assert result.task_type == "analysis"
    assert result.tier == TIER_STRONG_REASONING


def test_classifier_not_fitted_raises():
    clf = LocalClassifier()
    with pytest.raises(RuntimeError, match="fit"):
        clf.predict("hello")


def test_classifier_save_load(tmp_path, clf):
    path = str(tmp_path / "clf.pkl")
    clf.save(path)
    new_clf = LocalClassifier()
    new_clf.load(path)
    result = new_clf.predict("Translate this to Spanish")
    assert result.task_type in {"code", "qa", "summarize", "translate", "analysis"}
    assert result.tier  # non-empty tier


def test_classifier_custom_data():
    clf = LocalClassifier()
    clf.fit([
        ("cat", "I love cats"),
        ("cat", "My cat is fluffy"),
        ("dog", "Dogs are loyal animals"),
        ("dog", "My dog loves to play fetch"),
    ])
    result = clf.predict("cats are great pets")
    assert result.task_type in {"cat", "dog"}
    # Custom task types not in _TASK_TIER_MAP → default cheap_bulk
    assert result.tier == TIER_CHEAP_BULK


# ── ClassifierRoutingLayer ────────────────────────────────────────────────────

def test_layer_routes_above_threshold():
    layer = ClassifierRoutingLayer(min_confidence=0.0, auto_fit=True)
    result = layer.route("What is the tallest mountain?", {})
    assert result is not None
    assert result.layer_name == "layer3_classifier"
    assert result.decision.tier  # non-empty tier


def test_layer_abstains_below_threshold():
    layer = ClassifierRoutingLayer(min_confidence=1.01, auto_fit=True)
    result = layer.route("hello", {})
    assert result is None


def test_layer_injects_context():
    layer = ClassifierRoutingLayer(min_confidence=0.0, auto_fit=True)
    ctx: dict = {}
    layer.route("Write a function", ctx)
    assert "classifier_task" in ctx
    assert "classifier_confidence" in ctx


def test_layer_name():
    assert ClassifierRoutingLayer().name == "layer3_classifier"


def test_layer_decision_has_signals():
    layer = ClassifierRoutingLayer(min_confidence=0.0, auto_fit=True)
    result = layer.route("Write a Python class", {})
    assert result is not None
    assert "task_type" in result.decision.signals
