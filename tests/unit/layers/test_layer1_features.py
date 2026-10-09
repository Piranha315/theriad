"""Tests for Layer 1 feature extraction and routing."""
from __future__ import annotations

import pytest

from theriad.layers.layer1_features.extractor import FeatureExtractor
from theriad.layers.layer1_features.layer import FeatureRoutingLayer
from theriad.models.tiers import TIER_CHEAP_BULK, TIER_CHEAP_CODE


@pytest.fixture
def extractor() -> FeatureExtractor:
    return FeatureExtractor()


# ── FeatureExtractor ──────────────────────────────────────────────────────────

def test_basic_word_and_line_count(extractor):
    prompt = "Hello world\nThis is line two"
    f = extractor.extract(prompt)
    assert f.word_count == 6
    assert f.line_count == 2


def test_token_count_positive(extractor):
    f = extractor.extract("The quick brown fox")
    assert f.token_count > 0


def test_code_fence_detected(extractor):
    prompt = "Fix this:\n```python\ndef foo():\n    return 1\n```"
    f = extractor.extract(prompt)
    assert f.code_fence_count == 1
    assert f.code_ratio > 0


def test_json_probability_valid_json(extractor):
    f = extractor.extract('{"key": "value", "num": 42}')
    assert f.json_probability == 1.0


def test_json_probability_plain_text(extractor):
    f = extractor.extract("Tell me about the weather")
    assert f.json_probability == 0.0


def test_markdown_ratio(extractor):
    prompt = "# H1\n## H2\n- item\n- item\n- item"
    f = extractor.extract(prompt)
    assert f.markdown_heading_count == 2
    assert f.markdown_list_count == 3
    assert f.markdown_ratio >= 1.0


def test_punctuation_ratio(extractor):
    f = extractor.extract("Hello, world! How are you?")
    assert 0 < f.punctuation_ratio < 0.3


def test_code_keyword_count(extractor):
    prompt = "def foo(): return None\ndef bar(): pass\nclass Baz: pass\nimport os\nfrom sys import argv"
    f = extractor.extract(prompt)
    assert f.code_keyword_count >= 4


def test_to_dict_has_all_fields(extractor):
    f = extractor.extract("test prompt")
    d = f.to_dict()
    expected_keys = {
        "token_count", "word_count", "line_count", "char_count",
        "punctuation_ratio", "code_fence_count", "inline_code_count",
        "code_keyword_count", "code_intent_count", "code_ratio", "markdown_heading_count",
        "markdown_list_count", "markdown_bold_count", "markdown_ratio",
        "json_brace_ratio", "json_probability",
    }
    assert expected_keys.issubset(d.keys())


# ── FeatureRoutingLayer ───────────────────────────────────────────────────────

def test_routes_valid_json():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    result = layer.route('{"name": "Alice", "role": "admin"}', {})
    assert result is not None
    assert result.layer_name == "layer1_features"
    assert result.decision.tier == TIER_CHEAP_BULK


def test_short_prompt_abstains_without_success_evidence():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    result = layer.route("What is 2+2?", {})
    assert result is None


def test_routes_code_heavy_prompt():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    prompt = "\n".join([
        "```python",
        "def quicksort(arr):",
        "    if len(arr) <= 1:",
        "        return arr",
        "    pivot = arr[0]",
        "    left = [x for x in arr[1:] if x <= pivot]",
        "    right = [x for x in arr[1:] if x > pivot]",
        "    return quicksort(left) + [pivot] + quicksort(right)",
        "```",
    ])
    result = layer.route(prompt, {})
    assert result is not None
    assert result.decision.tier == TIER_CHEAP_CODE


def test_routes_natural_language_coding_request():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    prompt = (
        "Write a Python function that takes a list of URLs and returns their "
        "HTTP status codes concurrently using asyncio."
    )
    result = layer.route(prompt, {})
    assert result is not None
    assert result.decision.tier == TIER_CHEAP_CODE


def test_features_injected_into_context():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    ctx: dict = {}
    layer.route("Short prompt.", ctx)
    # Features must always be injected, regardless of routing decision
    assert "features" in ctx


def test_abstains_on_medium_english():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    sentence = "The conference was attended by many researchers and industry professionals. "
    medium = sentence * 20  # ~200 tokens — not short, not long, no code/json/markdown
    result = layer.route(medium, {})
    # Should abstain — no strong signal
    assert result is None


def test_decision_carries_signals():
    layer = FeatureRoutingLayer(min_confidence=0.75)
    result = layer.route('{"name": "Alice"}', {})
    assert result is not None
    assert len(result.decision.signals) > 0
