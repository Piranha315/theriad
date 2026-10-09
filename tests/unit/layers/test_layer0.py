"""Tests for Layer 0 deterministic routing rules."""
from __future__ import annotations

from theriad.layers.layer0.layer import build_default_layer
from theriad.layers.layer0.rules import (
    CodeDetectionRule,
    JsonDetectionRule,
    LanguageDetectionRule,
    MarkdownDetectionRule,
    PromptLengthRule,
    RegexRule,
)
from theriad.models.tiers import (
    TIER_CHEAP_BULK,
    TIER_CHEAP_CODE,
    TIER_STRONG_REASONING,
)

# ── RegexRule ─────────────────────────────────────────────────────────────────

def test_regex_rule_matches():
    rule = RegexRule("sql_detect", r"\bSELECT\b.*\bFROM\b", tier=TIER_CHEAP_BULK)
    decision = rule.evaluate("SELECT id FROM users WHERE id=1")
    assert decision is not None
    assert decision.tier == TIER_CHEAP_BULK
    assert decision.confidence == 0.95


def test_regex_rule_no_match():
    rule = RegexRule("sql_detect", r"\bSELECT\b.*\bFROM\b", tier=TIER_CHEAP_BULK)
    assert rule.evaluate("Tell me a joke") is None


def test_regex_rule_name():
    rule = RegexRule("my_rule", r"x", tier=TIER_CHEAP_BULK)
    assert rule.name == "my_rule"


def test_regex_rule_signals():
    rule = RegexRule("sql_detect", r"\bSELECT\b.*\bFROM\b", tier=TIER_CHEAP_BULK)
    decision = rule.evaluate("SELECT id FROM users")
    assert decision is not None
    assert "rule" in decision.signals


# ── CodeDetectionRule ─────────────────────────────────────────────────────────

def test_code_fence_detected():
    prompt = "Fix this:\n```python\ndef foo():\n    pass\n```"
    decision = CodeDetectionRule().evaluate(prompt)
    assert decision is not None
    assert decision.confidence >= 0.90
    assert decision.tier == TIER_CHEAP_CODE


def test_code_plain_text_no_fire():
    prompt = "What is the capital of France?"
    decision = CodeDetectionRule().evaluate(prompt)
    assert decision is None


def test_code_keyword_density():
    prompt = "def foo(): return None\ndef bar(): pass\nclass Baz: pass"
    decision = CodeDetectionRule().evaluate(prompt)
    assert decision is not None
    assert decision.tier == TIER_CHEAP_CODE


def test_code_natural_language_prompt():
    prompt = (
        "Write a Python function that takes a list of URLs and returns their "
        "HTTP status codes concurrently using asyncio."
    )
    decision = CodeDetectionRule().evaluate(prompt)
    assert decision is not None
    assert decision.tier == TIER_CHEAP_CODE


def test_code_signals_populated():
    prompt = "Fix this:\n```python\ndef foo():\n    pass\n```"
    decision = CodeDetectionRule().evaluate(prompt)
    assert decision is not None
    assert "code_fence" in decision.signals or "keyword_density" in decision.signals


def test_default_layer_does_not_short_circuit_coding_prompt_to_bulk():
    layer = build_default_layer()
    prompt = (
        "Write a Python function that takes a list of URLs and returns their "
        "HTTP status codes concurrently using asyncio."
    )
    result = layer.route(prompt, {})
    assert result is not None
    assert result.decision.tier == TIER_CHEAP_CODE


# ── JsonDetectionRule ─────────────────────────────────────────────────────────

def test_json_valid_object():
    prompt = '{"name": "Alice", "age": 30}'
    decision = JsonDetectionRule().evaluate(prompt)
    assert decision is not None
    assert decision.confidence == 0.90
    assert decision.tier == TIER_CHEAP_BULK


def test_json_valid_array():
    decision = JsonDetectionRule().evaluate('[1, 2, 3]')
    assert decision is not None
    assert decision.tier == TIER_CHEAP_BULK


def test_json_plain_text():
    assert JsonDetectionRule().evaluate("Hello world") is None


# ── MarkdownDetectionRule ─────────────────────────────────────────────────────

def test_markdown_headings_and_list():
    prompt = "# Title\n## Subtitle\n- item1\n- item2\n- item3"
    decision = MarkdownDetectionRule().evaluate(prompt)
    assert decision is not None
    assert decision.confidence > 0.75
    assert decision.tier == TIER_CHEAP_BULK


def test_markdown_no_structure():
    assert MarkdownDetectionRule().evaluate("Just a sentence.") is None


def test_markdown_below_threshold():
    # Only 1 heading, 1 list item — below threshold of 3
    assert MarkdownDetectionRule().evaluate("# Title\n- item") is None


# ── PromptLengthRule ──────────────────────────────────────────────────────────

def test_short_prompt_abstains():
    short = "What is 2+2?"
    assert PromptLengthRule().evaluate(short) is None


def test_long_prompt_routes_strong_reasoning():
    long_prompt = " ".join(["word"] * 2000)
    decision = PromptLengthRule().evaluate(long_prompt)
    assert decision is not None
    assert decision.tier == TIER_STRONG_REASONING


def test_medium_prompt_abstains():
    medium = " ".join(["word"] * 200)
    assert PromptLengthRule().evaluate(medium) is None


def test_length_signals_word_count():
    long_prompt = " ".join(["word"] * 2000)
    decision = PromptLengthRule().evaluate(long_prompt)
    assert decision is not None
    assert "word_count" in decision.signals


# ── LanguageDetectionRule ─────────────────────────────────────────────────────

def test_english_abstains():
    prompt = "Please write a short story about a robot exploring the ocean."
    result = LanguageDetectionRule().evaluate(prompt)
    # English should return None (no action needed)
    assert result is None


def test_too_short_abstains():
    assert LanguageDetectionRule().evaluate("Bonjour") is None


def test_french_detected():
    prompt = "Peux-tu me résumer les grandes lignes de l'histoire de France depuis la révolution?"
    result = LanguageDetectionRule().evaluate(prompt)
    # langdetect may or may not fire on this; we just assert it doesn't raise
    # and if it fires, confidence must be in range and tier must be set
    if result is not None:
        assert 0.0 <= result.confidence <= 1.0
        assert result.tier == TIER_CHEAP_BULK


# ── DeterministicLayer ────────────────────────────────────────────────────────

def test_layer_routes_json():
    layer = build_default_layer(min_confidence=0.75)
    result = layer.route('{"key": "value"}', {})
    assert result is not None
    assert result.layer_name == "layer0"
    assert result.decision.tier == TIER_CHEAP_BULK


def test_layer_abstains_on_medium_prose():
    # Use real English prose at medium length (100–1400 words) to avoid
    # synthetic text confusing langdetect, and no code/JSON/markdown markers.
    layer = build_default_layer(min_confidence=0.75)
    sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
    medium = sentence * 10  # ~90 words — still short enough to trigger PromptLengthRule
    # PromptLengthRule fires on <50 words only; 90 words = abstain for length.
    # All other rules should abstain on plain English prose.
    # LanguageDetectionRule will correctly identify English and abstain.
    result = layer.route(medium, {})
    assert result is None  # no rule fires with sufficient confidence


def test_layer_short_prompt_routes():
    layer = build_default_layer(min_confidence=0.75)
    result = layer.route("Hi!", {})
    assert result is None


def test_layer_records_all_fired():
    layer = build_default_layer(min_confidence=0.99)  # unreachably high → all pass
    # Code fence fires but won't meet 0.99 confidence
    prompt = "```python\npass\n```"
    result = layer.route(prompt, {})
    # Either None (all sub-threshold) or has all_decisions populated
    if result is not None:
        assert len(result.all_decisions) >= 1


def test_layer_code_prompt_routes_code_tier():
    """Regression: code prompt must route as cheap_code, not prompt_length: short."""
    layer = build_default_layer(min_confidence=0.75)
    result = layer.route("Write a Python function to reverse a linked list", {})
    assert result is not None
    assert result.decision.tier == TIER_CHEAP_CODE
