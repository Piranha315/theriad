"""
Tests for BuildProjectRule (Layer 0).

Key invariants:
- Full-stack app/build prompts → code_strong (never cheap_bulk)
- Short build prompts route to code_strong from build intent, not length
- Explanatory/educational prompts starting with skip-verbs → abstain
- Plain short prompts with no app signals → abstain
"""
from __future__ import annotations

import pytest

from theriad.layers.layer0.layer import build_default_layer
from theriad.layers.layer0.rules import BuildProjectRule
from theriad.models.tiers import TIER_CHEAP_BULK, TIER_CODE_STRONG


# ── BuildProjectRule unit tests ────────────────────────────────────────────────

class TestBuildProjectRule:

    def setup_method(self):
        self.rule = BuildProjectRule()

    # Should fire: build verb + app signal(s)

    def test_build_react_app_with_auth_fires(self):
        d = self.rule.evaluate("Build a React app with auth and tests.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_implement_backend_service_fires(self):
        d = self.rule.evaluate("Implement a backend service with a database.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_create_api_fires(self):
        d = self.rule.evaluate("Create a REST API service.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_develop_full_stack_app_fires(self):
        d = self.rule.evaluate("Develop a full-stack application with frontend and backend.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_scaffold_project_with_tests_fires(self):
        d = self.rule.evaluate("Scaffold a new project with tests and auth.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_build_with_database_and_auth_fires(self):
        d = self.rule.evaluate("Build an app with database, auth, and API endpoints.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG

    def test_two_app_signals_gives_higher_confidence(self):
        single = self.rule.evaluate("Create a service.")
        multi = self.rule.evaluate("Create an app with backend and database.")
        assert single is not None and multi is not None
        assert multi.confidence >= single.confidence

    # Should NOT fire: skip verbs at start

    def test_explain_how_to_build_abstains(self):
        d = self.rule.evaluate("Explain how to build a React app with auth.")
        assert d is None

    def test_describe_how_to_create_abstains(self):
        d = self.rule.evaluate("Describe how to create a backend service.")
        assert d is None

    def test_what_is_abstains(self):
        d = self.rule.evaluate("What is the best way to build an app?")
        assert d is None

    def test_list_abstains(self):
        d = self.rule.evaluate("List the steps to create a service.")
        assert d is None

    # Should NOT fire: build verb but no app signals

    def test_build_with_no_app_signal_abstains(self):
        d = self.rule.evaluate("Build a quick summary of this text.")
        assert d is None

    def test_create_list_abstains(self):
        d = self.rule.evaluate("Create a list of countries.")
        assert d is None

    # Should NOT fire: app signals but no build verb

    def test_app_signal_only_abstains(self):
        d = self.rule.evaluate("The frontend has a bug in the authentication flow.")
        assert d is None

    # Rule properties

    def test_name_is_build_project(self):
        assert self.rule.name == "build_project"

    def test_returns_routing_decision(self):
        from theriad.core.routing.interfaces import RoutingDecision
        d = self.rule.evaluate("Build a React app with auth.")
        assert isinstance(d, RoutingDecision)

    def test_signals_dict_in_decision(self):
        d = self.rule.evaluate("Build an app with backend and tests.")
        assert d is not None
        assert "build_verb_count" in d.signals
        assert "app_signal_count" in d.signals
        assert d.signals["build_verb_count"] >= 1
        assert d.signals["app_signal_count"] >= 1

    # Punctuation stripping

    def test_punctuation_does_not_break_detection(self):
        d = self.rule.evaluate("Build a backend, with auth, tests, and an API.")
        assert d is not None
        assert d.tier == TIER_CODE_STRONG


# ── Integration: BuildProjectRule fires BEFORE PromptLengthRule ──────────────

class TestBuildRulePreemptsPromptLength:
    """
    Short build prompts (< 50 words) must route to code_strong, not cheap_bulk.
    BuildProjectRule must remain the decisive signal for short build prompts.
    """

    def setup_method(self):
        self.layer = build_default_layer()

    def test_short_build_app_routes_to_code_strong_not_cheap_bulk(self):
        prompt = "Build a React app with auth and tests."
        assert len(prompt.split()) < 50, "prompt must be short for this test to be meaningful"
        result = self.layer.route(prompt, {})
        assert result is not None
        assert result.decision.tier == TIER_CODE_STRONG
        assert result.decision.tier != TIER_CHEAP_BULK

    def test_create_backend_service_is_not_cheap_bulk(self):
        prompt = "Create a backend API service with a database."
        result = self.layer.route(prompt, {})
        assert result is not None
        assert result.decision.tier != TIER_CHEAP_BULK

    def test_implement_auth_routes_to_code_strong(self):
        prompt = "Implement user authentication for the app."
        result = self.layer.route(prompt, {})
        assert result is not None
        assert result.decision.tier == TIER_CODE_STRONG

    def test_short_simple_prompt_does_not_assume_easy(self):
        """Shortness alone must not force the cheapest tier."""
        prompt = "What is 2 plus 2?"
        result = self.layer.route(prompt, {})
        assert result is None

    def test_layer_name_reflects_layer0_for_build_prompts(self):
        result = self.layer.route("Build an app with backend and auth.", {})
        assert result is not None
        assert result.layer_name == "layer0"

    def test_winning_rule_name_is_build_project(self):
        result = self.layer.route("Build a React app with auth, database, and API.", {})
        assert result is not None
        # The winning decision should come from build_project rule
        assert result.decision.reason.startswith("build_project:")


# ── Parametrized cross-check ──────────────────────────────────────────────────

@pytest.mark.parametrize("prompt, expect_tier", [
    # Full-stack → code_strong
    ("Build a React app with auth and tests.", TIER_CODE_STRONG),
    ("Create a backend service with database.", TIER_CODE_STRONG),
    ("Implement an API server with endpoints.", TIER_CODE_STRONG),
    ("Scaffold a project with tests and authentication.", TIER_CODE_STRONG),
    # Explanation and plain short prompts do not assume an easy model.
    ("Explain how to build a React app with auth.", None),
    ("What is the capital of France?", None),
])
def test_layer0_tier_for_build_prompts(prompt, expect_tier):
    layer = build_default_layer()
    result = layer.route(prompt, {})
    if expect_tier is None:
        assert result is None
        return
    assert result is not None, f"Layer abstained for: {prompt!r}"
    assert result.decision.tier == expect_tier, (
        f"Expected {expect_tier!r} for {prompt!r}, got {result.decision.tier!r} "
        f"(reason: {result.decision.reason!r})"
    )
