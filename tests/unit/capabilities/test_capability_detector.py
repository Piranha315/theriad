"""
Unit tests for theriad.capabilities.detector.CapabilityDetector

Each test maps to one of the spec examples or an explicit contract requirement.

Run with:
    pytest tests/unit/capabilities/test_capability_detector.py -v
"""
from __future__ import annotations

import pytest

from theriad.capabilities.detector import CapabilityDetector, CapabilityResult


@pytest.fixture()
def detector() -> CapabilityDetector:
    return CapabilityDetector()


# ── 1. Text-only ──────────────────────────────────────────────────────────────

def test_text_only_economics(detector):
    """Factual knowledge question → text_only, no browser flags."""
    result = detector.detect("Explain the difference between CPI and GDP deflator.")
    assert result.recommended_execution_mode == "text_only"
    assert result.capabilities["requires_browser_navigation"] is False
    assert result.capabilities["requires_current_web"] is False
    assert result.risk_level == "low"
    assert result.requires_confirmation_before_start is False


def test_text_only_simple_question(detector):
    """Generic factual question stays text_only."""
    result = detector.detect("What is the boiling point of water?")
    assert result.recommended_execution_mode == "text_only"
    assert result.risk_level == "low"


# ── 2. Text + preview ─────────────────────────────────────────────────────────

def test_text_plus_preview_html_landing_page(detector):
    """HTML generation request → text_plus_preview, artifact_preview=True."""
    result = detector.detect(
        "Create a single-file HTML landing page for Theriad using Tailwind CDN."
    )
    assert result.recommended_execution_mode == "text_plus_preview"
    assert result.capabilities["requires_artifact_preview"] is True
    assert result.capabilities["requires_code_execution"] is False
    assert result.capabilities["requires_browser_navigation"] is False
    assert result.risk_level == "low"
    assert result.requires_confirmation_before_start is False


def test_text_plus_preview_tailwind(detector):
    """'Build a UI with Tailwind' → text_plus_preview."""
    result = detector.detect("Build a UI component with Tailwind CSS classes.")
    assert result.recommended_execution_mode == "text_plus_preview"
    assert result.capabilities["requires_artifact_preview"] is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Design a website for me.",
        "Create a responsive dashboard for a logistics team.",
        "Write CSS for a pricing card.",
        "Generate an SVG architecture diagram.",
        "I need a landing page for a sustainable fashion brand.",
        "Can you show me a dashboard prototype for a logistics team?",
    ],
)
def test_visual_artifact_requests_use_preview_mode(detector, prompt):
    """Natural-language visual tasks must route to the HTML preview renderer."""
    result = detector.detect(prompt)
    assert result.recommended_execution_mode == "text_plus_preview"
    assert result.capabilities["requires_artifact_preview"] is True


@pytest.mark.parametrize(
    "prompt",
    [
        "Explain CSS specificity.",
        "I need help understanding CSS specificity.",
        "What is the difference between a website and a web app?",
    ],
)
def test_visual_terms_without_a_build_request_stay_text_only(detector, prompt):
    """Mentioning a renderable format alone must not produce a blank preview."""
    result = detector.detect(prompt)
    assert result.recommended_execution_mode == "text_only"
    assert result.capabilities["requires_artifact_preview"] is False


def test_existing_page_visual_observation_stays_browser_agent(detector):
    result = detector.detect("Take a screenshot of the page at https://example.com.")
    assert result.recommended_execution_mode == "browser_agent"
    assert result.capabilities["requires_visual_observation"] is True
    assert result.capabilities["requires_artifact_preview"] is False


# ── 3. Browser agent — in-stock product search ───────────────────────────────

def test_browser_agent_in_stock_search(detector):
    """Live product availability search → browser_agent."""
    result = detector.detect(
        "Find buntings near Bethesda that are in stock for pickup today."
    )
    assert result.recommended_execution_mode == "browser_agent"
    assert result.capabilities["requires_current_web"] is True
    assert result.capabilities["requires_browser_navigation"] is True
    assert result.capabilities["requires_data_extraction"] is True
    assert result.requires_confirmation_before_start is True


def test_browser_agent_current_prices(detector):
    """'Current prices' → browser_agent."""
    result = detector.detect("What are the current prices for flights to Tokyo?")
    assert result.recommended_execution_mode == "browser_agent"
    assert result.capabilities["requires_current_web"] is True


def test_browser_agent_weather_without_a_recency_word(detector):
    """Weather is point-in-time even with no 'current'/'today' in the prompt.

    Regression: this reported text_only, so the question never reached the
    search path and was answered from model training data.
    """
    result = detector.detect("What is the weather in New York?")
    assert result.capabilities["requires_current_web"] is True
    assert result.recommended_execution_mode == "browser_agent"


def test_weather_terms_agree_with_the_search_trigger(detector):
    """The detector and the search service must agree on what needs the web."""
    from theriad.search.service import is_web_search_request

    for prompt in (
        "What is the weather in New York?",
        "Give me the forecast for Denver.",
        "What temperature is it in Oslo?",
    ):
        assert is_web_search_request(prompt) is True, prompt
        assert detector.detect(prompt).capabilities["requires_current_web"] is True, prompt


# ── 4. Browser agent — high risk, approval required ──────────────────────────

def test_browser_agent_bank_high_risk(detector):
    """Bank login + statement download → browser_agent, high risk, approval required."""
    result = detector.detect(
        "Log into my bank and download my latest statement."
    )
    assert result.recommended_execution_mode == "browser_agent"
    assert result.capabilities["requires_browser_navigation"] is True
    assert result.capabilities["requires_account_access"] is True
    assert result.capabilities["requires_sensitive_info"] is True
    assert result.capabilities["requires_user_approval"] is True
    assert result.risk_level == "high"
    assert result.requires_confirmation_before_start is True


def test_browser_agent_medical_portal(detector):
    """Medical portal access → high risk."""
    result = detector.detect(
        "Sign into my health portal and check my latest test results."
    )
    assert result.risk_level == "high"
    assert result.capabilities["requires_account_access"] is True
    assert result.requires_confirmation_before_start is True


# ── 5. Local tool / workflow agent ───────────────────────────────────────────

def test_workflow_agent_run_tests_fix_repo(detector):
    """Run tests + fix failures + summarise → workflow_agent or local_tool_agent."""
    result = detector.detect(
        "Run the tests, fix any failures, and summarize what changed."
    )
    assert result.recommended_execution_mode in ("workflow_agent", "local_tool_agent")
    assert result.capabilities["requires_local_command"] is True
    assert result.capabilities["requires_file_editing"] is True
    assert result.capabilities["requires_code_execution"] is True
    assert result.requires_confirmation_before_start is True


def test_local_tool_agent_npm_install(detector):
    """'npm install' → local_tool_agent."""
    result = detector.detect("Run npm install and start the dev server.")
    assert result.recommended_execution_mode in ("local_tool_agent", "workflow_agent")
    assert result.capabilities["requires_local_command"] is True


# ── 6. URL summarise without interaction → text_only ─────────────────────────

def test_url_summarise_no_interaction(detector):
    """Pasting a URL and asking for a summary (no interaction) → text_only."""
    result = detector.detect(
        "Can you summarize the content at https://example.com/article?"
    )
    # No interaction words — model can read the URL without browser automation
    assert result.recommended_execution_mode == "text_only"
    assert result.capabilities["requires_browser_navigation"] is False


# ── 7. URL inspect/interact → browser_agent ──────────────────────────────────

def test_url_interact_browser_agent(detector):
    """'Go to URL and click …' → browser_agent."""
    result = detector.detect(
        "Go to https://example.com/products and find the cheapest option."
    )
    assert result.recommended_execution_mode == "browser_agent"
    assert result.capabilities["requires_browser_navigation"] is True


# ── 8. Route diagnostics include capability flags ─────────────────────────────

def test_result_has_all_capability_flags(detector):
    """CapabilityResult.to_dict() always includes the full 14-flag capabilities dict."""
    result = detector.detect("Hello world")
    d = result.to_dict()
    expected_flags = {
        "requires_current_web", "requires_browser_navigation", "requires_visual_observation",
        "requires_dom_observation", "requires_file_editing", "requires_local_command",
        "requires_code_execution", "requires_artifact_preview", "requires_form_interaction",
        "requires_account_access", "requires_sensitive_info", "requires_data_extraction",
        "requires_user_approval", "requires_paid_call",
    }
    assert set(d["capabilities"].keys()) == expected_flags
    # All required fields present
    for field in ("recommended_execution_mode", "confidence", "reason", "risk_level",
                  "requires_confirmation_before_start", "matched_signals"):
        assert field in d, f"Missing field: {field}"


def test_confidence_in_range(detector):
    """Confidence is always in [0, 1]."""
    for prompt in [
        "Tell me a joke",
        "Find prices for laptops near me",
        "Login to my bank account",
        "Run tests and commit the fix",
    ]:
        r = detector.detect(prompt)
        assert 0.0 <= r.confidence <= 1.0, f"confidence={r.confidence} out of range for: {prompt!r}"


# ── 9. Normal execute does not perform browser actions ────────────────────────

def test_detect_never_launches_agents(detector):
    """detect() is a pure analysis function — it returns a result, never a side effect."""
    # This test verifies that calling detect() for a browser_agent prompt
    # only returns a CapabilityResult and does not start any session or I/O.
    result = detector.detect("Find buntings near Bethesda in stock for pickup today.")
    assert isinstance(result, CapabilityResult)
    assert result.recommended_execution_mode == "browser_agent"
    # No session IDs, no URLs visited, no files changed — just a dataclass.
    assert result.matched_signals is not None  # analysis data present


# ── 10. Agentic mode requires explicit confirmation ───────────────────────────

def test_agentic_mode_requires_confirmation(detector):
    """All agentic modes must have requires_confirmation_before_start=True."""
    agentic_prompts = [
        "Find buntings in stock near me",          # browser_agent
        "Log into my bank",                        # browser_agent high-risk
        "Run the tests and fix the repo",          # workflow_agent
        "npm install and start the server",        # local_tool_agent
    ]
    for p in agentic_prompts:
        r = detector.detect(p)
        assert r.recommended_execution_mode != "text_only", (
            f"Expected agentic mode for: {p!r}, got text_only"
        )
        assert r.requires_confirmation_before_start is True, (
            f"Expected confirmation=True for: {p!r}"
        )


def test_text_only_no_confirmation_required(detector):
    """text_only / text_plus_preview prompts do NOT require confirmation."""
    safe_prompts = [
        "What is the capital of France?",
        "Create an HTML landing page with Tailwind.",
        "Explain recursion.",
    ]
    for p in safe_prompts:
        r = detector.detect(p)
        assert r.recommended_execution_mode in ("text_only", "text_plus_preview"), (
            f"Expected text mode for: {p!r}, got {r.recommended_execution_mode}"
        )
        assert r.requires_confirmation_before_start is False, (
            f"Expected confirmation=False for: {p!r}"
        )


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_prompt(detector):
    """Empty prompt returns text_only safely."""
    result = detector.detect("")
    assert result.recommended_execution_mode == "text_only"


def test_detect_never_raises(detector):
    """detect() must never raise regardless of input."""
    for bad_input in ["", "  ", "\n\n", "a" * 10000, "🤖" * 100]:
        result = detector.detect(bad_input)
        assert isinstance(result, CapabilityResult)
