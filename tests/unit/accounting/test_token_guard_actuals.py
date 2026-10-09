from theriad.accounting.actuals import (
    cached_prompt_tokens,
    provider_reported_cost,
    realized_cost,
    usage_is_available,
)
from theriad.accounting.pricing import PricingTable
from theriad.accounting.token_guard import assess_token_risk
from theriad.core.providers.interfaces import CompletionResponse


def _response(raw=None, model="groq/openai/gpt-oss-20b"):
    return CompletionResponse(
        model=model,
        content="done",
        prompt_tokens=1_000,
        completion_tokens=500,
        latency_ms=10,
        provider="openrouter",
        raw=raw or {},
    )


def test_large_context_requires_confirmation_but_normal_message_does_not():
    large = assess_token_risk(
        source_tokens=8_000,
        sent_tokens=4_000,
        expected_output_tokens=1_000,
        estimated_cost_usd=0.001,
        selected_model="paid/model",
        free_model="free/model",
    )
    normal = assess_token_risk(
        source_tokens=500,
        sent_tokens=500,
        expected_output_tokens=500,
        estimated_cost_usd=0.001,
        selected_model="paid/model",
        free_model="free/model",
    )

    assert large.requires_confirmation is True
    assert normal.requires_confirmation is False


def test_provider_reported_cost_and_cache_usage_take_priority():
    response = _response(
        {
            "usage": {
                "cost": 0.000123,
                "prompt_tokens_details": {"cached_tokens": 750},
            }
        }
    )

    assert provider_reported_cost(response) == 0.000123
    assert realized_cost(response, PricingTable()) == 0.000123
    assert cached_prompt_tokens(response) == 750


def test_realized_cost_falls_back_to_price_table():
    response = _response()
    assert realized_cost(response, PricingTable()) is not None


def test_missing_provider_usage_is_not_treated_as_an_estimate():
    response = _response(raw={"id": "provider-response", "choices": [{}]}, model="qwen/qwen3-14b")
    response.prompt_tokens = 0
    response.completion_tokens = 0

    assert usage_is_available(response) is False
    assert realized_cost(response, PricingTable()) is None
