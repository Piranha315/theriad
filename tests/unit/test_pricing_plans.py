"""Tests for the public OpenRouter-first pricing policy."""
from __future__ import annotations

from fastapi.testclient import TestClient

from theriad.api.server import app
from theriad.pricing_plans import PRICING_CATALOG_VERSION, get_pricing_catalog


def test_byok_policy_uses_openrouter_without_theriad_subscription_pricing():
    catalog = get_pricing_catalog()
    byok = next(item for item in catalog["offers"] if item["id"] == "byok")

    assert catalog["version"] == PRICING_CATALOG_VERSION
    assert catalog["provider"] == "OpenRouter"
    assert catalog["access_mode"] == "bring_your_own_key"
    assert catalog["local_hosting_required"] is False
    assert catalog["free_models_available"] is True
    assert catalog["billing_status"] == "byok"
    assert catalog["checkout_available"] is False
    assert catalog["usage_billing"] == "direct_to_openrouter"
    assert catalog["routing_fee"]["per_million_tokens_usd"] == 0.0
    assert byok["price_display"] == "$0 to Theriad"
    assert byok["annual_price_display"] == ""
    assert any("OpenRouter" in item for item in byok["includes"])


def test_catalog_returns_isolated_copy():
    first = get_pricing_catalog()
    first["offers"][0]["name"] = "mutated"

    second = get_pricing_catalog()
    assert second["offers"][0]["name"] == "Bring your own key"


def test_pricing_plans_endpoint_exposes_openrouter_policy():
    with TestClient(app) as client:
        response = client.get("/api/pricing/plans")

    assert response.status_code == 200
    payload = response.json()
    ids = {item["id"] for item in payload["offers"]}
    assert payload["billing_status"] == "byok"
    assert payload["provider"] == "OpenRouter"
    assert ids == {"byok"}
    assert payload["checkout_available"] is False
    assert any("bills the account" in note for note in payload["notes"])
    assert any("does not currently offer subscriptions" in note for note in payload["notes"])


def test_route_preview_does_not_add_a_platform_token_fee():
    with TestClient(app) as client:
        response = client.post(
            "/api/route",
            json={"prompt": "Summarize this short note.", "baseline": "openai/gpt-4o"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["platform_fee_usd"] == 0.0
    assert payload["estimated_cost_usd"] == payload["model_cost_usd"]
