"""Theriad's provider-only billing policy.

Theriad currently has no subscription or platform billing. Users bring an
OpenRouter key, and OpenRouter bills that account directly for model usage.
The endpoint remains as a compatibility surface for the landing page while
subscription billing is dormant.
"""
from __future__ import annotations

from copy import deepcopy


PRICING_CATALOG_VERSION = "2026-09-17-byok"

_PRICING_CATALOG = {
    "version": PRICING_CATALOG_VERSION,
    "currency": "USD",
    "billing_status": "byok",
    "checkout_available": False,
    "provider": "OpenRouter",
    "access_mode": "bring_your_own_key",
    "local_hosting_required": False,
    "provider_costs": "direct_to_openrouter",
    "routing_fee": {
        "per_million_tokens_usd": 0.0,
        "applies_to": "none",
        "guest_demo": "exempt",
    },
    "free_models_available": True,
    "usage_billing": "direct_to_openrouter",
    "offers": [
        {
            "id": "byok",
            "name": "Bring your own key",
            "category": "provider",
            "eyebrow": "No Theriad subscription",
            "featured": True,
            "price_display": "$0 to Theriad",
            "annual_price_display": "",
            "annual_savings_display": "",
            "summary": "Add your OpenRouter API key and pay OpenRouter directly for the models you use.",
            "includes": [
                "Use OpenRouter free and paid models",
                "Keep provider billing and limits in your OpenRouter account",
                "Use Theriad routing, projects, and cost visibility",
            ],
        },
    ],
    "notes": [
        "Add an OpenRouter API key in Account before running paid-model requests.",
        "OpenRouter bills the account associated with the key at its current model rates.",
        "Free models remain subject to OpenRouter availability and limits.",
        "Theriad does not currently offer subscriptions, included allowances, checkout, or platform usage fees.",
        "No local model hosting is required.",
    ],
}


def get_pricing_catalog() -> dict:
    """Return an isolated copy of the public pricing policy catalog."""
    return deepcopy(_PRICING_CATALOG)
