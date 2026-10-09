"""Theriad's OpenRouter-first self-serve and enterprise pricing policy.

The subscription covers Theriad access and a cost-based allowance for paid
OpenRouter calls. Free models remain available when listed by OpenRouter, and
paid usage continues pay-as-you-go after the allowance. Authenticated paid
calls also carry a small transparent Theriad routing fee when billing is live.
These plans do not require local model hosting. The catalog is truthful about
whether the optional billing boundary is configured.
"""
from __future__ import annotations

from copy import deepcopy


PRICING_CATALOG_VERSION = "2026-08-20"
PROVIDER_COST_SHARE = 0.90


def _usage_allowance(monthly_price: float | None, annual_price: float | None) -> dict:
    """Build a usage allowance from the price of each billing period.

    OpenRouter prices differ by model and by input/output token mix, so the
    policy stores a cost cap and labels the token limit as model-dependent.
    """
    monthly_allowance = (
        round(monthly_price * PROVIDER_COST_SHARE, 2)
        if monthly_price is not None else None
    )
    annual_allowance = (
        round(annual_price * PROVIDER_COST_SHARE, 2)
        if annual_price is not None else None
    )
    return {
        "provider_cost_share": PROVIDER_COST_SHARE,
        "monthly_provider_cost_usd": monthly_allowance,
        "annual_provider_cost_usd": annual_allowance,
        "monthly_display": (
            f"Up to ${monthly_allowance:.2f} / month"
            if monthly_allowance is not None else "Custom monthly allowance"
        ),
        "annual_display": (
            f"Up to ${annual_allowance:.2f} / year"
            if annual_allowance is not None else "Custom annual allowance"
        ),
        "token_limit_display": "Model-dependent token limit",
        "token_limit_basis": "actual OpenRouter input/output cost",
        "overage": "pay_as_you_go_through_openrouter",
    }

_PRICING_CATALOG = {
    "version": PRICING_CATALOG_VERSION,
    "currency": "USD",
    "billing_status": "policy_only",
    "provider": "OpenRouter",
    "access_mode": "hosted_via_openrouter",
    "local_hosting_required": False,
    "provider_costs": "included_allowance_then_openrouter_payg",
    "routing_fee": {
        "per_million_tokens_usd": 0.001,
        "applies_to": "authenticated_paid_calls",
        "guest_demo": "exempt",
    },
    "free_models_available": True,
    "usage_billing": "included_allowance_then_pay_as_you_go",
    "annual_discount": "2 months free",
    "pricing_rule": "annual price = 10 monthly payments",
    "usage_limit_policy": {
        "provider_cost_share": PROVIDER_COST_SHARE,
        "subscription_cost_share": round(1 - PROVIDER_COST_SHARE, 2),
        "calculation": "billing_period_price * 0.90",
        "basis": "actual OpenRouter model cost",
        "token_limit": "model_dependent",
        "overage": "pay_as_you_go_through_openrouter",
    },
    "offers": [
        {
            "id": "starter",
            "name": "Starter",
            "category": "subscription",
            "eyebrow": "For everyday use",
            "featured": True,
            "price_display": "$5 / month",
            "annual_price_display": "$50 / year",
            "annual_savings_display": "Save $10",
            "price_monthly_usd": 5,
            "price_annual_usd": 50,
            "price_min_usd": 5,
            "price_max_usd": 5,
            "annual_savings_usd": 10,
            "billing_period": "monthly_or_annual",
            "usage_allowance": _usage_allowance(5, 50),
            "summary": "A simple Theriad workspace through OpenRouter. No local model hosting required.",
            "includes": [
                "Use OpenRouter models, including free models when available",
                "Projects, routing, and cost visibility",
                "Paid models through OpenRouter, with pay-as-you-go after the allowance",
            ],
            "excludes": [
                "Usage beyond the included allowance",
                "Enterprise usage and support terms",
            ],
        },
        {
            "id": "pro",
            "name": "Pro",
            "category": "subscription",
            "eyebrow": "For more headroom",
            "featured": False,
            "price_display": "$10 / month",
            "annual_price_display": "$100 / year",
            "annual_savings_display": "Save $20",
            "price_monthly_usd": 10,
            "price_annual_usd": 100,
            "price_min_usd": 10,
            "price_max_usd": 10,
            "annual_savings_usd": 20,
            "billing_period": "monthly_or_annual",
            "usage_allowance": _usage_allowance(10, 100),
            "summary": "More room for regular workflows while you keep choosing the model path.",
            "includes": [
                "Everything in Starter",
                "Higher self-serve usage allowance",
                "Free and paid OpenRouter models",
            ],
            "excludes": [
                "Usage beyond the included allowance",
                "Enterprise support and custom terms",
            ],
        },
        {
            "id": "enterprise-usage",
            "name": "Enterprise",
            "category": "enterprise",
            "eyebrow": "For teams that need more",
            "featured": False,
            "price_display": "Contact us",
            "annual_price_display": "Annual terms available",
            "annual_savings_display": "",
            "price_monthly_usd": None,
            "price_annual_usd": None,
            "price_min_usd": None,
            "price_max_usd": None,
            "annual_savings_usd": None,
            "billing_period": "custom",
            "usage_allowance": _usage_allowance(None, None),
            "summary": "Custom usage and support through OpenRouter for teams that outgrow self-serve plans.",
            "includes": [
                "Higher usage through OpenRouter",
                "Team terms and defined support",
                "Custom routing, security, and operational scope",
            ],
            "excludes": [
                "Automatic authorization or compliance certification",
            ],
        },
    ],
    "notes": [
        "Each plan includes paid OpenRouter model usage up to 90% of the subscription price for that billing period.",
        "The token limit is model-dependent because OpenRouter input and output prices vary by model.",
        "Free OpenRouter models do not draw down the paid-model allowance; they remain subject to OpenRouter's own availability and limits.",
        "After the included allowance, paid model use continues pay-as-you-go through OpenRouter.",
        "No local model hosting is required for these plans.",
        "Subscription prices cover Theriad access; the included allowance is based on actual OpenRouter model cost, and authenticated paid calls add a $0.001 per 1M token Theriad routing fee when billing is live.",
        "Yearly plans are priced at 10 months of the monthly price.",
    ],
}


def get_pricing_catalog() -> dict:
    """Return an isolated copy of the public pricing policy catalog."""
    catalog = deepcopy(_PRICING_CATALOG)
    # Do not turn missing Stripe configuration into a simulated checkout. The
    # public catalog simply reflects whether a real checkout boundary exists.
    try:
        from theriad.billing.service import public_billing_status
        catalog["billing_status"] = public_billing_status()
        catalog["checkout_available"] = catalog["billing_status"] == "live"
    except Exception:
        catalog["checkout_available"] = False
    return catalog
