"""Small, explicit Stripe boundary for Theriad subscriptions.

Stripe is optional at development time. When it is not configured, the API
reports ``policy_only`` and never invents a checkout URL or subscription state.
OpenRouter usage remains BYOK/pay-as-you-go; a Theriad subscription does not
silently become a managed model allowance.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from datetime import datetime, timezone
from dataclasses import dataclass

import httpx

from theriad.accounting.pricing import PLATFORM_FEE_PER_1M_TOKENS, platform_fee
from theriad.storage.database import get_db

STRIPE_API_BASE = "https://api.stripe.com/v1"
WEBHOOK_TOLERANCE_SECONDS = 300
ROUTING_FEE_POLICY_VERSION = "2026-08-20-routing-fee-v1"


@dataclass(frozen=True)
class BillingConfig:
    secret_key: str | None
    webhook_secret: str | None
    success_url: str
    cancel_url: str
    prices: dict[tuple[str, str], str]

    @property
    def checkout_ready(self) -> bool:
        return bool(self.secret_key and all(self.prices.values()))

    @property
    def webhook_ready(self) -> bool:
        return bool(self.webhook_secret)

    @property
    def live_ready(self) -> bool:
        return self.checkout_ready and self.webhook_ready


def get_billing_config() -> BillingConfig:
    prices: dict[tuple[str, str], str] = {}
    for plan_id in ("starter", "pro"):
        for interval in ("monthly", "annual"):
            value = os.environ.get(
                f"THERIAD_STRIPE_{plan_id.upper()}_{interval.upper()}_PRICE_ID", ""
            ).strip()
            prices[(plan_id, interval)] = value
    return BillingConfig(
        secret_key=os.environ.get("STRIPE_SECRET_KEY", "").strip() or None,
        webhook_secret=os.environ.get("STRIPE_WEBHOOK_SECRET", "").strip() or None,
        success_url=os.environ.get(
            "THERIAD_BILLING_SUCCESS_URL", "http://localhost:5173/?billing=success"
        ).strip(),
        cancel_url=os.environ.get(
            "THERIAD_BILLING_CANCEL_URL", "http://localhost:5173/?billing=cancel"
        ).strip(),
        prices=prices,
    )


def public_billing_status() -> str:
    return "live" if get_billing_config().live_ready else "policy_only"


_PLAN_LIMITS: dict[str, dict] = {
    "free": {
        "free_models": True,
        "byok_model_calls": True,
        "managed_model_calls": False,
        "max_agent_steps": 0,
        "monthly_request_limit": 100,
    },
    "starter": {
        "free_models": True,
        "byok_model_calls": True,
        "managed_model_calls": False,
        "max_agent_steps": 0,
        "monthly_request_limit": 1000,
    },
    "pro": {
        "free_models": True,
        "byok_model_calls": True,
        "managed_model_calls": False,
        "max_agent_steps": 0,
        "monthly_request_limit": 5000,
    },
    "enterprise-usage": {
        "free_models": True,
        "byok_model_calls": True,
        "managed_model_calls": False,
        "max_agent_steps": 0,
        "monthly_request_limit": None,
    },
}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def routing_fee_policy() -> dict:
    """Return the public, versioned Theriad routing-fee contract."""
    return {
        "version": ROUTING_FEE_POLICY_VERSION,
        "currency": "USD",
        "per_million_tokens": PLATFORM_FEE_PER_1M_TOKENS,
        "applies_to": "authenticated_paid_calls",
        "guest_demo": "exempt",
        "managed_model_allowance": False,
        "basis": "actual_prompt_plus_completion_tokens",
    }


def _billing_period(value: str | None = None) -> str:
    if value:
        return value[:7]
    return datetime.now(timezone.utc).strftime("%Y-%m")


def record_usage_event(
    *,
    organization_id: str | None,
    user_id: str | None,
    run_id: str | None,
    provider: str | None,
    model: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    provider_cost_usd: float | None,
    billable: bool,
    usage_known: bool = True,
    metadata: dict | None = None,
) -> dict:
    """Persist one tenant-owned usage event exactly once.

    This is intentionally a ledger write, not a Stripe call. Stripe reporting
    is an asynchronous aggregation step so a provider response can never be
    held hostage by billing-network latency and a retry cannot double charge a
    run.
    """
    now = _now()
    stable_run_id = run_id or f"unpersisted:{user_id or 'anonymous'}:{now}"
    idempotency_key = f"routing-fee:{stable_run_id}:v1"
    input_value = int(input_tokens) if input_tokens is not None else None
    output_value = int(output_tokens) if output_tokens is not None else None
    known = bool(usage_known and input_value is not None and output_value is not None)
    fee = platform_fee(input_value, output_value) if billable and known else None
    status = "ready" if billable and known else "exempt" if not billable else "pending"
    total_billed = (
        round(float(provider_cost_usd or 0.0) + float(fee or 0.0), 8)
        if billable and known else 0.0
    )
    if _hosted_usage_enabled():
        if not organization_id or not user_id:
            return {
                "idempotency_key": idempotency_key,
                "status": "exempt",
                "routing_fee_usd": 0.0,
                "total_billed_usd": 0.0,
            }
        from theriad.storage.hosted import hosted_repositories

        repos = hosted_repositories()
        existing = repos["usage_events"].client.list_rows(
            "usage_events", filters={"idempotency_key": f"eq.{idempotency_key}"}, limit=1
        )
        if existing:
            return existing[0]
        return repos["usage_events"].insert_owned(
            {
                "id": str(uuid.uuid4()),
                "run_id": run_id,
                "provider": provider,
                "model": model,
                "input_tokens": input_value,
                "output_tokens": output_value,
                "provider_cost_usd": provider_cost_usd,
                "routing_fee_usd": fee,
                "total_billed_usd": total_billed,
                "status": status,
                "billing_period": _billing_period(now),
                "metadata": metadata or {},
                "created_at": now,
            },
            organization_id=organization_id,
            user_id=user_id,
        )
    db = get_db()
    db.execute(
        """INSERT INTO usage_events (
             id, idempotency_key, organization_id, user_id, run_id,
             provider, model, input_tokens, output_tokens, provider_cost_usd,
             routing_fee_usd, total_billed_usd, status, billing_period,
             metadata_json, created_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(idempotency_key) DO NOTHING""",
        (
            str(uuid.uuid4()), idempotency_key, organization_id, user_id, run_id,
            provider, model, input_value, output_value, provider_cost_usd,
            fee, total_billed, status, _billing_period(now),
            json.dumps(metadata or {}, sort_keys=True), now,
        ),
    )
    db.commit()
    row = db.execute(
        "SELECT * FROM usage_events WHERE idempotency_key = ?", (idempotency_key,)
    ).fetchone()
    return dict(row) if row else {
        "idempotency_key": idempotency_key,
        "status": status,
        "routing_fee_usd": fee,
    }


def usage_summary_for(*, organization_id: str | None, user_id: str | None) -> dict:
    """Return provider cost plus Theriad fee without exposing provider secrets."""
    if not user_id:
        return {
            "has_data": False,
            "total_runs": 0,
            "total_tokens": 0,
            "provider_cost_usd": 0.0,
            "routing_fee_usd": 0.0,
            "total_billed_usd": 0.0,
            "pending_reconciliation": 0,
            "billing_period": _billing_period(),
            "routing_fee": routing_fee_policy(),
        }
    if _hosted_usage_enabled():
        return _hosted_usage_summary(organization_id=organization_id, user_id=user_id)
    db = get_db()
    run = db.execute(
        """SELECT COUNT(*) AS runs,
                  COALESCE(SUM(input_tokens), 0) AS input_tokens,
                  COALESCE(SUM(output_tokens), 0) AS output_tokens,
                  COALESCE(SUM(actual_cost_usd), 0.0) AS provider_cost
           FROM runs WHERE user_id = ? AND status = 'success'""",
        (user_id,),
    ).fetchone()
    event = db.execute(
        """SELECT COALESCE(SUM(routing_fee_usd), 0.0) AS fee,
                  COALESCE(SUM(total_billed_usd), 0.0) AS billed,
                  SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending
           FROM usage_events WHERE user_id = ?""",
        (user_id,),
    ).fetchone()
    input_tokens = int(run["input_tokens"] or 0)
    output_tokens = int(run["output_tokens"] or 0)
    return {
        "has_data": int(run["runs"] or 0) > 0,
        "total_runs": int(run["runs"] or 0),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider_cost_usd": round(float(run["provider_cost"] or 0.0), 8),
        "routing_fee_usd": round(float(event["fee"] or 0.0), 8),
        "total_billed_usd": round(float(event["billed"] or 0.0), 8),
        "pending_reconciliation": int(event["pending"] or 0),
        "billing_period": _billing_period(),
        "routing_fee": routing_fee_policy(),
        "organization_id": organization_id,
    }


def _hosted_usage_enabled() -> bool:
    try:
        from theriad.storage.hosted import hosted_data_plane_enabled

        return hosted_data_plane_enabled()
    except Exception:
        return False


def _hosted_usage_summary(*, organization_id: str | None, user_id: str) -> dict:
    from theriad.storage.hosted import HostedStorageUnavailable, hosted_repositories

    if not organization_id:
        return {
            "has_data": False,
            "total_runs": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "provider_cost_usd": 0.0,
            "routing_fee_usd": 0.0,
            "total_billed_usd": 0.0,
            "pending_reconciliation": 0,
            "billing_period": _billing_period(),
            "routing_fee": routing_fee_policy(),
            "organization_id": organization_id,
        }
    try:
        repos = hosted_repositories()
        period = _billing_period()
        runs = repos["runs"].list_for_owner(
            organization_id=organization_id, user_id=user_id, limit=1000
        )
        runs = [row for row in runs if row.get("status") == "success" and str(row.get("created_at") or "")[:7] == period]
        events = repos["usage_events"].list_for_owner(
            organization_id=organization_id, user_id=user_id, limit=1000
        )
        events = [row for row in events if str(row.get("billing_period") or "") == period]
    except HostedStorageUnavailable:
        raise
    input_tokens = sum(int(row.get("input_tokens") or 0) for row in runs)
    output_tokens = sum(int(row.get("output_tokens") or 0) for row in runs)
    provider_cost = sum(float(row.get("actual_cost_usd") or 0.0) for row in runs)
    fee = sum(float(row.get("routing_fee_usd") or 0.0) for row in events)
    billed = sum(float(row.get("total_billed_usd") or 0.0) for row in events)
    return {
        "has_data": bool(runs),
        "total_runs": len(runs),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": input_tokens + output_tokens,
        "provider_cost_usd": round(provider_cost, 8),
        "routing_fee_usd": round(fee, 8),
        "total_billed_usd": round(billed, 8),
        "pending_reconciliation": sum(1 for row in events if row.get("status") == "pending"),
        "billing_period": period,
        "routing_fee": routing_fee_policy(),
        "organization_id": organization_id,
    }


def reconcile_usage_event(
    *,
    run_id: str,
    input_tokens: int,
    output_tokens: int,
    provider_cost_usd: float | None,
) -> dict | None:
    """Finalize a pending event once the provider returns actual usage."""
    if _hosted_usage_enabled():
        from theriad.storage.hosted import hosted_repositories

        from theriad.api.auth_context import current_auth_context

        context = current_auth_context.get()
        if not context or context.principal_type != "registered" or not context.organization_id or not context.user_id:
            return None
        repos = hosted_repositories()
        rows = repos["usage_events"].client.list_rows(
            "usage_events",
            filters={
                "run_id": f"eq.{run_id}",
                "status": "eq.pending",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            limit=1,
        )
        if not rows:
            return None
        fee = platform_fee(int(input_tokens), int(output_tokens))
        total = round(float(provider_cost_usd or 0.0) + fee, 8)
        updated = repos["usage_events"].client.update_rows(
            "usage_events",
            filters={"id": f"eq.{rows[0]['id']}", "status": "eq.pending"},
            values={
                "input_tokens": int(input_tokens),
                "output_tokens": int(output_tokens),
                "provider_cost_usd": provider_cost_usd,
                "routing_fee_usd": fee,
                "total_billed_usd": total,
                "status": "ready",
            },
        )
        return updated[0] if updated else {**rows[0], "status": "ready", "routing_fee_usd": fee, "total_billed_usd": total}
    db = get_db()
    row = db.execute(
        "SELECT * FROM usage_events WHERE run_id = ? AND status = 'pending' ORDER BY created_at DESC LIMIT 1",
        (run_id,),
    ).fetchone()
    if row is None:
        return None
    fee = platform_fee(int(input_tokens), int(output_tokens))
    total = round(float(provider_cost_usd or 0.0) + fee, 8)
    db.execute(
        """UPDATE usage_events
           SET input_tokens = ?, output_tokens = ?, provider_cost_usd = ?,
               routing_fee_usd = ?, total_billed_usd = ?, status = 'ready'
           WHERE id = ? AND status = 'pending'""",
        (int(input_tokens), int(output_tokens), provider_cost_usd, fee, total, row["id"]),
    )
    db.commit()
    refreshed = db.execute("SELECT * FROM usage_events WHERE id = ?", (row["id"],)).fetchone()
    return dict(refreshed) if refreshed else None


async def report_routing_usage(period: str | None = None, *, limit: int = 100) -> dict:
    """Aggregate ready routing fees into one Stripe invoice item per org.

    The worker is deliberately separate from request handling. Sub-cent
    balances remain in ``ready`` state until a period has at least one cent,
    so rounding never silently discards token-level usage.
    """
    config = get_billing_config()
    if not config.live_ready:
        raise BillingUnavailable("Stripe checkout, prices, and webhook signing must be live before usage reporting.")
    period = _billing_period(period)
    if _hosted_usage_enabled():
        from theriad.storage.hosted import hosted_repositories

        repos = hosted_repositories()
        rows = repos["usage_events"].client.list_rows(
            "usage_events",
            filters={"billing_period": f"eq.{period}", "status": "eq.ready"},
            select="id,organization_id,routing_fee_usd,status,stripe_invoice_item_id",
            limit=max(1, min(limit * 100, 5000)),
        )
        grouped: dict[str, list[dict]] = {}
        for row in rows:
            if row.get("organization_id") and not row.get("stripe_invoice_item_id"):
                grouped.setdefault(row["organization_id"], []).append(row)
        reported = []
        deferred = []
        for organization_id, events in list(grouped.items())[: max(1, min(limit, 500))]:
            fee = sum(float(event.get("routing_fee_usd") or 0.0) for event in events)
            amount_cents = int(round(fee * 100))
            if amount_cents < 1:
                deferred.append({"organization_id": organization_id, "fee_usd": fee, "events": len(events)})
                continue
            customer = repos["billing_customers"].client.list_rows(
                "billing_customers", filters={"organization_id": f"eq.{organization_id}"}, limit=1
            )
            if not customer:
                deferred.append({"organization_id": organization_id, "reason": "stripe_customer_missing"})
                continue
            payload = await _stripe_post(
                "/invoiceitems",
                {
                    "customer": customer[0]["stripe_customer_id"],
                    "amount": str(amount_cents),
                    "currency": "usd",
                    "description": f"Theriad routing usage for {period}",
                    "metadata[organization_id]": organization_id,
                    "metadata[billing_period]": period,
                },
                idempotency_key=f"routing-usage:{organization_id}:{period}",
            )
            invoice_item_id = str(payload.get("id") or "").strip()
            if not invoice_item_id:
                raise BillingUnavailable("Stripe returned no invoice item id for routing usage.")
            repos["usage_events"].client.update_rows(
                "usage_events",
                filters={
                    "organization_id": f"eq.{organization_id}",
                    "billing_period": f"eq.{period}",
                    "status": "eq.ready",
                    "stripe_invoice_item_id": "is.null",
                },
                values={"status": "reported", "stripe_invoice_item_id": invoice_item_id, "reported_at": _now()},
            )
            reported.append({"organization_id": organization_id, "invoice_item_id": invoice_item_id, "amount_cents": amount_cents})
        return {"period": period, "reported": reported, "deferred": deferred}
    db = get_db()
    rows = db.execute(
        """SELECT organization_id, SUM(routing_fee_usd) AS fee,
                  COUNT(*) AS events
           FROM usage_events
           WHERE billing_period = ? AND status = 'ready'
             AND stripe_invoice_item_id IS NULL
             AND organization_id IS NOT NULL
           GROUP BY organization_id ORDER BY organization_id LIMIT ?""",
        (period, max(1, min(limit, 500))),
    ).fetchall()
    reported = []
    deferred = []
    for row in rows:
        organization_id = row["organization_id"]
        fee = float(row["fee"] or 0.0)
        amount_cents = int(round(fee * 100))
        if amount_cents < 1:
            deferred.append({"organization_id": organization_id, "fee_usd": fee, "events": int(row["events"])})
            continue
        customer = db.execute(
            "SELECT stripe_customer_id FROM billing_customers WHERE organization_id = ?",
            (organization_id,),
        ).fetchone()
        if not customer:
            deferred.append({"organization_id": organization_id, "reason": "stripe_customer_missing"})
            continue
        payload = await _stripe_post(
            "/invoiceitems",
            {
                "customer": customer["stripe_customer_id"],
                "amount": str(amount_cents),
                "currency": "usd",
                "description": f"Theriad routing usage for {period}",
                "metadata[organization_id]": organization_id,
                "metadata[billing_period]": period,
            },
            idempotency_key=f"routing-usage:{organization_id}:{period}",
        )
        invoice_item_id = str(payload.get("id") or "").strip()
        if not invoice_item_id:
            raise BillingUnavailable("Stripe returned no invoice item id for routing usage.")
        db.execute(
            """UPDATE usage_events SET status = 'reported', stripe_invoice_item_id = ?, reported_at = ?
               WHERE organization_id = ? AND billing_period = ? AND status = 'ready'
                 AND stripe_invoice_item_id IS NULL""",
            (invoice_item_id, _now(), organization_id, period),
        )
        db.commit()
        reported.append({"organization_id": organization_id, "invoice_item_id": invoice_item_id, "amount_cents": amount_cents})
    return {"period": period, "reported": reported, "deferred": deferred}


def _active_subscription(organization_id: str) -> dict | None:
    if _hosted_usage_enabled():
        from theriad.storage.hosted import hosted_repositories

        rows = hosted_repositories()["billing_subscriptions"].client.list_rows(
            "billing_subscriptions",
            filters={
                "organization_id": f"eq.{organization_id}",
                "status": "in.(active,trialing,past_due)",
            },
            order="updated_at.desc",
            limit=1,
        )
        return rows[0] if rows else None
    row = get_db().execute(
        """SELECT * FROM billing_subscriptions
           WHERE organization_id = ? AND status IN ('active', 'trialing', 'past_due')
           ORDER BY updated_at DESC LIMIT 1""",
        (organization_id,),
    ).fetchone()
    return dict(row) if row else None


def entitlements_for(
    *, organization_id: str | None, user_id: str | None, guest: bool = False
) -> dict:
    """Return the server-side access decision without exposing credentials."""
    if guest or not organization_id:
        plan_id = "free"
        status = "guest" if guest else "anonymous"
        subscription = None
    else:
        if _hosted_usage_enabled():
            from theriad.storage.hosted import hosted_repositories

            rows = hosted_repositories()["entitlements"].client.list_rows(
                "organization_entitlements",
                filters={"organization_id": f"eq.{organization_id}"},
                limit=1,
            )
            row = rows[0] if rows else None
        else:
            db = get_db()
            local_row = db.execute(
                "SELECT plan_id, status FROM organization_entitlements WHERE organization_id = ?",
                (organization_id,),
            ).fetchone()
            row = dict(local_row) if local_row else None
        subscription = _active_subscription(organization_id)
        plan_id = subscription["plan_id"] if subscription else (row["plan_id"] if row else "free")
        status = subscription["status"] if subscription else (row["status"] if row else "active")

    limits = dict(_PLAN_LIMITS.get(plan_id, _PLAN_LIMITS["free"]))
    return {
        "plan_id": plan_id,
        "status": status,
        "billing_status": public_billing_status(),
        "free_models_allowed": bool(limits["free_models"]),
        "byok_model_calls_allowed": bool(limits["byok_model_calls"] and not guest),
        "managed_model_calls_allowed": bool(limits["managed_model_calls"]),
        "agent_execution_allowed": False,
        "routing_fee": routing_fee_policy(),
        "agent_execution_reason": (
            "Hosted real terminal and browser execution is disabled until an OS/container sandbox, "
            "tenant-isolated secrets, budgets, approvals, and append-only audit path are deployed."
        ),
        "limits": limits,
        "subscription": _safe_subscription(subscription),
        "user_id": user_id,
        "organization_id": organization_id,
    }


def _safe_subscription(subscription: dict | None) -> dict | None:
    if not subscription:
        return None
    return {
        "plan_id": subscription.get("plan_id"),
        "interval": subscription.get("interval"),
        "status": subscription.get("status"),
        "current_period_end": subscription.get("current_period_end"),
        "cancel_at_period_end": bool(subscription.get("cancel_at_period_end")),
    }


def request_limit_error(
    entitlements: dict, *, is_free_request: bool, provider_key_configured: bool
) -> dict | None:
    """Return a structured denial, or ``None`` when the request may proceed."""
    if (
        not is_free_request
        and entitlements.get("status") in {"past_due", "unpaid", "canceled", "incomplete_expired"}
    ):
        return {
            "code": "billing_action_required",
            "message": "Update the workspace billing method before running paid work.",
        }
    monthly_limit = entitlements["limits"].get("monthly_request_limit")
    user_id = entitlements.get("user_id")
    if monthly_limit is not None and user_id:
        month_start = datetime.now(timezone.utc).replace(
            day=1, hour=0, minute=0, second=0, microsecond=0
        ).isoformat()
        if _hosted_usage_enabled():
            organization_id = entitlements.get("organization_id")
            from theriad.storage.hosted import hosted_repositories

            rows = hosted_repositories()["runs"].list_for_owner(
                organization_id=organization_id, user_id=user_id, limit=1000
            )
            count = sum(
                1 for row in rows
                if row.get("status") == "success" and str(row.get("created_at") or "") >= month_start
            )
        else:
            row = get_db().execute(
                """SELECT COUNT(*) AS count FROM runs
                   WHERE user_id = ? AND status = 'success' AND created_at >= ?""",
                (user_id, month_start),
            ).fetchone()
            count = int(row["count"])
        if count >= int(monthly_limit):
            return {
                "code": "monthly_request_limit_reached",
                "message": "This account has reached its monthly request allowance.",
            }
    if is_free_request:
        if not entitlements["free_models_allowed"]:
            return {
                "code": "free_models_not_allowed",
                "message": "Free-model access is not available for this account.",
            }
        return None
    if not entitlements["byok_model_calls_allowed"]:
        return {
            "code": "paid_execution_requires_account",
            "message": "Create a registered account before using paid or provider-direct models.",
        }
    if not provider_key_configured:
        return {
            "code": "provider_key_required",
            "message": "Connect the selected provider key in Account before executing this paid request.",
        }
    return None


def billing_status_for(organization_id: str | None) -> dict:
    customer = None
    subscription = None
    if organization_id:
        if _hosted_usage_enabled():
            from theriad.storage.hosted import hosted_repositories

            rows = hosted_repositories()["billing_customers"].client.list_rows(
                "billing_customers", filters={"organization_id": f"eq.{organization_id}"}, limit=1
            )
            customer = bool(rows)
        else:
            row = get_db().execute(
                "SELECT stripe_customer_id FROM billing_customers WHERE organization_id = ?",
                (organization_id,),
            ).fetchone()
            customer = bool(row)
        subscription = _safe_subscription(_active_subscription(organization_id))
    config = get_billing_config()
    return {
        "billing_status": public_billing_status(),
        "checkout_available": config.live_ready,
        "portal_available": bool(config.live_ready and customer),
        "webhook_configured": config.webhook_ready,
        "has_stripe_customer": customer,
        "subscription": subscription,
        "reason": (
            None
            if config.live_ready
            else "Stripe checkout, all plan prices, and signed webhooks must be configured before billing is live."
        ),
    }


async def create_checkout_session(
    *, organization_id: str, user_id: str, email: str | None, plan_id: str, interval: str
) -> dict:
    config = get_billing_config()
    price_id = config.prices.get((plan_id, interval))
    if not config.live_ready or not price_id:
        raise BillingUnavailable("Stripe checkout is not configured for this plan.")
    data = {
        "mode": "subscription",
        "line_items[0][price]": price_id,
        "line_items[0][quantity]": "1",
        "success_url": config.success_url,
        "cancel_url": config.cancel_url,
        "client_reference_id": user_id,
        "metadata[organization_id]": organization_id,
        "metadata[plan_id]": plan_id,
        "metadata[interval]": interval,
    }
    if email:
        data["customer_email"] = email
    payload = await _stripe_post(
        "/checkout/sessions",
        data,
        idempotency_key=f"checkout:{organization_id}:{plan_id}:{interval}",
    )
    return {"id": payload.get("id"), "url": payload.get("url"), "billing_status": "live"}


async def create_portal_session(*, organization_id: str) -> dict:
    config = get_billing_config()
    if _hosted_usage_enabled():
        from theriad.storage.hosted import hosted_repositories

        rows = hosted_repositories()["billing_customers"].client.list_rows(
            "billing_customers", filters={"organization_id": f"eq.{organization_id}"}, limit=1
        )
        customer_id = rows[0].get("stripe_customer_id") if rows else None
    else:
        row = get_db().execute(
            "SELECT stripe_customer_id FROM billing_customers WHERE organization_id = ?",
            (organization_id,),
        ).fetchone()
        customer_id = row["stripe_customer_id"] if row else None
    if not config.live_ready or not customer_id:
        raise BillingUnavailable("A Stripe customer is not configured for this account.")
    payload = await _stripe_post(
        "/billing_portal/sessions",
        {"customer": customer_id, "return_url": config.success_url},
        idempotency_key=f"portal:{organization_id}:{int(time.time() // 60)}",
    )
    return {"id": payload.get("id"), "url": payload.get("url"), "billing_status": "live"}


class BillingUnavailable(RuntimeError):
    pass


async def _stripe_post(path: str, data: dict[str, str], *, idempotency_key: str) -> dict:
    config = get_billing_config()
    if not config.secret_key:
        raise BillingUnavailable("Stripe is not configured.")
    headers = {
        "Authorization": f"Bearer {config.secret_key}",
        "Idempotency-Key": idempotency_key,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            response = await client.post(f"{STRIPE_API_BASE}{path}", data=data, headers=headers)
        if response.status_code >= 400:
            raise BillingUnavailable("Stripe rejected the billing request.")
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise BillingUnavailable("Stripe billing is temporarily unavailable.") from exc
    if not isinstance(payload, dict):
        raise BillingUnavailable("Stripe returned an invalid billing response.")
    return payload


def verify_webhook_signature(payload: bytes, signature: str | None) -> bool:
    secret = get_billing_config().webhook_secret
    if not secret or not signature:
        return False
    timestamp = None
    signatures: list[str] = []
    for component in signature.split(","):
        key, _, value = component.strip().partition("=")
        if key == "t":
            timestamp = value
        elif key == "v1" and value:
            signatures.append(value)
    try:
        timestamp_int = int(timestamp or "0")
    except ValueError:
        return False
    if abs(time.time() - timestamp_int) > WEBHOOK_TOLERANCE_SECONDS:
        return False
    signed = f"{timestamp_int}.".encode() + payload
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    return any(hmac.compare_digest(expected, candidate) for candidate in signatures)


def apply_webhook_event(event: dict) -> tuple[bool, str]:
    """Project one verified Stripe event exactly once."""
    event_id = str(event.get("id") or "").strip()
    event_type = str(event.get("type") or "").strip()
    if not event_id or not event_type:
        raise ValueError("Stripe event is missing id or type")
    raw = json.dumps(event, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(raw).hexdigest()
    if _hosted_usage_enabled():
        return _apply_hosted_webhook_event(
            event_id=event_id, event_type=event_type, digest=digest, event=event
        )
    db = get_db()
    existing = db.execute(
        "SELECT processed_at FROM billing_events WHERE event_id = ?", (event_id,)
    ).fetchone()
    if existing and existing["processed_at"]:
        return False, "duplicate"
    if not existing:
        db.execute(
            "INSERT INTO billing_events (event_id, event_type, payload_sha256, received_at) "
            "VALUES (?, ?, ?, ?)",
            (event_id, event_type, digest, _now()),
        )

    obj = ((event.get("data") or {}).get("object") or {})
    if event_type == "checkout.session.completed":
        _apply_checkout(obj)
    elif event_type.startswith("customer.subscription."):
        _apply_subscription(obj)
    elif event_type == "invoice.paid":
        _mark_invoice(obj, "active")
    elif event_type == "invoice.payment_failed":
        _mark_invoice(obj, "past_due")

    db.execute(
        "UPDATE billing_events SET processed_at = ? WHERE event_id = ?",
        (_now(), event_id),
    )
    db.commit()
    return True, "processed"


def _apply_hosted_webhook_event(*, event_id: str, event_type: str, digest: str, event: dict) -> tuple[bool, str]:
    from theriad.storage.hosted import hosted_repositories

    repos = hosted_repositories()
    client = repos["billing_subscriptions"].client
    existing = client.list_rows("billing_events", filters={"event_id": f"eq.{event_id}"}, limit=1)
    if existing and existing[0].get("processed_at"):
        return False, "duplicate"
    if not existing:
        client.insert_row(
            "billing_events",
            {
                "event_id": event_id,
                "event_type": event_type,
                "payload_sha256": digest,
                "received_at": _now(),
            },
        )
    obj = ((event.get("data") or {}).get("object") or {})
    metadata = obj.get("metadata") or {}
    organization_id = str(metadata.get("organization_id") or "")
    if event_type == "checkout.session.completed" and organization_id:
        customer_id = str(obj.get("customer") or "").strip()
        if customer_id:
            client.upsert_row(
                "billing_customers",
                {
                    "organization_id": organization_id,
                    "stripe_customer_id": customer_id,
                    "updated_at": _now(),
                },
                conflict="organization_id",
            )
        subscription_id = str(obj.get("subscription") or "").strip()
        if subscription_id:
            subscription = client.upsert_row(
                "billing_subscriptions",
                {
                    "organization_id": organization_id,
                    "stripe_subscription_id": subscription_id,
                    "stripe_customer_id": customer_id or None,
                    "plan_id": str(metadata.get("plan_id") or "free"),
                    "interval": str(metadata.get("interval") or "monthly"),
                    "status": "active",
                    "updated_at": _now(),
                },
                conflict="stripe_subscription_id",
            )
            client.upsert_row(
                "organization_entitlements",
                {
                    "organization_id": organization_id,
                    "plan_id": subscription.get("plan_id") or str(metadata.get("plan_id") or "free"),
                    "status": subscription.get("status") or "active",
                    "updated_at": _now(),
                },
                conflict="organization_id",
            )
    elif event_type.startswith("customer.subscription."):
        subscription_id = str(obj.get("id") or "").strip()
        if subscription_id:
            if not organization_id:
                matches = client.list_rows(
                    "billing_subscriptions",
                    filters={"stripe_subscription_id": f"eq.{subscription_id}"},
                    limit=1,
                )
                organization_id = str(matches[0].get("organization_id") or "") if matches else ""
            if organization_id:
                items = ((obj.get("items") or {}).get("data") or [])
                price_id = (((items[0] if items else {}).get("price") or {}).get("id"))
                subscription = client.upsert_row(
                    "billing_subscriptions",
                    {
                        "organization_id": organization_id,
                        "stripe_subscription_id": subscription_id,
                        "stripe_customer_id": str(obj.get("customer") or "") or None,
                        "plan_id": str(metadata.get("plan_id") or _plan_for_price(price_id) or "free"),
                        "interval": str((((items[0] if items else {}).get("price") or {}).get("recurring") or {}).get("interval") or "monthly"),
                        "status": str(obj.get("status") or "active"),
                        "current_period_end": _timestamp_from_epoch(obj.get("current_period_end")),
                        "cancel_at_period_end": bool(obj.get("cancel_at_period_end")),
                        "updated_at": _now(),
                    },
                    conflict="stripe_subscription_id",
                )
                client.upsert_row(
                    "organization_entitlements",
                    {
                        "organization_id": organization_id,
                        "plan_id": subscription.get("plan_id") or str(metadata.get("plan_id") or "free"),
                        "status": subscription.get("status") or str(obj.get("status") or "active"),
                        "updated_at": _now(),
                    },
                    conflict="organization_id",
                )
    elif event_type in {"invoice.paid", "invoice.payment_failed"}:
        subscription_id = str(obj.get("subscription") or "").strip()
        if subscription_id:
            client.update_rows(
                "billing_subscriptions",
                filters={"stripe_subscription_id": f"eq.{subscription_id}"},
                values={"status": "active" if event_type == "invoice.paid" else "past_due", "updated_at": _now()},
            )
            subscription_rows = client.list_rows(
                "billing_subscriptions",
                filters={"stripe_subscription_id": f"eq.{subscription_id}"},
                limit=1,
            )
            if subscription_rows:
                client.upsert_row(
                    "organization_entitlements",
                    {
                        "organization_id": subscription_rows[0]["organization_id"],
                        "plan_id": subscription_rows[0].get("plan_id") or "free",
                        "status": "active" if event_type == "invoice.paid" else "past_due",
                        "updated_at": _now(),
                    },
                    conflict="organization_id",
                )
    client.update_rows(
        "billing_events",
        filters={"event_id": f"eq.{event_id}"},
        values={"processed_at": _now()},
    )
    return True, "processed"


def _apply_checkout(obj: dict) -> None:
    metadata = obj.get("metadata") or {}
    organization_id = str(metadata.get("organization_id") or "")
    if not organization_id:
        return
    customer_id = str(obj.get("customer") or "").strip()
    if customer_id:
        _upsert_customer(organization_id, customer_id)
    subscription_id = obj.get("subscription")
    if subscription_id:
        _upsert_subscription(
            organization_id=organization_id,
            subscription_id=str(subscription_id),
            customer_id=customer_id,
            plan_id=str(metadata.get("plan_id") or "free"),
            interval=str(metadata.get("interval") or "monthly"),
            status="active",
            current_period_end=None,
            cancel_at_period_end=False,
        )


def _apply_subscription(obj: dict) -> None:
    subscription_id = str(obj.get("id") or "").strip()
    if not subscription_id:
        return
    metadata = obj.get("metadata") or {}
    organization_id = str(metadata.get("organization_id") or "")
    db = get_db()
    if not organization_id:
        row = db.execute(
            "SELECT organization_id FROM billing_subscriptions WHERE stripe_subscription_id = ?",
            (subscription_id,),
        ).fetchone()
        organization_id = row["organization_id"] if row else ""
    if not organization_id:
        return
    items = ((obj.get("items") or {}).get("data") or [])
    price_id = (((items[0] if items else {}).get("price") or {}).get("id"))
    plan_id = str(metadata.get("plan_id") or _plan_for_price(price_id) or "free")
    interval = str((((items[0] if items else {}).get("price") or {}).get("recurring") or {}).get("interval") or "monthly")
    _upsert_subscription(
        organization_id=organization_id,
        subscription_id=subscription_id,
        customer_id=str(obj.get("customer") or "") or None,
        plan_id=plan_id,
        interval=interval,
        status=str(obj.get("status") or "active"),
        current_period_end=_timestamp_from_epoch(obj.get("current_period_end")),
        cancel_at_period_end=bool(obj.get("cancel_at_period_end")),
    )


def _mark_invoice(obj: dict, status: str) -> None:
    subscription_id = str(obj.get("subscription") or "").strip()
    if not subscription_id:
        return
    get_db().execute(
        "UPDATE billing_subscriptions SET status = ?, updated_at = ? WHERE stripe_subscription_id = ?",
        (status, _now(), subscription_id),
    )


def _upsert_customer(organization_id: str, customer_id: str) -> None:
    db = get_db()
    now = _now()
    db.execute(
        """INSERT INTO billing_customers (organization_id, stripe_customer_id, created_at, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(organization_id) DO UPDATE SET
             stripe_customer_id = excluded.stripe_customer_id, updated_at = excluded.updated_at""",
        (organization_id, customer_id, now, now),
    )


def _upsert_subscription(
    *, organization_id: str, subscription_id: str, customer_id: str | None,
    plan_id: str, interval: str, status: str, current_period_end: str | None,
    cancel_at_period_end: bool,
) -> None:
    db = get_db()
    now = _now()
    db.execute(
        """INSERT INTO billing_subscriptions (
             id, organization_id, stripe_subscription_id, stripe_customer_id,
             plan_id, interval, status, current_period_end,
             cancel_at_period_end, created_at, updated_at
           ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(stripe_subscription_id) DO UPDATE SET
             organization_id = excluded.organization_id,
             stripe_customer_id = excluded.stripe_customer_id,
             plan_id = excluded.plan_id, interval = excluded.interval,
             status = excluded.status, current_period_end = excluded.current_period_end,
             cancel_at_period_end = excluded.cancel_at_period_end,
             updated_at = excluded.updated_at""",
        (
            str(uuid.uuid4()), organization_id, subscription_id, customer_id,
            plan_id, interval, status, current_period_end,
            int(cancel_at_period_end), now, now,
        ),
    )
    db.execute(
        """INSERT INTO organization_entitlements (organization_id, plan_id, status, updated_at)
           VALUES (?, ?, ?, ?)
           ON CONFLICT(organization_id) DO UPDATE SET
             plan_id = excluded.plan_id, status = excluded.status, updated_at = excluded.updated_at""",
        (organization_id, plan_id, status, now),
    )


def _plan_for_price(price_id: str | None) -> str | None:
    if not price_id:
        return None
    for key, value in get_billing_config().prices.items():
        if value == price_id:
            return key[0]
    return None


def _timestamp_from_epoch(value) -> str | None:
    try:
        return datetime.fromtimestamp(int(value), tz=timezone.utc).isoformat()
    except (TypeError, ValueError, OSError, OverflowError):
        return None
