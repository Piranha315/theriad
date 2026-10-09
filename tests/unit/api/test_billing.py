"""Billing boundary tests: policy-only behavior and signed webhook projection."""
from __future__ import annotations

import hashlib
import hmac
import json
import time

import pytest
from fastapi.testclient import TestClient

from theriad.api.server import app
from theriad.billing.service import get_billing_config
from theriad.storage import database as db_module


@pytest.fixture()
def billing_db(tmp_path, monkeypatch):
    db_module.close_db()
    db_module.configure(tmp_path / "billing.db")
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "local")
    for name in (
        "STRIPE_SECRET_KEY",
        "STRIPE_WEBHOOK_SECRET",
        "THERIAD_STRIPE_STARTER_MONTHLY_PRICE_ID",
        "THERIAD_STRIPE_STARTER_ANNUAL_PRICE_ID",
        "THERIAD_STRIPE_PRO_MONTHLY_PRICE_ID",
        "THERIAD_STRIPE_PRO_ANNUAL_PRICE_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    yield
    db_module.close_db()


def _registered_client() -> tuple[TestClient, str, dict]:
    client = TestClient(app)
    response = client.post(
        "/api/auth/register",
        json={"username": "billing-user", "password": "password123"},
    )
    assert response.status_code == 201
    payload = response.json()
    return client, payload["token"], payload["user"]


def test_subscription_billing_is_disabled_in_byok_mode(billing_db):
    client, token, _user = _registered_client()
    headers = {"Authorization": f"Bearer {token}"}

    status = client.get("/api/billing/status", headers=headers)
    assert status.status_code == 200
    assert status.json()["billing_status"] == "byok"
    assert status.json()["checkout_available"] is False

    entitlements = client.get("/api/entitlements", headers=headers)
    assert entitlements.status_code == 200
    assert entitlements.json()["byok_model_calls_allowed"] is True
    assert entitlements.json()["managed_model_calls_allowed"] is False

    checkout = client.post(
        "/api/billing/checkout",
        headers=headers,
        json={"plan_id": "starter", "interval": "monthly"},
    )
    assert checkout.status_code == 503
    assert checkout.json()["billing_status"] == "byok"
    assert "url" not in checkout.json()


def test_stripe_environment_cannot_enable_subscriptions_in_byok_mode(billing_db, monkeypatch):
    monkeypatch.setenv("STRIPE_SECRET_KEY", "sk_test_ignored")
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_ignored")
    monkeypatch.setenv("THERIAD_STRIPE_STARTER_MONTHLY_PRICE_ID", "price_ignored")
    monkeypatch.setenv("THERIAD_STRIPE_STARTER_ANNUAL_PRICE_ID", "price_ignored")
    monkeypatch.setenv("THERIAD_STRIPE_PRO_MONTHLY_PRICE_ID", "price_ignored")
    monkeypatch.setenv("THERIAD_STRIPE_PRO_ANNUAL_PRICE_ID", "price_ignored")

    config = get_billing_config()
    assert config.checkout_ready is False
    assert config.webhook_ready is False
    assert config.live_ready is False


def test_signed_webhook_is_projected_once(billing_db, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    client, _token, user = _registered_client()
    organization_id = user["organization_id"]
    event = {
        "id": "evt_checkout_test_1",
        "type": "checkout.session.completed",
        "data": {
            "object": {
                "customer": "cus_test_1",
                "subscription": "sub_test_1",
                "metadata": {
                    "organization_id": organization_id,
                    "plan_id": "starter",
                    "interval": "monthly",
                },
            }
        },
    }
    raw = json.dumps(event, separators=(",", ":")).encode()
    timestamp = str(int(time.time()))
    digest = hmac.new(
        b"whsec_test", f"{timestamp}.".encode() + raw, hashlib.sha256
    ).hexdigest()
    headers = {"Stripe-Signature": f"t={timestamp},v1={digest}"}

    first = client.post("/api/billing/webhook", content=raw, headers=headers)
    second = client.post("/api/billing/webhook", content=raw, headers=headers)
    assert first.status_code == 200
    assert first.json()["processed"] is True
    assert second.status_code == 200
    assert second.json()["status"] == "duplicate"

    db = db_module.get_db()
    customer = db.execute(
        "SELECT stripe_customer_id FROM billing_customers WHERE organization_id = ?",
        (organization_id,),
    ).fetchone()
    subscription = db.execute(
        "SELECT plan_id, status FROM billing_subscriptions WHERE organization_id = ?",
        (organization_id,),
    ).fetchone()
    assert customer["stripe_customer_id"] == "cus_test_1"
    assert dict(subscription) == {"plan_id": "starter", "status": "active"}


def test_historical_stripe_events_do_not_restrict_byok_work(billing_db, monkeypatch):
    monkeypatch.setenv("STRIPE_WEBHOOK_SECRET", "whsec_test")
    client, token, user = _registered_client()
    organization_id = user["organization_id"]
    checkout = {
        "id": "evt_checkout_test_2",
        "type": "checkout.session.completed",
        "data": {"object": {
            "customer": "cus_test_2", "subscription": "sub_test_2",
            "metadata": {"organization_id": organization_id, "plan_id": "starter", "interval": "monthly"},
        }},
    }

    def post_event(event):
        raw = json.dumps(event, separators=(",", ":")).encode()
        timestamp = str(int(time.time()))
        digest = hmac.new(b"whsec_test", f"{timestamp}.".encode() + raw, hashlib.sha256).hexdigest()
        return client.post(
            "/api/billing/webhook", content=raw,
            headers={"Stripe-Signature": f"t={timestamp},v1={digest}"},
        )

    assert post_event(checkout).status_code == 200
    failed = {
        "id": "evt_invoice_failed_2", "type": "invoice.payment_failed",
        "data": {"object": {"subscription": "sub_test_2"}},
    }
    assert post_event(failed).status_code == 200
    paid = client.post("/api/chat/send", headers={"Authorization": f"Bearer {token}"}, json={
        "message": "paid test", "mode": "execute", "allow_paid_calls": True,
        "cost_decision": "paid",
    })
    assert paid.json().get("error") != "billing_action_required"

    deleted = {
        "id": "evt_subscription_deleted_2", "type": "customer.subscription.deleted",
        "data": {"object": {"id": "sub_test_2", "status": "canceled", "metadata": {"organization_id": organization_id}}},
    }
    assert post_event(deleted).status_code == 200
    entitlement = db_module.get_db().execute(
        "SELECT status FROM organization_entitlements WHERE organization_id = ?", (organization_id,)
    ).fetchone()
    assert entitlement["status"] == "canceled"
