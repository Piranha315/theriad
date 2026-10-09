"""BYOK usage-ledger contracts."""
from __future__ import annotations

from theriad.billing.service import (
    entitlements_for,
    record_usage_event,
    reconcile_usage_event,
    request_limit_error,
    usage_summary_for,
)
from theriad.storage import database as db_module


def _db(tmp_path, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "local")
    monkeypatch.setenv("THERIAD_STORAGE_BACKEND", "sqlite")
    db_module.close_db()
    db_module.configure(tmp_path / "routing-fee.db")
    return db_module.get_db()


def test_byok_usage_is_not_billable_and_is_idempotent(tmp_path, monkeypatch):
    db = _db(tmp_path, monkeypatch)
    event = record_usage_event(
        organization_id="org-1",
        user_id="user-1",
        run_id="run-1",
        provider="openrouter",
        model="model-1",
        input_tokens=1_000_000,
        output_tokens=1_000_000,
        provider_cost_usd=0.12,
        billable=True,
    )

    assert event["status"] == "exempt"
    assert event["routing_fee_usd"] is None
    assert event["total_billed_usd"] == 0.0

    duplicate = record_usage_event(
        organization_id="org-1", user_id="user-1", run_id="run-1",
        provider="openrouter", model="model-1", input_tokens=1,
        output_tokens=1, provider_cost_usd=99.0, billable=True,
    )
    assert duplicate["id"] == event["id"]
    assert db.execute("SELECT COUNT(*) AS count FROM usage_events").fetchone()["count"] == 1


def test_missing_usage_stays_pending_until_reconciled(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    pending = record_usage_event(
        organization_id="org-1", user_id="user-1", run_id="run-2",
        provider="openrouter", model="model-1", input_tokens=None,
        output_tokens=None, provider_cost_usd=None, billable=True,
        usage_known=False,
    )
    assert pending["status"] == "exempt"
    assert pending["routing_fee_usd"] is None

    assert reconcile_usage_event(
        run_id="run-2", input_tokens=500_000, output_tokens=500_000, provider_cost_usd=0.03
    ) is None
    assert usage_summary_for(organization_id="org-1", user_id="user-1")["pending_reconciliation"] == 0


def test_local_guest_byok_is_not_treated_as_hosted_paid_execution():
    entitlements = entitlements_for(
        organization_id=None,
        user_id="local-guest",
        guest=True,
    )

    assert entitlements["byok_model_calls_allowed"] is True
    assert request_limit_error(
        entitlements,
        is_free_request=False,
        provider_key_configured=True,
    ) is None


def test_guest_traffic_is_exempt_and_past_due_denies_paid_work(tmp_path, monkeypatch):
    _db(tmp_path, monkeypatch)
    event = record_usage_event(
        organization_id=None, user_id="guest-1", run_id="run-guest",
        provider="theriad", model="free", input_tokens=10,
        output_tokens=10, provider_cost_usd=0.0, billable=False,
    )
    assert event["status"] == "exempt"
    assert event["routing_fee_usd"] is None
    assert request_limit_error(
        {"status": "past_due", "limits": {"monthly_request_limit": None}, "byok_model_calls_allowed": True},
        is_free_request=False,
        provider_key_configured=True,
    )["code"] == "billing_action_required"
