"""Email-based recovery flows for local Theriad accounts."""
from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient

import theriad.api.auth as auth_module
import theriad.api.server as server_module
from theriad.storage import database as db_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("THERIAD_SMTP_HOST", "smtp.test")
    monkeypatch.setenv("THERIAD_SMTP_PORT", "587")
    monkeypatch.setenv("THERIAD_SMTP_FROM", "accounts@test.invalid")
    monkeypatch.setenv("THERIAD_SMTP_USE_TLS", "true")
    db_module.close_db()
    db_module.configure(tmp_path / "auth-recovery.db")
    with TestClient(server_module.app) as value:
        yield value
    db_module.close_db()


def register_local_account(client, *, username="recovery-user", email="recovery@example.com"):
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "old password", "email": email},
    )
    assert response.status_code == 201
    return response.json()


def test_registration_records_recovery_email_without_exposing_a_key(client):
    registered = register_local_account(client)

    assert registered["user"]["email"] == "recovery@example.com"
    assert "recovery_key" not in registered


def test_email_recovery_resets_password_and_invalidates_old_sessions(client, monkeypatch):
    delivered = {}

    def capture_delivery(**kwargs):
        delivered.update(kwargs)

    monkeypatch.setattr(auth_module, "_send_recovery_email", capture_delivery)
    registered = register_local_account(client, username="reset-user", email="reset@example.com")
    old_token = registered["token"]

    requested = client.post("/api/auth/recovery/request", json={"identifier": "reset-user"})

    assert requested.status_code == 200
    assert "If that account has a recovery email" in requested.json()["message"]
    assert delivered["recipient"] == "reset@example.com"
    assert delivered["username"] == "reset-user"
    recovery_key = delivered["recovery_key"]
    assert re.fullmatch(r"THR(?:-[0-9A-F]{4}){8}", recovery_key)

    reset = client.post(
        "/api/auth/recovery/complete",
        json={"recovery_key": recovery_key, "password": "new password"},
    )

    assert reset.status_code == 200
    reset_body = reset.json()
    assert reset_body["user"]["username"] == "reset-user"
    assert reset_body["token"] != old_token
    assert "recovery_key" not in reset_body

    assert client.get(
        "/api/auth/me",
        headers={"Authorization": f"Bearer {old_token}"},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "reset-user", "password": "old password"},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"username": "reset-user", "password": "new password"},
    ).status_code == 200
    assert client.post(
        "/api/auth/recovery/complete",
        json={"recovery_key": recovery_key, "password": "another password"},
    ).status_code == 400


def test_recovery_request_does_not_disclose_unknown_or_unemailed_accounts(client, monkeypatch):
    sent = []
    monkeypatch.setattr(auth_module, "_send_recovery_email", lambda **kwargs: sent.append(kwargs))
    register_local_account(client, username="without-email", email="")

    unknown = client.post("/api/auth/recovery/request", json={"identifier": "not-an-account"})
    missing_email = client.post("/api/auth/recovery/request", json={"identifier": "without-email"})

    assert unknown.status_code == 200
    assert missing_email.status_code == 200
    assert unknown.json() == missing_email.json()
    assert sent == []


def test_failed_email_delivery_removes_the_unreceived_key(client, monkeypatch):
    def fail_delivery(**_kwargs):
        raise auth_module.RecoveryEmailDeliveryError("SMTP unavailable")

    monkeypatch.setattr(auth_module, "_send_recovery_email", fail_delivery)
    register_local_account(client, username="delivery-user", email="delivery@example.com")

    requested = client.post("/api/auth/recovery/request", json={"identifier": "delivery-user"})

    assert requested.status_code == 200
    db = db_module.get_db()
    assert db.execute("SELECT COUNT(*) FROM auth_recovery_tokens").fetchone()[0] == 0


def test_unconfigured_email_recovery_reports_server_state(client, monkeypatch):
    monkeypatch.delenv("THERIAD_SMTP_HOST")

    requested = client.post("/api/auth/recovery/request", json={"identifier": "any-account"})

    assert requested.status_code == 503
    assert requested.json()["detail"] == "Email recovery is not configured on this Theriad server."


def test_expired_email_verification_key_cannot_reset_a_password(client, monkeypatch):
    delivered = {}
    monkeypatch.setattr(auth_module, "_send_recovery_email", lambda **kwargs: delivered.update(kwargs))
    register_local_account(client, username="expired-user", email="expired@example.com")
    assert client.post("/api/auth/recovery/request", json={"identifier": "expired-user"}).status_code == 200

    db = db_module.get_db()
    db.execute("UPDATE auth_recovery_tokens SET expires_at = '2000-01-01T00:00:00+00:00'")
    db.commit()

    reset = client.post(
        "/api/auth/recovery/complete",
        json={"recovery_key": delivered["recovery_key"], "password": "new password"},
    )

    assert reset.status_code == 400


def test_signed_in_legacy_account_can_still_issue_a_legacy_key(client):
    registered = register_local_account(client, username="legacy-user", email="legacy@example.com")

    issued = client.post(
        "/api/auth/recovery/key",
        headers={"Authorization": f"Bearer {registered['token']}"},
    )

    assert issued.status_code == 200
    assert client.post(
        "/api/auth/recovery/username",
        json={"recovery_key": issued.json()["recovery_key"]},
    ).json() == {"username": "legacy-user"}
