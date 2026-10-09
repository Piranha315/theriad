"""Anonymous site analytics collection and private owner summary tests."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import theriad.api.server as server_module
from theriad.storage import database as db_module


@pytest.fixture()
def client(tmp_path):
    db_module.close_db()
    db_module.configure(tmp_path / "analytics.db")
    with TestClient(server_module.app) as value:
        yield value
    db_module.close_db()


def _register(client, username: str):
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "correct horse battery"},
    )
    assert response.status_code == 201
    return response.json()["token"]


def test_collect_sets_anonymous_cookie_and_aggregates_sessions(client, monkeypatch):
    first = client.post(
        "/api/analytics/collect",
        json={
            "event": "page_view",
            "session_id": "session-001",
            "path": "#/app",
            "referrer": "https://search.example/results?email=private@example.com",
        },
    )
    assert first.status_code == 200
    assert "theriad_visitor_id=" in first.headers["set-cookie"]

    heartbeat = client.post(
        "/api/analytics/collect",
        json={"event": "heartbeat", "session_id": "session-001", "path": "#/app"},
    )
    assert heartbeat.status_code == 200

    page = client.post(
        "/api/analytics/collect",
        json={"event": "page_view", "session_id": "session-001", "path": "/pricing"},
    )
    assert page.status_code == 200

    monkeypatch.setenv("THERIAD_ANALYTICS_OWNER_USERNAME", "owner")
    token = _register(client, "owner")
    summary = client.get(
        "/api/analytics/summary",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert summary.status_code == 200
    body = summary.json()
    assert body["total_visitors"] == 1
    assert body["total_visits"] == 1
    assert body["total_page_views"] == 2
    assert body["top_pages"][0] == {"path": "/#/app", "views": 1}
    assert body["top_referrers"] == [{"referrer": "https://search.example", "visits": 1}]
    assert "email=private@example.com" not in str(body)


def test_summary_is_owner_only_and_fails_closed_when_unconfigured(client, monkeypatch):
    owner_token = _register(client, "owner")
    other_token = _register(client, "other")
    monkeypatch.setenv("THERIAD_ANALYTICS_OWNER_USERNAME", "")
    # "Unconfigured" means neither gate is set; a developer .env loaded by the
    # server import can supply the access code, so clear it explicitly.
    monkeypatch.delenv("THERIAD_ANALYTICS_ACCESS_CODE", raising=False)

    unconfigured = client.get(
        "/api/analytics/summary",
        headers={"Authorization": f"Bearer {owner_token}"},
    )
    assert unconfigured.status_code == 503

    monkeypatch.setenv("THERIAD_ANALYTICS_OWNER_USERNAME", "owner")
    forbidden = client.get(
        "/api/analytics/summary",
        headers={"Authorization": f"Bearer {other_token}"},
    )
    assert forbidden.status_code == 403


def test_registered_account_can_unlock_analytics_with_access_code(client, monkeypatch):
    monkeypatch.setenv("THERIAD_ANALYTICS_OWNER_USERNAME", "")
    monkeypatch.setenv("THERIAD_ANALYTICS_ACCESS_CODE", "site-analytics-recovery")
    token = _register(client, "recovery-user")
    headers = {"Authorization": f"Bearer {token}"}

    locked = client.get("/api/analytics/summary", headers=headers)
    assert locked.status_code == 403

    wrong = client.post(
        "/api/analytics/unlock",
        headers=headers,
        json={"passcode": "wrong-code"},
    )
    assert wrong.status_code == 403

    unlocked = client.post(
        "/api/analytics/unlock",
        headers=headers,
        json={"passcode": "site-analytics-recovery"},
    )
    assert unlocked.status_code == 200
    assert unlocked.json()["ok"] is True
    assert "expires_at" in unlocked.json()

    summary = client.get("/api/analytics/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["owner"] == "recovery-user"


def test_guests_cannot_unlock_analytics(client, monkeypatch):
    monkeypatch.setenv("THERIAD_ANALYTICS_ACCESS_CODE", "site-analytics-recovery")
    guest = client.post("/api/auth/guest").json()
    response = client.post(
        "/api/analytics/unlock",
        headers={"Authorization": f"Bearer {guest['token']}"},
        json={"passcode": "site-analytics-recovery"},
    )
    assert response.status_code == 403


def test_unlock_endpoint_is_private_in_hosted_mode(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setenv("THERIAD_ANALYTICS_ACCESS_CODE", "site-analytics-recovery")
    response = client.post(
        "/api/analytics/unlock",
        json={"passcode": "site-analytics-recovery"},
    )
    assert response.status_code == 401


def test_owner_can_exclude_this_browser_from_future_collection(client, monkeypatch):
    monkeypatch.setenv("THERIAD_ANALYTICS_OWNER_USERNAME", "owner")
    owner_token = _register(client, "owner")
    headers = {"Authorization": f"Bearer {owner_token}"}

    enabled = client.post("/api/analytics/exclusion", headers=headers, json={"enabled": True})
    assert enabled.status_code == 200
    assert enabled.json() == {"ok": True, "excluded": True}
    assert "theriad_analytics_excluded=1" in enabled.headers["set-cookie"]

    excluded_event = client.post(
        "/api/analytics/collect",
        json={"event": "page_view", "session_id": "excluded-session"},
    )
    assert excluded_event.status_code == 200
    assert excluded_event.json() == {"ok": True, "excluded": True}

    status = client.get("/api/analytics/exclusion", headers=headers)
    assert status.status_code == 200
    assert status.json() == {"excluded": True}

    disabled = client.post("/api/analytics/exclusion", headers=headers, json={"enabled": False})
    assert disabled.status_code == 200
    assert disabled.json() == {"ok": True, "excluded": False}

    included_event = client.post(
        "/api/analytics/collect",
        json={"event": "page_view", "session_id": "included-session"},
    )
    assert included_event.status_code == 200
    assert included_event.json() == {"ok": True}

    summary = client.get("/api/analytics/summary", headers=headers)
    assert summary.status_code == 200
    assert summary.json()["total_page_views"] == 1


def test_collect_rejects_invalid_event_and_session_id(client):
    invalid_event = client.post(
        "/api/analytics/collect",
        json={"event": "identify_user", "session_id": "session-001"},
    )
    assert invalid_event.status_code == 422

    invalid_session = client.post(
        "/api/analytics/collect",
        json={"event": "page_view", "session_id": "short"},
    )
    assert invalid_session.status_code == 422
