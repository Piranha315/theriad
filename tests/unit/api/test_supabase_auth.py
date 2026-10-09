"""Supabase Google session bridging, email verification, and legacy AAL2 tests."""
from __future__ import annotations

import base64
import json
import sqlite3

import pytest
from fastapi.testclient import TestClient

import theriad.api.auth as auth_module
import theriad.api.server as server_module
from theriad.storage import database as db_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    db_module.close_db()
    db_module.configure(tmp_path / "supabase-auth.db")
    with TestClient(server_module.app) as value:
        yield value
    db_module.close_db()


def _token(aal: str) -> str:
    def segment(value):
        return base64.urlsafe_b64encode(json.dumps(value).encode()).rstrip(b"=").decode()

    return f"{segment({'alg': 'HS256', 'typ': 'JWT'})}.{segment({'aal': aal})}.signature"


def _supabase_response(token: str):
    user = {
        "id": "supabase-user-123",
        "email": "ada@example.com",
        "app_metadata": {"provider": "google", "providers": ["google"]},
        "user_metadata": {"full_name": "Ada Lovelace", "avatar_url": "https://example.com/ada.png"},
    }

    def fake_get(url, headers, timeout):
        assert url == "https://example.supabase.co/auth/v1/user"
        assert headers["apikey"] == "sb_publishable_test"
        assert headers["Authorization"] == f"Bearer {token}"
        assert timeout == 8.0
        return auth_module.httpx.Response(200, json=user)

    return fake_get


def test_google_exchange_requires_aal2(client, monkeypatch):
    token = _token("aal1")
    monkeypatch.setattr(auth_module.httpx, "get", _supabase_response(token))

    response = client.post("/api/auth/supabase/exchange", json={"access_token": token})

    assert response.status_code == 403
    assert "Two-factor authentication" in response.json()["detail"]


def test_google_email_code_is_sent_rate_limited_and_consumed_once(client, monkeypatch):
    token = _token("aal1")
    monkeypatch.setattr(auth_module.httpx, "get", _supabase_response(token))
    delivered = {}

    def capture_delivery(**kwargs):
        delivered.update(kwargs)

    monkeypatch.setattr(auth_module, "_send_email_mfa_code", capture_delivery)

    requested = client.post(
        "/api/auth/supabase/email/request",
        json={"access_token": token},
    )
    assert requested.status_code == 200
    challenge = requested.json()
    assert challenge["email"] == "a**@example.com"
    assert delivered["recipient"] == "ada@example.com"
    assert len(delivered["code"]) == 6
    assert delivered["code"].isdigit()

    repeated = client.post(
        "/api/auth/supabase/email/request",
        json={"access_token": token},
    )
    assert repeated.status_code == 200
    assert repeated.json()["challenge_id"] == challenge["challenge_id"]

    verified = client.post(
        "/api/auth/supabase/email/verify",
        json={
            "access_token": token,
            "challenge_id": challenge["challenge_id"],
            "code": delivered["code"],
        },
    )
    assert verified.status_code == 200
    assert verified.json()["user"]["mfa_verified"] is True
    assert verified.json()["supabase"]["method"] == "email"

    replay = client.post(
        "/api/auth/supabase/email/verify",
        json={
            "access_token": token,
            "challenge_id": challenge["challenge_id"],
            "code": delivered["code"],
        },
    )
    assert replay.status_code == 400
    assert "not recognized" in replay.json()["detail"]


def test_google_email_code_rejects_wrong_code_and_caps_attempts(client, monkeypatch):
    token = _token("aal1")
    monkeypatch.setattr(auth_module.httpx, "get", _supabase_response(token))
    delivered = {}
    monkeypatch.setattr(auth_module, "_send_email_mfa_code", lambda **kwargs: delivered.update(kwargs))

    requested = client.post(
        "/api/auth/supabase/email/request",
        json={"access_token": token},
    )
    challenge_id = requested.json()["challenge_id"]
    wrong_code = "000000" if delivered["code"] != "000000" else "000001"
    for _ in range(auth_module._EMAIL_MFA_MAX_ATTEMPTS):
        response = client.post(
            "/api/auth/supabase/email/verify",
            json={"access_token": token, "challenge_id": challenge_id, "code": wrong_code},
        )
        assert response.status_code == 400

    capped = client.post(
        "/api/auth/supabase/email/verify",
        json={"access_token": token, "challenge_id": challenge_id, "code": wrong_code},
    )
    assert capped.status_code == 400
    assert "Too many" in capped.json()["detail"]


def test_guest_session_bootstraps_pre_supabase_database(tmp_path, monkeypatch):
    """A legacy database must migrate before the first guest session write."""
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_PUBLISHABLE_KEY", "sb_publishable_test")
    database_path = tmp_path / "legacy-auth.db"
    with sqlite3.connect(database_path) as connection:
        connection.executescript(
            """
            CREATE TABLE users (
                id TEXT PRIMARY KEY,
                username TEXT UNIQUE,
                password_hash TEXT,
                is_guest INTEGER NOT NULL DEFAULT 0 CHECK(is_guest IN (0, 1)),
                created_at TEXT NOT NULL
            );
            CREATE TABLE auth_sessions (
                token TEXT PRIMARY KEY,
                user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                created_at TEXT NOT NULL
            );
            """
        )

    db_module.close_db()
    db_module.configure(database_path)
    try:
        with TestClient(server_module.app) as client:
            response = client.post("/api/auth/guest")

        assert response.status_code == 201
        assert response.json()["user"]["is_guest"] is True

        db = db_module.get_db()
        assert "supabase_user_id" in {
            row[1] for row in db.execute("PRAGMA table_info(users)").fetchall()
        }
        assert "mfa_verified" in {
            row[1] for row in db.execute("PRAGMA table_info(auth_sessions)").fetchall()
        }
    finally:
        db_module.close_db()


def test_google_exchange_persists_profile_and_scopes_session(client, monkeypatch):
    guest = client.post("/api/auth/guest").json()
    token = _token("aal2")
    monkeypatch.setattr(auth_module.httpx, "get", _supabase_response(token))

    response = client.post(
        "/api/auth/supabase/exchange",
        headers={"Authorization": f"Bearer {guest['token']}"},
        json={"access_token": token},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["user"]["email"] == "ada@example.com"
    assert body["user"]["display_name"] == "Ada Lovelace"
    assert body["user"]["auth_provider"] == "google"
    assert body["user"]["mfa_verified"] is True
    assert body["supabase"] == {
        "id": "supabase-user-123",
        "email": "ada@example.com",
        "mfa_verified": True,
    }

    db = db_module.get_db()
    row = db.execute(
        "SELECT username, email, display_name, avatar_url, auth_provider, "
        "supabase_user_id FROM users WHERE supabase_user_id = ?",
        ("supabase-user-123",),
    ).fetchone()
    session = db.execute(
        "SELECT mfa_verified FROM auth_sessions WHERE token = ?", (body["token"],)
    ).fetchone()
    assert dict(row) == {
        "username": "ada@example.com",
        "email": "ada@example.com",
        "display_name": "Ada Lovelace",
        "avatar_url": "https://example.com/ada.png",
        "auth_provider": "google",
        "supabase_user_id": "supabase-user-123",
    }
    assert session["mfa_verified"] == 1

    me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {body['token']}"})
    assert me.status_code == 200
    assert me.json()["user"]["supabase_user_id"] == "supabase-user-123"


def test_google_exchange_fails_closed_without_server_configuration(client, monkeypatch):
    monkeypatch.delenv("SUPABASE_URL")
    token = _token("aal2")

    response = client.post("/api/auth/supabase/exchange", json={"access_token": token})

    assert response.status_code == 503
