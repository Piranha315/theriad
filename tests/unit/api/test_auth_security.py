from __future__ import annotations

from fastapi.testclient import TestClient

import theriad.api.server as server_module
import theriad.settings.secrets as secrets_module
from theriad.storage import database as db_module


def test_security_summary_and_revoke_others(tmp_path, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "local")
    db_module.close_db()
    db_module.configure(tmp_path / "security.db")
    monkeypatch.setattr(secrets_module, "_THERIAD_DIR", tmp_path / ".theriad")
    monkeypatch.setattr(secrets_module, "_SECRETS_FILE", tmp_path / ".theriad" / "secrets.json")
    with TestClient(server_module.app) as client:
        first = client.post(
            "/api/auth/register", json={"username": "security-user", "password": "correct horse"}
        ).json()
        second = client.post(
            "/api/auth/login", json={"username": "security-user", "password": "correct horse"}
        ).json()
        headers = {"Authorization": f"Bearer {first['token']}"}
        summary = client.get("/api/auth/security", headers=headers)
        assert summary.status_code == 200
        assert summary.json()["session_count"] == 2
        revoked = client.post("/api/auth/sessions/revoke-others", headers=headers)
        assert revoked.status_code == 200
        assert revoked.json()["revoked"] == 1
        assert client.get("/api/auth/me", headers=headers).status_code == 200
        assert client.get(
            "/api/auth/me", headers={"Authorization": f"Bearer {second['token']}"}
        ).status_code == 401
    db_module.close_db()
