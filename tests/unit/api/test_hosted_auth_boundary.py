"""Hosted authentication, organization ownership, and tenant-scope contracts."""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import theriad.api.server as server_module
import theriad.settings.secrets as secrets_module
from theriad.storage import database as db_module


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "local")
    db_module.close_db()
    db_module.configure(tmp_path / "hosted-auth.db")
    monkeypatch.setattr(secrets_module, "_THERIAD_DIR", tmp_path / ".theriad")
    monkeypatch.setattr(secrets_module, "_SECRETS_FILE", tmp_path / ".theriad" / "secrets.json")
    with TestClient(server_module.app) as value:
        yield value
    db_module.close_db()


def _register(client: TestClient, username: str) -> dict:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "password": "correct horse"},
    )
    assert response.status_code == 201, response.text
    return response.json()


def _headers(token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"}


def test_registered_payload_creates_one_personal_owner_org(client):
    guest = client.post("/api/auth/guest").json()
    assert guest["user"]["principal_type"] == "guest"
    assert guest["user"]["organization_id"] is None
    assert guest["user"]["role"] is None

    registered = client.post(
        "/api/auth/register",
        headers=_headers(guest["token"]),
        json={"username": "owner-user", "password": "correct horse"},
    ).json()
    user = registered["user"]

    assert user["principal_type"] == "registered"
    assert user["role"] == "owner"
    assert user["organization_id"]
    assert user["deployment_mode"] == "local"

    db = db_module.get_db()
    memberships = db.execute(
        "SELECT organization_id, user_id, role FROM organization_memberships WHERE user_id = ?",
        (user["id"],),
    ).fetchall()
    assert len(memberships) == 1
    assert dict(memberships[0]) == {
        "organization_id": user["organization_id"],
        "user_id": user["id"],
        "role": "owner",
    }

    me = client.get("/api/auth/me", headers=_headers(registered["token"]))
    assert me.status_code == 200
    assert me.json()["user"]["organization_id"] == user["organization_id"]


def test_hosted_mode_requires_auth_but_keeps_public_catalog_and_auth(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")

    assert client.get("/api/projects").status_code == 401
    assert client.get("/api/usage/summary").status_code == 401
    assert client.get("/api/health").status_code == 200
    assert client.get("/api/models").status_code == 200
    assert client.get("/api/pricing").status_code == 200
    assert client.get("/api/pricing/plans").status_code == 200
    assert client.get("/v1/models").status_code == 200
    assert client.post("/v1/chat/completions", json={}).status_code == 401

    guest = client.post("/api/auth/guest")
    assert guest.status_code == 201
    private = client.get(
        "/api/projects", headers=_headers(guest.json()["token"])
    )
    assert private.status_code == 503
    assert private.json()["error"] == "hosted_storage_unavailable"


def test_hosted_mode_rejects_invalid_bearer_tokens(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    response = client.get("/api/projects", headers=_headers("not-a-session"))
    assert response.status_code == 401
    assert "Authentication" in response.json()["detail"] or "authentication" in response.json()["detail"]


def test_hosted_mode_refuses_registration_without_supabase(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    response = client.post(
        "/api/auth/register",
        json={"username": "first-user", "password": "correct horse"},
    )
    assert response.status_code == 503
    assert "Hosted account storage is unavailable" in response.json()["detail"]


def test_hosted_mode_does_not_save_provider_keys_without_supabase(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    response = client.post(
        "/api/auth/register",
        json={"username": "key-user-one", "password": "correct horse"},
    )
    assert response.status_code == 503


def test_invalid_deployment_mode_fails_closed(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "production-ish")
    response = client.get("/api/health")
    assert response.status_code == 503
    assert "THERIAD_DEPLOYMENT_MODE" in response.json()["detail"]


def test_hosted_agents_are_blocked_and_capabilities_are_truthful(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    response = client.post(
        "/api/auth/register",
        json={"username": "agent-gate-user", "password": "correct horse"},
    )
    assert response.status_code == 503


def test_hosted_chat_agent_blocks_browser_after_body_aware_boundary(client, monkeypatch):
    registered = _register(client, "chat-agent-gate-user")
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setattr(
        "theriad.storage.hosted.hosted_data_plane_enabled",
        lambda: False,
    )
    events: list[tuple[str, dict]] = []

    def capture_event(event_type, **kwargs):
        events.append((event_type, kwargs))

    monkeypatch.setattr("theriad.storage.audit.record_event", capture_event)

    response = client.post(
        "/api/chat/agent",
        headers=_headers(registered["token"]),
        json={"message": "Go to example.com and summarize it"},
    )

    assert response.status_code == 503
    assert response.json()["error"] == "hosted_agent_executor_unavailable"
    assert any(event_type == "hosted_agent_blocked" for event_type, _ in events)
    event = next(kwargs for event_type, kwargs in events if event_type == "hosted_agent_blocked")
    assert event["resource_type"] == "http_path"
    assert event["resource_id"] == "/api/chat/agent"
    assert event["metadata"]["intent"] == "browser_agent"


def test_hosted_registered_chat_search_is_free_only(client, monkeypatch):
    registered = _register(client, "hosted-search-budget-user")
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setattr(
        "theriad.storage.hosted.hosted_data_plane_enabled",
        lambda: False,
    )
    observed: dict[str, bool] = {}

    async def search(message, *, telemetry, free_only=False):
        observed["free_only"] = free_only
        telemetry.update({
            "execution_kind": "web_search",
            "source_count": 1,
            "evidence_quality": "grounded",
            "free_only": free_only,
        })
        return (
            "[Weather source](https://example.org/weather)",
            [{
                "title": "Weather source",
                "url": "https://example.org/weather",
                "snippet": "Current weather information.",
                "source": "test-api",
            }],
            [message],
        )

    monkeypatch.setattr(
        "theriad.agents.search_synthesis.synthesize_web_search",
        search,
    )
    response = client.post(
        "/api/chat/agent",
        headers=_headers(registered["token"]),
        json={"message": "Find today's weather", "max_cost_usd": 0},
    )

    assert response.status_code == 200, response.text
    assert observed["free_only"] is True
    assert response.json()["trace"]["free_only"] is True


def test_hosted_guest_chat_search_reaches_api_path_and_stays_free(client, monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setattr(
        "theriad.storage.hosted.hosted_data_plane_enabled",
        lambda: False,
    )

    async def search(message, *, telemetry, free_only=False):
        assert message == "Find today's weather"
        assert free_only is True
        telemetry.update({
            "execution_kind": "web_search",
            "source_count": 1,
            "evidence_quality": "grounded",
        })
        return (
            "[Weather source](https://example.org/weather)",
            [{
                "title": "Weather source",
                "url": "https://example.org/weather",
                "snippet": "Current weather information.",
                "source": "test-api",
            }],
            [message],
        )

    monkeypatch.setattr(
        "theriad.agents.search_synthesis.synthesize_web_search",
        search,
    )
    guest = client.post("/api/auth/guest")
    assert guest.status_code == 201, guest.text

    response = client.post(
        "/api/chat/agent",
        headers=_headers(guest.json()["token"]),
        json={"message": "Find today's weather"},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    assert data["phase"] == "completed"
    assert data["browser_session_id"] is None
    assert data["trace"]["execution_kind"] == "web_search"


def test_hosted_api_search_failure_does_not_open_browser(client, monkeypatch):
    registered = _register(client, "chat-search-failure-user")
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setattr(
        "theriad.storage.hosted.hosted_data_plane_enabled",
        lambda: False,
    )

    async def fail_search(_message, *, telemetry):
        telemetry["execution_kind"] = "web_search"
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr(
        "theriad.agents.search_synthesis.synthesize_web_search",
        fail_search,
    )
    response = client.post(
        "/api/chat/agent",
        headers=_headers(registered["token"]),
        json={"message": "Find today's weather"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["phase"] == "failed"
    assert data["browser_session_id"] is None
    assert "Automatic web search is unavailable" in data["trace"]["final_answer"]


@pytest.mark.asyncio
async def test_browser_agent_sessions_are_owner_scoped():
    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    from theriad.agents.browser_session_store import BrowserSessionStore

    store = BrowserSessionStore()
    session = await store.create(
        BrowserAgentStartRequest(goal="owner one task"),
        owner_user_id="user-one",
        owner_organization_id="org-one",
    )

    assert await store.get(session.session_id, owner_user_id="user-one") is session
    assert await store.get(session.session_id, owner_user_id="user-two") is None
    assert await store.get(session.session_id) is None
