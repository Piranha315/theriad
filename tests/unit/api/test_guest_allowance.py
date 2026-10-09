"""Guests use a platform key for a small allowance, then get cut off."""
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

import theriad.api.server as server_module
from theriad.billing import guest_allowance as allowance_module
from theriad.core.providers.interfaces import CompletionResponse


def _clear_ledger():
    db = allowance_module._db()
    db.execute("DELETE FROM guest_platform_spend")
    db.commit()


@pytest.fixture
def client(monkeypatch):
    from fastapi.testclient import TestClient
    from theriad.api.server import app

    monkeypatch.setattr(server_module, "_guest_free_only", lambda user: bool(user and user.get("is_guest")))
    monkeypatch.setenv("THERIAD_GUEST_OPENROUTER_API_KEY", "platform-key")
    monkeypatch.setenv("THERIAD_CHAT_BROWSER_TOOLS", "0")
    monkeypatch.delenv("THERIAD_CHAT_WEB_TOOLS", raising=False)
    import theriad.agents.chat_web_access as access

    monkeypatch.setattr(access, "model_supports_tools", lambda slug: True)
    with TestClient(app) as test_client:
        _clear_ledger()
        yield test_client
        _clear_ledger()


def _guest(client):
    token = client.post("/api/auth/guest").json()["token"]
    return {"Authorization": f"Bearer {token}"}


def _fake_complete(seen, cost=0.4):
    async def fake(self, request):
        seen.append({"key": self._api_key, "tools": [t.get("type") for t in request.extra.get("tools", [])]})
        return CompletionResponse(
            model=request.model, content="Sunny.", prompt_tokens=5, completion_tokens=2,
            latency_ms=1.0, provider="openrouter",
            raw={"choices": [{"message": {"content": "Sunny."}}],
                 "usage": {"server_tool_use_details": {"web_search_requests": 1}}},
            cost_usd=cost,
        )
    return fake


def _send(client, headers, message="Weather in Philadelphia today?"):
    return client.post(
        "/api/chat/send", headers=headers,
        json={"message": message, "mode": "execute", "allow_paid_calls": True},
    )


def test_guest_gets_web_tools_on_platform_key_then_is_cut_off(client):
    headers = _guest(client)
    seen: list[dict] = []
    with patch("theriad.core.providers.openrouter.OpenRouterProvider.complete", new=_fake_complete(seen)):
        for _ in range(3):  # 3 × $0.40 crosses the $1 allowance
            assert _send(client, headers).status_code == 200
        blocked = _send(client, headers)

    assert all(call["key"] == "platform-key" for call in seen)
    assert "openrouter:web_search" in seen[0]["tools"]
    assert blocked.status_code == 402
    assert blocked.json()["error"] == "guest_allowance_exhausted"
    assert "Create a free account" in blocked.json()["message"]
    assert len(seen) == 3  # the cut-off request never reached the provider


def test_new_guest_from_same_ip_shares_the_daily_cap(client):
    seen: list[dict] = []
    with patch("theriad.core.providers.openrouter.OpenRouterProvider.complete", new=_fake_complete(seen, cost=1.0)):
        assert _send(client, _guest(client)).status_code == 200
        assert _send(client, _guest(client)).status_code == 402


def test_without_platform_key_guests_keep_free_chat_without_web(client, monkeypatch):
    monkeypatch.delenv("THERIAD_GUEST_OPENROUTER_API_KEY")
    response = client.post(
        "/api/chat/send", headers=_guest(client),
        json={"message": "Weather in Philadelphia today?", "mode": "dry_run"},
    )
    assert response.status_code == 200
    assert response.json()["chat_intent"] != "normal_chat"


def _request(peer, headers=None):
    return SimpleNamespace(client=SimpleNamespace(host=peer), headers=headers or {})


def test_client_ip_trusts_proxy_headers_only_from_loopback():
    spoofed = {"x-forwarded-for": "1.1.1.1, 9.9.9.9", "x-real-ip": "8.8.8.8"}
    assert allowance_module.client_ip(_request("203.0.113.5", spoofed)) == "203.0.113.5"
    assert allowance_module.client_ip(_request("127.0.0.1", spoofed)) == "8.8.8.8"
    assert allowance_module.client_ip(_request("127.0.0.1", {"x-forwarded-for": "1.1.1.1, 9.9.9.9"})) == "9.9.9.9"
