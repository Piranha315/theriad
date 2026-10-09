from __future__ import annotations

import httpx

from theriad.routing.observation_store import mirror_to_supabase


def test_shared_mirror_uses_service_secret_without_exposing_it(monkeypatch):
    captured = {}

    def fake_post(url, *, json, headers, timeout):
        captured.update(url=url, json=json, headers=headers, timeout=timeout)
        return httpx.Response(201, request=httpx.Request("POST", url))

    monkeypatch.setenv("THERIAD_ROUTING_OBSERVATION_STORE", "supabase")
    monkeypatch.setenv("SUPABASE_URL", "https://example.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "sb_secret_example")
    monkeypatch.setattr(httpx, "post", fake_post)
    assert mirror_to_supabase(
        [
            {
                "decision_id": "decision",
                "prompt_hash": "hash",
                "router_version": "v2",
                "lane": "general",
                "model": "model",
                "role": "selected",
                "expected_cost_usd": float("inf"),
            }
        ],
        run_id="run",
        principal_id="supabase:user",
    )
    assert captured["headers"]["apikey"] == "sb_secret_example"
    assert "Authorization" not in captured["headers"]
    assert captured["json"][0]["expected_cost_usd"] is None
    assert captured["json"][0]["principal_id"] == "supabase:user"


def test_missing_shared_configuration_is_a_safe_noop(monkeypatch):
    monkeypatch.setenv("THERIAD_ROUTING_OBSERVATION_STORE", "auto")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)
    assert not mirror_to_supabase([], run_id="run", principal_id=None)
