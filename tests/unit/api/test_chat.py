"""
Tests for Chat / Recents APIs.

Covers:
  - POST /api/conversations   (standalone, no project)
  - GET  /api/recents         (merged feed)
  - POST /api/chat/send       dry_run mode
  - POST /api/chat/send       execute mode gated behind allow_paid_calls
  - POST /api/chat/send       execute mode with allow_paid_calls=true (mocked provider)
  - POST /api/chat/send       creates conversation automatically when none supplied
  - ALLOWED_EXTENSIONS includes .sql / .yaml / .yml
  - Existing /api/route and /api/execute still work
"""

from __future__ import annotations

import io
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import theriad.api.projects as projects_module
import theriad.api.server as server_module
from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.storage import database as db_module
from theriad.workflow.detector import WorkflowDetector

# ── Fixtures ──────────────────────────────────────────────────────────────────


def _mock_wf_detector():
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


@pytest.fixture()
def tmp_db(tmp_path):
    db_path = tmp_path / "test.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    db_module.close_db()
    db_module.configure(db_path)
    projects_module.set_upload_base(upload_dir)

    yield db_path, upload_dir

    db_module.close_db()


@pytest.fixture()
def client(monkeypatch, tmp_db):
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    return TestClient(server_module.app)


def _mock_response(content="Test response"):
    from theriad.core.providers.interfaces import CompletionResponse

    return CompletionResponse(
        model="qwen/qwen3-14b",
        content=content,
        prompt_tokens=10,
        completion_tokens=20,
        latency_ms=300.0,
        provider="openrouter",
    )


def _make_project(client, name="Test Project"):
    r = client.post("/api/projects", json={"name": name})
    assert r.status_code == 201
    return r.json()


def _make_conversation(client, project_id, title="Conv 1"):
    r = client.post(f"/api/projects/{project_id}/conversations", json={"title": title})
    assert r.status_code == 201
    return r.json()


def test_hosted_logs_map_runs_to_owner_scoped_user_prompts(monkeypatch):
    import theriad.api.auth_context as auth_context_module
    import theriad.storage.hosted as hosted_module

    organization_id = "11111111-1111-4111-8111-111111111111"
    user_id = "hosted-user-1"
    conversation_id = "22222222-2222-4222-8222-222222222222"
    assistant_message_id = "33333333-3333-4333-8333-333333333333"
    calls = []

    class FakeMessageClient:
        def list_rows(self, table, **kwargs):
            calls.append((table, kwargs))
            return [
                {
                    "id": "44444444-4444-4444-8444-444444444444",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "Find my durable chat log",
                    "created_at": "2026-09-01T12:00:01+00:00",
                },
                {
                    "id": assistant_message_id,
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": "Here is the saved log.",
                    "created_at": "2026-09-01T12:00:02+00:00",
                },
            ]

    class FakeRunRepository:
        def list_for_owner(self, **kwargs):
            assert kwargs == {
                "organization_id": organization_id,
                "user_id": user_id,
                "order": "created_at.desc",
                "limit": 20,
            }
            return [{
                "id": "55555555-5555-4555-8555-555555555555",
                "conversation_id": conversation_id,
                "message_id": assistant_message_id,
                "model": "qwen/qwen3-14b",
                "provider": "openrouter",
                "tier": "cheap_bulk",
                "input_tokens": 7,
                "output_tokens": 9,
                "actual_cost_usd": 0.012,
                "total_latency_ms": 240,
                "fallback_used": False,
                "status": "success",
                "error": None,
                "created_at": "2026-09-01T12:00:02+00:00",
            }]

    context = SimpleNamespace(
        organization_id=organization_id,
        user_id=user_id,
        principal_type="registered",
    )
    repositories = {
        "runs": FakeRunRepository(),
        "messages": SimpleNamespace(client=FakeMessageClient()),
    }
    monkeypatch.setattr(auth_context_module, "get_auth_context", lambda _request: context)
    monkeypatch.setattr(hosted_module, "hosted_repositories", lambda: repositories)

    records = server_module._hosted_inference_records(object(), 20)

    assert len(records) == 1
    assert records[0]["routing_reason"] == "Find my durable chat log"
    assert records[0]["total_tokens"] == 16
    assert records[0]["estimated_cost_usd"] == 0.012
    assert calls[0][0] == "messages"
    assert calls[0][1]["filters"] == {
        "conversation_id": f"in.({conversation_id})",
        "organization_id": f"eq.{organization_id}",
        "user_id": f"eq.{user_id}",
    }


# ── Standalone conversation (no project) ──────────────────────────────────────


class TestStandaloneConversation:
    def test_create_without_project(self, client):
        r = client.post("/api/conversations", json={"title": "Quick chat"})
        assert r.status_code == 201
        body = r.json()
        assert body["id"]
        assert body["title"] == "Quick chat"
        assert body["project_id"] is None

    def test_create_without_title_gets_default(self, client):
        r = client.post("/api/conversations", json={})
        assert r.status_code == 201
        assert r.json()["title"]  # has some default

    def test_standalone_conv_appears_in_recents(self, client):
        r = client.post("/api/conversations", json={"title": "Solo conv"})
        assert r.status_code == 201
        recents = client.get("/api/recents").json()["items"]
        ids = [i["id"] for i in recents if i["type"] == "conversation"]
        assert r.json()["id"] in ids


# ── Durable conversation history index ────────────────────────────────────────


class TestConversationHistory:
    def test_history_lists_previews_counts_and_searches_all_messages(self, client):
        first = client.post(
            "/api/conversations", json={"title": "Research notebook"}
        ).json()
        for role, content in (
            ("user", "first prompt"),
            ("assistant", "latest answer"),
            ("user", "middle-only searchable detail"),
        ):
            response = client.post(
                f"/api/conversations/{first['id']}/messages",
                json={"role": role, "content": content},
            )
            assert response.status_code == 201

        client.post(
            "/api/conversations", json={"title": "Another chat"}
        ).json()

        page = client.get("/api/conversations?limit=1")
        assert page.status_code == 200
        assert page.json()["limit"] == 1
        assert page.json()["has_more"] is True

        result = client.get(
            "/api/conversations?limit=1&q=middle-only"
        )
        assert result.status_code == 200
        body = result.json()
        assert body["has_more"] is False
        assert len(body["items"]) == 1
        item = body["items"][0]
        assert item["id"] == first["id"]
        assert item["title"] == "Research notebook"
        assert item["snippet"] == "latest answer"
        assert item["first_message"] == "first prompt"
        assert item["message_count"] == 3

        recents = client.get("/api/recents").json()["items"]
        recent = next(row for row in recents if row["id"] == first["id"])
        assert recent["snippet"] == "latest answer"


def test_hosted_logs_map_runs_to_owner_scoped_user_prompts(monkeypatch):
    import theriad.api.auth_context as auth_context_module
    import theriad.storage.hosted as hosted_module

    organization_id = "11111111-1111-4111-8111-111111111111"
    user_id = "hosted-user-1"
    conversation_id = "22222222-2222-4222-8222-222222222222"
    assistant_message_id = "33333333-3333-4333-8333-333333333333"
    calls = []

    class FakeMessageClient:
        def list_rows(self, table, **kwargs):
            calls.append((table, kwargs))
            return [
                {
                    "id": "44444444-4444-4444-8444-444444444444",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "content": "Find my durable chat log",
                    "created_at": "2026-09-01T12:00:01+00:00",
                },
                {
                    "id": assistant_message_id,
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "content": "Here is the saved log.",
                    "created_at": "2026-09-01T12:00:02+00:00",
                },
            ]

    class FakeRunRepository:
        def list_for_owner(self, **kwargs):
            assert kwargs == {
                "organization_id": organization_id,
                "user_id": user_id,
                "order": "created_at.desc",
                "limit": 20,
            }
            return [{
                "id": "55555555-5555-4555-8555-555555555555",
                "conversation_id": conversation_id,
                "message_id": assistant_message_id,
                "model": "qwen/qwen3-14b",
                "provider": "openrouter",
                "tier": "cheap_bulk",
                "input_tokens": 7,
                "output_tokens": 9,
                "actual_cost_usd": 0.012,
                "total_latency_ms": 240,
                "fallback_used": False,
                "status": "success",
                "error": None,
                "created_at": "2026-09-01T12:00:02+00:00",
            }]

    context = SimpleNamespace(
        organization_id=organization_id,
        user_id=user_id,
        principal_type="registered",
    )
    repositories = {
        "runs": FakeRunRepository(),
        "messages": SimpleNamespace(client=FakeMessageClient()),
    }
    monkeypatch.setattr(auth_context_module, "get_auth_context", lambda _request: context)
    monkeypatch.setattr(hosted_module, "hosted_repositories", lambda: repositories)

    records = server_module._hosted_inference_records(object(), 20)

    assert len(records) == 1
    assert records[0]["routing_reason"] == "Find my durable chat log"
    assert records[0]["total_tokens"] == 16
    assert records[0]["estimated_cost_usd"] == 0.012
    assert calls[0][0] == "messages"
    assert calls[0][1]["filters"] == {
        "conversation_id": f"in.({conversation_id})",
        "organization_id": f"eq.{organization_id}",
        "user_id": f"eq.{user_id}",
    }


# ── GET /api/recents ──────────────────────────────────────────────────────────


class TestRecents:
    def test_empty_recents(self, client):
        r = client.get("/api/recents")
        assert r.status_code == 200
        assert r.json()["items"] == []

    def test_recents_includes_project(self, client):
        proj = _make_project(client, "My proj")
        items = client.get("/api/recents").json()["items"]
        types = [i["type"] for i in items]
        assert "project" in types
        ids = [i["id"] for i in items if i["type"] == "project"]
        assert proj["id"] in ids

    def test_recents_includes_conversation(self, client):
        proj = _make_project(client)
        conv = _make_conversation(client, proj["id"])
        items = client.get("/api/recents").json()["items"]
        conv_ids = [i["id"] for i in items if i["type"] == "conversation"]
        assert conv["id"] in conv_ids

    def test_recents_includes_file(self, client):
        proj = _make_project(client)
        r = client.post(
            f"/api/projects/{proj['id']}/files",
            files={"file": ("note.txt", io.BytesIO(b"hello"), "text/plain")},
        )
        assert r.status_code == 201
        items = client.get("/api/recents").json()["items"]
        file_ids = [i["id"] for i in items if i["type"] == "file"]
        assert r.json()["id"] in file_ids

    def test_recents_limit_respected(self, client):
        for i in range(5):
            _make_project(client, f"P{i}")
        items = client.get("/api/recents?limit=3").json()["items"]
        assert len(items) <= 3

    def test_recents_item_shape(self, client):
        _make_project(client, "Shape test")
        item = client.get("/api/recents").json()["items"][0]
        for key in ("id", "type", "title", "updated_at", "metadata"):
            assert key in item, f"missing key: {key}"


# ── POST /api/chat/send — dry_run ─────────────────────────────────────────────


class TestChatSendDryRun:
    def test_dry_run_returns_route_no_assistant(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "hello world",
                "mode": "dry_run",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "dry_run"
        assert body["assistant_message"] is None
        assert body["route"] is not None
        assert body["route"]["tier"]
        assert body["route"]["model"]

    def test_dry_run_creates_conversation(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "test",
                "mode": "dry_run",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["conversation_id"]

    def test_dry_run_saves_user_message(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "save me",
                "mode": "dry_run",
            },
        )
        assert r.status_code == 200
        body = r.json()
        assert body["user_message"]["role"] == "user"
        assert body["user_message"]["content"] == "save me"
        # Verify message persisted
        conv_id = body["conversation_id"]
        conv = client.get(f"/api/conversations/{conv_id}").json()
        msgs = conv["messages"]
        assert len(msgs) == 1
        assert msgs[0]["content"] == "save me"

    def test_dry_run_with_project_context(self, client):
        proj = _make_project(client, "Ctx proj")
        r = client.post(
            "/api/chat/send",
            json={
                "message": "with project",
                "project_id": proj["id"],
                "mode": "dry_run",
            },
        )
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    def test_dry_run_uses_existing_conversation(self, client):
        # First send — creates conversation
        r1 = client.post("/api/chat/send", json={"message": "first", "mode": "dry_run"})
        conv_id = r1.json()["conversation_id"]

        # Second send — uses same conversation
        r2 = client.post(
            "/api/chat/send",
            json={
                "message": "second",
                "mode": "dry_run",
                "conversation_id": conv_id,
            },
        )
        assert r2.json()["conversation_id"] == conv_id

        conv = client.get(f"/api/conversations/{conv_id}").json()
        assert len(conv["messages"]) == 2

    def test_default_mode_is_dry_run(self, client):
        r = client.post("/api/chat/send", json={"message": "default mode"})
        assert r.status_code == 200
        assert r.json()["status"] == "dry_run"

    @pytest.mark.parametrize(
        "message",
        [
            "What is Theriad?",
            "What is Theriad?",
            "What does this app do?",
            "Tell me about Theriad",
        ],
    )
    def test_about_theriad_browser_is_consistent_and_deterministic(self, client, message):
        response = client.post(
            "/api/chat/send",
            json={
                "message": message,
                "mode": "execute",
                "allow_paid_calls": False,
            },
        )

        assert response.status_code == 200
        data = response.json()
        answer = data["assistant_message"]["content"]
        assert "cost-first AI router" in answer
        assert "cheapest model" in answer
        assert data["route"]["provider"] == "local"
        assert data["run"] is None


class TestRuntimeSettings:
    def test_runtime_settings_have_safe_defaults(self, client):
        response = client.get("/api/settings/runtime")
        assert response.status_code == 200
        data = response.json()
        assert data["regular_chat_returns_answer"] is True
        assert data["allow_follow_up_questions"] is True
        assert data["route_only_available"] is False
        assert data["disable_premium_models"] is False
        assert data["model_origin_policy"] == "any"
        assert data["max_cost_per_request_usd"] is None
        assert data["max_daily_spend_usd"] == 5.0
        assert data["updated_at"]

    def test_runtime_settings_persist(self, client):
        response = client.put("/api/settings/runtime", json={
            "regular_chat_returns_answer": False,
            "allow_follow_up_questions": True,
            "route_only_available": True,
            "disable_premium_models": True,
            "model_origin_policy": "us_only",
            "max_cost_per_request_usd": 0.01,
            "max_daily_spend_usd": 1.25,
        })
        assert response.status_code == 200
        data = response.json()
        assert data["regular_chat_returns_answer"] is False
        assert data["allow_follow_up_questions"] is True
        assert data["route_only_available"] is True
        assert data["disable_premium_models"] is True
        assert data["model_origin_policy"] == "us_only"
        assert data["max_cost_per_request_usd"] == 0.01
        assert data["max_daily_spend_usd"] == 1.25

        reread = client.get("/api/settings/runtime").json()
        assert reread["regular_chat_returns_answer"] is False
        assert reread["allow_follow_up_questions"] is True
        assert reread["max_daily_spend_usd"] == 1.25

    def test_us_only_origin_policy_is_honored_by_chat_route(self, client):
        client.put("/api/settings/runtime", json={"model_origin_policy": "us_only"})
        response = client.post("/api/chat/send", json={
            "message": "Write a short summary.",
            "mode": "dry_run",
        })
        assert response.status_code == 200
        data = response.json()
        assert data["route"]["model_origin_policy"] == "us_only"
        assert data["route"]["model"]
        from theriad.models.registry import DEFAULT_REGISTRY

        origin_by_model = {
            candidate.model: candidate.origin
            for candidates in DEFAULT_REGISTRY._tiers.values()
            for candidate in candidates
        }
        routed_models = [data["route"]["model"]] + [
            model for _provider, model in data["route"]["fallback_chain"]
        ]
        assert all(origin_by_model.get(model) == "US" for model in routed_models)

    def test_regular_chat_setting_forces_route_only(self, client):
        client.put("/api/settings/runtime", json={"regular_chat_returns_answer": False})
        response = client.post("/api/chat/send", json={
            "message": "Explain how routing works.",
            "mode": "execute",
            "allow_paid_calls": True,
        })
        assert response.status_code == 200
        assert response.json()["status"] == "route_only"
        assert response.json()["assistant_message"] is None

    def test_route_only_does_not_start_browser_agent(self, client):
        client.put("/api/settings/runtime", json={"route_only_available": True})
        response = client.post("/api/chat/send", json={
            "message": "Find current news about Philadelphia.",
            "mode": "dry_run",
        })
        assert response.status_code == 200
        assert response.json()["status"] == "dry_run"
        # Read-only web questions stay in chat: the model has web tools and
        # decides whether to search, so no browser agent is started.
        assert response.json()["chat_intent"] == "normal_chat"

    def test_daily_spend_limit_blocks_before_provider_call(self, client):
        client.put("/api/settings/runtime", json={
            "max_cost_per_request_usd": None,
            "max_daily_spend_usd": 0.0,
        })
        db_module.get_db().execute(
            """INSERT INTO runs (id, actual_cost_usd, status, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                "spent-before-limit",
                0.01,
                "success",
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        db_module.get_db().commit()
        response = client.post("/api/chat/send", json={
            "message": "Explain how routing works.",
            "mode": "execute",
            "allow_paid_calls": True,
        })
        assert response.status_code == 409
        assert response.json()["error"] == "max_daily_spend_exceeded"


# ── POST /api/chat/send — execute mode gating ─────────────────────────────────


class TestChatSendExecuteGating:
    def test_large_request_requires_confirmation_before_saving(self, client):
        message = "context " * 8_100
        response = client.post(
            "/api/chat/send",
            json={
                "message": message,
                "mode": "execute",
                "allow_paid_calls": True,
            },
        )

        assert response.status_code == 409
        body = response.json()
        assert body["status"] == "cost_confirmation_required"
        assert body["cost_preview"]["requires_confirmation"] is True
        assert client.get("/api/recents").json()["items"] == []

    def test_free_fallback_cannot_be_selected_without_warning(self, client):
        response = client.post(
            "/api/chat/send",
            json={
                "message": "A small request",
                "mode": "execute",
                "allow_paid_calls": True,
                "cost_decision": "free",
            },
        )

        assert response.status_code == 422
        assert "only after" in response.json()["error"]

    def test_execute_without_allow_paid_calls_returns_403(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "do it",
                "mode": "execute",
                "allow_paid_calls": False,
            },
        )
        assert r.status_code == 403
        body = r.json()
        assert "allow_paid_calls" in body["error"]

    def test_execute_403_still_returns_route_info(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "do it",
                "mode": "execute",
                "allow_paid_calls": False,
            },
        )
        assert r.status_code == 403
        body = r.json()
        # The 403 body should include route info so the UI can display it
        assert "route" in body
        assert body["route"]["tier"]

    def test_execute_403_still_creates_conversation(self, client):
        r = client.post(
            "/api/chat/send",
            json={
                "message": "blocked",
                "mode": "execute",
                "allow_paid_calls": False,
            },
        )
        assert r.status_code == 403
        body = r.json()
        assert body.get("conversation_id")


class TestGuestChatRestrictions:
    def test_hosted_guest_preview_never_enters_paid_confirmation(self, client, monkeypatch):
        monkeypatch.setattr(server_module, "_guest_free_only", lambda user: True)
        guest = client.post("/api/auth/guest").json()
        headers = {"Authorization": f"Bearer {guest['token']}"}
        response = client.post(
            "/api/chat/cost-preview",
            headers=headers,
            json={
                "message": "context " * 8_100,
                "mode": "execute",
                "allow_paid_calls": True,
                "cost_decision": "paid",
            },
        )

        assert response.status_code == 200
        body = response.json()
        assert body["guest_free_only"] is True
        assert body["requires_confirmation"] is False
        assert body["estimated_cost_usd"] == 0.0

    def test_hosted_guest_dry_run_forces_free_route(self, client, monkeypatch):
        monkeypatch.setattr(server_module, "_guest_free_only", lambda user: True)
        guest = client.post("/api/auth/guest").json()
        headers = {"Authorization": f"Bearer {guest['token']}"}
        response = client.post(
            "/api/chat/send",
            headers=headers,
            json={
                "message": "Explain how routing works.",
                "mode": "dry_run",
                "allow_paid_calls": True,
                "cost_decision": "paid",
            },
        )

        assert response.status_code == 200
        route = response.json()["route"]
        assert route["guest_free_only"] is True
        assert route["assistant_provider"] == "theriad"
        assert route["model"].endswith(":free")

    def test_hosted_guest_chat_has_no_sample_query_limit(self, client, monkeypatch):
        monkeypatch.setattr(server_module, "_guest_free_only", lambda user: True)
        guest = client.post("/api/auth/guest").json()
        headers = {"Authorization": f"Bearer {guest['token']}"}

        for index in range(3):
            response = client.post(
                "/api/chat/send",
                headers=headers,
                json={
                    "message": f"Free guest request {index}",
                    "mode": "dry_run",
                    "cost_decision": "paid",
                },
            )

            assert response.status_code == 200
            assert response.json()["route"]["guest_free_only"] is True

    def test_local_guest_uses_configured_paid_route(self, client, monkeypatch):
        monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "local")
        guest = client.post("/api/auth/guest").json()
        headers = {"Authorization": f"Bearer {guest['token']}"}
        response = client.post(
            "/api/chat/send",
            headers=headers,
            json={
                "message": "Analyze a difficult tradeoff and state your assumptions.",
                "mode": "dry_run",
                "allow_paid_calls": True,
                "cost_decision": "paid",
            },
        )

        assert response.status_code == 200
        route = response.json()["route"]
        assert route["guest_free_only"] is False
        assert route["assistant_provider"] == "theriad"
        assert not route["model"].endswith(":free")


# ── POST /api/chat/send — execute with mocked provider ────────────────────────


class TestChatSendExecute:
    def test_execute_returns_assistant_message(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response("42"))

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            r = client.post(
                "/api/chat/send",
                json={
                    "message": "what is 6*7",
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["assistant_message"] is not None
        assert body["assistant_message"]["content"] == "42"

    def test_renderable_chat_request_injects_renderer_contract(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(
            return_value=_mock_response("```html\n<!doctype html><html><body>Preview</body></html>\n```")
        )

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            response = client.post(
                "/api/chat/send",
                json={
                    "message": "Design a website for me.",
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["capability"]["recommended_execution_mode"] == "text_plus_preview"
        assert data["rendering"]["enabled"] is True
        request = mock_provider.complete.await_args.args[0]
        assert "RENDERABLE_ARTIFACT_OUTPUT" in request.messages[0].content

    def test_execute_renderable_request_injects_renderer_contract(self, client, monkeypatch):
        """The legacy execution endpoint must not bypass the preview contract."""
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new=AsyncMock(
                return_value=_mock_response("```html\n<!doctype html><html><body>Preview</body></html>\n```")
            ),
        ) as mock_complete:
            response = client.post(
                "/api/execute",
                json={
                    "prompt": "Design a website for me.",
                    "allow_paid_calls": True,
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["rendering"]["enabled"] is True
        request = mock_complete.await_args.args[0]
        assert "RENDERABLE_ARTIFACT_OUTPUT" in request.messages[0].content

    def test_follow_up_setting_adds_shared_system_instruction(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        settings = client.put(
            "/api/settings/runtime",
            json={"allow_follow_up_questions": True},
        )
        assert settings.status_code == 200

        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response("What details should I use?"))

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            response = client.post(
                "/api/chat/send",
                json={
                    "message": "Help me plan a launch.",
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["route"]["follow_up_questions_enabled"] is True
        request = mock_provider.complete.await_args.args[0]
        assert "ask one concise clarifying question" in request.messages[0].content

    def test_follow_up_request_override_works_when_setting_is_off(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response("Ready."))

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            response = client.post(
                "/api/chat/send",
                json={
                    "message": "Help me plan a launch.",
                    "mode": "execute",
                    "allow_paid_calls": True,
                    "allow_follow_up_questions": True,
                },
            )

        assert response.status_code == 200
        assert response.json()["route"]["follow_up_questions_enabled"] is True
        request = mock_provider.complete.await_args.args[0]
        assert "ask one concise clarifying question" in request.messages[0].content

    def test_execute_saves_both_messages(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response("pong"))

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            r = client.post(
                "/api/chat/send",
                json={
                    "message": "ping",
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        assert r.status_code == 200
        body = r.json()
        conv_id = body["conversation_id"]
        conv = client.get(f"/api/conversations/{conv_id}").json()
        assert len(conv["messages"]) == 2
        roles = [m["role"] for m in conv["messages"]]
        assert "user" in roles
        assert "assistant" in roles

    def test_execute_returns_run_metadata(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response())

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            r = client.post(
                "/api/chat/send",
                json={
                    "message": "run meta test",
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        body = r.json()
        run = body.get("run")
        assert run is not None
        assert run["model"]
        assert run["tier"]
        assert "input_tokens" in run
        assert "output_tokens" in run

    def test_execute_attaches_file_context(self, client, monkeypatch):
        """Files attached via file_ids are included in context."""
        proj = _make_project(client)
        upload_r = client.post(
            f"/api/projects/{proj['id']}/files",
            files={"file": ("notes.txt", io.BytesIO(b"project notes"), "text/plain")},
        )
        file_id = upload_r.json()["id"]

        monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
        mock_provider = AsyncMock()
        mock_provider.complete = AsyncMock(return_value=_mock_response("noted"))

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider",
            return_value=mock_provider,
        ):
            r = client.post(
                "/api/chat/send",
                json={
                    "message": "summarize",
                    "project_id": proj["id"],
                    "file_ids": [file_id],
                    "mode": "execute",
                    "allow_paid_calls": True,
                },
            )

        assert r.status_code == 200
        # provider was called (implicitly verified by mock not raising)
        body = r.json()
        assert body["context"]["tokens"] > 0  # file text contributed to context


# ── ALLOWED_EXTENSIONS additions ──────────────────────────────────────────────


class TestAllowedExtensions:
    @pytest.mark.parametrize(
        "ext,filename",
        [
            (".sql", "schema.sql"),
            (".yaml", "config.yaml"),
            (".yml", "docker.yml"),
        ],
    )
    def test_new_extensions_accepted(self, client, ext, filename):
        proj = _make_project(client)
        r = client.post(
            f"/api/projects/{proj['id']}/files",
            files={"file": (filename, io.BytesIO(b"content"), "text/plain")},
        )
        assert r.status_code == 201, f"{ext} should be accepted: {r.text}"

    def test_pdf_still_rejected(self, client):
        proj = _make_project(client)
        r = client.post(
            f"/api/projects/{proj['id']}/files",
            files={"file": ("report.pdf", io.BytesIO(b"%PDF"), "application/pdf")},
        )
        assert r.status_code == 415


# ── Existing endpoints backward-compatibility ─────────────────────────────────


class TestBackwardCompat:
    def test_route_still_works(self, client):
        r = client.post("/api/route", json={"prompt": "hello"})
        assert r.status_code == 200
        assert "tier" in r.json()

    def test_execute_still_requires_allow_paid_calls(self, client):
        r = client.post(
            "/api/execute",
            json={
                "prompt": "hello",
                "allow_paid_calls": False,
            },
        )
        assert r.status_code == 403

    def test_execute_large_request_requires_cost_confirmation_before_provider(self, client):
        response = client.post(
            "/api/execute",
            json={
                "prompt": "context " * 8_100,
                "allow_paid_calls": True,
            },
        )

        assert response.status_code == 409
        assert response.json()["status"] == "cost_confirmation_required"

    def test_projects_list_still_works(self, client):
        r = client.get("/api/projects")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_conversations_endpoint_still_works(self, client):
        proj = _make_project(client)
        r = client.get(f"/api/projects/{proj['id']}/conversations")
        assert r.status_code == 200
        assert isinstance(r.json(), list)
