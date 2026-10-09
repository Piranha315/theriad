"""
Tests for Project / Conversation / File / Run / Context APIs.

Covers every requirement from the Definition of Done spec:
  - create / list / update / delete project
  - create / list conversations
  - append / read messages
  - upload / list / delete supported file
  - reject unsupported file type
  - context builder includes selected file text
  - context builder truncates over budget
  - execute with project/conversation saves messages and run metadata
  - execute without project still works as before
  - paid-call safety still works
  - existing /api/route still works
"""
from __future__ import annotations

import io
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import theriad.api.server as server_module
import theriad.api.projects as projects_module
from theriad.storage import database as db_module
from theriad.layers.layer2_embedding.backends import MockEmbeddingBackend
from theriad.workflow.detector import WorkflowDetector


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _mock_wf_detector():
    return WorkflowDetector(embedding_backend=MockEmbeddingBackend(dim=64))


@pytest.fixture()
def tmp_db(tmp_path):
    """Give each test an isolated SQLite database and upload directory."""
    db_path = tmp_path / "test.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    # Reset thread-local connection so tests don't share state
    db_module.close_db()
    db_module.configure(db_path)
    projects_module.set_upload_base(upload_dir)

    yield db_path, upload_dir

    db_module.close_db()


@pytest.fixture()
def client(monkeypatch, tmp_db):
    monkeypatch.setattr(server_module, "_create_workflow_detector", _mock_wf_detector)
    return TestClient(server_module.app)


# ── Helper ────────────────────────────────────────────────────────────────────

def _make_mock_response(
    model="qwen/qwen3-14b",
    content="Hello world",
    prompt_tokens=10,
    completion_tokens=20,
    latency_ms=400.0,
):
    from theriad.core.providers.interfaces import CompletionResponse
    return CompletionResponse(
        model=model,
        content=content,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        latency_ms=latency_ms,
        provider="openrouter",
    )


# ═════════════════════════════════════════════════════════════════════════════
# Projects CRUD
# ═════════════════════════════════════════════════════════════════════════════

class TestProjectsCRUD:

    def test_list_projects_empty(self, client):
        r = client.get("/api/projects")
        assert r.status_code == 200
        assert r.json() == []

    def test_create_project(self, client):
        r = client.post("/api/projects", json={"name": "Alpha", "description": "First project"})
        assert r.status_code == 201
        data = r.json()
        assert data["name"] == "Alpha"
        assert data["description"] == "First project"
        assert "id" in data

    def test_list_projects_after_create(self, client):
        client.post("/api/projects", json={"name": "P1"})
        client.post("/api/projects", json={"name": "P2"})
        r = client.get("/api/projects")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_update_project(self, client):
        p = client.post("/api/projects", json={"name": "Old"}).json()
        r = client.patch(f"/api/projects/{p['id']}", json={"name": "New", "instructions": "Be concise."})
        assert r.status_code == 200
        assert r.json()["name"] == "New"
        assert r.json()["instructions"] == "Be concise."

    def test_delete_project(self, client):
        p = client.post("/api/projects", json={"name": "ToDelete"}).json()
        r = client.delete(f"/api/projects/{p['id']}")
        assert r.status_code == 204
        assert client.get("/api/projects").json() == []

    def test_update_nonexistent_project(self, client):
        r = client.patch("/api/projects/does-not-exist", json={"name": "X"})
        assert r.status_code == 404

    def test_delete_nonexistent_project(self, client):
        r = client.delete("/api/projects/does-not-exist")
        assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# Conversations
# ═════════════════════════════════════════════════════════════════════════════

class TestConversations:

    def _project(self, client):
        return client.post("/api/projects", json={"name": "ConvProject"}).json()

    def test_create_conversation(self, client):
        p = self._project(client)
        r = client.post(f"/api/projects/{p['id']}/conversations", json={"title": "Session 1"})
        assert r.status_code == 201
        assert r.json()["title"] == "Session 1"
        assert r.json()["project_id"] == p["id"]

    def test_list_conversations(self, client):
        p = self._project(client)
        client.post(f"/api/projects/{p['id']}/conversations", json={})
        client.post(f"/api/projects/{p['id']}/conversations", json={"title": "Two"})
        r = client.get(f"/api/projects/{p['id']}/conversations")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_get_conversation_with_messages(self, client):
        p = self._project(client)
        c = client.post(f"/api/projects/{p['id']}/conversations", json={}).json()
        client.post(f"/api/conversations/{c['id']}/messages", json={"role": "user", "content": "Hi"})
        r = client.get(f"/api/conversations/{c['id']}")
        assert r.status_code == 200
        assert len(r.json()["messages"]) == 1
        assert r.json()["messages"][0]["content"] == "Hi"

    def test_conversation_not_found(self, client):
        r = client.get("/api/conversations/missing")
        assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# Messages
# ═════════════════════════════════════════════════════════════════════════════

class TestMessages:

    def _setup(self, client):
        p = client.post("/api/projects", json={"name": "MsgProject"}).json()
        c = client.post(f"/api/projects/{p['id']}/conversations", json={}).json()
        return p, c

    def test_append_user_message(self, client):
        _, c = self._setup(client)
        r = client.post(
            f"/api/conversations/{c['id']}/messages",
            json={"role": "user", "content": "Hello!"},
        )
        assert r.status_code == 201
        assert r.json()["role"] == "user"
        assert r.json()["content"] == "Hello!"

    def test_append_assistant_message(self, client):
        _, c = self._setup(client)
        r = client.post(
            f"/api/conversations/{c['id']}/messages",
            json={"role": "assistant", "content": "Hi back"},
        )
        assert r.status_code == 201
        assert r.json()["role"] == "assistant"

    def test_invalid_role_rejected(self, client):
        _, c = self._setup(client)
        r = client.post(
            f"/api/conversations/{c['id']}/messages",
            json={"role": "robot", "content": "x"},
        )
        assert r.status_code == 422

    def test_messages_ordered_by_created_at(self, client):
        _, c = self._setup(client)
        for i in range(3):
            client.post(f"/api/conversations/{c['id']}/messages",
                        json={"role": "user", "content": str(i)})
        conv = client.get(f"/api/conversations/{c['id']}").json()
        contents = [m["content"] for m in conv["messages"]]
        assert contents == ["0", "1", "2"]


# ═════════════════════════════════════════════════════════════════════════════
# Files
# ═════════════════════════════════════════════════════════════════════════════

SUPPORTED_FILES = [
    ("hello.txt",  b"hello world",       "text/plain"),
    ("readme.md",  b"# Hello",           "text/markdown"),
    ("data.json",  b'{"key": "val"}',    "application/json"),
    ("script.py",  b"print('hi')",       "text/x-python"),
    ("app.js",     b"console.log('hi')", "application/javascript"),
    ("style.css",  b"body { color: red }", "text/css"),
]

# Images are supported now: they upload as bytes and are sent to a
# vision-capable model as image content blocks rather than text-extracted.
IMAGE_FILES = [
    ("shot.png",   b"\x89PNG\r\n\x1a\n",   "image/png"),
    ("photo.jpg",  b"\xff\xd8\xff\xe0",     "image/jpeg"),
    ("anim.gif",   b"GIF89a",            "image/gif"),
    ("pic.webp",   b"RIFF0000WEBP",      "image/webp"),
]

UNSUPPORTED_FILES = [
    ("doc.docx",   b"PK",               "application/octet-stream"),
    ("data.xlsx",  b"PK",               "application/octet-stream"),
    ("binary.exe", b"\x4d\x5a",          "application/octet-stream"),
]


class TestFiles:

    def _project(self, client):
        return client.post("/api/projects", json={"name": "FileProject"}).json()

    def test_list_files_empty(self, client):
        p = self._project(client)
        r = client.get(f"/api/projects/{p['id']}/files")
        assert r.status_code == 200
        assert r.json() == []

    @pytest.mark.parametrize("fname,content,mime", SUPPORTED_FILES)
    def test_upload_supported_file(self, client, fname, content, mime):
        p = self._project(client)
        r = client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": (fname, io.BytesIO(content), mime)},
        )
        assert r.status_code == 201, r.text
        data = r.json()
        assert data["original_name"] == fname
        assert data["size_bytes"] == len(content)
        assert data["project_id"] == p["id"]

    @pytest.mark.parametrize("fname,content,mime", UNSUPPORTED_FILES)
    def test_reject_unsupported_file_type(self, client, fname, content, mime):
        p = self._project(client)
        r = client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": (fname, io.BytesIO(content), mime)},
        )
        assert r.status_code == 415, r.text

    def test_list_files_after_upload(self, client):
        p = self._project(client)
        client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": ("a.txt", io.BytesIO(b"a"), "text/plain")},
        )
        client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": ("b.txt", io.BytesIO(b"b"), "text/plain")},
        )
        r = client.get(f"/api/projects/{p['id']}/files")
        assert r.status_code == 200
        assert len(r.json()) == 2

    def test_delete_file(self, client):
        p = self._project(client)
        f = client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": ("del.txt", io.BytesIO(b"bye"), "text/plain")},
        ).json()
        r = client.delete(f"/api/files/{f['id']}")
        assert r.status_code == 204
        assert client.get(f"/api/projects/{p['id']}/files").json() == []

    def test_delete_nonexistent_file(self, client):
        r = client.delete("/api/files/no-such-id")
        assert r.status_code == 404


# ═════════════════════════════════════════════════════════════════════════════
# Context Builder
# ═════════════════════════════════════════════════════════════════════════════

class TestContextBuilder:

    def _setup(self, client, instructions="Use Python 3.11."):
        p = client.post("/api/projects", json={
            "name": "CtxProject",
            "instructions": instructions,
        }).json()
        c = client.post(f"/api/projects/{p['id']}/conversations", json={}).json()
        f = client.post(
            f"/api/projects/{p['id']}/files",
            files={"file": ("util.py", io.BytesIO(b"def add(a, b): return a + b"), "text/x-python")},
        ).json()
        return p, c, f

    def test_context_includes_instructions(self, client):
        p, c, _ = self._setup(client)
        r = client.post("/api/context/build", json={"project_id": p["id"]})
        assert r.status_code == 200
        assert "Use Python 3.11." in r.json()["context_text"]

    def test_context_includes_file_text(self, client):
        p, c, f = self._setup(client)
        r = client.post("/api/context/build", json={
            "project_id": p["id"],
            "file_ids": [f["id"]],
        })
        assert r.status_code == 200
        body = r.json()
        assert "def add" in body["context_text"]

    def test_context_includes_messages(self, client):
        p, c, _ = self._setup(client)
        client.post(f"/api/conversations/{c['id']}/messages",
                    json={"role": "user", "content": "What is 2+2?"})
        client.post(f"/api/conversations/{c['id']}/messages",
                    json={"role": "assistant", "content": "4"})
        r = client.post("/api/context/build", json={
            "project_id": p["id"],
            "conversation_id": c["id"],
        })
        assert r.status_code == 200
        text = r.json()["context_text"]
        assert "What is 2+2?" in text
        assert "4" in text

    def test_context_truncates_over_budget(self, client):
        p = client.post("/api/projects", json={
            "name": "BigProject",
            "instructions": "x " * 2000,  # ~2000 tokens
        }).json()
        # Tiny budget — will force truncation
        r = client.post("/api/context/build", json={
            "project_id": p["id"],
            "max_context_tokens": 50,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["truncated"] is True
        assert body["total_tokens"] <= 50
        assert body["truncation_warning"] is not None

    def test_context_no_truncation_when_fits(self, client):
        p, c, f = self._setup(client, instructions="Short instruction.")
        r = client.post("/api/context/build", json={
            "project_id": p["id"],
            "file_ids": [f["id"]],
            "max_context_tokens": 4096,
        })
        assert r.status_code == 200
        assert r.json()["truncated"] is False

    def test_context_default_does_not_apply_an_application_cap(self, client):
        p = client.post("/api/projects", json={
            "name": "UnboundedProject",
            "instructions": "head " * 2_000 + "tail-marker",
        }).json()

        r = client.post("/api/context/build", json={"project_id": p["id"]})

        assert r.status_code == 200
        body = r.json()
        assert body["truncated"] is False
        assert body["budget_tokens"] is None
        assert "tail-marker" in body["context_text"]

    def test_paid_provider_warning_present_when_files_attached(self, client):
        p, _, f = self._setup(client)
        r = client.post("/api/context/build", json={
            "project_id": p["id"],
            "file_ids": [f["id"]],
        })
        assert r.json()["paid_provider_warning"] is not None

    def test_no_paid_warning_without_files(self, client):
        p, _, _ = self._setup(client)
        r = client.post("/api/context/build", json={"project_id": p["id"]})
        assert r.json()["paid_provider_warning"] is None


# ═════════════════════════════════════════════════════════════════════════════
# Execute — project context + message/run persistence
# ═════════════════════════════════════════════════════════════════════════════

class TestExecuteWithProject:

    def _setup(self, client):
        p = client.post("/api/projects", json={"name": "ExecProject"}).json()
        c = client.post(f"/api/projects/{p['id']}/conversations", json={}).json()
        return p, c

    def test_execute_with_project_saves_messages(self, client, monkeypatch):
        p, c = self._setup(client)
        mock_resp = _make_mock_response(content="The answer is 42")

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ), patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            r = client.post("/api/execute", json={
                "prompt": "What is 6 * 7?",
                "allow_paid_calls": True,
                "project_id": p["id"],
                "conversation_id": c["id"],
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        assert body["persistence"]["status"] == "saved"
        assert body["saved_user_message_id"] is not None
        assert body["saved_assistant_message_id"] is not None
        assert body["saved_run_id"] is not None

        # Verify messages are actually in the DB
        conv = client.get(f"/api/conversations/{c['id']}").json()
        msgs = conv["messages"]
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "What is 6 * 7?"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "The answer is 42"

    def test_execute_surfaces_persistence_failure(self, client, monkeypatch):
        p, c = self._setup(client)

        def fail_save_run(**_kwargs):
            raise RuntimeError("storage unavailable")

        monkeypatch.setattr(projects_module, "save_run", fail_save_run)
        mock_resp = _make_mock_response(content="The provider answered")
        with (
            patch(
                "theriad.core.providers.openrouter.OpenRouterProvider.complete",
                new_callable=AsyncMock,
                return_value=mock_resp,
            ),
            patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}),
        ):
            response = client.post(
                "/api/execute",
                json={
                    "prompt": "Please persist this answer",
                    "allow_paid_calls": True,
                    "project_id": p["id"],
                    "conversation_id": c["id"],
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "success"
        assert body["persistence"] == {
            "status": "failed",
            "error": "Execution completed, but the conversation record could not be saved.",
        }

    def test_hosted_persistence_does_not_fallback_to_sqlite_without_context(
        self, client, monkeypatch
    ):
        monkeypatch.setattr(projects_module, "hosted_data_plane_enabled", lambda: True)

        with pytest.raises(projects_module.HostedStorageUnavailable):
            projects_module.save_run(
                project_id=None,
                conversation_id=None,
                model="test-model",
                tier="cheap",
                provider="test",
                policy="balanced",
                input_tokens=1,
                output_tokens=1,
                actual_cost_usd=0.0,
                routing_latency_ms=0.0,
                provider_latency_ms=0.0,
                total_latency_ms=0.0,
                fallback_used=False,
                status="success",
                user_id="hosted-user",
            )

        with pytest.raises(projects_module.HostedStorageUnavailable):
            projects_module.save_message(
                conversation_id="missing-conversation",
                role="assistant",
                content="Should not be written locally.",
            )

    def test_execute_with_project_saves_run(self, client):
        p, c = self._setup(client)
        mock_resp = _make_mock_response()

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ), patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            r = client.post("/api/execute", json={
                "prompt": "Hello",
                "allow_paid_calls": True,
                "project_id": p["id"],
                "conversation_id": c["id"],
            })

        assert r.status_code == 200
        run_id = r.json()["saved_run_id"]
        assert run_id is not None

        # Verify run is in project runs list
        runs = client.get(f"/api/projects/{p['id']}/runs").json()
        assert any(r["id"] == run_id for r in runs)
        run = next(r for r in runs if r["id"] == run_id)
        assert run["status"] == "success"
        assert run["project_id"] == p["id"]
        assert run["conversation_id"] == c["id"]

    def test_execute_without_project_still_works(self, client):
        """Non-project execute path must be fully unaffected."""
        mock_resp = _make_mock_response()

        with patch(
            "theriad.core.providers.openrouter.OpenRouterProvider.complete",
            new_callable=AsyncMock,
            return_value=mock_resp,
        ), patch.dict("os.environ", {"OPENROUTER_API_KEY": "test-key"}):
            r = client.post("/api/execute", json={
                "prompt": "Simple question",
                "allow_paid_calls": True,
            })

        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "success"
        # No persistence fields when no project context
        assert body["saved_run_id"] is None
        assert body["saved_user_message_id"] is None
        assert body["saved_assistant_message_id"] is None


# ═════════════════════════════════════════════════════════════════════════════
# Paid-call safety
# ═════════════════════════════════════════════════════════════════════════════

class TestPaidCallSafety:

    def test_execute_requires_allow_paid_calls(self, client):
        r = client.post("/api/execute", json={"prompt": "test", "allow_paid_calls": False})
        assert r.status_code == 403

    def test_execute_with_project_requires_allow_paid_calls(self, client):
        p = client.post("/api/projects", json={"name": "SafeProject"}).json()
        c = client.post(f"/api/projects/{p['id']}/conversations", json={}).json()
        r = client.post("/api/execute", json={
            "prompt": "test",
            "allow_paid_calls": False,
            "project_id": p["id"],
            "conversation_id": c["id"],
        })
        assert r.status_code == 403

    def test_execute_without_api_key_returns_503(self, client, monkeypatch):
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        # Also patch secrets file lookup so a saved key doesn't bypass the gate
        monkeypatch.setattr("theriad.settings.secrets.get_openrouter_api_key", lambda: None)
        with patch("theriad.core.config.loader.load_config") as mock_cfg:
            mock_cfg.side_effect = Exception("no config")
            r = client.post("/api/execute", json={
                "prompt": "test",
                "allow_paid_calls": True,
            })
        assert r.status_code == 503


# ═════════════════════════════════════════════════════════════════════════════
# Existing endpoints unaffected
# ═════════════════════════════════════════════════════════════════════════════

# ═════════════════════════════════════════════════════════════════════════════
# Response-shape contract (guards the frontend "direct object" assumption)
# ═════════════════════════════════════════════════════════════════════════════

class TestResponseShape:
    """
    The frontend expects a direct project object from POST /api/projects,
    not a wrapper like {"project": {...}}.  These tests pin the shape so a
    backend refactor can't silently break the frontend without a red test.
    """

    def test_create_returns_direct_object_not_wrapped(self, client):
        r = client.post("/api/projects", json={"name": "ShapeTest"})
        assert r.status_code == 201
        body = r.json()
        # Must be a flat dict with the project fields at the top level
        assert "id" in body
        assert "name" in body
        assert "created_at" in body
        assert "updated_at" in body
        # Must NOT be wrapped in a "project" key
        assert "project" not in body

    def test_create_returns_all_required_fields(self, client):
        r = client.post("/api/projects", json={
            "name": "FullFields",
            "description": "desc",
            "instructions": "Do X",
        })
        body = r.json()
        assert body["name"] == "FullFields"
        assert body["description"] == "desc"
        assert body["instructions"] == "Do X"
        assert body["id"]          # non-empty string
        assert body["created_at"]  # non-empty timestamp
        assert body["updated_at"]  # non-empty timestamp

    def test_create_missing_name_returns_422(self, client):
        r = client.post("/api/projects", json={"description": "no name"})
        assert r.status_code == 422

    def test_create_content_type_json_required(self, client):
        # Sending a plain string body (not JSON) must fail with 422, not 502
        r = client.post(
            "/api/projects",
            content=b"not json",
            headers={"Content-Type": "application/json"},
        )
        assert r.status_code == 422

    def test_list_returns_array_not_object(self, client):
        client.post("/api/projects", json={"name": "A"})
        r = client.get("/api/projects")
        assert r.status_code == 200
        body = r.json()
        assert isinstance(body, list)
        assert len(body) == 1
        # Each element is a flat project dict
        assert "id" in body[0]
        assert "name" in body[0]

    def test_create_project_appears_in_list_immediately(self, client):
        """Simulates the optimistic-update path: the returned project must
        exactly match the corresponding entry in the subsequent GET list."""
        created = client.post("/api/projects", json={"name": "ListCheck"}).json()
        listed  = client.get("/api/projects").json()
        assert len(listed) == 1
        assert listed[0]["id"]   == created["id"]
        assert listed[0]["name"] == created["name"]


# ═════════════════════════════════════════════════════════════════════════════
# Existing endpoints unaffected
# ═════════════════════════════════════════════════════════════════════════════

class TestExistingEndpoints:

    def test_route_still_works(self, client):
        r = client.post("/api/route", json={
            "prompt": "Write a Python function to parse JSON",
            "policy": "balanced",
            "baseline": "openai/gpt-4o",
        })
        assert r.status_code == 200
        body = r.json()
        assert "tier" in body
        assert "model" in body
        assert "routing_path" in body

    def test_models_still_works(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_simulate_still_works(self, client):
        r = client.post("/api/simulate", json={
            "prompts": [{"prompt": "Hello world", "expected_output_tokens": 50}],
            "baseline": "openai/gpt-4o",
            "policy": "balanced",
        })
        assert r.status_code == 200
        assert "prompts" in r.json()
