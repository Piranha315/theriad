"""Tests for named memory import and storage endpoints."""

from __future__ import annotations

import importlib

import pytest
from fastapi.testclient import TestClient
from starlette.requests import Request

import theriad.api.server as server_module
from theriad.memory.store import MemoryStore


@pytest.fixture()
def client(tmp_path, monkeypatch):
    store_module = importlib.import_module("theriad.memory.store")
    monkeypatch.setattr(store_module, "_store", MemoryStore(tmp_path / "memory.json"))
    return TestClient(server_module.app)


def test_conversation_import_uses_explicit_name_and_neutral_source(client):
    response = client.post(
        "/api/memory/import",
        json={
            "name": "Personal conversations",
            "data": [{
                "name": "Recipes",
                "created_at": "2026-01-01T00:00:00Z",
                "chat_messages": [{"sender": "human", "text": "I am vegetarian"}],
            }],
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["format"] == "conversation"
    assert body["upload"]["name"] == "personal-conversations"

    memories = client.get("/api/memory").json()["memories"]
    assert memories[0]["source"] == "import"
    assert "model_id" not in memories[0]


def test_authenticated_memory_files_are_scoped_by_user(tmp_path, monkeypatch):
    request = Request({"type": "http", "headers": []})
    scope = {"id": "user-a"}
    monkeypatch.setenv("THERIAD_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(server_module, "user_scope_id", lambda _request: scope["id"])

    first = server_module._memory_store_for_request(request)
    assert first.name == "memory"
    first.add("Only user A should see this")

    scope["id"] = "user-b"
    second = server_module._memory_store_for_request(request)
    assert second.path != first.path
    assert second.list() == []
