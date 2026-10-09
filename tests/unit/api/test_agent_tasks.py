from __future__ import annotations

import os
from unittest.mock import patch

from fastapi.testclient import TestClient

os.environ.setdefault("THERIAD_LOCAL_API_TOKEN", "test-agent-token")

from theriad.api.server import app
from theriad.search.service import SearchResult

client = TestClient(app, headers={"Authorization": "Bearer test-agent-token"})


def test_capability_manifest_is_truthful_about_host_execution():
    response = client.get("/api/agent/capabilities")
    assert response.status_code == 200
    data = response.json()
    assert data["production_mock_success_allowed"] is False
    assert data["terminal"]["mode"] == "guarded_host"
    assert data["terminal"]["sandboxed"] is False
    assert data["web"]["tools"] == ["web.search", "web.fetch"]
    assert data["web"]["live_verified"] is False
    assert data["task_supervisor"]["canonical"] is True
    assert data["task_supervisor"]["selected"] == "local"
    assert data["delegation"] == {
        "available": True,
        "mode": "on",
        "tool": "task.plan",
        "max_depth": 1,
        "min_children": 2,
        "max_children": 3,
        "max_concurrency": 2,
        "admission_requires_approval": False,
        "child_side_effects_follow_parent_policy": True,
    }


def test_task_storage_status_reports_local_fallback_without_secrets():
    response = client.get("/api/agent/storage")
    assert response.status_code == 200
    data = response.json()
    assert data["backend"] == "local"
    assert data["shared"] is False
    assert data["ready"] is True
    assert data["supabase_secret_configured"] is False


def test_task_api_rejects_unauthenticated_local_requests():
    response = TestClient(app).post("/api/agent/tasks", json={"goal": "no token"})
    assert response.status_code == 401


def test_task_lifecycle_and_project_read():
    created = client.post("/api/agent/tasks", json={
        "goal": "read the readme",
        "permission_profile": "autonomous",
        "requested_capabilities": ["project"],
    })
    assert created.status_code == 200
    task_id = created.json()["task_id"]
    result = client.post(f"/api/agent/tasks/{task_id}/tools", json={
        "name": "project.read",
        "arguments": {"path": "README.md"},
    })
    assert result.status_code == 200
    assert result.json()["ok"] is True
    events = client.get(f"/api/agent/tasks/{task_id}/events").json()["events"]
    assert [event["kind"] for event in events] == ["task.created", "tool.started", "tool.completed"]


def test_task_api_exposes_delegation_children_and_tree():
    created = client.post("/api/agent/tasks", json={
        "goal": "research one bounded question",
        "permission_profile": "autonomous",
        "requested_capabilities": ["web"],
        "budgets": {"max_model_calls": 4},
    }).json()
    task_id = created["task_id"]
    planned = client.post(f"/api/agent/tasks/{task_id}/tools", json={
        "name": "task.plan",
        "arguments": {
            "nodes": [
                {
                    "id": "research",
                    "objective": "Return the bounded research result.",
                    "role": "research",
                    "required_capabilities": ["web"],
                    "budget": {"max_model_calls": 1, "max_tool_calls": 1},
                },
                {
                    "id": "verify",
                    "objective": "Verify the bounded research result.",
                    "role": "verification",
                    "depends_on": ["research"],
                    "required_capabilities": [],
                    "budget": {"max_model_calls": 1, "max_tool_calls": 1},
                },
            ],
        },
    })

    assert planned.status_code == 200
    assert planned.json()["ok"] is True
    children = client.get(f"/api/agent/tasks/{task_id}/children").json()["children"]
    tree = client.get(f"/api/agent/tasks/{task_id}/tree").json()
    assert len(children) == 2
    assert children[0]["execution_role"] == "research"
    assert tree["task_id"] == task_id
    assert tree["children"][0]["task_id"] == children[0]["task_id"]


def test_task_api_exposes_canonical_web_search():
    created = client.post("/api/agent/tasks", json={
        "goal": "research public documentation",
        "permission_profile": "autonomous",
        "requested_capabilities": ["web"],
    })
    assert created.status_code == 200
    task_id = created.json()["task_id"]

    async def search(_service, _query):
        return [SearchResult("Docs", "https://example.org/docs", "Reference", "duckduckgo")]

    with patch("theriad.search.service.SearchService.search", new=search):
        result = client.post(f"/api/agent/tasks/{task_id}/tools", json={
            "name": "web.search",
            "arguments": {"query": "public docs"},
        })

    assert result.status_code == 200
    assert result.json()["ok"] is True
    assert result.json()["output"]["provider"] == "duckduckgo"


def test_extension_pair_code_is_task_bound_and_one_time():
    created = client.post("/api/agent/tasks", json={"goal": "use browser"}).json()
    response = client.post(f"/api/agent/tasks/{created['task_id']}/extension/pair", json={})
    assert response.status_code == 200
    assert response.json()["task_id"] == created["task_id"]
    assert response.json()["pairing_code"]
