from __future__ import annotations

import io

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from theriad.api.skills import router as skills_router
from theriad.storage import database as db_module


@pytest.fixture()
def client(tmp_path):
    db_module.close_db()
    db_module.configure(tmp_path / "skills-api.db")
    app = FastAPI()
    app.include_router(skills_router)
    yield TestClient(app)
    db_module.close_db()


def test_crud_contract_and_enable_toggle(client):
    created = client.post(
        "/api/skills",
        json={"name": "Concise", "instructions": "Keep answers short."},
    )
    assert created.status_code == 201
    skill = created.json()["skill"]
    assert skill["enabled"] is False

    listed = client.get("/api/skills").json()
    assert listed["count"] == 2
    assert {item["id"] for item in listed["skills"]} >= {skill["id"]}
    caveman = next(item for item in listed["skills"] if item["name"] == "Caveman")
    assert caveman["source_type"] == "builtin"
    assert caveman["enabled"] is False

    updated = client.patch(f"/api/skills/{skill['id']}", json={"enabled": True, "priority": 10})
    assert updated.status_code == 200
    assert updated.json()["skill"]["enabled"] is True
    assert client.get("/api/skills?enabled=true").json()["count"] == 1

    deleted = client.delete(f"/api/skills/{skill['id']}")
    assert deleted.json() == {"deleted": skill["id"]}
    assert client.get(f"/api/skills/{skill['id']}").status_code == 404


def test_multipart_import_is_disabled_by_default_and_dedupes(client):
    upload = (
        "files",
        (
            "careful/SKILL.md",
            io.BytesIO(b"---\nname: Careful\n---\nVerify claims before answering."),
            "text/markdown",
        ),
    )
    response = client.post("/api/skills/import/files", files=[upload])
    assert response.status_code == 201
    body = response.json()
    assert body["skills_created"] == 1
    assert body["enabled"] is False
    assert body["skills"][0]["enabled"] is False

    duplicate = client.post(
        "/api/skills/import",
        files=[
            (
                "files",
                (
                    "careful/SKILL.md",
                    io.BytesIO(b"---\nname: Careful\n---\nVerify claims before answering."),
                    "text/markdown",
                ),
            )
        ],
    )
    assert duplicate.status_code == 201
    assert duplicate.json()["duplicates_skipped"] == 1


def test_multipart_import_cannot_bypass_review_with_enabled_flag(client):
    response = client.post(
        "/api/skills/import/files",
        data={"enabled": "true"},
        files={"files": ("careful.md", b"# Careful\nReview me first.", "text/markdown")},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["enabled"] is False
    assert body["review_required"] is True
    assert body["skills"][0]["enabled"] is False
    assert any("remain disabled" in warning for warning in body["warnings"])


def test_import_rejects_unknown_project_scope(client):
    response = client.post(
        "/api/skills/import/files",
        data={"project_id": "missing"},
        files={"files": ("skill.md", b"# Test\nInstructions", "text/markdown")},
    )
    assert response.status_code == 422
    assert "project 'missing' not found" in response.json()["detail"]

    listed = client.get("/api/skills?project_id=missing")
    assert listed.status_code == 422
