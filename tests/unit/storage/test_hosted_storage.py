"""Hosted Supabase boundary and ownership contract tests."""
from __future__ import annotations

import pytest

from theriad.storage.hosted import (
    HostedStorageUnavailable,
    ProjectRepository,
    assert_hosted_storage_ready,
    get_hosted_storage_config,
    hosted_data_plane_enabled,
)


class _FakeClient:
    def __init__(self):
        self.calls = []

    def list_rows(self, table, **kwargs):
        self.calls.append(("list", table, kwargs))
        return []

    def insert_row(self, table, row):
        self.calls.append(("insert", table, row))
        return row


def test_hosted_mode_enables_data_plane_and_fails_closed_without_supabase(monkeypatch):
    monkeypatch.setenv("THERIAD_DEPLOYMENT_MODE", "hosted")
    monkeypatch.setenv("THERIAD_STORAGE_BACKEND", "sqlite")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    assert hosted_data_plane_enabled() is True
    assert get_hosted_storage_config().required is True
    with pytest.raises(HostedStorageUnavailable):
        assert_hosted_storage_ready(probe=False)


def test_hosted_repository_always_filters_organization_and_user():
    client = _FakeClient()
    repository = ProjectRepository(client)

    repository.get_owned(
        "project-1", organization_id="org-1", user_id="user-1"
    )
    repository.insert_owned(
        {"id": "project-2", "name": "Scoped"},
        organization_id="org-1",
        user_id="user-1",
    )

    list_call = client.calls[0]
    assert list_call[2]["filters"] == {
        "id": "eq.project-1",
        "organization_id": "eq.org-1",
        "user_id": "eq.user-1",
    }
    assert client.calls[1][2]["organization_id"] == "org-1"
    assert client.calls[1][2]["user_id"] == "user-1"
