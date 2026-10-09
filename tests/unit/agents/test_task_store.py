from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from theriad.agents.task_engine import TaskEngine, TaskEngineError
from theriad.agents.task_store import (
    LocalTaskStore,
    SupabaseTaskStore,
    SupabaseTaskStoreConfig,
    TaskStoreError,
    build_task_store,
    task_store_configuration,
)


class Response:
    def __init__(self, payload=None):
        self.payload = payload

    def raise_for_status(self):
        return None

    def json(self):
        return self.payload


def test_explicit_supabase_mode_fails_closed_without_secret(monkeypatch, tmp_path):
    monkeypatch.setenv("THERIAD_TASK_STORE", "supabase")
    monkeypatch.delenv("SUPABASE_URL", raising=False)
    monkeypatch.delenv("SUPABASE_SECRET_KEY", raising=False)
    monkeypatch.delenv("SUPABASE_SERVICE_ROLE_KEY", raising=False)

    with pytest.raises(TaskStoreError, match="requires SUPABASE_URL"):
        build_task_store(tmp_path)


def test_auto_mode_selects_shared_store_only_with_server_secret(monkeypatch):
    monkeypatch.setenv("THERIAD_TASK_STORE", "auto")
    monkeypatch.setenv("SUPABASE_URL", "https://project.supabase.co")
    monkeypatch.setenv("SUPABASE_SECRET_KEY", "server-secret")

    status = task_store_configuration()

    assert status["selected"] == "supabase"
    assert status["shared"] is True


def test_supabase_persist_uses_single_restricted_rpc_call():
    store = SupabaseTaskStore(SupabaseTaskStoreConfig(
        url="https://project.supabase.co",
        secret_key="server-secret",
    ))
    snapshot = {"version": 1, "task": {"task_id": "task-1"}}
    event = {"event_id": "event-1", "task_id": "task-1"}

    with patch("theriad.agents.task_store.httpx.request", return_value=Response()) as request:
        store.persist(snapshot, event)

    kwargs = request.call_args.kwargs
    assert request.call_args.args[:2] == (
        "POST",
        "https://project.supabase.co/rest/v1/rpc/persist_agent_task",
    )
    assert kwargs["json"] == {"p_snapshot": snapshot, "p_event": event}
    assert kwargs["headers"]["apikey"] == "server-secret"
    assert "Authorization" not in kwargs["headers"]
    assert kwargs["headers"]["Prefer"] == "return=minimal"


def test_legacy_service_role_jwt_is_used_as_bearer():
    legacy_jwt = "eyJlegacy-service-role"
    store = SupabaseTaskStore(SupabaseTaskStoreConfig(
        url="https://project.supabase.co",
        secret_key=legacy_jwt,
    ))

    assert store._headers()["Authorization"] == f"Bearer {legacy_jwt}"


def test_supabase_load_events_batches_task_ids():
    store = SupabaseTaskStore(SupabaseTaskStoreConfig(
        url="https://project.supabase.co",
        secret_key="server-secret",
    ))
    event = {
        "event_id": "00000000-0000-0000-0000-000000000002",
        "task_id": "00000000-0000-0000-0000-000000000001",
        "sequence": 1,
        "kind": "task.created",
        "payload": {},
        "created_at": 1.0,
    }
    with patch(
        "theriad.agents.task_store.httpx.request",
        return_value=Response([{"event": event}]),
    ) as request:
        grouped = store.load_events([event["task_id"]])

    assert grouped[event["task_id"]] == [event]
    assert request.call_args.kwargs["params"]["task_id"] == f"in.({event['task_id']})"


def test_local_store_lease_prevents_two_supervisors(tmp_path):
    store = LocalTaskStore(tmp_path)
    assert store.claim("task", "worker-a", 30) is True
    assert store.claim("task", "worker-b", 30) is False
    assert store.heartbeat("task", "worker-b", 30) is False
    store.release("task", "worker-a")
    assert store.claim("task", "worker-b", 30) is True


@pytest.mark.asyncio
async def test_task_engine_is_the_claiming_supervisor(tmp_path):
    store = LocalTaskStore(tmp_path / "events")
    engine = TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
        store=store,
    )
    task = engine.create(goal="supervise me", workspace_root=str(tmp_path))

    with patch(
        "theriad.agents.task_runner.run_task",
        new=AsyncMock(return_value={"status": "completed"}),
    ) as run:
        result = await engine.supervise(task.task_id)

    assert result == {"status": "completed"}
    assert run.await_args.kwargs == {"engine": engine, "supervised": True}
    kinds = [event.kind for event in task.events]
    assert "supervisor.claimed" in kinds
    assert "supervisor.released" in kinds
    assert store.claim(task.task_id, "another-worker", 30) is True


@pytest.mark.asyncio
async def test_task_engine_rejects_another_active_supervisor(tmp_path):
    store = LocalTaskStore(tmp_path / "events")
    engine = TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
        store=store,
    )
    task = engine.create(goal="one worker only", workspace_root=str(tmp_path))
    assert store.claim(task.task_id, "other-worker", 30) is True

    with pytest.raises(TaskEngineError, match="already claimed"):
        await engine.supervise(task.task_id)


def test_specialists_roll_up_under_one_task_lifecycle(tmp_path):
    engine = TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
    )
    task = engine.create(goal="browse and inspect", workspace_root=str(tmp_path))
    engine.attach_specialist(task.task_id, kind="browser", specialist_id="browser-1")
    engine.attach_specialist(task.task_id, kind="terminal", specialist_id="terminal-1")

    engine.sync_specialist(task.task_id, kind="terminal", status="completed")
    assert task.status == "running"
    engine.sync_specialist(
        task.task_id,
        kind="browser",
        status="completed",
        final_answer="combined result",
    )

    assert task.status == "completed"
    assert task.final_answer == "combined result"


@pytest.mark.asyncio
async def test_shared_running_task_is_recovered_only_after_lease_claim(tmp_path):
    class SharedStore(LocalTaskStore):
        def status(self):
            return {"backend": "test-shared", "configured": True, "shared": True}

    store = SharedStore(tmp_path / "events")
    first = TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
        store=store,
    )
    task = first.create(goal="recover safely", workspace_root=str(tmp_path))
    task.status = "running"
    first.checkpoint(task.task_id)

    second = TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
        store=store,
    )
    restored = second.get(task.task_id)
    assert restored.status == "running"
    assert "task.recovered" not in [event.kind for event in restored.events]

    with patch(
        "theriad.agents.task_runner.run_task",
        new=AsyncMock(return_value={"status": "ready"}),
    ):
        await second.supervise(task.task_id)

    assert "task.recovered" in [event.kind for event in restored.events]


def test_supabase_migration_is_owner_scoped_and_server_write_only():
    migration = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "supabase/migrations/20260821000000_agent_task_supervisor.sql"
    ).read_text(encoding="utf-8").lower()

    assert "enable row level security" in migration
    assert "principal_id = 'supabase:' || (select auth.uid())::text" in migration
    assert "grant execute on function public.persist_agent_task" in migration
    assert "to service_role" in migration
    assert "grant select on public.agent_tasks to authenticated" not in migration
    assert "grant insert on public.agent_tasks to authenticated" not in migration
