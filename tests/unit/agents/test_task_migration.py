from __future__ import annotations

import importlib.util
from pathlib import Path

from theriad.agents.task_engine import TaskEngine


def _module():
    path = Path(__file__).resolve().parents[3] / "scripts/migrate_agent_tasks_to_supabase.py"
    spec = importlib.util.spec_from_file_location("migrate_agent_tasks", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_migration_plan_reads_snapshots_and_events_without_mutation(tmp_path):
    source = tmp_path / "agent-tasks"
    engine = TaskEngine(event_dir=source, allowed_workspace_root=tmp_path)
    task = engine.create(goal="migrate me", workspace_root=str(tmp_path))

    snapshots, events = _module().migration_plan(source)

    assert [row["task"]["task_id"] for row in snapshots] == [task.task_id]
    assert events[task.task_id][0]["kind"] == "task.created"
    assert list(source.glob("*.snapshot.json"))
