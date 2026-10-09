from __future__ import annotations

import pytest

from theriad.agents.task_engine import TaskEngine, TaskEngineError, ToolCall


def engine_for(tmp_path):
    return TaskEngine(
        event_dir=tmp_path / "events",
        allowed_workspace_root=tmp_path,
    )


def plan_nodes():
    return [
        {
            "id": "research",
            "objective": "Collect the bounded evidence needed by the verifier.",
            "role": "research",
            "required_capabilities": ["web"],
            "success_criteria": ["Return a concise evidence summary."],
            "output_schema": {"type": "string"},
            "budget": {"max_model_calls": 3, "max_tool_calls": 3},
        },
        {
            "id": "verify",
            "objective": "Verify the research result and report the conclusion.",
            "role": "verification",
            "depends_on": ["research"],
            "required_capabilities": [],
            "success_criteria": ["State whether the evidence is sufficient."],
            "output_schema": {"type": "string"},
            "budget": {"max_model_calls": 3, "max_tool_calls": 3},
        },
    ]


def test_plan_admission_is_automatic_idempotent_and_durable(tmp_path):
    engine = engine_for(tmp_path)
    parent = engine.create(
        goal="research and verify",
        workspace_root=str(tmp_path),
        permission_profile="strict",
        requested_capabilities=["web"],
        budgets={"max_model_calls": 8, "max_tool_calls": 6},
    )

    first = engine.execute(parent.task_id, ToolCall("task.plan", {"nodes": plan_nodes()}))
    second = engine.execute(parent.task_id, ToolCall("task.plan", {"nodes": plan_nodes()}))

    assert first.ok is True
    assert first.approval_required is False
    assert second.ok is True
    assert second.output["idempotent_replay"] is True
    assert parent.status == "waiting_for_children"
    assert len(parent.child_task_ids) == 2
    assert not any(event.kind == "approval.required" for event in parent.events)

    restored = engine_for(tmp_path)
    restored_parent = restored.get(parent.task_id)
    restored_children = restored.children(parent.task_id)
    assert restored_parent.child_task_ids == parent.child_task_ids
    assert [child.execution_role for child in restored_children] == [
        "research",
        "verification",
    ]
    assert restored_children[1].dependency_ids == [restored_children[0].task_id]


def test_child_cannot_broaden_parent_capabilities_or_budget(tmp_path):
    engine = engine_for(tmp_path)
    parent = engine.create(
        goal="bounded web task",
        workspace_root=str(tmp_path),
        requested_capabilities=["web"],
        budgets={"max_model_calls": 4},
    )

    with pytest.raises(TaskEngineError, match="broaden parent capabilities"):
        engine.plan_children(
            parent.task_id,
            [
                {
                    "id": "write",
                    "objective": "Write a file.",
                    "role": "coding",
                    "required_capabilities": ["project"],
                    "budget": {"max_model_calls": 1},
                },
                {
                    "id": "research",
                    "objective": "Research.",
                    "role": "research",
                    "required_capabilities": ["web"],
                    "budget": {"max_model_calls": 1},
                },
            ],
        )

    with pytest.raises(TaskEngineError, match="exceed the parent budget"):
        engine.plan_children(
            parent.task_id,
            [
                {
                    "id": "research-a",
                    "objective": "Research A.",
                    "role": "research",
                    "required_capabilities": ["web"],
                    "budget": {"max_model_calls": 2},
                },
                {
                    "id": "research-b",
                    "objective": "Research B.",
                    "role": "research",
                    "required_capabilities": ["web"],
                    "budget": {"max_model_calls": 2},
                },
            ],
        )
    assert engine.children(parent.task_id) == []


def test_parent_stop_cascades_to_children(tmp_path):
    engine = engine_for(tmp_path)
    parent = engine.create(
        goal="research and verify",
        workspace_root=str(tmp_path),
        requested_capabilities=["web"],
        budgets={"max_model_calls": 8},
    )
    engine.plan_children(parent.task_id, plan_nodes())

    engine.stop(parent.task_id)

    assert parent.status == "stopped"
    assert {child.status for child in engine.children(parent.task_id)} == {"stopped"}


@pytest.mark.asyncio
async def test_runner_executes_child_dag_and_resumes_parent_without_polling(
    monkeypatch, tmp_path
):
    import theriad.agents.task_runner as runner

    engine = engine_for(tmp_path)
    parent = engine.create(
        goal="research and verify",
        workspace_root=str(tmp_path),
        permission_profile="autonomous",
        requested_capabilities=["web"],
        budgets={
            "max_model_calls": 8,
            "max_tool_calls": 4,
            "max_runtime_seconds": 60,
        },
    )
    calls: list[tuple[str, str]] = []

    async def plan(task, tools):
        calls.append((task.execution_role, task.status))
        if task.depth == 0 and not task.child_task_ids:
            assert "task.plan" in tools
            return ToolCall("task.plan", {"nodes": plan_nodes()})
        if task.depth == 0:
            assert any(row.get("tool") == "task.children" for row in task.context)
            return ToolCall("task.complete", {"answer": "combined result"})
        assert "task.plan" not in tools
        return ToolCall(
            "task.complete",
            {"answer": f"{task.execution_role} result"},
        )

    monkeypatch.setattr(runner, "_plan", plan)

    result = await runner.run_task(parent.task_id, engine=engine)

    assert result["status"] == "completed"
    assert result["final_answer"] == "combined result"
    children = engine.children(parent.task_id)
    assert [child.status for child in children] == ["completed", "completed"]
    assert [child.final_answer for child in children] == [
        "research result",
        "verification result",
    ]
    assert [role for role, _status in calls] == [
        "orchestrator",
        "research",
        "verification",
        "orchestrator",
    ]
    assert not any(event.kind == "approval.required" for event in parent.events)
    assert any(event.kind == "task.children_completed" for event in parent.events)


def test_delegated_roles_use_existing_model_tiers(tmp_path):
    from theriad.agents.task_runner import _model_for_task
    from theriad.core.providers.runtime import OPENAI_CHEAP_MODEL

    engine = engine_for(tmp_path)
    child = engine.create(
        goal="implement",
        workspace_root=str(tmp_path),
        depth=1,
        execution_role="coding",
    )

    model, tier, reason = _model_for_task(child, "openai")

    assert tier == "cheap_code"
    assert model == OPENAI_CHEAP_MODEL
    assert "coding" in reason
