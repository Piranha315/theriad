from __future__ import annotations

import pytest

from theriad.skills.context import build_skill_context, render_skill_context
from theriad.skills.store import SkillConflict, SkillStore, SkillValidationError
from theriad.storage import database as db_module


@pytest.fixture()
def store(tmp_path):
    db_module.close_db()
    db_module.configure(tmp_path / "skills.db")
    skill_store = SkillStore()
    yield skill_store
    db_module.close_db()


def _project(store: SkillStore, project_id: str = "project-1") -> str:
    db = store._db()
    db.execute(
        "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        (project_id, "Project", "2026-01-01", "2026-01-01"),
    )
    db.commit()
    return project_id


def test_crud_enable_and_duplicate_protection(store):
    skill = store.create(name="Concise", instructions="Keep answers concise.")
    assert skill.enabled is False
    assert store.get(skill.id) == skill

    updated = store.update(skill.id, enabled=True, description="Short answers")
    assert updated.enabled is True
    assert updated.description == "Short answers"

    with pytest.raises(SkillConflict):
        store.create(name="Duplicate name irrelevant", instructions="Keep answers concise.")

    assert store.delete(skill.id) is True
    assert store.delete(skill.id) is False


def test_effective_scope_is_global_plus_matching_project_only(store):
    project_id = _project(store)
    other_id = _project(store, "project-2")
    global_skill = store.create(
        name="Global", instructions="Use headings.", enabled=True, priority=20
    )
    project_skill = store.create(
        name="Project", instructions="Prefer Python.", project_id=project_id, enabled=True
    )
    store.create(
        name="Other project", instructions="Prefer Rust.", project_id=other_id, enabled=True
    )
    store.create(name="Disabled", instructions="Never inject.", enabled=False)

    assert [item.id for item in store.resolve_enabled(None)] == [global_skill.id]
    assert [item.id for item in store.resolve_enabled(project_id)] == [
        global_skill.id,
        project_skill.id,
    ]


def test_project_delete_cascades_scoped_skill_but_not_global(store):
    project_id = _project(store)
    global_skill = store.create(name="Global", instructions="Global instructions")
    scoped = store.create(
        name="Scoped", instructions="Scoped instructions", project_id=project_id
    )

    db = store._db()
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()

    assert store.get(scoped.id) is None
    assert store.get(global_skill.id) is not None


def test_validation_rejects_control_and_format_characters(store):
    with pytest.raises(SkillValidationError):
        store.create(name="Bad", instructions="hidden\u202einstruction")
    with pytest.raises(SkillValidationError):
        store.create(name="Bad\nName", instructions="valid")


def test_context_reports_injected_and_omitted_skills(store):
    first = store.create(name="First", instructions="A" * 80, enabled=True)
    second = store.create(name="Second", instructions="B" * 80, enabled=True)

    one = render_skill_context([first, second], max_chars=500, max_count=1)
    assert one.count == 1
    assert one.ids == [first.id]
    assert one.omitted_ids == [second.id]
    assert one.truncated is True
    assert "Enabled skill: First" in one.text
    assert one.total_chars == len(one.text) <= 500

    built = build_skill_context(store=store)
    assert built.count == 2
    assert built.metadata()["skills_injected"] == 2


def test_context_never_partially_injects_an_oversized_skill(store):
    skill = store.create(name="Large", instructions="x" * 2_000, enabled=True)
    context = render_skill_context([skill], max_chars=400)
    assert context.text == ""
    assert context.omitted_ids == [skill.id]
    assert context.truncated is True


def test_account_scoped_skills_do_not_cross_accounts(store):
    account_a = SkillStore(connection=store._db(), scope_id="account-a")
    account_b = SkillStore(connection=store._db(), scope_id="account-b")
    skill_a = account_a.create(name="A", instructions="Only account A.")

    assert account_a.get(skill_a.id) is not None
    assert account_b.get(skill_a.id) is None
    assert all(skill.id != skill_a.id for skill in account_b.list())
