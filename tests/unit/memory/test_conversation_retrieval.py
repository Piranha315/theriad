from theriad.memory.retrieval import retrieve_conversations
from theriad.storage import database as db_module


def test_specific_conversation_retrieval_is_explicit_and_project_scoped(tmp_path):
    db_module.close_db()
    db_module.configure(tmp_path / "retrieval.db")
    db = db_module.get_db()
    db.execute(
        "INSERT INTO projects (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
        ("p1", "Project", "2026-01-01", "2026-01-01"),
    )
    db.execute(
        "INSERT INTO conversations (id, project_id, title, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("c1", "p1", "Alpaca launch", "2026-01-01", "2026-01-01"),
    )
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        ("m1", "c1", "user", "The alpaca launch uses a blue rocket.", "2026-01-01"),
    )
    db.commit()

    implicit_text, implicit_sources = retrieve_conversations(
        "Build a chat interface for alpacas", project_id="p1"
    )
    explicit_text, explicit_sources = retrieve_conversations(
        "Look at my conversation about the alpaca launch", project_id="p1"
    )
    other_project_text, _ = retrieve_conversations(
        "Look at my conversation about the alpaca launch", project_id="other"
    )

    assert implicit_text == ""
    assert implicit_sources == []
    assert "blue rocket" in explicit_text
    assert explicit_sources[0].id == "c1"
    assert other_project_text == ""
    db_module.close_db()
