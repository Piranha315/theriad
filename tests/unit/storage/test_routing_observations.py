from __future__ import annotations

from pathlib import Path

from theriad.api.projects import save_run
from theriad.storage.database import close_db, get_db


def test_save_run_persists_normalized_routing_observations(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("THERIAD_ROUTING_OBSERVATION_STORE", "local")
    close_db()
    db_path = tmp_path / "test.db"
    get_db(db_path)
    # save_run uses the configured default path, so point the module at this DB.
    from theriad.storage import database

    database.configure(db_path)
    run_id = save_run(
        project_id=None,
        conversation_id=None,
        model="cheap",
        tier="cheap_bulk",
        provider="test",
        policy="balanced",
        input_tokens=10,
        output_tokens=20,
        actual_cost_usd=0.01,
        routing_latency_ms=1.0,
        provider_latency_ms=2.0,
        total_latency_ms=3.0,
        fallback_used=False,
        status="success",
        routing_observations=[
            {
                "decision_id": "decision",
                "prompt_hash": "hash",
                "router_version": "v2",
                "lane": "general",
                "model": "cheap",
                "role": "selected",
                "p_success": 0.9,
                "eligible": True,
            }
        ],
    )
    row = get_db().execute(
        "SELECT run_id, prompt_hash, model, role FROM routing_observations"
    ).fetchone()
    assert dict(row) == {
        "run_id": run_id,
        "prompt_hash": "hash",
        "model": "cheap",
        "role": "selected",
    }
    close_db()
