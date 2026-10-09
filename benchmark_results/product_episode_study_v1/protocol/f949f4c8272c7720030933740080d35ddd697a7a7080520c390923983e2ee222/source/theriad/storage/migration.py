"""Explicit, ownership-preserving local-to-Supabase migration helpers."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from theriad.storage.database import get_db
from theriad.storage.hosted import HostedStorageUnavailable, hosted_repositories


def _copy_if_missing(client, table: str, row: dict[str, Any], *, record_id: str) -> bool:
    existing = client.list_rows(table, filters={"id": f"eq.{record_id}"}, limit=1)
    if existing:
        return False
    client.insert_row(table, row)
    return True


def migrate_user_workspace_to_hosted(
    *, user_id: str, organization_id: str, organization_name: str, data_dir: str | Path = "data"
) -> dict[str, Any]:
    """Copy only explicitly user-owned rows into the hosted data plane.

    The function is idempotent and intentionally does not delete the local
    source. Operators can compare row counts and ownership first, then remove
    the staging rows as a separate, recoverable migration step.
    """
    repos = hosted_repositories()
    client = repos["projects"].client
    counts = {key: 0 for key in (
        "organization", "membership", "projects", "conversations", "messages",
        "files", "runs", "usage_events", "memory",
        "project_settings",
    )}
    org_rows = client.list_rows("organizations", filters={"id": f"eq.{organization_id}"}, limit=1)
    if not org_rows:
        client.insert_row(
            "organizations",
            {"id": organization_id, "name": organization_name, "kind": "personal"},
        )
        counts["organization"] += 1
    membership_rows = client.list_rows(
        "organization_memberships",
        filters={"organization_id": f"eq.{organization_id}", "user_id": f"eq.{user_id}"},
        limit=1,
    )
    if not membership_rows:
        client.insert_row(
            "organization_memberships",
            {
                "organization_id": organization_id,
                "user_id": user_id,
                "role": "owner",
                "personal": True,
            },
        )
        counts["membership"] += 1

    db = get_db()
    projects = [dict(row) for row in db.execute("SELECT * FROM projects WHERE user_id = ?", (user_id,)).fetchall()]
    for row in projects:
        if _copy_if_missing(
            client,
            "projects",
            {
                "id": row["id"], "organization_id": organization_id, "user_id": user_id,
                "name": row["name"], "description": row["description"], "instructions": row["instructions"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
            },
            record_id=row["id"],
        ):
            counts["projects"] += 1

    conversations = [dict(row) for row in db.execute("SELECT * FROM conversations WHERE user_id = ?", (user_id,)).fetchall()]
    for row in conversations:
        if _copy_if_missing(
            client,
            "conversations",
            {
                "id": row["id"], "organization_id": organization_id, "user_id": user_id,
                "project_id": row["project_id"], "title": row["title"],
                "created_at": row["created_at"], "updated_at": row["updated_at"],
            },
            record_id=row["id"],
        ):
            counts["conversations"] += 1

    settings_rows = db.execute(
        """SELECT ps.* FROM project_settings ps
           JOIN projects p ON p.id = ps.project_id
           WHERE p.user_id = ?""",
        (user_id,),
    ).fetchall()
    for row in settings_rows:
        if client.list_rows(
            "project_settings", filters={"project_id": f"eq.{row['project_id']}"}, limit=1
        ):
            continue
        client.insert_row(
            "project_settings",
            {
                "project_id": row["project_id"],
                "organization_id": organization_id,
                "user_id": user_id,
                "allowed_model_patterns": json.loads(row["allowed_model_patterns"] or "[]"),
                "blocked_model_patterns": json.loads(row["blocked_model_patterns"] or "[]"),
                "max_cost_per_call_usd": row["max_cost_per_call_usd"],
                "allow_premium": bool(row["allow_premium"]),
                "local_only": bool(row["local_only"]),
                "enable_openrouter_auto_fallback": bool(row["enable_openrouter_auto_fallback"]),
                "updated_at": row["updated_at"],
            },
        )
        counts["project_settings"] += 1

    messages = db.execute(
        """SELECT m.* FROM messages m JOIN conversations c ON c.id = m.conversation_id
           WHERE c.user_id = ?""",
        (user_id,),
    ).fetchall()
    for row in messages:
        if _copy_if_missing(
            client,
            "messages",
            {
                "id": row["id"], "organization_id": organization_id, "user_id": user_id,
                "conversation_id": row["conversation_id"], "role": row["role"],
                "content": row["content"], "created_at": row["created_at"],
            },
            record_id=row["id"],
        ):
            counts["messages"] += 1

    from theriad.api.projects import get_upload_base

    for row in db.execute("SELECT * FROM project_files WHERE user_id = ?", (user_id,)).fetchall():
        if client.list_rows("project_files", filters={"id": f"eq.{row['id']}"}, limit=1):
            continue
        directory = row["project_id"] or "_standalone"
        source = get_upload_base() / directory / row["filename"]
        if not source.exists():
            raise HostedStorageUnavailable("A local attachment is missing; hosted migration stopped safely.")
        object_path = client.upload_object(
            organization_id=organization_id,
            object_id=row["id"],
            filename=row["filename"],
            content=source.read_bytes(),
            content_type=row["mime_type"],
        )
        client.insert_row(
            "project_files",
            {
                "id": row["id"], "organization_id": organization_id, "user_id": user_id,
                "project_id": row["project_id"], "storage_path": object_path,
                "filename": row["filename"], "original_name": row["original_name"],
                "mime_type": row["mime_type"], "size_bytes": row["size_bytes"],
                "extracted_text": row["extracted_text"], "created_at": row["created_at"],
            },
        )
        counts["files"] += 1

    for row in db.execute("SELECT * FROM runs WHERE user_id = ?", (user_id,)).fetchall():
        if _copy_if_missing(
            client,
            "runs",
            {
                "id": row["id"], "organization_id": organization_id, "user_id": user_id,
                "project_id": row["project_id"], "conversation_id": row["conversation_id"],
                "message_id": row["message_id"], "model": row["model"], "provider": row["provider"],
                "tier": row["tier"], "policy": row["policy"], "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"], "actual_cost_usd": row["actual_cost_usd"],
                "routing_latency_ms": row["routing_latency_ms"], "provider_latency_ms": row["provider_latency_ms"],
                "total_latency_ms": row["total_latency_ms"], "fallback_used": bool(row["fallback_used"]),
                "status": row["status"] or "unknown", "error": row["error"], "created_at": row["created_at"],
            },
            record_id=row["id"],
        ):
            counts["runs"] += 1

    for row in db.execute("SELECT * FROM usage_events WHERE user_id = ?", (user_id,)).fetchall():
        if client.list_rows("usage_events", filters={"idempotency_key": f"eq.{row['idempotency_key']}"}, limit=1):
            continue
        client.insert_row(
            "usage_events",
            {
                "id": row["id"], "idempotency_key": row["idempotency_key"],
                "organization_id": organization_id, "user_id": user_id, "run_id": row["run_id"],
                "provider": row["provider"], "model": row["model"], "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"], "provider_cost_usd": row["provider_cost_usd"],
                "routing_fee_usd": row["routing_fee_usd"], "total_billed_usd": row["total_billed_usd"],
                "status": row["status"], "billing_period": row["billing_period"],
                "metadata": json.loads(row["metadata_json"] or "{}"), "created_at": row["created_at"],
                "reported_at": row["reported_at"],
            },
        )
        counts["usage_events"] += 1

    # Memory files are copied through the hosted store only when present. This
    # path is hashed by user id and never accepts a caller-provided filename.
    memory_path = Path(data_dir) / "memory-users" / f"{hashlib.sha256(user_id.encode()).hexdigest()[:32]}.json"
    if memory_path.exists():
        from theriad.memory import get_memory_store
        from theriad.memory.hosted import HostedMemoryStore

        source_store = get_memory_store(memory_path)
        target_store = HostedMemoryStore(organization_id, user_id, client)
        for entry in source_store.list():
            if target_store.get(entry.id):
                continue
            target_store.entries.insert_owned(
                {
                    "id": entry.id,
                    "project_id": None,
                    "model_id": entry.model_id,
                    "label": entry.title or "Memory",
                    "content": entry.text,
                    "source": entry.source,
                    "metadata": target_store._metadata(entry),
                    "created_at": entry.created_at,
                    "updated_at": entry.updated_at,
                },
                organization_id=organization_id,
                user_id=user_id,
            )
            counts["memory"] += 1

    return {"status": "copied", "counts": counts, "source_user_id": user_id, "organization_id": organization_id}
