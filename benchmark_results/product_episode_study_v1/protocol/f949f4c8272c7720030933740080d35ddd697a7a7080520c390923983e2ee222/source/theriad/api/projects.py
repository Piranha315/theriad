"""
Project / Conversation / File / Run API routes.

Mounted on the main FastAPI app via app.include_router(projects_router).

All routes are under /api/:

    Projects
        GET     /api/projects
        POST    /api/projects
        PATCH   /api/projects/{project_id}
        DELETE  /api/projects/{project_id}

    Conversations
        GET     /api/projects/{project_id}/conversations
        POST    /api/projects/{project_id}/conversations
        GET     /api/conversations/{conversation_id}

    Messages
        POST    /api/conversations/{conversation_id}/messages

    Files
        GET     /api/projects/{project_id}/files
        POST    /api/projects/{project_id}/files         (multipart upload)
        GET     /api/files/{file_id}
        DELETE  /api/files/{file_id}

    Context
        POST    /api/context/build

    Runs
        GET     /api/projects/{project_id}/runs
        GET     /api/runs/{run_id}
"""
from __future__ import annotations

import json
import math
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, UploadFile, File as FastAPIFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from theriad.api.auth import current_scope_id, user_scope_id
from theriad.core.logging.logger import get_logger
from theriad.storage.context import build_context
from theriad.storage.hosted import HostedStorageUnavailable, hosted_data_plane_enabled, hosted_repositories
from theriad.storage.database import get_db

logger = get_logger("api.projects")

projects_router = APIRouter(prefix="/api")

# Image extensions. These are stored as bytes and sent to the model as image
# content blocks; they are deliberately NOT text-extracted (see _extract_text).
IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
}

# Maps extension -> the MIME type used in the data: URI handed to the provider.
# We derive this from the extension rather than trusting the browser-supplied
# content type, which is client-controlled and frequently wrong or absent.
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
}

TEXT_EXTENSIONS = {
    ".txt", ".md", ".json", ".csv",
    ".py", ".js", ".jsx", ".ts", ".tsx",
    ".html", ".css",
    ".sql", ".yaml", ".yml",
}

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = TEXT_EXTENSIONS | IMAGE_EXTENSIONS

# Cap on a single image, applied after upload. Images become base64 data URIs in
# the prompt, inflating ~33% on the wire, so an unbounded image is a cost and
# latency problem rather than merely a big file.
MAX_IMAGE_BYTES = 8 * 1024 * 1024


def is_image(filename: str | None) -> bool:
    return Path(filename or "").suffix.lower() in IMAGE_EXTENSIONS


def _extract_text(content: bytes, suffix: str) -> str | None:
    """Text for the prompt, or None when the bytes are not text.

    Decoding an image as UTF-8 with errors="replace" yields pages of replacement
    characters that would be injected straight into the prompt as if they were
    document content. Images carry their meaning as pixels; they go through the
    image content block instead.
    """
    if suffix in IMAGE_EXTENSIONS:
        return None
    try:
        return content.decode("utf-8", errors="replace")
    except Exception:
        return None

# Upload directory base — relative to CWD unless overridden
_UPLOAD_BASE: Path | None = None


def get_upload_base() -> Path:
    base = _UPLOAD_BASE or Path("data") / "uploads"
    base.mkdir(parents=True, exist_ok=True)
    return base


def set_upload_base(path: str | Path) -> None:
    """Override upload directory (useful in tests)."""
    global _UPLOAD_BASE
    _UPLOAD_BASE = Path(path)
    _UPLOAD_BASE.mkdir(parents=True, exist_ok=True)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return str(uuid.uuid4())


def _row_to_dict(row) -> dict:
    return dict(row) if row else {}


def _hosted_scope(request: Request | None):
    """Return the hosted repository scope for a registered request.

    Guest sessions remain a short-lived SQLite staging scope so the existing
    guest-to-account upgrade can preserve their workspace. Registered hosted
    accounts always use Supabase and never fall back silently.
    """
    if not hosted_data_plane_enabled() or request is None:
        return None
    from theriad.api.auth_context import get_auth_context

    context = get_auth_context(request)
    if context.principal_type not in {"registered", "guest"} or not context.organization_id or not context.user_id:
        return None
    try:
        return context, hosted_repositories()
    except HostedStorageUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def _hosted_scope_from_current(user_id: str | None):
    """Resolve the hosted scope and fail closed when it cannot be proven."""
    if not hosted_data_plane_enabled():
        return None
    from theriad.api.auth_context import current_auth_context

    if not user_id:
        raise HostedStorageUnavailable("Hosted persistence requires an authenticated user.")
    context = current_auth_context.get()
    if (
        not context
        or context.principal_type not in {"registered", "guest"}
        or context.user_id != user_id
        or not context.organization_id
    ):
        raise HostedStorageUnavailable("Hosted persistence context is unavailable.")
    return context, hosted_repositories()


def _current_row_scope(scope_id: str | None = None) -> tuple[str, tuple[str, ...]]:
    """Return a predicate for internal helpers running inside an API request."""
    scope = scope_id if scope_id is not None else current_scope_id()
    return ("user_id = ?", (scope,)) if scope else ("user_id IS NULL", ())


def _conversation_message_summaries(
    rows,
    *,
    search: str | None = None,
) -> dict[str, dict]:
    """Build small, owner-scoped previews without returning message bodies."""
    summaries: dict[str, dict] = {}
    for raw_row in rows:
        row = dict(raw_row)
        conversation_id = row.get("conversation_id")
        if not conversation_id:
            continue
        summary = summaries.setdefault(
            str(conversation_id),
            {
                "message_count": 0,
                "latest_user": None,
                "latest_assistant": None,
                "first_user": None,
                "search_match": False,
            },
        )
        summary["message_count"] += 1
        role = row.get("role")
        created_at = str(row.get("created_at") or "")
        content = str(row.get("content") or "")
        if search and search in content.lower():
            summary["search_match"] = True
        candidate = {
            "content": content,
            "created_at": created_at,
        }
        if role in {"user", "assistant"}:
            latest_key = f"latest_{role}"
            previous = summary[latest_key]
            if previous is None or created_at >= previous["created_at"]:
                summary[latest_key] = candidate
        if role == "user":
            previous = summary["first_user"]
            if previous is None or created_at < previous["created_at"]:
                summary["first_user"] = candidate
    return summaries


def _hosted_message_summaries(
    repos,
    context,
    conversation_ids: list[str],
    *,
    search: str | None = None,
) -> dict[str, dict]:
    """Load bounded previews for hosted conversations in one owner-scoped query."""
    if not conversation_ids:
        return {}
    # IDs originate from the hosted conversations table, so they are safe to
    # place in PostgREST's `in.(...)` filter. The transport still applies the
    # organization and user filters as a second ownership boundary.
    ids = ",".join(str(conversation_id) for conversation_id in conversation_ids)
    rows = repos["messages"].client.list_rows(
        "messages",
        filters={
            "conversation_id": f"in.({ids})",
            "organization_id": f"eq.{context.organization_id}",
            "user_id": f"eq.{context.user_id}",
        },
        select="conversation_id,role,content,created_at",
        order="created_at.asc",
        limit=1000,
    )
    return _conversation_message_summaries(rows, search=search)


def _preview_item(summary: dict | None) -> tuple[str | None, str | None, int]:
    """Return latest preview, first user message, and message count."""
    summary = summary or {}
    latest = summary.get("latest_assistant") or summary.get("latest_user")
    first_user = summary.get("first_user")
    snippet = str(latest.get("content") or "")[:140] if latest else None
    first_message = str(first_user.get("content") or "")[:140] if first_user else None
    return snippet, first_message, int(summary.get("message_count") or 0)


def _history_limit(value: int) -> int:
    return max(1, min(int(value), 100))


def _history_offset(value: int) -> int:
    return max(0, int(value))


def _history_search_pattern(value: str | None) -> tuple[str | None, str | None]:
    """Normalize a history query and escape SQL LIKE metacharacters."""
    needle = " ".join((value or "").split()).lower()
    if not needle:
        return None, None
    escaped = needle.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
    return needle, f"%{escaped}%"
# ── Projects ──────────────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    name: str
    description: Optional[str] = None
    instructions: Optional[str] = None


class ProjectUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None


@projects_router.get("/projects")
def list_projects(request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        return repos["projects"].list_for_owner(
            organization_id=context.organization_id,
            user_id=context.user_id,
            order="updated_at.desc",
        )
    db = get_db()
    rows = db.execute(
        "SELECT * FROM projects WHERE user_id IS ? ORDER BY updated_at DESC",
        (user_scope_id(request),),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@projects_router.post("/projects", status_code=201)
def create_project(body: ProjectCreate, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        now = _now()
        return repos["projects"].insert_owned(
            {
                "id": _new_id(),
                "name": body.name,
                "description": body.description,
                "instructions": body.instructions,
                "created_at": now,
                "updated_at": now,
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
    db = get_db()
    project_id = _new_id()
    now = _now()
    db.execute(
        "INSERT INTO projects (id, name, description, instructions, created_at, updated_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)",
        (project_id, body.name, body.description, body.instructions, now, now,
         user_scope_id(request)),
    )
    db.commit()
    row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_dict(row)


@projects_router.patch("/projects/{project_id}")
def update_project(project_id: str, body: ProjectUpdate, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        current = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if current is None:
            raise HTTPException(status_code=404, detail="Project not found")
        values = {"updated_at": _now()}
        if body.name is not None:
            values["name"] = body.name
        if body.description is not None:
            values["description"] = body.description
        if body.instructions is not None:
            values["instructions"] = body.instructions
        rows = repos["projects"].update_owned(
            project_id,
            values,
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        return rows[0] if rows else {**current, **values}
    db = get_db()
    _require_project(db, project_id, user_scope_id(request))

    fields = {}
    if body.name is not None:
        fields["name"] = body.name
    if body.description is not None:
        fields["description"] = body.description
    if body.instructions is not None:
        fields["instructions"] = body.instructions
    fields["updated_at"] = _now()

    set_clause = ", ".join(f"{k} = ?" for k in fields)
    values = list(fields.values()) + [project_id]
    db.execute(f"UPDATE projects SET {set_clause} WHERE id = ?", values)
    db.commit()
    row = db.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
    return _row_to_dict(row)


@projects_router.delete("/projects/{project_id}", status_code=204)
def delete_project(project_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        current = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if current is None:
            raise HTTPException(status_code=404, detail="Project not found")
        files = repos["files"].client.list_rows(
            "project_files",
            filters={
                "project_id": f"eq.{project_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            select="id,storage_path",
        )
        for file_row in files:
            path = file_row.get("storage_path")
            if path:
                try:
                    repos["files"].client.delete_object(path)
                except HostedStorageUnavailable:
                    pass
        repos["projects"].delete_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        return None
    db = get_db()
    _require_project(db, project_id, user_scope_id(request))
    db.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    db.commit()
    # Clean up uploaded files directory
    upload_dir = get_upload_base() / project_id
    if upload_dir.exists():
        import shutil
        shutil.rmtree(upload_dir, ignore_errors=True)
    return None


# ── Conversations ─────────────────────────────────────────────────────────────

class ConversationCreate(BaseModel):
    title: Optional[str] = None


@projects_router.get("/conversations")
def list_conversation_history(
    request: Request,
    limit: int = 50,
    offset: int = 0,
    q: str | None = None,
):
    """Return durable, owner-scoped chat history with safe list previews.

    ``/api/recents`` is a mixed activity feed and intentionally has a small
    global limit. This route is the canonical conversation index for the chat
    UI, so searching and paging do not hide chats behind projects, files, or
    inference runs.
    """
    limit = _history_limit(limit)
    offset = _history_offset(offset)
    needle, pattern = _history_search_pattern(q)

    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        rows = repos["conversations"].list_for_owner(
            organization_id=context.organization_id,
            user_id=context.user_id,
            order="updated_at.desc",
            limit=1000,
        )
        project_rows = repos["projects"].list_for_owner(
            organization_id=context.organization_id,
            user_id=context.user_id,
            select="id,name",
            limit=1000,
        )
        project_names = {str(row.get("id")): row.get("name") for row in project_rows}
        summaries = _hosted_message_summaries(
            repos,
            context,
            [str(row.get("id")) for row in rows if row.get("id")],
            search=needle,
        )
        items: list[dict] = []
        for row in rows:
            conversation_id = str(row.get("id") or "")
            snippet, first_message, message_count = _preview_item(summaries.get(conversation_id))
            project_name = project_names.get(str(row.get("project_id")))
            if needle:
                haystack = " ".join(
                    str(value or "").lower()
                    for value in (row.get("title"), project_name, snippet, first_message)
                )
                if needle not in haystack and not summaries.get(conversation_id, {}).get("search_match"):
                    continue
            items.append({
                "id": row.get("id"),
                "title": row.get("title") or "Untitled Conversation",
                "project_id": row.get("project_id"),
                "project_name": project_name,
                "created_at": row.get("created_at"),
                "updated_at": row.get("updated_at"),
                "snippet": snippet,
                "first_message": first_message,
                "message_count": message_count,
            })
        return {
            "items": items[offset:offset + limit],
            "limit": limit,
            "offset": offset,
            "has_more": len(items) > offset + limit,
        }

    db = get_db()
    scope = user_scope_id(request)
    search_clause = """
        AND (
            ? IS NULL
            OR LOWER(COALESCE(c.title, '')) LIKE ? ESCAPE '\\'
            OR LOWER(COALESCE(p.name, '')) LIKE ? ESCAPE '\\'
            OR EXISTS (
                SELECT 1 FROM messages sm
                WHERE sm.conversation_id = c.id
                  AND LOWER(sm.content) LIKE ? ESCAPE '\\'
            )
        )
    """
    rows = db.execute(
        f"""
        SELECT c.id, c.title, c.project_id, c.created_at, c.updated_at,
               p.name AS project_name,
               COUNT(m.id) AS message_count,
               COALESCE(
                   (SELECT substr(m1.content, 1, 140) FROM messages m1
                    WHERE m1.conversation_id = c.id AND m1.role = 'assistant'
                    ORDER BY m1.created_at DESC LIMIT 1),
                   (SELECT substr(m2.content, 1, 140) FROM messages m2
                    WHERE m2.conversation_id = c.id AND m2.role = 'user'
                    ORDER BY m2.created_at DESC LIMIT 1)
               ) AS snippet,
               (SELECT substr(m3.content, 1, 140) FROM messages m3
                WHERE m3.conversation_id = c.id AND m3.role = 'user'
                ORDER BY m3.created_at ASC LIMIT 1) AS first_message
        FROM conversations c
        LEFT JOIN projects p ON p.id = c.project_id
        LEFT JOIN messages m ON m.conversation_id = c.id
        WHERE c.user_id IS ?
        {search_clause}
        GROUP BY c.id
        ORDER BY c.updated_at DESC
        LIMIT ? OFFSET ?
        """,
        (scope, needle, pattern, pattern, pattern, limit + 1, offset),
    ).fetchall()
    items = []
    for row in rows:
        item = _row_to_dict(row)
        item["message_count"] = int(item.get("message_count") or 0)
        items.append(item)
    return {
        "items": items[:limit],
        "limit": limit,
        "offset": offset,
        "has_more": len(items) > limit,
    }


@projects_router.get("/projects/{project_id}/conversations")
def list_conversations(project_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return repos["conversations"].client.list_rows(
            "conversations",
            filters={
                "project_id": f"eq.{project_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            order="updated_at.desc",
        )
    db = get_db()
    _require_project(db, project_id, user_scope_id(request))
    rows = db.execute(
        "SELECT * FROM conversations WHERE project_id = ? ORDER BY updated_at DESC",
        (project_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@projects_router.post("/projects/{project_id}/conversations", status_code=201)
def create_conversation(project_id: str, body: ConversationCreate, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        now = _now()
        return repos["conversations"].insert_owned(
            {
                "id": _new_id(),
                "project_id": project_id,
                "title": body.title or "New Conversation",
                "created_at": now,
                "updated_at": now,
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
    db = get_db()
    scope = user_scope_id(request)
    _require_project(db, project_id, scope)
    conv_id = _new_id()
    now = _now()
    db.execute(
        "INSERT INTO conversations (id, project_id, title, created_at, updated_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, project_id, body.title or "New Conversation", now, now, scope),
    )
    db.commit()
    row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return _row_to_dict(row)


@projects_router.get("/conversations/{conversation_id}")
def get_conversation(conversation_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        conv = repos["conversations"].get_owned(
            conversation_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        message_rows = repos["messages"].client.list_rows(
            "messages",
            filters={
                "conversation_id": f"eq.{conversation_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            order="created_at.asc",
        )
        run_rows = repos["runs"].client.list_rows(
            "runs",
            filters={
                "conversation_id": f"eq.{conversation_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            select=(
                "id,message_id,model,tier,actual_cost_usd,provider,"
                "total_latency_ms,status"
            ),
            order="created_at.asc",
        )
        runs_by_message = {
            str(row.get("message_id")): row
            for row in run_rows
            if row.get("message_id")
        }
        messages = []
        for raw_message in message_rows:
            message = dict(raw_message)
            run = runs_by_message.get(str(message.get("id")))
            if run:
                message.update({
                    "run_id": run.get("id"),
                    "run_model": run.get("model"),
                    "run_tier": run.get("tier"),
                    "run_cost": run.get("actual_cost_usd"),
                    "run_provider": run.get("provider"),
                    "run_latency_ms": run.get("total_latency_ms"),
                    "run_status": run.get("status"),
                })
            messages.append(message)
        conv["messages"] = messages
        return conv
    db = get_db()
    row = db.execute(
        "SELECT * FROM conversations WHERE id = ? AND user_id IS ?",
        (conversation_id, user_scope_id(request)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    conv = _row_to_dict(row)
    messages = db.execute(
        """SELECT m.*, r.id AS run_id, r.model AS run_model, r.tier AS run_tier,
                      r.actual_cost_usd AS run_cost, r.provider AS run_provider
               FROM messages m
               LEFT JOIN runs r ON r.message_id = m.id
               WHERE m.conversation_id = ?
               ORDER BY m.created_at ASC""",
        (conversation_id,),
    ).fetchall()
    conv["messages"] = [_row_to_dict(m) for m in messages]
    return conv


# ── Messages ──────────────────────────────────────────────────────────────────

class MessageCreate(BaseModel):
    role: str
    content: str


@projects_router.post("/conversations/{conversation_id}/messages", status_code=201)
def append_message(conversation_id: str, body: MessageCreate, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        conv = repos["conversations"].get_owned(
            conversation_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if conv is None:
            raise HTTPException(status_code=404, detail="Conversation not found")
        if body.role not in ("user", "assistant", "system"):
            raise HTTPException(status_code=422, detail="role must be user, assistant, or system")
        now = _now()
        message = repos["messages"].insert_owned(
            {
                "id": _new_id(),
                "conversation_id": conversation_id,
                "role": body.role,
                "content": body.content,
                "created_at": now,
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        repos["conversations"].update_owned(
            conversation_id,
            {"updated_at": now},
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        return message
    db = get_db()
    row = db.execute(
        "SELECT id FROM conversations WHERE id = ? AND user_id IS ?",
        (conversation_id, user_scope_id(request)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    if body.role not in ("user", "assistant", "system"):
        raise HTTPException(status_code=422, detail="role must be user, assistant, or system")

    msg_id = _new_id()
    now = _now()
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, body.role, body.content, now),
    )
    db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    db.commit()
    row = db.execute("SELECT * FROM messages WHERE id = ?", (msg_id,)).fetchone()
    return _row_to_dict(row)


# ── Files ─────────────────────────────────────────────────────────────────────

@projects_router.get("/projects/{project_id}/files")
def list_files(project_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return repos["files"].client.list_rows(
            "project_files",
            filters={
                "project_id": f"eq.{project_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            select="id,project_id,filename,original_name,mime_type,size_bytes,created_at",
            order="created_at.desc",
        )
    db = get_db()
    _require_project(db, project_id, user_scope_id(request))
    rows = db.execute(
        "SELECT id, project_id, filename, original_name, mime_type, size_bytes, created_at "
        "FROM project_files WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@projects_router.post("/projects/{project_id}/files", status_code=201)
async def upload_file(project_id: str, request: Request, file: UploadFile = FastAPIFile(...)):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=415,
                detail=f"Unsupported file type: '{suffix}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
            )
        content = await file.read()
        size_bytes = len(content)
        if suffix in IMAGE_EXTENSIONS and size_bytes > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the hosted upload limit.")
        file_id = _new_id()
        stored_name = f"{file_id}{suffix}"
        extracted_text = _extract_text(content, suffix)
        object_path = repos["files"].client.upload_object(
            organization_id=context.organization_id,
            object_id=file_id,
            filename=stored_name,
            content=content,
            content_type=file.content_type,
        )
        try:
            row = repos["files"].insert_owned(
                {
                    "id": file_id,
                    "project_id": project_id,
                    "storage_path": object_path,
                    "filename": stored_name,
                    "original_name": file.filename,
                    "mime_type": file.content_type,
                    "size_bytes": size_bytes,
                    "extracted_text": extracted_text,
                    "created_at": _now(),
                },
                organization_id=context.organization_id,
                user_id=context.user_id,
            )
        except Exception:
            try:
                repos["files"].client.delete_object(object_path)
            except Exception:
                pass
            raise
        return {key: row.get(key) for key in (
            "id", "project_id", "filename", "original_name", "mime_type", "size_bytes", "created_at"
        )}
    db = get_db()
    scope = user_scope_id(request)
    _require_project(db, project_id, scope)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    content = await file.read()
    size_bytes = len(content)
    if suffix in IMAGE_EXTENSIONS and size_bytes > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image too large: {size_bytes / 1_048_576:.1f} MB. "
                f"Maximum is {MAX_IMAGE_BYTES // 1_048_576} MB."
            ),
        )

    # Save file to disk
    upload_dir = get_upload_base() / project_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = _new_id()
    stored_name = f"{file_id}{suffix}"
    dest = upload_dir / stored_name
    dest.write_bytes(content)

    # Extract text (None for images — they travel as image blocks, not prose)
    extracted_text = _extract_text(content, suffix)

    now = _now()
    db.execute(
        "INSERT INTO project_files "
        "(id, project_id, filename, original_name, mime_type, size_bytes, extracted_text, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, project_id, stored_name, file.filename, file.content_type,
         size_bytes, extracted_text, now, scope),
    )
    db.commit()

    row = db.execute(
        "SELECT id, project_id, filename, original_name, mime_type, size_bytes, created_at "
        "FROM project_files WHERE id = ?",
        (file_id,),
    ).fetchone()
    return _row_to_dict(row)


@projects_router.post("/files", status_code=201)
async def upload_standalone_file(request: Request, file: UploadFile = FastAPIFile(...)):
    """Upload a file with no parent project, for attachments in quick chat."""
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=415, detail="Unsupported hosted file type.")
        content = await file.read()
        size_bytes = len(content)
        if suffix in IMAGE_EXTENSIONS and size_bytes > MAX_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail="Image exceeds the hosted upload limit.")
        file_id = _new_id()
        stored_name = f"{file_id}{suffix}"
        object_path = repos["files"].client.upload_object(
            organization_id=context.organization_id,
            object_id=file_id,
            filename=stored_name,
            content=content,
            content_type=file.content_type,
        )
        try:
            row = repos["files"].insert_owned(
                {
                    "id": file_id,
                    "project_id": None,
                    "storage_path": object_path,
                    "filename": stored_name,
                    "original_name": file.filename,
                    "mime_type": file.content_type,
                    "size_bytes": size_bytes,
                    "extracted_text": _extract_text(content, suffix),
                    "created_at": _now(),
                },
                organization_id=context.organization_id,
                user_id=context.user_id,
            )
        except Exception:
            try:
                repos["files"].client.delete_object(object_path)
            except Exception:
                pass
            raise
        return {key: row.get(key) for key in (
            "id", "project_id", "filename", "original_name", "mime_type", "size_bytes", "created_at"
        )}
    db = get_db()
    scope = user_scope_id(request)

    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported file type: '{suffix}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
            ),
        )

    content = await file.read()
    size_bytes = len(content)
    if suffix in IMAGE_EXTENSIONS and size_bytes > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Image too large: {size_bytes / 1_048_576:.1f} MB. "
                f"Maximum is {MAX_IMAGE_BYTES // 1_048_576} MB."
            ),
        )

    upload_dir = get_upload_base() / "_standalone"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_id = _new_id()
    stored_name = f"{file_id}{suffix}"
    (upload_dir / stored_name).write_bytes(content)

    extracted_text = _extract_text(content, suffix)

    now = _now()
    db.execute(
        "INSERT INTO project_files "
        "(id, project_id, filename, original_name, mime_type, size_bytes, extracted_text, created_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (file_id, None, stored_name, file.filename, file.content_type,
         size_bytes, extracted_text, now, scope),
    )
    db.commit()

    row = db.execute(
        "SELECT id, project_id, filename, original_name, mime_type, size_bytes, created_at "
        "FROM project_files WHERE id = ?",
        (file_id,),
    ).fetchone()
    return _row_to_dict(row)


@projects_router.get("/files/{file_id}")
def get_file(file_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        row = repos["files"].get_owned(
            file_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail="File not found")
        return row
    db = get_db()
    row = db.execute(
        "SELECT * FROM project_files WHERE id = ? AND user_id IS ?",
        (file_id, user_scope_id(request)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")
    return _row_to_dict(row)


@projects_router.delete("/files/{file_id}", status_code=204)
def delete_file(file_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        row = repos["files"].get_owned(
            file_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail="File not found")
        path = row.get("storage_path")
        if path:
            repos["files"].client.delete_object(path)
        repos["files"].delete_owned(
            file_id, organization_id=context.organization_id, user_id=context.user_id
        )
        return None
    db = get_db()
    row = db.execute(
        "SELECT id, project_id, filename FROM project_files WHERE id = ? AND user_id IS ?",
        (file_id, user_scope_id(request)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="File not found")

    db.execute("DELETE FROM project_files WHERE id = ?", (file_id,))
    db.commit()

    # Remove physical file
    file_path = get_upload_base() / (row["project_id"] or "_standalone") / row["filename"]
    if file_path.exists():
        file_path.unlink(missing_ok=True)

    return None


# ── Context builder ───────────────────────────────────────────────────────────

class ContextBuildRequest(BaseModel):
    project_id: Optional[str] = None
    conversation_id: Optional[str] = None
    file_ids: list[str] = []
    max_context_tokens: int = 4096
    recent_message_limit: int = 20


@projects_router.post("/context/build")
def build_project_context(body: ContextBuildRequest, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        instructions: str | None = None
        if body.project_id:
            project = repos["projects"].get_owned(
                body.project_id, organization_id=context.organization_id, user_id=context.user_id
            )
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")
            instructions = project.get("instructions")
        messages: list[dict] = []
        if body.conversation_id:
            conversation = repos["conversations"].get_owned(
                body.conversation_id,
                organization_id=context.organization_id,
                user_id=context.user_id,
            )
            if conversation is None:
                raise HTTPException(status_code=404, detail="Conversation not found")
            rows = repos["messages"].client.list_rows(
                "messages",
                filters={
                    "conversation_id": f"eq.{body.conversation_id}",
                    "organization_id": f"eq.{context.organization_id}",
                    "user_id": f"eq.{context.user_id}",
                },
                select="role,content",
                order="created_at.asc",
            )
            messages = rows[-body.recent_message_limit:]
        file_texts: list[tuple[str, str]] = []
        for file_id in body.file_ids:
            row = repos["files"].get_owned(
                file_id, organization_id=context.organization_id, user_id=context.user_id
            )
            if row and row.get("extracted_text"):
                file_texts.append((str(row.get("original_name") or row.get("filename")), row["extracted_text"]))
        ctx = build_context(
            instructions=instructions,
            messages=messages,
            file_texts=file_texts,
            max_context_tokens=body.max_context_tokens,
            include_paid_warning=bool(file_texts),
        )
        return {
            "context_text": ctx.text,
            "total_tokens": ctx.total_tokens,
            "budget_tokens": ctx.budget_tokens,
            "truncated": ctx.truncated,
            "truncation_warning": ctx.truncation_warning,
            "paid_provider_warning": ctx.paid_provider_warning,
            "pieces": [{"label": piece.label, "tokens": piece.tokens} for piece in ctx.pieces],
        }
    db = get_db()
    scope = user_scope_id(request)

    instructions: str | None = None
    if body.project_id:
        proj = db.execute(
            "SELECT instructions FROM projects WHERE id = ? AND user_id IS ?",
            (body.project_id, scope),
        ).fetchone()
        if proj is None:
            raise HTTPException(status_code=404, detail="Project not found")
        instructions = proj["instructions"]

    messages: list[dict] = []
    if body.conversation_id:
        rows = db.execute(
            "SELECT role, content FROM messages "
            "WHERE conversation_id = ? AND conversation_id IN "
            "(SELECT id FROM conversations WHERE user_id IS ?) "
            "ORDER BY created_at ASC",
            (body.conversation_id, scope),
        ).fetchall()
        all_msgs = [{"role": r["role"], "content": r["content"]} for r in rows]
        # Take most recent N
        messages = all_msgs[-body.recent_message_limit:]

    file_texts: list[tuple[str, str]] = []
    if body.file_ids:
        for fid in body.file_ids:
            row = db.execute(
                "SELECT original_name, extracted_text FROM project_files "
                "WHERE id = ? AND user_id IS ?",
                (fid, scope),
            ).fetchone()
            if row and row["extracted_text"]:
                file_texts.append((row["original_name"], row["extracted_text"]))

    ctx = build_context(
        instructions=instructions,
        messages=messages,
        file_texts=file_texts,
        max_context_tokens=body.max_context_tokens,
        include_paid_warning=bool(file_texts),
    )

    return {
        "context_text": ctx.text,
        "total_tokens": ctx.total_tokens,
        "budget_tokens": ctx.budget_tokens,
        "truncated": ctx.truncated,
        "truncation_warning": ctx.truncation_warning,
        "paid_provider_warning": ctx.paid_provider_warning,
        "pieces": [
            {"label": p.label, "tokens": p.tokens}
            for p in ctx.pieces
        ],
    }


# ── Project settings ──────────────────────────────────────────────────────────

class ProjectSettingsUpdate(BaseModel):
    """Payload for PUT /api/projects/{project_id}/settings."""
    allowed_model_patterns: Optional[list[str]] = None
    blocked_model_patterns: Optional[list[str]] = None
    max_cost_per_call_usd: Optional[float] = None
    allow_premium: Optional[bool] = None
    local_only: Optional[bool] = None
    enable_openrouter_auto_fallback: Optional[bool] = None


@projects_router.get("/projects/{project_id}/settings")
def get_project_settings(project_id: str, request: Request = None):
    """
    Return per-project model filter and budget settings.

    Returns HTTP 200 with empty/default values when no settings row exists yet.
    """
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        row = repos["project_settings"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if row is None:
            return {
                "project_id": project_id,
                "allowed_model_patterns": [],
                "blocked_model_patterns": [],
                "max_cost_per_call_usd": None,
                "allow_premium": True,
                "local_only": False,
                "enable_openrouter_auto_fallback": False,
                "updated_at": None,
            }
        d = dict(row)
        for key in ("allowed_model_patterns", "blocked_model_patterns"):
            if isinstance(d.get(key), str):
                d[key] = json.loads(d[key] or "[]")
            d[key] = d.get(key) or []
        for key in ("allow_premium", "local_only", "enable_openrouter_auto_fallback"):
            d[key] = bool(d.get(key))
        return d

    db = get_db()
    _require_project(db, project_id, user_scope_id(request) if request is not None else None)
    row = db.execute(
        "SELECT * FROM project_settings WHERE project_id = ?", (project_id,)
    ).fetchone()
    if row is None:
        return {
            "project_id": project_id,
            "allowed_model_patterns": [],
            "blocked_model_patterns": [],
            "max_cost_per_call_usd": None,
            "allow_premium": True,
            "local_only": False,
            "enable_openrouter_auto_fallback": False,
            "updated_at": None,
        }
    d = _row_to_dict(row)
    d["allowed_model_patterns"] = json.loads(d["allowed_model_patterns"] or "[]")
    d["blocked_model_patterns"] = json.loads(d["blocked_model_patterns"] or "[]")
    d["allow_premium"] = bool(d["allow_premium"])
    d["local_only"] = bool(d["local_only"])
    d["enable_openrouter_auto_fallback"] = bool(d["enable_openrouter_auto_fallback"])
    return d


@projects_router.put("/projects/{project_id}/settings")
def update_project_settings(project_id: str, body: ProjectSettingsUpdate, request: Request):
    """
    Create or replace per-project model filter and budget settings.

    Only fields explicitly set in the request body are updated; omitted fields
    retain their current value (PATCH semantics despite using PUT verb, to keep
    the endpoint idempotent and safe for first-time creation).
    """
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        existing = repos["project_settings"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        now = _now()
        if existing is None:
            values = {
                "project_id": project_id,
                "allowed_model_patterns": body.allowed_model_patterns or [],
                "blocked_model_patterns": body.blocked_model_patterns or [],
                "max_cost_per_call_usd": body.max_cost_per_call_usd,
                "allow_premium": body.allow_premium is not False,
                "local_only": bool(body.local_only),
                "enable_openrouter_auto_fallback": bool(body.enable_openrouter_auto_fallback),
                "updated_at": now,
            }
            repos["project_settings"].insert_owned(
                values, organization_id=context.organization_id, user_id=context.user_id
            )
        else:
            def _value(incoming, default):
                return default if incoming is None else incoming

            def _json_value(incoming, default):
                return default if incoming is None else incoming

            previous = dict(existing)
            values = {
                "allowed_model_patterns": _json_value(
                    body.allowed_model_patterns,
                    previous.get("allowed_model_patterns") or [],
                ),
                "blocked_model_patterns": _json_value(
                    body.blocked_model_patterns,
                    previous.get("blocked_model_patterns") or [],
                ),
                "max_cost_per_call_usd": _value(
                    body.max_cost_per_call_usd,
                    previous.get("max_cost_per_call_usd"),
                ),
                "allow_premium": _value(
                    body.allow_premium,
                    bool(previous.get("allow_premium", True)),
                ),
                "local_only": _value(
                    body.local_only,
                    bool(previous.get("local_only", False)),
                ),
                "enable_openrouter_auto_fallback": _value(
                    body.enable_openrouter_auto_fallback,
                    bool(previous.get("enable_openrouter_auto_fallback", False)),
                ),
                "updated_at": now,
            }
            repos["project_settings"].update_owned(
                project_id, values,
                organization_id=context.organization_id, user_id=context.user_id,
            )
        return get_project_settings(project_id, request)

    db = get_db()
    _require_project(db, project_id, user_scope_id(request))

    existing = db.execute(
        "SELECT * FROM project_settings WHERE project_id = ?", (project_id,)
    ).fetchone()

    now = _now()
    if existing is None:
        # First write: insert with defaults for unset fields
        db.execute(
            """INSERT INTO project_settings (
                project_id, allowed_model_patterns, blocked_model_patterns,
                max_cost_per_call_usd, allow_premium, local_only,
                enable_openrouter_auto_fallback, updated_at
            ) VALUES (?,?,?,?,?,?,?,?)""",
            (
                project_id,
                json.dumps(body.allowed_model_patterns or []),
                json.dumps(body.blocked_model_patterns or []),
                body.max_cost_per_call_usd,
                1 if (body.allow_premium is not False) else 0,
                1 if body.local_only else 0,
                1 if body.enable_openrouter_auto_fallback else 0,
                now,
            ),
        )
    else:
        # Partial update: merge with existing values
        prev = _row_to_dict(existing)
        allowed = (
            json.dumps(body.allowed_model_patterns)
            if body.allowed_model_patterns is not None
            else prev["allowed_model_patterns"]
        )
        blocked = (
            json.dumps(body.blocked_model_patterns)
            if body.blocked_model_patterns is not None
            else prev["blocked_model_patterns"]
        )
        allow_premium = (
            (1 if body.allow_premium else 0)
            if body.allow_premium is not None
            else prev["allow_premium"]
        )
        local_only = (
            (1 if body.local_only else 0)
            if body.local_only is not None
            else prev["local_only"]
        )
        auto_fallback = (
            (1 if body.enable_openrouter_auto_fallback else 0)
            if body.enable_openrouter_auto_fallback is not None
            else prev["enable_openrouter_auto_fallback"]
        )
        max_cost = (
            body.max_cost_per_call_usd
            if body.max_cost_per_call_usd is not None
            else prev["max_cost_per_call_usd"]
        )
        db.execute(
            """UPDATE project_settings SET
                allowed_model_patterns = ?,
                blocked_model_patterns = ?,
                max_cost_per_call_usd = ?,
                allow_premium = ?,
                local_only = ?,
                enable_openrouter_auto_fallback = ?,
                updated_at = ?
            WHERE project_id = ?""",
            (allowed, blocked, max_cost, allow_premium, local_only, auto_fallback, now, project_id),
        )
    db.commit()
    return get_project_settings(project_id, request)


# ── Runs ──────────────────────────────────────────────────────────────────────

@projects_router.get("/projects/{project_id}/runs")
def list_runs(project_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        project = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return repos["runs"].client.list_rows(
            "runs",
            filters={
                "project_id": f"eq.{project_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            order="created_at.desc",
        )
    db = get_db()
    _require_project(db, project_id, user_scope_id(request))
    rows = db.execute(
        "SELECT * FROM runs WHERE project_id = ? ORDER BY created_at DESC",
        (project_id,),
    ).fetchall()
    return [_row_to_dict(r) for r in rows]


@projects_router.get("/runs/{run_id}")
def get_run(run_id: str, request: Request):
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        row = repos["runs"].get_owned(
            run_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if row is None:
            raise HTTPException(status_code=404, detail="Run not found")
        return row
    db = get_db()
    row = db.execute(
        "SELECT * FROM runs WHERE id = ? AND user_id IS ?",
        (run_id, user_scope_id(request)),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Run not found")
    return _row_to_dict(row)


# ── Standalone conversation (no project required) ─────────────────────────────

@projects_router.post("/conversations", status_code=201)
def create_standalone_conversation(body: ConversationCreate, request: Request):
    """Create a conversation without a parent project (for quick/anonymous chats)."""
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        now = _now()
        return repos["conversations"].insert_owned(
            {
                "id": _new_id(),
                "project_id": None,
                "title": body.title or "New Conversation",
                "created_at": now,
                "updated_at": now,
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
    db = get_db()
    conv_id = _new_id()
    now = _now()
    db.execute(
        "INSERT INTO conversations (id, project_id, title, created_at, updated_at, user_id) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (conv_id, None, body.title or "New Conversation", now, now,
         user_scope_id(request)),
    )
    db.commit()
    row = db.execute("SELECT * FROM conversations WHERE id = ?", (conv_id,)).fetchone()
    return _row_to_dict(row)


# ── Recents feed ──────────────────────────────────────────────────────────────

@projects_router.get("/recents")
def get_recents(request: Request, limit: int = 20):
    """
    Return recently touched conversations, projects, files, and runs,
    merged and sorted by updated_at descending. Scoped to the current user.
    """
    hosted = _hosted_scope(request)
    if hosted:
        context, repos = hosted
        projects = repos["projects"].list_for_owner(
            organization_id=context.organization_id, user_id=context.user_id, limit=limit,
        )
        conversations = repos["conversations"].list_for_owner(
            organization_id=context.organization_id, user_id=context.user_id, limit=limit,
        )
        files = repos["files"].list_for_owner(
            organization_id=context.organization_id, user_id=context.user_id, limit=limit,
        )
        runs = repos["runs"].list_for_owner(
            organization_id=context.organization_id, user_id=context.user_id, limit=limit,
        )
        project_names = {item.get("id"): item.get("name") for item in projects}
        conversation_summaries = _hosted_message_summaries(
            repos, context, [str(row.get("id")) for row in conversations if row.get("id")]
        )
        items: list[dict] = []
        for row in conversations:
            snippet, first_message, message_count = _preview_item(
                conversation_summaries.get(str(row.get("id")))
            )
            items.append({
                "id": row.get("id"), "type": "conversation", "title": row.get("title") or "Untitled Conversation",
                "project_id": row.get("project_id"), "project_name": project_names.get(row.get("project_id")),
                "updated_at": row.get("updated_at"), "snippet": snippet,
                "first_message": first_message,
                "metadata": {"message_count": message_count},
            })
        for row in projects:
            items.append({
                "id": row.get("id"), "type": "project", "title": row.get("name"),
                "project_id": row.get("id"), "project_name": row.get("name"),
                "updated_at": row.get("updated_at"), "metadata": {},
            })
        for row in files:
            items.append({
                "id": row.get("id"), "type": "file", "title": row.get("original_name"),
                "project_id": row.get("project_id"), "project_name": project_names.get(row.get("project_id")),
                "updated_at": row.get("created_at"),
                "metadata": {"size_bytes": row.get("size_bytes"), "mime_type": row.get("mime_type")},
            })
        for row in runs:
            items.append({
                "id": row.get("id"), "type": "run", "title": f"Run · {row.get('model') or 'unknown'}",
                "project_id": row.get("project_id"), "project_name": project_names.get(row.get("project_id")),
                "updated_at": row.get("created_at"),
                "metadata": {key: row.get(key) for key in (
                    "model", "tier", "actual_cost_usd", "total_latency_ms", "status"
                )},
            })
        items.sort(key=lambda item: item.get("updated_at") or "", reverse=True)
        return {"items": items[:limit]}

    db = get_db()
    scope = user_scope_id(request)
    items: list[dict] = []

    # Recent conversations
    rows = db.execute("""
        SELECT c.id, c.title, c.project_id, c.updated_at,
               p.name AS project_name,
               COALESCE(
                   (SELECT substr(m.content, 1, 140) FROM messages m
                    WHERE m.conversation_id = c.id AND m.role = 'assistant'
                    ORDER BY m.created_at DESC LIMIT 1),
                   (SELECT substr(m.content, 1, 140) FROM messages m
                    WHERE m.conversation_id = c.id AND m.role = 'user'
                    ORDER BY m.created_at DESC LIMIT 1)
               ) AS snippet
        FROM conversations c
        LEFT JOIN projects p ON c.project_id = p.id
        WHERE c.user_id IS ?
        ORDER BY c.updated_at DESC LIMIT ?
    """, (scope, limit)).fetchall()
    for r in rows:
        items.append({
            "id": r["id"],
            "type": "conversation",
            "title": r["title"] or "Untitled Conversation",
            "project_id": r["project_id"],
            "project_name": r["project_name"],
            "updated_at": r["updated_at"],
            "snippet": r["snippet"],
            "metadata": {},
        })

    # Recent projects
    rows = db.execute(
        "SELECT * FROM projects WHERE user_id IS ? ORDER BY updated_at DESC LIMIT ?",
        (scope, limit),
    ).fetchall()
    for r in rows:
        items.append({
            "id": r["id"],
            "type": "project",
            "title": r["name"],
            "project_id": r["id"],
            "project_name": r["name"],
            "updated_at": r["updated_at"],
            "metadata": {},
        })

    # Recent files
    rows = db.execute(
        "SELECT pf.id, pf.original_name, pf.project_id, pf.size_bytes, "
        "pf.mime_type, pf.created_at, p.name AS project_name "
        "FROM project_files pf "
        "LEFT JOIN projects p ON pf.project_id = p.id "
        "WHERE pf.user_id IS ? "
        "ORDER BY pf.created_at DESC LIMIT ?",
        (scope, limit),
    ).fetchall()
    for r in rows:
        items.append({
            "id": r["id"],
            "type": "file",
            "title": r["original_name"],
            "project_id": r["project_id"],
            "project_name": r["project_name"],
            "updated_at": r["created_at"],
            "metadata": {"size_bytes": r["size_bytes"], "mime_type": r["mime_type"]},
        })

    # Recent runs
    rows = db.execute(
        "SELECT r.id, r.project_id, r.model, r.tier, r.actual_cost_usd, "
        "r.total_latency_ms, r.status, r.created_at, p.name AS project_name "
        "FROM runs r "
        "LEFT JOIN projects p ON r.project_id = p.id "
        "WHERE r.user_id IS ? "
        "ORDER BY r.created_at DESC LIMIT ?",
        (scope, limit),
    ).fetchall()
    for r in rows:
        items.append({
            "id": r["id"],
            "type": "run",
            "title": f"Run · {r['model'] or 'unknown'}",
            "project_id": r["project_id"],
            "project_name": r["project_name"],
            "updated_at": r["created_at"],
            "metadata": {
                "model": r["model"],
                "tier": r["tier"],
                "actual_cost_usd": r["actual_cost_usd"],
                "total_latency_ms": r["total_latency_ms"],
                "status": r["status"],
            },
        })

    # Merge-sort by updated_at descending, return top N
    items.sort(key=lambda x: x.get("updated_at") or "", reverse=True)
    return {"items": items[:limit]}


# ── Internal helpers (used by execute endpoint in server.py) ──────────────────

def save_run(
    *,
    project_id: str | None,
    conversation_id: str | None,
    model: str | None,
    tier: str | None,
    provider: str | None,
    policy: str | None,
    input_tokens: int | None,
    output_tokens: int | None,
    actual_cost_usd: float | None,
    routing_latency_ms: float | None,
    provider_latency_ms: float | None,
    total_latency_ms: float | None,
    fallback_used: bool,
    status: str,
    error: str | None = None,
    message_id: str | None = None,
    # Extended routing metadata (added for feature: actual selected model metadata)
    routing_source: str | None = None,
    routing_reason: str | None = None,
    sticky_route_used: bool = False,
    cost_quality_tradeoff: int | None = None,
    fallback_chain: list[str] | None = None,
    user_id: str | None = None,
    routing_observations: list[dict] | None = None,
) -> str:
    """Insert a run record and return its id."""
    hosted = _hosted_scope_from_current(user_id)
    if hosted:
        context, repos = hosted
        run_id = _new_id()
        row = repos["runs"].insert_owned(
            {
                "id": run_id,
                "project_id": project_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "model": model,
                "tier": tier,
                "provider": provider,
                "policy": policy,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "actual_cost_usd": actual_cost_usd,
                "routing_latency_ms": routing_latency_ms,
                "provider_latency_ms": provider_latency_ms,
                "total_latency_ms": total_latency_ms,
                "fallback_used": bool(fallback_used),
                "status": status,
                "error": error,
                "created_at": _now(),
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        return str(row.get("id") or run_id)
    db = get_db()
    run_id = _new_id()
    now = _now()
    db.execute(
        """INSERT INTO runs (
            id, project_id, conversation_id, message_id,
            model, tier, provider, policy,
            input_tokens, output_tokens, actual_cost_usd,
            routing_latency_ms, provider_latency_ms, total_latency_ms,
            fallback_used, status, error, created_at,
            routing_source, routing_reason, sticky_route_used,
            cost_quality_tradeoff, fallback_chain, user_id
        ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            run_id, project_id, conversation_id, message_id,
            model, tier, provider, policy,
            input_tokens, output_tokens, actual_cost_usd,
            routing_latency_ms, provider_latency_ms, total_latency_ms,
            1 if fallback_used else 0, status, error, now,
            routing_source, routing_reason,
            1 if sticky_route_used else 0,
            cost_quality_tradeoff,
            json.dumps(fallback_chain) if fallback_chain is not None else None,
            user_id,
        ),
    )
    if routing_observations:
        _save_routing_observations(
            db,
            run_id=run_id,
            user_id=user_id,
            rows=routing_observations,
            created_at=now,
        )
    db.commit()
    if routing_observations:
        from theriad.routing.observation_store import mirror_to_supabase

        if not mirror_to_supabase(routing_observations, run_id=run_id, principal_id=user_id):
            logger.debug(
                "routing observations retained locally; shared mirror unavailable",
                extra={"run_id": run_id},
            )
    return run_id


def _finite(value):
    if value is None:
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _save_routing_observations(
    db,
    *,
    run_id: str | None,
    user_id: str | None,
    rows: list[dict],
    created_at: str,
) -> None:
    """Persist normalized router observations in the caller's transaction."""

    required = {"decision_id", "prompt_hash", "router_version", "lane", "model", "role"}
    for row in rows:
        missing = required - row.keys()
        if missing:
            raise ValueError(f"routing observation missing {sorted(missing)}")
        db.execute(
            """INSERT INTO routing_observations (
                id, decision_id, run_id, user_id, prompt_hash, prompt_embedding,
                router_version, lane, model, role, propensity,
                p_success, p_success_lower, expected_output_tokens,
                expected_cost_usd, expected_latency_ms, uncertainty,
                eligible, exclusion_reason, realized_success, realized_score,
                realized_cost_usd, realized_latency_ms, input_tokens,
                output_tokens, evaluator, evaluator_version, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                _new_id(), row["decision_id"], run_id, user_id, row["prompt_hash"],
                json.dumps(row.get("prompt_embedding")) if row.get("prompt_embedding") is not None else None,
                row["router_version"], row["lane"], row["model"], row["role"],
                _finite(row.get("propensity")), _finite(row.get("p_success")),
                _finite(row.get("p_success_lower")), _finite(row.get("expected_output_tokens")),
                _finite(row.get("expected_cost_usd")), _finite(row.get("expected_latency_ms")),
                _finite(row.get("uncertainty")), 1 if row.get("eligible", True) else 0,
                row.get("exclusion_reason"), _finite(row.get("realized_success")),
                _finite(row.get("realized_score")), _finite(row.get("realized_cost_usd")),
                _finite(row.get("realized_latency_ms")), row.get("input_tokens"),
                row.get("output_tokens"), row.get("evaluator"),
                row.get("evaluator_version"), created_at,
            ),
        )


def save_message(
    *,
    conversation_id: str,
    role: str,
    content: str,
) -> str:
    """Insert a message and update conversation.updated_at. Returns message id."""
    from theriad.api.auth_context import current_auth_context

    context = current_auth_context.get()
    hosted = _hosted_scope_from_current(context.user_id if context else None)
    if hosted:
        context, repos = hosted
        msg_id = _new_id()
        now = _now()
        repos["messages"].insert_owned(
            {
                "id": msg_id,
                "conversation_id": conversation_id,
                "role": role,
                "content": content,
                "created_at": now,
            },
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        repos["conversations"].update_owned(
            conversation_id,
            {"updated_at": now},
            organization_id=context.organization_id,
            user_id=context.user_id,
        )
        return msg_id
    db = get_db()
    msg_id = _new_id()
    now = _now()
    db.execute(
        "INSERT INTO messages (id, conversation_id, role, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (msg_id, conversation_id, role, content, now),
    )
    db.execute(
        "UPDATE conversations SET updated_at = ? WHERE id = ?",
        (now, conversation_id),
    )
    db.commit()
    return msg_id


def get_file_texts(
    file_ids: list[str], scope_id: str | None = None
) -> list[tuple[str, str]]:
    """Return [(original_name, extracted_text), ...] for the given file ids."""
    if not file_ids:
        return []
    hosted = _hosted_scope_from_current(scope_id)
    if hosted:
        context, repos = hosted
        result: list[tuple[str, str]] = []
        for file_id in file_ids:
            row = repos["files"].get_owned(
                file_id, organization_id=context.organization_id, user_id=context.user_id
            )
            if row and row.get("extracted_text"):
                result.append((str(row.get("original_name") or row.get("filename")), row["extracted_text"]))
        return result
    db = get_db()
    scope_clause, scope_values = _current_row_scope(scope_id)
    result: list[tuple[str, str]] = []
    for fid in file_ids:
        row = db.execute(
            f"SELECT original_name, extracted_text FROM project_files "
            f"WHERE id = ? AND {scope_clause}",
            (fid, *scope_values),
        ).fetchone()
        if row and row["extracted_text"]:
            result.append((row["original_name"], row["extracted_text"]))
    return result


def get_file_images(
    file_ids: list[str], scope_id: str | None = None
) -> list[tuple[str, str]]:
    """Return [(original_name, data_uri), ...] for image attachments.

    Non-image attachments are skipped — they are handled by
    :func:`get_file_texts`. Reads the bytes back off disk rather than storing a
    base64 copy in the database, so an image costs one row and one file.

    A file whose row exists but whose bytes are missing is skipped with a
    warning rather than raising: one unreadable attachment should not fail the
    whole message.
    """
    if not file_ids:
        return []
    import base64

    hosted = _hosted_scope_from_current(scope_id)
    if hosted:
        context, repos = hosted
        result: list[tuple[str, str]] = []
        for file_id in file_ids:
            row = repos["files"].get_owned(
                file_id, organization_id=context.organization_id, user_id=context.user_id
            )
            if not row or not is_image(row.get("filename")) or not row.get("storage_path"):
                continue
            try:
                raw = repos["files"].client.download_object(row["storage_path"])
            except HostedStorageUnavailable:
                logger.warning("hosted image attachment unreadable", extra={"file_id": file_id})
                continue
            suffix = Path(row["filename"]).suffix.lower()
            mime = IMAGE_MIME_TYPES.get(suffix, "application/octet-stream")
            result.append((row.get("original_name") or row["filename"], f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"))
        return result

    db = get_db()
    scope_clause, scope_values = _current_row_scope(scope_id)
    result: list[tuple[str, str]] = []
    for fid in file_ids:
        row = db.execute(
            f"SELECT project_id, filename, original_name FROM project_files "
            f"WHERE id = ? AND {scope_clause}",
            (fid, *scope_values),
        ).fetchone()
        if not row or not is_image(row["filename"]):
            continue
        suffix = Path(row["filename"]).suffix.lower()
        directory = row["project_id"] or "_standalone"
        path = get_upload_base() / directory / row["filename"]
        try:
            raw = path.read_bytes()
        except OSError:
            logger.warning("image attachment unreadable", extra={"file_id": fid})
            continue
        mime = IMAGE_MIME_TYPES.get(suffix, "application/octet-stream")
        encoded = base64.b64encode(raw).decode("ascii")
        result.append((row["original_name"], f"data:{mime};base64,{encoded}"))
    return result


def get_recent_messages(
    conversation_id: str, limit: int = 20, scope_id: str | None = None
) -> list[dict]:
    hosted = _hosted_scope_from_current(scope_id)
    if hosted:
        context, repos = hosted
        conversation = repos["conversations"].get_owned(
            conversation_id, organization_id=context.organization_id, user_id=context.user_id
        )
        if not conversation:
            return []
        rows = repos["messages"].client.list_rows(
            "messages",
            filters={
                "conversation_id": f"eq.{conversation_id}",
                "organization_id": f"eq.{context.organization_id}",
                "user_id": f"eq.{context.user_id}",
            },
            select="role,content",
            order="created_at.asc",
        )
        return rows[-limit:]
    db = get_db()
    scope_clause, scope_values = _current_row_scope(scope_id)
    rows = db.execute(
        f"SELECT m.role, m.content FROM messages m "
        f"JOIN conversations c ON c.id = m.conversation_id "
        f"WHERE m.conversation_id = ? AND c.{scope_clause} ORDER BY m.created_at ASC",
        (conversation_id, *scope_values),
    ).fetchall()
    msgs = [{"role": r["role"], "content": r["content"]} for r in rows]
    return msgs[-limit:]


def get_project_instructions(project_id: str, scope_id: str | None = None) -> str | None:
    hosted = _hosted_scope_from_current(scope_id)
    if hosted:
        context, repos = hosted
        row = repos["projects"].get_owned(
            project_id, organization_id=context.organization_id, user_id=context.user_id
        )
        return row.get("instructions") if row else None
    db = get_db()
    scope_clause, scope_values = _current_row_scope(scope_id)
    row = db.execute(
        f"SELECT instructions FROM projects WHERE id = ? AND {scope_clause}",
        (project_id, *scope_values),
    ).fetchone()
    return row["instructions"] if row else None


def load_project_settings(project_id: str, scope_id: str | None = None) -> dict | None:
    """
    Return project settings dict or None if no settings row exists.

    Returned dict has Python-typed values (lists, bools, float|None).
    This function is safe to call without an active request (no HTTPException).
    """
    try:
        if hosted_data_plane_enabled() and scope_id:
            from theriad.api.auth_context import current_auth_context

            context = current_auth_context.get()
            if context and context.principal_type in {"registered", "guest"} and context.user_id == scope_id and context.organization_id:
                repos = hosted_repositories()
                row = repos["project_settings"].get_owned(
                    project_id, organization_id=context.organization_id, user_id=scope_id
                )
                if row is None:
                    return None
                d = dict(row)
                for key in ("allowed_model_patterns", "blocked_model_patterns"):
                    if isinstance(d.get(key), str):
                        d[key] = json.loads(d[key] or "[]")
                    d[key] = d.get(key) or []
                for key in ("allow_premium", "local_only", "enable_openrouter_auto_fallback"):
                    d[key] = bool(d.get(key))
                return d
        db = get_db()
        scope_clause, scope_values = _current_row_scope(scope_id)
        row = db.execute(
            f"SELECT ps.* FROM project_settings ps "
            f"JOIN projects p ON p.id = ps.project_id "
            f"WHERE ps.project_id = ? AND p.{scope_clause}",
            (project_id, *scope_values),
        ).fetchone()
        if row is None:
            return None
        d = _row_to_dict(row)
        d["allowed_model_patterns"] = json.loads(d["allowed_model_patterns"] or "[]")
        d["blocked_model_patterns"] = json.loads(d["blocked_model_patterns"] or "[]")
        d["allow_premium"] = bool(d["allow_premium"])
        d["local_only"] = bool(d["local_only"])
        d["enable_openrouter_auto_fallback"] = bool(d["enable_openrouter_auto_fallback"])
        return d
    except Exception:
        return None


# ── Private ───────────────────────────────────────────────────────────────────

def _require_project(db, project_id: str, scope: str | None = None) -> None:
    row = db.execute(
        "SELECT id FROM projects WHERE id = ? AND user_id IS ?",
        (project_id, scope),
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Project not found")
