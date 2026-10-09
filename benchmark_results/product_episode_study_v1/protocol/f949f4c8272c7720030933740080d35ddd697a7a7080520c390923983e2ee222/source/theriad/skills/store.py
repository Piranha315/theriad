"""SQLite-backed CRUD and scope resolution for Theriad skills."""

from __future__ import annotations

import hashlib
import sqlite3
import unicodedata
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

from theriad.storage.database import get_db

MAX_NAME_CHARS = 80
MAX_DESCRIPTION_CHARS = 500
MAX_INSTRUCTIONS_CHARS = 40_000
MIN_PRIORITY = -10_000
MAX_PRIORITY = 10_000

_UNSET = object()


def _current_scope_id() -> str | None:
    """Read the request account scope without importing auth at module load."""
    try:
        from theriad.api.auth import current_scope_id
        return current_scope_id()
    except ImportError:
        return None


class SkillValidationError(ValueError):
    """A skill field or scope is invalid."""


class SkillNotFound(LookupError):
    """A requested skill does not exist."""


class SkillConflict(ValueError):
    """An equivalent skill already exists in the requested scope."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _reject_unsafe_unicode(value: str, field_name: str) -> None:
    for char in value:
        category = unicodedata.category(char)
        if category == "Cf" or (category == "Cc" and char not in {"\n", "\t"}):
            raise SkillValidationError(
                f"{field_name} contains unsupported control or formatting characters"
            )


def normalize_name(value: str) -> str:
    name = unicodedata.normalize("NFC", str(value or "")).strip()
    _reject_unsafe_unicode(name, "name")
    if not name:
        raise SkillValidationError("skill name must not be empty")
    if "\n" in name or "\t" in name:
        raise SkillValidationError("skill name must be a single line")
    if len(name) > MAX_NAME_CHARS:
        raise SkillValidationError(f"skill name must be at most {MAX_NAME_CHARS} characters")
    return name


def normalize_description(value: str | None) -> str:
    description = unicodedata.normalize("NFC", str(value or "")).strip()
    _reject_unsafe_unicode(description, "description")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise SkillValidationError(
            f"skill description must be at most {MAX_DESCRIPTION_CHARS} characters"
        )
    return description


def normalize_instructions(value: str) -> str:
    instructions = str(value or "").removeprefix("\ufeff")
    instructions = instructions.replace("\r\n", "\n").replace("\r", "\n")
    instructions = unicodedata.normalize("NFC", instructions).strip()
    _reject_unsafe_unicode(instructions, "instructions")
    if not instructions:
        raise SkillValidationError("skill instructions must not be empty")
    if len(instructions) > MAX_INSTRUCTIONS_CHARS:
        raise SkillValidationError(
            f"skill instructions must be at most {MAX_INSTRUCTIONS_CHARS:,} characters"
        )
    return instructions


def normalize_priority(value: int) -> int:
    try:
        priority = int(value)
    except (TypeError, ValueError) as exc:
        raise SkillValidationError("priority must be an integer") from exc
    if not MIN_PRIORITY <= priority <= MAX_PRIORITY:
        raise SkillValidationError(
            f"priority must be between {MIN_PRIORITY:,} and {MAX_PRIORITY:,}"
        )
    return priority


def content_sha256(instructions: str) -> str:
    normalized = normalize_instructions(instructions)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SkillRecord:
    id: str
    project_id: str | None
    name: str
    description: str
    instructions: str
    enabled: bool
    priority: int
    source_type: str
    source_name: str | None
    content_sha256: str
    created_at: str
    updated_at: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _record(row: sqlite3.Row) -> SkillRecord:
    return SkillRecord(
        id=row["id"],
        project_id=row["project_id"],
        name=row["name"],
        description=row["description"] or "",
        instructions=row["instructions"],
        enabled=bool(row["enabled"]),
        priority=int(row["priority"]),
        source_type=row["source_type"],
        source_name=row["source_name"],
        content_sha256=row["content_sha256"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
    )


class SkillStore:
    """CRUD over the shared SQLite database, with optional connection injection."""

    def __init__(
        self,
        connection: sqlite3.Connection | None = None,
        scope_id: str | None = None,
    ):
        self._connection = connection
        self._scope_id = scope_id if scope_id is not None else _current_scope_id()

    def _db(self) -> sqlite3.Connection:
        return self._connection or get_db()

    def _scope_filter(self, column: str = "user_id") -> tuple[str, list[Any]]:
        if self._scope_id:
            return f"({column} IS NULL OR {column} = ?)", [self._scope_id]
        return f"{column} IS NULL", []

    def _owned_scope_filter(self, column: str = "user_id") -> tuple[str, list[Any]]:
        if self._scope_id:
            return f"{column} = ?", [self._scope_id]
        return f"{column} IS NULL", []

    @staticmethod
    def _normalize_project_id(project_id: str | None) -> str | None:
        if project_id is None:
            return None
        normalized = str(project_id).strip()
        return normalized or None

    def _require_project(self, project_id: str | None) -> str | None:
        project_id = self._normalize_project_id(project_id)
        if project_id is None:
            return None
        clauses = ["id = ?"]
        values: list[Any] = [project_id]
        if self._scope_id:
            clauses.append("user_id = ?")
            values.append(self._scope_id)
        row = self._db().execute(
            f"SELECT id FROM projects WHERE {' AND '.join(clauses)}", values
        ).fetchone()
        if row is None:
            raise SkillValidationError(f"project '{project_id}' not found")
        return project_id

    def validate_project_id(self, project_id: str | None) -> str | None:
        """Validate and normalize a global/project scope without changing state."""
        return self._require_project(project_id)

    def list(
        self,
        *,
        project_id: str | None = None,
        effective: bool = False,
        enabled: bool | None = None,
    ) -> list[SkillRecord]:
        from .builtins import ensure_builtin_skills

        ensure_builtin_skills(self._db())
        scope_clause, scope_values = self._scope_filter()
        clauses: list[str] = [scope_clause]
        values: list[Any] = list(scope_values)
        project_id = self._normalize_project_id(project_id)

        if effective:
            if project_id is None:
                clauses.append("project_id IS NULL")
            else:
                clauses.append("(project_id IS NULL OR project_id = ?)")
                values.append(project_id)
        if enabled is not None:
            clauses.append("enabled = ?")
            values.append(1 if enabled else 0)

        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        rows = (
            self._db()
            .execute(
                "SELECT * FROM skills"
                + where
                + " ORDER BY CASE WHEN project_id IS NULL THEN 0 ELSE 1 END, "
                "priority ASC, name COLLATE NOCASE ASC, created_at ASC",
                values,
            )
            .fetchall()
        )
        return [_record(row) for row in rows]

    def resolve_enabled(self, project_id: str | None = None) -> list[SkillRecord]:
        return self.list(project_id=project_id, effective=True, enabled=True)

    def get(self, skill_id: str) -> SkillRecord | None:
        scope_clause, scope_values = self._scope_filter()
        row = self._db().execute(
            f"SELECT * FROM skills WHERE id = ? AND {scope_clause}",
            [skill_id, *scope_values],
        ).fetchone()
        return _record(row) if row else None

    def create(
        self,
        *,
        name: str,
        instructions: str,
        description: str = "",
        project_id: str | None = None,
        enabled: bool = False,
        priority: int = 100,
        source_type: str = "manual",
        source_name: str | None = None,
    ) -> SkillRecord:
        name = normalize_name(name)
        description = normalize_description(description)
        instructions = normalize_instructions(instructions)
        project_id = self._require_project(project_id)
        priority = normalize_priority(priority)
        source_type = str(source_type or "manual").strip().lower()[:40] or "manual"
        source_name = str(source_name).strip()[:500] if source_name else None
        digest = content_sha256(instructions)
        skill_id = str(uuid.uuid4())
        now = _utcnow()

        db = self._db()
        try:
            db.execute(
                """INSERT INTO skills (
                    id, user_id, project_id, name, description, instructions, enabled,
                    priority, source_type, source_name, content_sha256,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    skill_id,
                    self._scope_id,
                    project_id,
                    name,
                    description,
                    instructions,
                    1 if enabled else 0,
                    priority,
                    source_type,
                    source_name,
                    digest,
                    now,
                    now,
                ),
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            if "UNIQUE" in str(exc).upper():
                raise SkillConflict("an equivalent skill already exists in this scope") from exc
            raise SkillValidationError(str(exc)) from exc
        return self.get(skill_id)  # type: ignore[return-value]

    def update(
        self,
        skill_id: str,
        *,
        name: str | None = None,
        description: str | None = None,
        instructions: str | None = None,
        project_id: str | None | object = _UNSET,
        enabled: bool | None = None,
        priority: int | None = None,
    ) -> SkillRecord:
        current = self.get(skill_id)
        if current is None:
            raise SkillNotFound(f"skill '{skill_id}' not found")

        fields: dict[str, Any] = {}
        if name is not None:
            fields["name"] = normalize_name(name)
        if description is not None:
            fields["description"] = normalize_description(description)
        if instructions is not None:
            normalized = normalize_instructions(instructions)
            fields["instructions"] = normalized
            fields["content_sha256"] = content_sha256(normalized)
        if project_id is not _UNSET:
            fields["project_id"] = self._require_project(project_id)  # type: ignore[arg-type]
        if enabled is not None:
            fields["enabled"] = 1 if enabled else 0
        if priority is not None:
            fields["priority"] = normalize_priority(priority)

        if not fields:
            return current
        fields["updated_at"] = _utcnow()
        values = list(fields.values()) + [skill_id]
        assignments = ", ".join(f"{field} = ?" for field in fields)

        db = self._db()
        scope_clause, scope_values = self._owned_scope_filter()
        try:
            updated = db.execute(
                f"UPDATE skills SET {assignments} WHERE id = ? AND {scope_clause}",
                [*values, *scope_values],
            )
            db.commit()
        except sqlite3.IntegrityError as exc:
            db.rollback()
            if "UNIQUE" in str(exc).upper():
                raise SkillConflict("an equivalent skill already exists in this scope") from exc
            raise SkillValidationError(str(exc)) from exc
        if updated.rowcount != 1:
            raise SkillNotFound(f"skill '{skill_id}' not found")
        return self.get(skill_id)  # type: ignore[return-value]

    def delete(self, skill_id: str) -> bool:
        db = self._db()
        scope_clause, scope_values = self._owned_scope_filter()
        cursor = db.execute(
            f"DELETE FROM skills WHERE id = ? AND {scope_clause}",
            [skill_id, *scope_values],
        )
        db.commit()
        return cursor.rowcount > 0


def get_enabled_skills(
    project_id: str | None = None, *, store: SkillStore | None = None
) -> list[SkillRecord]:
    return (store or SkillStore()).resolve_enabled(project_id)
