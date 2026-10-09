"""Synchronize bundled, reviewable skills into the normal skill store."""

from __future__ import annotations

import hashlib
import uuid
from datetime import datetime, timezone
from pathlib import Path

_ROOT = Path(__file__).with_name("builtin")
_BUILTINS = (
    {
        "name": "Caveman",
        "description": "Try an ultra-compressed response style that removes filler while preserving technical details.",
        "path": _ROOT / "caveman" / "SKILL.md",
        "source_name": "https://github.com/JuliusBrussee/caveman@0d95a81d35a9f2d123a5e9430d1cfc43d55f1bb0",
    },
)


def ensure_builtin_skills(db) -> None:
    now = datetime.now(timezone.utc).isoformat()
    for item in _BUILTINS:
        raw = item["path"].read_text(encoding="utf-8").strip()
        instructions = raw.split("---", 2)[-1].strip() if raw.startswith("---") else raw
        digest = hashlib.sha256(instructions.encode("utf-8")).hexdigest()
        row = db.execute(
            "SELECT id, content_sha256 FROM skills WHERE source_type = 'builtin' AND source_name = ?",
            (item["source_name"],),
        ).fetchone()
        if row is None:
            db.execute(
                """INSERT INTO skills (
                    id, project_id, name, description, instructions, enabled,
                    priority, source_type, source_name, content_sha256,
                    created_at, updated_at
                ) VALUES (?, NULL, ?, ?, ?, 0, 50, 'builtin', ?, ?, ?, ?)""",
                (
                    str(uuid.uuid4()),
                    item["name"],
                    item["description"],
                    instructions,
                    item["source_name"],
                    digest,
                    now,
                    now,
                ),
            )
        elif row["content_sha256"] != digest:
            db.execute(
                "UPDATE skills SET instructions = ?, content_sha256 = ?, updated_at = ? WHERE id = ?",
                (instructions, digest, now, row["id"]),
            )
    db.commit()
