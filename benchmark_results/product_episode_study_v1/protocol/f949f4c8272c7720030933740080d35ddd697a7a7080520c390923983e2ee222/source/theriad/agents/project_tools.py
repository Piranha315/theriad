"""Workspace-confined file operations for the unified local agent harness."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


class ProjectToolError(ValueError):
    pass


def resolve_project_path(root: str, value: str = ".") -> Path:
    base = Path(root).expanduser().resolve()
    candidate = (base / value).resolve() if not Path(value).is_absolute() else Path(value).resolve()
    if candidate != base and base not in candidate.parents:
        raise ProjectToolError("Path escapes the selected workspace.")
    return candidate


def read_file(root: str, path: str, *, max_bytes: int = 500_000) -> dict:
    target = resolve_project_path(root, path)
    raw = target.read_bytes()
    if len(raw) > max_bytes:
        raise ProjectToolError(f"File exceeds {max_bytes} byte read limit.")
    return {"path": str(target.relative_to(Path(root).resolve())), "content": raw.decode("utf-8", errors="replace")}


def search(root: str, query: str, *, limit: int = 100) -> dict:
    if not query:
        raise ProjectToolError("query is required")
    matches: list[dict] = []
    skip = {".git", "node_modules", "dist", "build", "__pycache__", ".venv", "venv"}
    sensitive_names = {".env", ".env.local", "credentials", "credentials.json", "secrets.json"}
    for dirpath, dirnames, filenames in os.walk(Path(root).resolve()):
        dirnames[:] = [name for name in dirnames if name not in skip]
        for filename in filenames:
            if filename.lower() in sensitive_names or filename.lower().endswith((".pem", ".key", ".p12")):
                continue
            target = Path(dirpath) / filename
            try:
                for line_no, line in enumerate(target.read_text(errors="ignore").splitlines(), 1):
                    if query in line:
                        matches.append({
                            "path": str(target.relative_to(Path(root).resolve())),
                            "line": line_no,
                            "text": line[:500],
                        })
                        if len(matches) >= limit:
                            return {"matches": matches, "truncated": True}
            except (OSError, UnicodeError):
                continue
    return {"matches": matches, "truncated": False}


def write_file(root: str, path: str, content: str, *, overwrite: bool = False) -> dict:
    target = resolve_project_path(root, path)
    if target.exists() and not overwrite:
        raise ProjectToolError("File exists; set overwrite=true or use project.replace.")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content)
    return {"path": str(target.relative_to(Path(root).resolve())), "bytes_written": len(content.encode())}


def replace_text(root: str, path: str, expected: str, replacement: str, *, count: int = 1) -> dict:
    target = resolve_project_path(root, path)
    content = target.read_text()
    occurrences = content.count(expected)
    if occurrences < count or count < 1:
        raise ProjectToolError(f"Expected text occurrence mismatch: found {occurrences}, required {count}.")
    target.write_text(content.replace(expected, replacement, count))
    return {"path": str(target.relative_to(Path(root).resolve())), "replacements": count}


def move_path(root: str, source: str, destination: str) -> dict:
    src = resolve_project_path(root, source)
    dst = resolve_project_path(root, destination)
    if dst.exists():
        raise ProjectToolError("Destination already exists.")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    return {"source": source, "destination": destination}


def delete_path(root: str, path: str) -> dict:
    target = resolve_project_path(root, path)
    if target == Path(root).resolve():
        raise ProjectToolError("Cannot delete the workspace root.")
    if target.is_dir():
        shutil.rmtree(target)
    else:
        target.unlink()
    return {"path": path, "deleted": True}
