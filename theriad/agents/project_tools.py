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


def list_paths(root: str, path: str = ".", *, recursive: bool = False, limit: int = 500) -> dict:
    """List workspace-relative names without opening files or following links."""
    if type(recursive) is not bool or type(limit) is not int or not 1 <= limit <= 2000:
        raise ProjectToolError("recursive must be boolean; limit must be an integer from 1 to 2000")
    base = Path(root).resolve()
    target = resolve_project_path(root, path)
    if not target.is_dir():
        raise ProjectToolError("project.list requires a directory; use project.read for files")
    entries = []
    skipped_symlinks = False
    def walk(directory):
        nonlocal skipped_symlinks
        for item in sorted(directory.iterdir(), key=lambda p: p.name):
            if item.is_symlink():
                skipped_symlinks = True
                continue
            # Recheck confinement for each visited path, including descendants.
            resolved = resolve_project_path(root, str(item))
            kind = "directory" if resolved.is_dir() else "file" if resolved.is_file() else None
            if kind is None:
                continue
            yield {"path": item.relative_to(base).as_posix(), "type": kind}
            if recursive and kind == "directory":
                yield from walk(resolved)
    for entry in walk(target):
        if len(entries) == limit:
            return {"entries": entries, "truncated": True, "skipped_symlinks": skipped_symlinks}
        entries.append(entry)
    return {"entries": entries, "truncated": False, "skipped_symlinks": skipped_symlinks}


def read_file(root: str, path: str, *, max_bytes: int = 500_000) -> dict:
    target = resolve_project_path(root, path)
    if target.is_dir():
        raise ProjectToolError("project.read requires a file; use project.list to discover directory entries")
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
