"""Freeze and verify the executable protocol of a paid episode study.

Only application Python sources, the study runner, and explicit dependency
manifests are copied. Credentials, environment files, responses and other study
data are never part of this snapshot. Capturing disk sources does not establish
which versions an already-running process previously imported; callers must
capture before collection and bind this manifest to their test lock.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import shutil
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

VERSION = "product-episode-protocol-v1"
_SOURCE_ROOT = Path(__file__).resolve().parents[2]
_DEPENDENCY_FILES = (
    "pyproject.toml",
    "uv.lock",
    "poetry.lock",
    "pdm.lock",
    "Pipfile.lock",
    "requirements.lock",
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "bun.lock",
    "bun.lockb",
)


def _encoded(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _environment() -> dict[str, Any]:
    # Versions only: direct URLs, distribution locations and environment values
    # can contain credentials and are intentionally never inspected or copied.
    packages = sorted(
        {
            (re.sub(r"[-_.]+", "-", distribution.metadata["Name"]).lower(), distribution.version)
            for distribution in importlib.metadata.distributions()
            if distribution.metadata.get("Name")
        }
    )
    return {
        "python": {
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
            "cache_tag": sys.implementation.cache_tag,
            "executable": str(Path(sys.executable).resolve()),
            "platform": sys.platform,
            "machine": platform.machine(),
        },
        "dependencies": [{"name": name, "version": version} for name, version in packages],
    }


def _sources(root: Path) -> dict[str, bytes]:
    paths = list((root / "theriad").rglob("*.py"))
    runner = root / "scripts" / "run_router_episode_study.py"
    if not paths or not runner.is_file():
        raise ValueError("protocol application sources or study runner are missing")
    paths.append(runner)
    paths.extend(root / name for name in _DEPENDENCY_FILES if (root / name).exists())
    values = {}
    for path in sorted(paths):
        relative = path.relative_to(root)
        if any(part.startswith(".") for part in relative.parts):
            # Hidden trees are not importable product sources and can contain
            # local configuration. Refuse rather than capture unexpected data.
            raise ValueError("hidden source path is outside the study protocol")
        if any(parent.is_symlink() for parent in (path, *path.parents) if parent != root):
            raise ValueError("protocol sources must not be symbolic links")
        if not path.is_file() or not path.resolve().is_relative_to(root):
            raise ValueError("protocol source is outside its repository")
        values[relative.as_posix()] = path.read_bytes()
    return values


def _file_manifest(sources: Mapping[str, bytes]) -> dict[str, dict[str, Any]]:
    return {
        name: {"sha256": _sha(content), "size_bytes": len(content)}
        for name, content in sources.items()
    }


def _write(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def _unsigned(manifest: Mapping[str, Any]) -> dict[str, Any]:
    if set(manifest) != {
        "version",
        "source_root",
        "files",
        "environment",
        "manifest_hash",
        "snapshot_directory",
    }:
        raise ValueError("invalid protocol manifest schema")
    return {
        key: value
        for key, value in manifest.items()
        if key not in {"manifest_hash", "snapshot_directory"}
    }


def capture_protocol(directory: str | Path) -> dict[str, Any]:
    """Create or reuse a complete immutable snapshot, without changing old ones."""
    root = _SOURCE_ROOT.resolve()
    destination = Path(directory).resolve() / "protocol"
    if destination.is_relative_to(root / "theriad"):
        raise ValueError("protocol snapshots cannot be stored inside application sources")
    sources = _sources(root)
    payload = {
        "version": VERSION,
        "source_root": str(root),
        "files": _file_manifest(sources),
        "environment": _environment(),
    }
    manifest_hash = _sha(_encoded(payload))
    target = destination / manifest_hash
    manifest = {**payload, "manifest_hash": manifest_hash, "snapshot_directory": str(target)}
    if target.exists():
        verify_protocol(manifest)
        return manifest
    destination.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".capture-", dir=destination))
    try:
        for name, content in sources.items():
            _write(temporary / "source" / name, content)
        _write(temporary / "manifest.json", _encoded(manifest) + b"\n")
        if (
            _file_manifest(_sources(root)) != payload["files"]
            or _environment() != payload["environment"]
        ):
            raise ValueError("protocol changed during capture")
        try:
            temporary.rename(target)
        except OSError:
            # A simultaneous identical capture may have won publication. Never
            # replace its snapshot; require every byte and its manifest to agree.
            if not target.exists():
                raise
        verify_protocol(manifest)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return manifest


def verify_protocol(manifest: Mapping[str, Any], *, check_live: bool = True) -> None:
    """Verify archived bytes, and by default the sources/environment about to run.

    ``check_live=False`` supports reading a historical report after the checkout
    or installed dependencies change. It still validates every archived byte;
    it does not authorize collecting more episodes using the old protocol.
    """
    payload = _unsigned(manifest)
    if manifest.get("version") != VERSION or _sha(_encoded(payload)) != manifest.get(
        "manifest_hash"
    ):
        raise ValueError("protocol manifest digest changed")
    root = _SOURCE_ROOT.resolve()
    if check_live and manifest["source_root"] != str(root):
        raise ValueError("protocol repository location changed")
    snapshot = Path(manifest["snapshot_directory"])
    if (
        not snapshot.is_absolute()
        or snapshot.name != manifest["manifest_hash"]
        or snapshot.parent.name != "protocol"
    ):
        raise ValueError("invalid protocol snapshot location")
    try:
        if (
            snapshot.is_symlink()
            or (snapshot / "manifest.json").is_symlink()
            or json.loads((snapshot / "manifest.json").read_text()) != dict(manifest)
        ):
            raise ValueError("protocol snapshot manifest changed")
        inventory = {
            path.relative_to(snapshot / "source").as_posix()
            for path in (snapshot / "source").rglob("*")
            if path.is_file()
        }
        if inventory != set(manifest["files"]):
            raise ValueError("protocol snapshot source inventory changed")
        for name, expected in manifest["files"].items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("invalid protocol source path")
            source = snapshot / "source" / relative
            if source.is_symlink() or not source.resolve().is_relative_to(snapshot.resolve()):
                raise ValueError("protocol snapshot source escaped its directory")
            content = source.read_bytes()
            if expected != {"sha256": _sha(content), "size_bytes": len(content)}:
                raise ValueError(f"protocol snapshot source changed: {name}")
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("protocol snapshot is missing or unreadable") from error
    if not check_live:
        return
    if _file_manifest(_sources(root)) != manifest["files"]:
        raise ValueError("protocol source inventory or content changed")
    if _environment() != manifest["environment"]:
        raise ValueError("protocol Python or dependency versions changed")

