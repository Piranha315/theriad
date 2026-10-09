"""Protocol snapshots use only temporary sources and never call providers."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from theriad.routing import episode_protocol as protocol


@pytest.fixture
def source_tree(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "theriad" / "routing").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "theriad" / "routing" / "decision.py").write_text("def choose(): return 'model'\n")
    (root / "scripts" / "run_router_episode_study.py").write_text("print('study')\n")
    (root / "pyproject.toml").write_text('[project]\nname="test"\n')
    (root / "uv.lock").write_text("version = 1\n")
    (root / ".env").write_text("PRIVATE_SENTINEL_DO_NOT_COPY=yes\n")
    (root / "theriad" / "private.json").write_text('{"secret":"do not copy"}')
    monkeypatch.setattr(protocol, "_SOURCE_ROOT", root)
    monkeypatch.setattr(
        protocol,
        "_environment",
        lambda: {
            "python": {"version": "3.12.0"},
            "dependencies": [{"name": "httpx", "version": "0.28"}],
        },
    )
    return root, tmp_path / "study"


def test_capture_is_deterministic_complete_and_excludes_credentials(source_tree):
    root, directory = source_tree
    manifest = protocol.capture_protocol(directory)
    assert protocol.capture_protocol(directory) == manifest
    assert set(manifest["files"]) == {
        "theriad/routing/decision.py",
        "scripts/run_router_episode_study.py",
        "pyproject.toml",
        "uv.lock",
    }
    target = Path(manifest["snapshot_directory"])
    assert target == directory / "protocol" / manifest["manifest_hash"]
    assert json.loads((target / "manifest.json").read_text()) == manifest
    for name in manifest["files"]:
        assert (target / "source" / name).read_bytes() == (root / name).read_bytes()
    assert all(
        b"PRIVATE_SENTINEL" not in path.read_bytes() for path in target.rglob("*") if path.is_file()
    )
    protocol.verify_protocol(manifest)


@pytest.mark.parametrize("change", ["edit", "addition", "deletion", "lockfile"])
def test_source_drift_fails_without_rewriting_snapshot(source_tree, change):
    root, directory = source_tree
    manifest = protocol.capture_protocol(directory)
    old = Path(manifest["snapshot_directory"]) / "source" / "theriad/routing/decision.py"
    before = old.read_bytes()
    source = root / "theriad/routing/decision.py"
    if change == "edit":
        source.write_text("def choose(): return 'different'\n")
    elif change == "addition":
        (root / "theriad/new.py").write_text("NEW = True\n")
    elif change == "deletion":
        source.unlink()
    else:
        (root / "uv.lock").write_text("version = 2\n")
    with pytest.raises(ValueError, match="source"):
        protocol.verify_protocol(manifest)
    assert old.read_bytes() == before
    if change != "deletion":
        newer = protocol.capture_protocol(directory)
        assert newer["manifest_hash"] != manifest["manifest_hash"]
        assert old.read_bytes() == before


def test_dependency_or_python_drift_fails(source_tree, monkeypatch):
    _, directory = source_tree
    manifest = protocol.capture_protocol(directory)
    monkeypatch.setattr(
        protocol, "_environment", lambda: {"python": {"version": "3.13.0"}, "dependencies": []}
    )
    with pytest.raises(ValueError, match="Python or dependency"):
        protocol.verify_protocol(manifest)


def test_historical_verification_accepts_live_drift_but_rejects_archive_damage(
    source_tree, monkeypatch
):
    root, directory = source_tree
    manifest = protocol.capture_protocol(directory)
    (root / "theriad/routing/decision.py").write_text("changed after study\n")
    monkeypatch.setattr(protocol, "_environment", lambda: {"python": {}, "dependencies": []})
    protocol.verify_protocol(manifest, check_live=False)
    with pytest.raises(ValueError, match="source"):
        protocol.verify_protocol(manifest)
    monkeypatch.setattr(protocol, "_SOURCE_ROOT", root / "different_checkout")
    protocol.verify_protocol(manifest, check_live=False)
    archived = Path(manifest["snapshot_directory"]) / "source/theriad/routing/decision.py"
    archived.write_text("corrupted archive\n")
    with pytest.raises(ValueError, match="snapshot source changed"):
        protocol.verify_protocol(manifest, check_live=False)


@pytest.mark.parametrize("corruption", ["manifest", "snapshot", "extra_file", "missing_file"])
def test_corrupted_snapshot_or_manifest_fails_closed(source_tree, corruption):
    _, directory = source_tree
    manifest = protocol.capture_protocol(directory)
    target = Path(manifest["snapshot_directory"])
    if corruption == "manifest":
        changed = copy.deepcopy(manifest)
        changed["environment"] = {}
        with pytest.raises(ValueError, match="digest"):
            protocol.verify_protocol(changed)
        return
    source = target / "source/theriad/routing/decision.py"
    if corruption == "snapshot":
        source.write_text("altered")
    elif corruption == "extra_file":
        (target / "source/untracked.py").write_text("untracked")
    else:
        source.unlink()
    with pytest.raises(ValueError, match="snapshot"):
        protocol.verify_protocol(manifest)
    with pytest.raises(ValueError, match="snapshot"):
        protocol.capture_protocol(directory)


def test_symlink_cannot_capture_external_content(source_tree):
    root, directory = source_tree
    (root / "theriad/external.py").symlink_to(root / ".env")
    with pytest.raises(ValueError, match="symbolic links"):
        protocol.capture_protocol(directory)
    assert not (directory / "protocol").exists()


def test_missing_or_changed_manifest_schema_fails(source_tree):
    _, directory = source_tree
    with pytest.raises(ValueError, match="schema"):
        protocol.verify_protocol({})
    manifest = protocol.capture_protocol(directory)
    (Path(manifest["snapshot_directory"]) / "manifest.json").unlink()
    with pytest.raises(ValueError, match="missing or unreadable"):
        protocol.verify_protocol(manifest)


def test_capture_detects_source_changes_during_snapshot(source_tree, monkeypatch):
    root, directory = source_tree
    original = protocol._write

    def changing_write(path, content):
        original(path, content)
        if path.name == "manifest.json":
            (root / "uv.lock").write_text("version = 9\n")

    monkeypatch.setattr(protocol, "_write", changing_write)
    with pytest.raises(ValueError, match="during capture"):
        protocol.capture_protocol(directory)
    assert list((directory / "protocol").iterdir()) == []

