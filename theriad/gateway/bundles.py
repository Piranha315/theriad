"""Portable signed bundles including their hash-addressed local router assets."""

import hashlib
import hmac
import io
import json
import tarfile
from pathlib import Path

from .releases import canonical, digest


def export_bundle(releases, release, path, asset_root):
    payload = releases.load(release)
    root = Path(asset_root).expanduser().resolve()
    with tarfile.open(path, "w:gz") as archive:
        data = canonical(releases.store.get("release/" + release))
        item = tarfile.TarInfo("bundle.json")
        item.size = len(data)
        archive.addfile(item, io.BytesIO(data))
        for name, expected in payload.get("router_assets", {}).items():
            source = (root / name).resolve()
            if not source.is_relative_to(root):
                raise ValueError("router asset escaped its root")
            data = source.read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("router asset does not verify")
            item = tarfile.TarInfo("assets/" + name)
            item.size = len(data)
            archive.addfile(item, io.BytesIO(data))
    return {"release": release, "bundle": str(Path(path).resolve())}


def import_bundle(releases, path, asset_root):
    root = Path(asset_root).expanduser().resolve()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        if (
            len({m.name for m in members}) != len(members)
            or any(not m.isfile() or m.size > 256_000_000 for m in members)
            or sum(m.size for m in members) > 512_000_000
        ):
            raise ValueError("bundle contains invalid or oversized members")
        manifest = archive.extractfile("bundle.json")
        if manifest is None:
            raise ValueError("bundle manifest is missing")
        envelope = json.load(manifest)
        payload = envelope["payload"]
        signature = hmac.new(releases.key, canonical(payload), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(signature, envelope.get("signature", "")):
            raise ValueError("bundle signature mismatch")
        expected_members = {"bundle.json"} | {
            "assets/" + name for name in payload.get("router_assets", {})
        }
        if {m.name for m in members} != expected_members:
            raise ValueError("bundle contains unexpected assets")
        # Verify every byte before importing or changing the active pointer.
        assets = []
        for name, expected in payload.get("router_assets", {}).items():
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("router asset escaped its root")
            source = archive.extractfile("assets/" + name)
            data = source.read()
            if hashlib.sha256(data).hexdigest() != expected:
                raise ValueError("bundle asset does not verify")
            if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError("immutable asset collision")
            assets.append((target, data))
        release = releases.install(payload)
        if release != digest(payload):
            raise ValueError("bundle identity mismatch")
        for target, data in assets:
            target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not target.exists():
                target.write_bytes(data)
                target.chmod(0o444)
        return releases.activate(release)
