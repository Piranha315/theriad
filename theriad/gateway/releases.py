"""Content-addressed signed bundles and atomic active-release pointers."""

from __future__ import annotations

import hashlib
import hmac
import json
import time

from .contracts import Deployment, EndpointPolicy, GatewayError


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


class Releases:
    def __init__(self, store, key: str):
        if len(key) < 32:
            raise ValueError("release signing key must contain at least 32 characters")
        self.store, self.key = store, key.encode()

    def install(self, payload: dict) -> str:
        deployments = [Deployment.model_validate(row) for row in payload["deployments"]]
        if len({d.id for d in deployments}) != len(deployments):
            raise ValueError("duplicate deployment IDs")
        EndpointPolicy.model_validate(payload["policy"])
        artifacts = ([payload["router"]] if payload.get("router") else []) + [
            r["artifact"] for r in payload.get("replacements", [])
        ]
        for raw in artifacts:
            from theriad.routing.v2 import RouterV2Artifact

            artifact = RouterV2Artifact.from_dict(raw)
            if artifact.research_only:
                raise ValueError("research-only router cannot be installed")
            ids = {d.id for d in deployments}
            if set(artifact.models) - ids:
                raise ValueError("router heads must use deployment IDs")
        for replacement in payload.get("replacements", []):
            if (
                replacement["candidate"] not in replacement["artifact"]["models"]
                or replacement["incumbent"] not in replacement["artifact"]["models"]
            ):
                raise ValueError(
                    "replacement requires independently fitted candidate and incumbent heads"
                )
        release_id = digest(payload)
        signature = hmac.new(self.key, canonical(payload), hashlib.sha256).hexdigest()
        self.store.put("release/" + release_id, {"payload": payload, "signature": signature})
        return release_id

    def load(self, release_id):
        record = self.store.get("release/" + release_id)
        if not record or digest(record["payload"]) != release_id:
            raise GatewayError("release_unavailable", "Release is unavailable", 503)
        expected = hmac.new(self.key, canonical(record["payload"]), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected, record.get("signature", "")):
            raise GatewayError("release_invalid", "Release signature does not verify", 503)
        return record["payload"]

    def active(self):
        pointer = self.store.get("active")
        if not pointer:
            raise GatewayError("release_unavailable", "No approved release is installed", 503)
        return pointer, self.load(pointer["release"])

    def activate(self, release_id, expected=None):
        self.load(release_id)

        def change(old):
            if expected is not None and (old or {}).get("release") != expected:
                raise GatewayError("release_conflict", "Active release changed", 409)
            return {
                "release": release_id,
                "previous": (old or {}).get("release"),
                "activated_at": time.time(),
            }

        return self.store.mutate("active", change)

    def rollback(self):
        old = self.store.get("active") or {}
        if not old.get("previous"):
            raise GatewayError("rollback_unavailable", "No previous release", 409)
        return self.activate(old["previous"], old["release"])
