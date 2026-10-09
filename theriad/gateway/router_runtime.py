"""Only signed, hash-verified local assets can process request text."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

from .contracts import GatewayError


def load_router(raw, payload):
    from theriad.routing.v2 import OnnxEmbeddingBackend, RouterV2, RouterV2Artifact

    artifact = RouterV2Artifact.from_dict(raw)
    if not artifact.embedding_model:
        return RouterV2(artifact)
    runtime = artifact.embedding_runtime
    if not runtime or runtime.get("backend") != "onnx":
        raise GatewayError(
            "router_runtime_unavailable",
            "Only bundled local ONNX embedding runtimes are supported",
            503,
        )
    rss, latency = artifact.embedding_incremental_rss_mb, artifact.embedding_p95_latency_ms
    if rss is None or latency is None or rss > 120 or latency > 20:
        raise GatewayError(
            "router_runtime_unavailable", "Embedding runtime has not passed resource gates", 503
        )
    root = (
        Path(os.environ.get("THERIAD_GATEWAY_ROUTER_ASSETS", "~/.theriad/gateway/router-assets"))
        .expanduser()
        .resolve()
    )
    manifest = payload.get("router_assets", {})
    paths = []
    for field in ("model_path", "tokenizer_path"):
        name = runtime[field]
        path = (root / name).resolve()
        if not path.is_relative_to(root) or not path.is_file() or name not in manifest:
            raise GatewayError(
                "router_runtime_unavailable", "Signed router asset is unavailable", 503
            )
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[name]:
            raise GatewayError("router_asset_invalid", "Router asset hash does not verify", 503)
        paths.append(path)
    try:
        backend = OnnxEmbeddingBackend(
            paths[0],
            paths[1],
            int(runtime.get("max_seq_length", 256)),
            int(runtime.get("intra_op_threads", 1)),
        )
    except (ImportError, ValueError) as exc:
        raise GatewayError(
            "router_runtime_unavailable", "Install the gateway-onnx runtime extra", 503
        ) from exc
    return RouterV2(artifact, embedding_backend=backend)


def package_assets(artifact, source, target):
    """Copy immutable local assets; never download an encoder in inference."""
    runtime = artifact.get("embedding_runtime")
    if not artifact.get("embedding_model"):
        return {}
    if not runtime or runtime.get("backend") != "onnx":
        raise ValueError("only a local ONNX embedding artifact can be packaged")
    source, target = Path(source).resolve(), Path(target).expanduser()
    manifest = {}
    for field in ("model_path", "tokenizer_path"):
        path = (source / runtime[field]).resolve()
        if not path.is_relative_to(source):
            raise ValueError("router assets must be inside the artifact directory")
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        name = "blobs/" + sha
        destination = target / name
        destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() != sha:
            raise ValueError("immutable asset collision")
        if not destination.exists():
            destination.write_bytes(data)
            destination.chmod(0o444)
        runtime[field] = name
        manifest[name] = sha
    return manifest
