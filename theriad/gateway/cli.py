"""Gateway operator commands. No credentials are printed."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import secrets
import time
from pathlib import Path

from .contracts import Deployment, GatewayError
from .releases import digest
from .worker import UpdateWorker, fetch_catalogs


def runtime():
    os.environ.setdefault("THERIAD_DISABLE_LIVE_PRICING", "1")
    # Use the same initialization path as the ASGI application, without a request.
    from .engine import GatewayEngine
    from .releases import Releases
    from .store import LocalStore
    from .transport import WireTransport

    if os.environ.get("THERIAD_GATEWAY_STORE") == "gcp":
        from .gcp import GoogleStore

        store = GoogleStore(
            os.environ["GOOGLE_CLOUD_PROJECT"], os.environ["THERIAD_GATEWAY_BUCKET"]
        )
    else:
        store = LocalStore(os.environ.get("THERIAD_GATEWAY_DB", "~/.theriad/gateway/state.sqlite"))
    from .ports import evaluation_store

    return GatewayEngine(
        Releases(store, os.environ["THERIAD_RELEASE_SIGNING_KEY"]),
        WireTransport(),
        evaluation_store(store),
    )


def init(directory):
    directory = Path(directory).expanduser()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    path = directory / "gateway.env"
    if path.exists():
        raise ValueError("gateway environment already exists; refusing to overwrite credentials")
    token, signer = secrets.token_urlsafe(32), secrets.token_urlsafe(48)
    config = {
        "THERIAD_GATEWAY_DB": str(directory / "state.sqlite"),
        "THERIAD_RELEASE_SIGNING_KEY": signer,
        "THERIAD_GATEWAY_TOKENS": json.dumps(
            {hashlib.sha256(token.encode()).hexdigest(): "local-owner"}
        ),
        "THERIAD_API_KEY": token,
    }
    import shlex

    data = (
        "\n".join("export " + key + "=" + shlex.quote(value) for key, value in config.items())
        + "\n"
    )
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as handle:
        handle.write(data)
    return {"environment_file": str(path), "credential_created": True}


async def bootstrap(engine, router_path=None):
    catalogs = await fetch_catalogs()
    from theriad.models.registry import DEFAULT_REGISTRY

    models = {m["id"]: m for m in catalogs["openrouter"]["data"]}
    from .catalog import initial_endpoint

    catalogs = await fetch_catalogs(
        model_ids=[
            c.model
            for t in DEFAULT_REGISTRY.all_tiers()
            for c in DEFAULT_REGISTRY.candidates_for_tier(t)
            if c.provider == "openrouter"
        ]
    )
    deployed = {}
    for tier in DEFAULT_REGISTRY.all_tiers():
        for candidate in DEFAULT_REGISTRY.candidates_for_tier(tier):
            if candidate.provider != "openrouter" or candidate.model not in models:
                continue
            row = models[candidate.model]
            if candidate.model in deployed:
                deployed[candidate.model]["tiers"].append(tier)
                continue
            try:
                host = initial_endpoint(
                    candidate.model, catalogs["deployment_endpoints"].get(candidate.model, {})
                )
                parameters = host.get("supported_parameters", [])
                deployment = Deployment(
                    id="openrouter-"
                    + digest(
                        {
                            "model": candidate.model,
                            "revision": row["canonical_slug"],
                            "host": host["tag"],
                            "variant": host["quantization"],
                        }
                    )[:24],
                    model=candidate.model,
                    revision=row["canonical_slug"],
                    provider="openrouter",
                    endpoint="https://openrouter.ai/api/v1",
                    region="provider-managed",
                    tiers=[tier],
                    hosting_provider=host["tag"],
                    supported_parameters=parameters,
                    serving_variant=host["quantization"],
                    context_window=host["context_length"],
                    max_input_tokens=host.get("max_prompt_tokens"),
                    max_output_tokens=host["max_completion_tokens"],
                    tools={"tools", "tool_choice"}.issubset(parameters),
                    schemas="response_format" in parameters,
                    vision="image" in row.get("architecture", {}).get("input_modalities", []),
                    reasoning="reasoning" in parameters,
                    native_responses=False,
                    input_price=float(host["pricing"]["prompt"]) * 1e6,
                    output_price=float(host["pricing"]["completion"]) * 1e6,
                    priced_at=catalogs["fetched_at"],
                    credential_env="OPENROUTER_API_KEY",
                    origin=candidate.origin,
                    quality_prior=candidate.quality_prior or 0.5,
                )
            except (KeyError, ValueError, TypeError, GatewayError):
                continue
            deployed[candidate.model] = deployment.model_dump()
    payload = {
        "deployments": list(deployed.values()),
        "policy": {
            "allowed_endpoints": ["https://openrouter.ai/api/v1"],
            "allowed_regions": ["provider-managed"],
            "max_price_age_seconds": 86400,
            "escalation_ceiling": "premium",
        },
        "source": "existing-approved-registry",
        "created_at": time.time(),
    }
    if router_path:
        artifact = json.loads(Path(router_path).read_text())
        translated = {}
        for model, head in artifact["models"].items():
            if model in deployed:
                provenance = head.get("deployment_identity", {})
                exact = deployed[model]
                if any(
                    provenance.get(k) != exact.get(k)
                    for k in (
                        "revision",
                        "hosting_provider",
                        "serving_variant",
                        "reasoning_profile",
                    )
                ):
                    continue
                translated[deployed[model]["id"]] = head
        if not translated:
            raise ValueError(
                "router heads lack exact deployment provenance; collect new heads rather than assigning model-level outcomes to a new host"
            )
        artifact["models"] = translated
        for lane in artifact["lanes"].values():
            lane["models"] = [deployed[m]["id"] for m in lane.get("models", []) if m in deployed]
            for field in ("fallback_model", "fixed_model"):
                if lane.get(field) in deployed:
                    lane[field] = deployed[lane[field]]["id"]
        payload["router"] = artifact
        from .router_runtime import package_assets

        payload["router_assets"] = package_assets(
            artifact,
            Path(router_path).parent,
            os.environ.get("THERIAD_GATEWAY_ROUTER_ASSETS", "~/.theriad/gateway/router-assets"),
        )
    if not payload["deployments"]:
        raise ValueError("catalog did not validate any approved deployment")
    release = engine.releases.install(payload)
    engine.releases.activate(release)
    return {"release": release, "deployments": len(payload["deployments"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    init_parser = sub.add_parser("init")
    init_parser.add_argument("--directory", default="~/.theriad/gateway")
    bootstrap_parser = sub.add_parser("bootstrap")
    bootstrap_parser.add_argument("--router")
    calibration = sub.add_parser("calibrate")
    calibration.add_argument("--suite", required=True)
    update = sub.add_parser("update")
    update.add_argument("--suite")
    update.add_argument("--identities")
    sub.add_parser("status")
    sub.add_parser("pause")
    sub.add_parser("resume")
    sub.add_parser("rollback")
    install = sub.add_parser("install")
    install.add_argument("bundle")
    export = sub.add_parser("export")
    export.add_argument("bundle")
    export.add_argument("--release")
    args = parser.parse_args()
    try:
        if args.command == "init":
            result = init(args.directory)
        else:
            engine = runtime()
            if args.command == "bootstrap":
                result = asyncio.run(bootstrap(engine, args.router))
            elif args.command == "calibrate":
                result = asyncio.run(
                    UpdateWorker(
                        engine.releases, engine.transport, engine.accounting_store
                    ).calibrate(json.loads(Path(args.suite).read_text()))
                )
            elif args.command == "update":
                suite = json.loads(Path(args.suite).read_text()) if args.suite else None
                identities = (
                    json.loads(Path(args.identities).read_text()) if args.identities else []
                )
                result = asyncio.run(
                    UpdateWorker(engine.releases, engine.transport, engine.accounting_store).run(
                        suite, identities
                    )
                )
            elif args.command == "status":
                from .adoption import MonthlyBudget

                result = {
                    "active": engine.store.get("active"),
                    "paused": engine.store.get("updates/paused"),
                    "latest_update": engine.store.get("update/latest"),
                    "evaluation_budget": MonthlyBudget(engine.accounting_store).summary(),
                }
            elif args.command == "pause":
                engine.store.put("updates/paused", {"reason": "operator"})
                engine.accounting_store.put("updates/paused", {"reason": "operator"})
                from .adoption import Canary

                canary = Canary(engine.releases).tick()
                result = {"paused": True, "canary": canary.get("state")}
            elif args.command == "resume":
                engine.store.put("updates/paused", None)
                engine.accounting_store.put("updates/paused", None)
                result = {"paused": False}
            elif args.command == "rollback":
                pointer = engine.store.get("active")
                if pointer and pointer.get("canary"):
                    engine.store.mutate(
                        "active", lambda old: {k: v for k, v in old.items() if k != "canary"}
                    )
                    result = {"canary_cancelled": True}
                else:
                    result = engine.releases.rollback()
            elif args.command == "export":
                release = args.release or engine.releases.active()[0]["release"]
                from .bundles import export_bundle

                result = export_bundle(
                    engine.releases,
                    release,
                    args.bundle,
                    os.environ.get(
                        "THERIAD_GATEWAY_ROUTER_ASSETS", "~/.theriad/gateway/router-assets"
                    ),
                )
            else:
                from .bundles import import_bundle

                result = import_bundle(
                    engine.releases,
                    args.bundle,
                    os.environ.get(
                        "THERIAD_GATEWAY_ROUTER_ASSETS", "~/.theriad/gateway/router-assets"
                    ),
                )

        print(json.dumps(result, indent=2))
    except (ValueError, KeyError, GatewayError) as exc:
        parser.exit(1, f"Gateway command failed: {exc}\n")


if __name__ == "__main__":
    main()
