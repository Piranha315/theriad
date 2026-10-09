import hashlib
import io
import json
import tarfile

import httpx
import pytest

from theriad.gateway.bundles import export_bundle, import_bundle
from theriad.gateway.contracts import Deployment, EndpointPolicy, GatewayError
from theriad.gateway.releases import Releases
from theriad.gateway.store import LocalStore
from theriad.gateway.transport import WireTransport
from theriad.gateway.worker import UpdateWorker, fetch_catalogs, validate_suite

from .test_adoption import bundle


@pytest.mark.asyncio
async def test_multiturn_fixture_charges_each_call_and_reuses_only_real_measurements(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("TEST_KEY", "provider-key")
    setup = bundle()
    setup["deployments"][0]["credential_env"] = "TEST_KEY"
    deployment = Deployment.model_validate(setup["deployments"][0])
    calls = []

    def handler(request):
        body = json.loads(request.content)
        calls.append(body)
        index = len(calls)
        message = {"role": "assistant", "content": None}
        if index < 3:
            name = "read_file" if index == 1 else "write_file"
            args = {"path": "answer.txt"}
            if index == 2:
                args["content"] = "correct\n"
            message["tool_calls"] = [
                {
                    "id": "call-" + str(index),
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        else:
            message["content"] = "Done"
        return httpx.Response(
            200,
            json={
                "id": "generation-" + str(index),
                "model": deployment.model,
                "choices": [
                    {"message": message, "finish_reason": "tool_calls" if index < 3 else "stop"}
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "cost": 0.001},
            },
        )

    worker = UpdateWorker(
        Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32),
        WireTransport(transport=httpx.MockTransport(handler)),
    )
    task = {
        "id": "fixture",
        "group": "real-group",
        "family": "code",
        "split": "qualification",
        "grader": "fixture",
        "request": {
            "messages": [
                {
                    "role": "user",
                    "content": "Read answer.txt, fix it, preserve input.txt, and finish",
                }
            ],
            "max_tokens": 256,
        },
        "fixture": {
            "files": {"answer.txt": "wrong\n", "input.txt": "preserve"},
            "expected_files": {"answer.txt": "correct\n"},
        },
        "max_calls": 4,
    }
    first = await worker.measure(
        deployment, task, "p", EndpointPolicy.model_validate(setup["policy"])
    )
    assert first["success"] == 1 and first["calls"] == 3 and first["cost"] == pytest.approx(0.003)
    assert first["evidence"]["preserved_inputs"] == {"input.txt": True}
    assert calls[1]["messages"][-1]["tool_call_id"] == "call-1"
    assert calls[2]["messages"][-1]["tool_call_id"] == "call-2"
    assert (
        await worker.measure(deployment, task, "p", EndpointPolicy.model_validate(setup["policy"]))
        == first
    )
    assert len(calls) == 3
    ledger = worker.store.get(worker.budget.key())["entries"]
    assert len(ledger) == 3 and sum(e["actual"] for e in ledger.values()) == pytest.approx(0.003)


@pytest.mark.asyncio
async def test_missing_compatibility_stops_before_paid_calls(tmp_path):
    class Transport:
        async def complete(self, *args):
            raise AssertionError("must not dispatch")

    setup = bundle()
    worker = UpdateWorker(Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32), Transport())
    suite = {
        "version": "v1",
        "source_revision": "reviewed-sha",
        "tasks": [
            {
                "id": "a",
                "group": "g",
                "family": "code",
                "split": "qualification",
                "grader": "exact",
                "expected": "yes",
                "request": {"messages": [{"role": "user", "content": "yes"}]},
            }
        ],
    }
    with pytest.raises(GatewayError, match="critical checks"):
        await worker.evaluate(
            Deployment.model_validate(setup["deployments"][1]),
            Deployment.model_validate(setup["deployments"][0]),
            suite,
            EndpointPolicy.model_validate(setup["policy"]),
            1,
        )
    assert not worker.store.get(worker.budget.key())


def test_native_and_streaming_probe_labels_cannot_replace_actual_protocol():
    task = {
        "id": "a",
        "group": "g",
        "family": "code",
        "split": "train",
        "grader": "exact",
        "expected": "yes",
        "critical": True,
        "request": {"messages": [{"role": "user", "content": "yes"}]},
    }
    for case in ("streaming", "native_responses"):
        with pytest.raises(ValueError):
            validate_suite(
                {
                    "version": "v",
                    "source_revision": "sha",
                    "tasks": [task | {"compatibility_case": case}],
                }
            )


def test_offline_bundle_contains_verified_assets_and_rejects_tampering(tmp_path):
    setup = bundle()
    data = b"verified-local-router-asset"
    name = "blobs/" + hashlib.sha256(data).hexdigest()
    assets = tmp_path / "assets"
    (assets / "blobs").mkdir(parents=True)
    (assets / name).write_bytes(data)
    setup["router_assets"] = {name: hashlib.sha256(data).hexdigest()}
    source = Releases(LocalStore(tmp_path / "source.sqlite"), "s" * 32)
    release = source.install(setup)
    source.activate(release)
    archive = tmp_path / "release.tar.gz"
    export_bundle(source, release, archive, assets)
    target = Releases(LocalStore(tmp_path / "target.sqlite"), "s" * 32)
    assert import_bundle(target, archive, tmp_path / "target-assets")["release"] == release
    assert (tmp_path / "target-assets" / name).read_bytes() == data
    with tarfile.open(archive, "r:gz") as reader:
        contents = [(m, reader.extractfile(m).read()) for m in reader.getmembers()]
    with tarfile.open(archive, "w:gz") as writer:
        for member, original in contents:
            if member.name.startswith("assets/"):
                original = b"corrupted"
            member.size = len(original)
            writer.addfile(member, io.BytesIO(original))
    before = target.active()[0]
    with pytest.raises(ValueError, match="verify"):
        import_bundle(target, archive, tmp_path / "target-assets")
    assert target.active()[0] == before


@pytest.mark.asyncio
async def test_benchmark_pagination_version_change_fails_closed(monkeypatch):
    monkeypatch.setenv("ARTIFICIAL_ANALYSIS_API_KEY", "free-key")

    def handler(request):
        if request.url.host == "openrouter.ai":
            return httpx.Response(200, json={"data": []})
        if request.url.path.endswith("/docs"):
            return httpx.Response(200, text="Current version: v4.3.2")
        page = request.url.params["page"]
        return httpx.Response(
            200,
            json={
                "intelligence_index_version": 4.3 if page == "1" else 4.4,
                "data": [],
                "pagination": {"has_more": page == "1"},
            },
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(GatewayError, match="Benchmark changed"):
            await fetch_catalogs(client)


@pytest.mark.asyncio
async def test_incumbent_calibration_never_reads_locked_qualification_labels(tmp_path):
    setup = bundle()
    setup.pop("router")
    releases = Releases(LocalStore(tmp_path / "state.sqlite"), "s" * 32)
    first = releases.install(setup)
    releases.activate(first)
    seen = []

    class Transport:
        async def complete(self, deployment, request):
            prompt = request.messages[-1].content
            assert "locked" not in prompt
            seen.append(prompt)
            return {
                "id": "g",
                "model": deployment.model,
                "choices": [{"message": {"content": "yes"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 3, "cost": 0.0001},
            }

    tasks = [
        {
            "id": f"{split}-{family}-{i}",
            "group": f"{split}-{family}-{i}",
            "split": split,
            "family": family,
            "grader": "exact",
            "expected": "yes",
            "request": {
                "messages": [{"role": "user", "content": f"{split} {family} prompt {i}"}],
                **(
                    {
                        "tools": [
                            {
                                "type": "function",
                                "function": {"name": "read_file", "parameters": {"type": "object"}},
                            }
                        ]
                    }
                    if family == "code"
                    else {}
                ),
                "max_tokens": 32,
            },
        }
        for split, count in [("train", 40), ("validation", 20)]
        for family in ["general", "code"]
        for i in range(count)
    ]
    tasks.append(
        {
            "id": "locked",
            "group": "held-out",
            "family": "code",
            "split": "qualification",
            "grader": "exact",
            "expected": "never inspect",
            "request": {"messages": [{"role": "user", "content": "locked label"}]},
        }
    )
    worker = UpdateWorker(releases, Transport())
    result = await worker.calibrate(
        {
            "version": "test",
            "source_revision": "pinned",
            "tasks": tasks,
            "provenance": {
                "license_status": "internal",
                "license_name": "Authored unit fixtures",
                "collection_vintage": "2026-10-01",
                "provenance_url": "repo://tests/unit/gateway/test_evaluation.py",
            },
        }
    )
    assert result["state"] == "calibrated"
    pointer, payload = releases.active()
    assert pointer["release"] != first and len(seen) == 240
    assert set(payload["router"]["models"]) == {"baseline", "challenger"}
    assert payload["router"]["models"]["baseline"]["deployment_identity"]["revision"] == "v1"
