"""Synthetic, zero-network rehearsal of the production episode study pipeline.

These uniquely identified fixtures are API plumbing inputs, not independently
sampled scientific tasks. Provider replies are explicit fakes; the code sandbox
can be real when explicitly requested. The actual trainer, calibrator, RouterV2, product task engine,
paired executor, source lock and report writer run without provider credentials.
"""
from __future__ import annotations

import importlib.util
import json
import os
import socket
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from theriad.core.providers.interfaces import CompletionResponse
from theriad.routing.episode_hooks import current_episode
from theriad.routing.episode_study import Episode, FAMILIES, atomic_json, digest

_SPEC = importlib.util.spec_from_file_location(
    "episode_rehearsal_cli", Path(__file__).resolve().parents[3] / "scripts/run_router_episode_study.py"
)
cli = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(cli)

BASELINE = "rehearsal/baseline"
CHEAP = "rehearsal/cheap"


def rehearsal_rows():
    """Enough distinct plumbing inputs for the unchanged trainer support gates."""
    rows = []
    for split, count in (("pilot", 1), ("train", 6), ("calibration", 2),
                         ("threshold_selection", 1), ("test", 1), ("smoke", 1)):
        for family in FAMILIES:
            for index in range(count):
                identity = f"synthetic-rehearsal.{split}.{family}.{index}"
                fixture = {"source_metadata": {"synthetic_rehearsal": True}}
                grader = {"kind": "exact", "answer": "correct"}
                prompt = f"Synthetic rehearsal fixture {identity}. Return correct."
                if family == "coding":
                    prompt = f"Synthetic rehearsal fixture {identity}. Write a Python increment function."
                    fixture.update(tests="def check(f): assert f(2) == 3", entry_point="increment")
                    grader = {"kind": "code"}
                elif family == "search":
                    prompt = f"Synthetic rehearsal fixture {identity}. Find the launch year from the evidence."
                    fixture["documents"] = [{"url": "https://fixture.invalid/launch", "title": "Launch", "text": "Launch occurred in 2007."}]
                elif family in {"tools", "agents"}:
                    prompt = f"Synthetic rehearsal fixture {identity}. Read input.txt, write output.json with answer 42, and finish."
                    fixture.update(files={"input.txt": "public synthetic fixture"}, expected_json={"output.json": {"answer": 42}})
                    grader = {"kind": "state"}
                rows.append(Episode(identity, identity, family, split, prompt, fixture, grader,
                    source="newly_authored", template_id=identity, tags=("synthetic_rehearsal",),
                    max_calls=4, max_input_tokens=16000, max_output_tokens=512))
    return rows


def frozen_plan(row):
    shared = {"kind": "current", "assistant_provider": "theriad", "request_extras": {}}
    if row.family in {"chat", "coding"}:
        return {**shared, "chain": [["openrouter", BASELINE]], "system_prompt": "Synthetic rehearsal."}
    if row.family == "search":
        return {**shared, "models": {"search_query": BASELINE, "search_synthesis": BASELINE}, "synthesis_enabled": True}
    return {**shared, "models": {"agent_planner": BASELINE}}


class RehearsalProvider:
    name = "openrouter"

    def __init__(self, rows, *, reported_billing=False):
        self.rows = {row.episode_id: row for row in rows}
        self.requests = []
        self.session_calls = {}
        self.reported_billing = reported_billing

    async def complete(self, request):
        session = current_episode()
        assert session is not None, "A provider call escaped episode reservation/accounting"
        row = self.rows[session.episode_id]
        call = self.session_calls.get(session, 0)
        self.session_calls[session] = call + 1
        self.requests.append((row.episode_id, session.policy_id, request.model))
        if row.family == "coding":
            output = "def increment(x): return x + 1"
        elif row.family == "search":
            output = '{"queries":[]}' if call == 0 else "correct"
        elif row.family in {"tools", "agents"}:
            steps = [
                {"tool": "project.read", "arguments": {"path": "input.txt"}},
                {"tool": "project.write", "arguments": {"path": "output.json", "content": '{"answer":42}'}},
                {"tool": "task.complete", "arguments": {"answer": "Done"}},
            ]
            assert call < len(steps), "Unexpected task planner retry"
            output = json.dumps(steps[call])
        else:
            output = "correct"
        cost = (0.0 if request.model == CHEAP else 0.00013) if self.reported_billing else None
        if self.reported_billing:
            assert request.extra['provider']['max_price']['request'] == 0
        return CompletionResponse(model=request.model, content=output, prompt_tokens=100,
            completion_tokens=30, latency_ms=1, provider=self.name, finish_reason="stop",
            cost_usd=cost,
            raw={"usage": {"prompt_tokens": 100, "completion_tokens": 30}})


@pytest.mark.asyncio
@pytest.mark.parametrize('profile', ['research', 'engineering_canary'])
async def test_full_offline_rehearsal_trains_routes_locks_and_rejects_changed_replay(tmp_path, monkeypatch, profile):
    import httpx
    from theriad.routing import episode_runtime, sandbox
    from theriad.routing.v2 import RouterV2

    real_sandbox = os.environ.get("THERIAD_EPISODE_REHEARSAL_REAL_SANDBOX") == "1"
    if real_sandbox and not sandbox.sandbox_available():
        pytest.fail("Real sandbox requested but unavailable; no mocked substitution")
    if real_sandbox:
        import sys
        monkeypatch.setenv(sandbox.CODE_GRADER_RUNTIME_ENV, sys.executable)
    rows = rehearsal_rows()
    engineering = profile == 'engineering_canary'
    if engineering:
        # Only this network-denied test replaces corpus/review identities and
        # allocation constants. Production retains its exact v9 admission.
        from theriad.routing import episode_engineering
        review_hash = digest({'synthetic_rehearsal_review': True})
        rows = [replace(row, fixture={**row.fixture, 'semantic_review': {
            'review_complete': True, 'status': 'reviewed_research_only',
            'review_hash': review_hash, 'disposition': 'retained_research_only',
            'collection_ready': False,
        }}) for row in rows if row.split != 'test']
        monkeypatch.setattr(episode_engineering, 'REVIEW_HASHES', frozenset({review_hash}))
        monkeypatch.setattr(episode_engineering, 'ALLOCATIONS', {
            stage: {family: count for family in FAMILIES}
            for stage, count in [('pilot', 1), ('train', 6), ('calibration', 2),
                                 ('threshold_selection', 1), ('smoke', 1)]})
    corpus = tmp_path / "synthetic-rehearsal.jsonl"
    corpus.write_text("\n".join(json.dumps(asdict(row)) for row in rows) + "\n")
    if engineering:
        import hashlib
        monkeypatch.setattr(episode_engineering, 'CORPUS_SHA256', hashlib.sha256(corpus.read_bytes()).hexdigest())
    output_override = os.environ.get("THERIAD_EPISODE_REHEARSAL_DIRECTORY")
    directory = Path(output_override).resolve() if output_override else tmp_path / "offline-rehearsal"
    if output_override and directory.exists() and any(directory.iterdir()):
        pytest.fail("Standalone rehearsal requires a new empty output directory")
    atomic_json(directory / "rehearsal.json", {
        "execution_mode": "offline_rehearsal", "synthetic_provider_responses": True,
        "synthetic_corpus": True, "scientific_evidence": False,
        "real_provider_spend_usd": 0, "code_sandbox_mocked": not real_sandbox,
        "promotion_eligible": False,
        "profile": profile,
        "synthetic_admission_identities": engineering,
        "synthetic_billing": engineering,
    })
    network_attempts = []

    def forbid_network(*args, **kwargs):
        network_attempts.append(True)
        raise AssertionError("Offline rehearsal forbids all network and DNS")

    original_socket = socket.socket

    class OfflineSocket(original_socket):
        def __init__(self, family=socket.AF_INET, *args, **kwargs):
            if family in {socket.AF_INET, socket.AF_INET6}:
                forbid_network()
            super().__init__(family, *args, **kwargs)

    monkeypatch.setattr(socket, "socket", OfflineSocket)
    monkeypatch.setattr(socket, "create_connection", forbid_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_network)
    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket.socket, "connect_ex", forbid_network)
    monkeypatch.setattr(socket.socket, "sendto", forbid_network)
    monkeypatch.setattr(cli, "load_credentials", lambda: None)
    monkeypatch.setattr("theriad.settings.secrets.get_active_assistant_provider", lambda: "theriad")
    monkeypatch.setenv("THERIAD_TEST_LIVE_MODELS", "true")
    monkeypatch.setenv("THERIAD_API_SEARCH_SYNTHESIS", "true")
    monkeypatch.setenv("THERIAD_AGENT_MODEL", BASELINE)
    monkeypatch.setattr("theriad.agents.search_synthesis._DEFAULT_SEARCH_MODEL", BASELINE)
    monkeypatch.setattr(episode_runtime, "freeze_baseline", frozen_plan)
    monkeypatch.setattr(cli, "registry_model_pool", lambda: {
        model: {"model": model, "provider": "openrouter"} for model in (BASELINE, CHEAP)
    })

    class CatalogResponse:
        def raise_for_status(self):
            pass

        def json(self):
            return {"data": [{"id": model,
                "pricing": {"prompt": str(price), "completion": str(price)},
                "architecture": {"input_modalities": ["text"]},
                "supported_parameters": ["tools"], "context_length": 100000}
                for model, price in ((BASELINE, 0.000001), (CHEAP, 0.0))]}

    class CatalogClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            assert url == "https://openrouter.ai/api/v1/models"
            return CatalogResponse()

    monkeypatch.setattr(httpx, "AsyncClient", CatalogClient)
    code_grades = []

    def fake_code_sandbox(code, tests, *, entry_point):
        # This verifies fixture wiring only. Never execute untrusted code here.
        assert code == "def increment(x): return x + 1"
        assert "f(2) == 3" in tests and entry_point == "increment"
        code_grades.append(True)
        return SimpleNamespace(passed=True, status="passed",
            to_dict=lambda: {"status": "passed", "executor_id": "synthetic-rehearsal-stub", "real_code_execution": False})

    if real_sandbox:
        original_grader = sandbox.run_python_tests
        def observed_grader(*args, **kwargs):
            result = original_grader(*args, **kwargs)
            code_grades.append(result.passed)
            return result
        monkeypatch.setattr(sandbox, "run_python_tests", observed_grader)
    else:
        monkeypatch.setattr(sandbox, "run_python_tests", fake_code_sandbox)
        monkeypatch.setattr(cli, "preflight", lambda target: atomic_json(target / "preflight.json", {
            "execution_mode": "offline_rehearsal", "provider_key_present": False,
            "code_sandbox": {"status": "mocked", "real_code_execution": False},
        }))
    provider = RehearsalProvider(rows, reported_billing=engineering)
    original_runtime = episode_runtime.EpisodeRuntime
    runtimes = []

    def offline_runtime(**kwargs):
        result = original_runtime(**kwargs, provider_factory=lambda *args, **kwargs: provider)
        runtimes.append(result)
        return result

    monkeypatch.setattr(episode_runtime, "EpisodeRuntime", offline_runtime)
    original_predict = RouterV2.predict
    predictions = []

    def observed_predict(self, *args, **kwargs):
        result = original_predict(self, *args, **kwargs)
        predictions.append(result.selected_model)
        return result

    monkeypatch.setattr(RouterV2, "predict", observed_predict)
    await cli.prepare(directory, [BASELINE, CHEAP] if engineering else [], corpus, profile=profile)
    if engineering:
        contract = cli.read_json(directory / 'contract.json')
        atomic_json(directory / 'authorization-development.json', {
            'phase': 'development', 'paid_calls_authorized': True,
            'user_authorization_reference': 'TEST ONLY: synthetic models, admission, billing; network denied',
            'contract_hash': digest(contract), 'budget_proposal_hash': contract['budget_proposal_hash'],
            'budget_config': contract['budget_config'], 'protocol_manifest_hash': contract['protocol_manifest_hash']})
        loaded = []
        original_load = cli.load_rows
        def guarded_load(target, splits):
            loaded.extend(splits)
            assert not {'smoke', 'test'}.intersection(splits), 'Later split loaded during engineering fitting'
            return original_load(target, splits)
        monkeypatch.setattr(cli, 'load_rows', guarded_load)
    await cli.run_study(directory, stop_after="fit" if engineering else "selection")
    challenger = cli.read_json(directory / "challenger.json")
    lock_bytes = None if engineering else (directory / "test-lock.json").read_bytes()
    assert challenger["kind"] == "surface"
    assert challenger["production_enabled"] is False
    assert not (directory / "test-outcomes.jsonl").exists()
    artifact = challenger["artifacts"]["chat"]
    assert artifact["effective_sample_size"]["train"] == 30
    assert artifact["effective_sample_size"]["calibration"] == 10
    assert artifact["effective_sample_size"]["threshold_selection"] == 5
    assert artifact["effective_sample_size"]["test"] == 0
    assert artifact["split_lineage"]["preprocessing_fit"] == "train_only"
    assert all("fixed_model" not in lane for lane in artifact["lanes"].values())
    assert predictions, "The actual RouterV2.predict implementation was never exercised"

    if engineering:
        assert not (directory / 'fixtures/test.json').exists()
        assert not (directory / 'test-lock.json').exists()
        assert not (directory / 'smoke-outcomes.jsonl').exists()
        report = cli.render_report(directory)
        assert report['decision'] == 'rehearsal_only'
        assert report['protocol_verified'] is False  # No locked research evaluation.
        assert report['engineering']['development_source_snapshot_verified'] is True
        assert report['engineering']['deployment_eligible'] is False
        assert all(report['stages'][stage]['unknown_cost_episodes'] == 0 for stage in ('pilot', 'development'))
        request_count = len(provider.requests)
        prediction_count = len(predictions)
        outcome_bytes = (directory / 'development-outcomes.jsonl').read_bytes()
        await cli.run_study(directory, stop_after='fit')
        assert len(provider.requests) == request_count
        assert (directory / 'development-outcomes.jsonl').read_bytes() == outcome_bytes
        with pytest.raises(ValueError, match='smoke lock'):
            await cli.run_study(directory, stop_after='smoke')
        assert len(provider.requests) == request_count
        assert loaded and code_grades and network_attempts == []
        if real_sandbox:
            assert all(code_grades)
        atomic_json(directory / 'engineering-rehearsal-verification.json', {
            'profile': profile, 'synthetic_corpus': True, 'synthetic_providers': True,
            'synthetic_admission_identities': True, 'real_provider_spend_usd': 0,
            'actual_trainer_calibration_and_router': True, 'router_prediction_count': prediction_count,
            'provider_request_count': request_count, 'network_attempts': len(network_attempts),
            'code_sandbox_mocked': not real_sandbox, 'idempotent_resume': True,
            'later_split_loads': [], 'paid_smoke_dispatch_blocked': True,
            'scientific_evidence': False, 'promotion_eligible': False})
        return

    await cli.run_study(directory, stop_after="test")
    assert (directory / "test-lock.json").read_bytes() == lock_bytes
    results = cli.outcomes(directory / "test-outcomes.jsonl")
    assert len(results) == 10 and all(value.success for value in results)
    for value in results:
        if value.policy_id == "challenger":
            assert value.evidence["decision_trace"]
            assert value.evidence["selected_configuration"] == [CHEAP]
            if value.family in {"tools", "agents", "search"}:
                assert len(value.evidence["decision_trace"]) >= 2
            if value.family in {"tools", "agents"}:
                assert value.evidence["final_state"] == {"output.json": {"answer": 42}}
                assert value.evidence["preserved_inputs"] == {"input.txt": True}
    report = cli.render_report(directory)
    assert report["decision"] == "rehearsal_only"
    assert report["execution_mode"] == "offline_rehearsal"
    assert "simulated" in report["accounting_interpretation"]
    assert "OFFLINE REHEARSAL" in (directory / "REPORT.md").read_text()
    assert report["production_changed"] is False
    assert report["protocol_verified"] is True
    assert report["stages"]["test"]["unknown_cost_episodes"] == 0
    assert not report["coverage"]["live_user_route_verified"]
    assert code_grades and network_attempts == []
    if real_sandbox:
        assert all(code_grades)
        for value in results:
            if value.family == "coding":
                assert value.evidence["sandbox"]["executor_id"] == sandbox.SANDBOX_EXECUTOR_ID

    # A valid resume reads durable results, never repeats provider dispatches.
    request_count = len(provider.requests)
    test_bytes = (directory / "test-outcomes.jsonl").read_bytes()
    await cli.run_study(directory, stop_after="test")
    assert len(provider.requests) == request_count
    assert (directory / "test-outcomes.jsonl").read_bytes() == test_bytes
    assert (directory / "test-lock.json").read_bytes() == lock_bytes

    # A changed policy cannot replay even the first already-completed pair.
    lock = cli.read_json(directory / "test-lock.json")
    row = next(row for row in rows if row.episode_id == lock["schedule"][0]["episode_id"])
    policy = {"baseline": {"kind": "current"}, "challenger": {**challenger, "version": "tampered"}}
    with pytest.raises(ValueError, match="frozen policy contract"):
        await cli.execute_pair(row=row, policies=policy, order=lock["schedule"][0]["policy_order"],
            runner=runtimes[-1], maximum_costs={key: runtimes[-1].maximum_cost(row, value) for key, value in policy.items()},
            budget=cli.ProductStudyBudget(directory / "budget"), stage="test",
            output=directory / "test-outcomes.jsonl", lock_hash=lock["lock_hash"], lock=lock)
    with pytest.raises(ValueError, match="frozen"):
        cli.write_once(directory / "challenger.json", {**challenger, "version": "tampered"})
    assert len(provider.requests) == request_count
    assert digest(cli.read_json(directory / "challenger.json")) == digest(challenger)
