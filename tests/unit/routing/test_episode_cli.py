"""Offline orchestration tests: no provider credentials or live calls."""

from __future__ import annotations

import importlib.util
import json
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from theriad.routing.episode_study import FAMILIES, Episode, EpisodeOutcome, atomic_json, digest
from theriad.routing.study_budget import StudyBudgetExceeded

spec = importlib.util.spec_from_file_location(
    "episode_cli", Path(__file__).resolve().parents[3] / "scripts/run_router_episode_study.py"
)
cli = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cli)


def row(family="chat", split="validation"):
    return Episode(
        f"{family}.{split}",
        f"{family}.{split}",
        family,
        split,
        f"{family} {split} task",
        {},
        {"kind": "exact", "answer": "yes"},
        source="newly_authored",
        template_id=f"{family}.{split}",
    )


def result(episode, policy, *, success=True, cost=0.01):
    return EpisodeOutcome(
        episode.episode_id,
        episode.group_id,
        episode.family,
        policy,
        episode.content_hash,
        success,
        "completed",
        cost,
        10,
        1,
        evidence={"output_tokens": 20},
    )


class Runtime:
    def __init__(self, unknown=False):
        self.calls = []
        self.unknown = unknown

    def maximum_cost(self, row, policy):
        return 0.1

    async def __call__(self, row, policy_id, policy):
        self.calls.append(policy_id)
        return result(row, policy_id, cost=None if self.unknown else 0.01)


def skipped_collection_fixture(directory, *, pending=False):
    task = row(split='train')
    contract = {'profile': 'engineering_canary'}
    configs = {'baseline': {'kind': 'current'}, 'luna': {'kind': 'model', 'model': 'luna'}}
    identity = digest([digest(contract), 'development', task.content_hash, configs])
    budget = cli.ProductStudyBudget(directory / 'budget')
    reservation = budget.reserve('development', identity, .2, calls=2)
    start = {'event': 'collection_started', 'collection_id': identity,
             'episode_id': task.episode_id, 'configurations': configs,
             'reservation_id': reservation, 'maximum_cost_usd': .2}
    events = [start, {'event': 'episode_completed', 'collection_id': identity,
                     'outcome': asdict(result(task, 'luna', success=False, cost=None))}]
    attempts = [{'event': 'attempt_started', 'attempt_id': 'one',
                 'episode_id': task.episode_id, 'policy_id': 'luna'}]
    if not pending:
        attempts.append({**attempts[0], 'event': 'attempt_finished', 'cost_usd': None})
    for name, values in [('development-outcomes.jsonl', events), ('attempts.jsonl', attempts)]:
        for value in values:
            cli.append_event(directory / name, value)
    authorization = {'approved': True, 'contract_hash': digest(contract),
        'user_authorization_reference': 'TEST ONLY: skip this failed task',
        'collections': [{'collection_id': identity, 'stage': 'development',
            'original_events_hash': digest(events), 'attempt_events_hash': digest(attempts)}]}
    atomic_json(directory / 'authorized-collection-skips.json', authorization)
    return task, configs, contract


def defer_luna(directory, contract):
    atomic_json(directory / 'deferred-model-authorization.json', {
        'approved': True, 'contract_hash': digest(contract), 'models': ['luna'],
        'retain_terminal_unknown_bounds': True,
        'user_authorization_reference': 'TEST ONLY: defer Luna and leave holes'})
    atomic_json(directory / 'partial-collection-authorization.json', {
        'approved': True, 'contract_hash': digest(contract), 'retain_terminal_unknown_bounds': True,
        'deferred_model_authorization_hash': digest(cli.read_json(directory / 'deferred-model-authorization.json')),
        'user_authorization_reference': 'TEST ONLY: retain full bounds for terminal unknown charges'})


@pytest.mark.asyncio
async def test_deferred_model_has_no_dispatch_no_fake_outcome_and_safe_resume(tmp_path):
    task = row(split='train')
    contract = {'profile': 'engineering_canary', 'alternatives': ['luna', 'deepseek']}
    configs = {'baseline': {'kind': 'current'}, 'luna': {'kind': 'model', 'model': 'luna'},
               'deepseek': {'kind': 'model', 'model': 'deepseek'}}
    defer_luna(tmp_path, contract)
    runtime = Runtime()
    for _ in range(2):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert sorted(runtime.calls) == ['baseline', 'deepseek']
    events = cli.read_events(tmp_path / 'development-outcomes.jsonl')
    assert cli.effective_completions(events) == []
    assert len(cli.effective_outcomes(events)) == 2
    deferred = [e for e in events if e['event'] == 'collection_deferred']
    assert len(deferred) == 1 and deferred[0]['missing_configurations'] == ['luna']
    assert deferred[0]['fitting_eligible'] is False
    assert cli.ProductStudyBudget(tmp_path / 'budget').summary()['realized_cost_usd'] == .02
    assert cli.authorized_deferred_configurations(tmp_path, contract, 'pilot', configs) == {}
    assert cli.authorized_deferred_configurations(tmp_path, contract, 'test', configs) == {}


@pytest.mark.asyncio
async def test_defer_existing_terminal_luna_failure_keeps_charge_and_outcomes(tmp_path):
    task = row(split='calibration')
    contract = {'profile': 'engineering_canary', 'alternatives': ['luna']}
    configs = {'baseline': {'kind': 'current'}, 'luna': {'kind': 'model', 'model': 'luna'}}
    identity = digest([digest(contract), 'development', task.content_hash, configs])
    budget = cli.ProductStudyBudget(tmp_path / 'budget')
    reservation = budget.reserve('development', identity, .2, calls=2)
    start = {'event': 'collection_started', 'collection_id': identity,
             'episode_id': task.episode_id, 'configurations': configs,
             'reservation_id': reservation, 'maximum_cost_usd': .2}
    cli.append_event(tmp_path / 'development-outcomes.jsonl', start)
    for policy, cost in [('baseline', .01), ('luna', None)]:
        cli.append_event(tmp_path / 'development-outcomes.jsonl', {
            'event': 'episode_completed', 'collection_id': identity,
            'outcome': asdict(result(task, policy, success=policy != 'luna', cost=cost))})
        for event in ['attempt_started', 'attempt_finished']:
            cli.append_event(tmp_path / 'attempts.jsonl', {'event': event,
                'episode_id': task.episode_id, 'policy_id': policy, 'attempt_id': policy, 'cost_usd': cost})
    defer_luna(tmp_path, contract)
    runtime = Runtime()
    for _ in range(2):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert runtime.calls == []
    assert budget.summary()['outstanding_reserved_cost_usd'] == .2
    assert budget.summary()['realized_cost_usd'] == 0
    budget.reserve('development', 'next-row', .1, calls=1)
    assert budget.summary()['committed_usd'] == pytest.approx(.3)
    assert next(r for r in cli.outcomes(tmp_path / 'development-outcomes.jsonl') if r.policy_id == 'luna').cost_usd is None


@pytest.mark.asyncio
async def test_deferred_luna_requires_terminal_evidence_for_unknown_active_cost(tmp_path):
    task = row(split='train')
    contract = {'profile': 'engineering_canary', 'alternatives': ['luna']}
    configs = {'baseline': {'kind': 'current'}, 'luna': {'kind': 'model', 'model': 'luna'}}
    defer_luna(tmp_path, contract)
    runtime = Runtime(unknown=True)
    with pytest.raises(ValueError, match='terminal attempt evidence'):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert runtime.calls == ['baseline']
    value = cli.read_json(tmp_path / 'deferred-model-authorization.json')
    value['approved'] = False
    atomic_json(tmp_path / 'deferred-model-authorization.json', value)
    with pytest.raises(ValueError, match='invalid deferred'):
        cli.authorized_deferred_configurations(tmp_path, contract, 'development', configs)


@pytest.mark.asyncio
async def test_partial_collection_retains_active_unknown_cost_without_repeating(tmp_path):
    task = row(split='train')
    contract = {'profile': 'engineering_canary', 'alternatives': ['luna', 'deepseek']}
    configs = {'baseline': {'kind': 'current'}, 'deepseek': {'kind': 'model', 'model': 'deepseek'},
               'luna': {'kind': 'model', 'model': 'luna'}}
    defer_luna(tmp_path, contract)
    class TerminalRuntime(Runtime):
        async def __call__(self, row, policy_id, policy):
            self.calls.append(policy_id)
            for event in ['attempt_started', 'attempt_finished']:
                cli.append_event(tmp_path / 'attempts.jsonl', {'event': event,
                    'episode_id': row.episode_id, 'policy_id': policy_id,
                    'attempt_id': policy_id, 'cost_usd': None if policy_id == 'baseline' else .01})
            return result(row, policy_id, cost=None if policy_id == 'baseline' else .01)
    runtime = TerminalRuntime()
    for _ in range(2):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert sorted(runtime.calls) == ['baseline', 'deepseek']
    budget = cli.ProductStudyBudget(tmp_path / 'budget')
    assert budget.summary()['outstanding_reserved_cost_usd'] == pytest.approx(.3)
    assert budget.summary()['realized_cost_usd'] == 0
    budget.reserve('development', 'next', .1, calls=1)
    assert budget.summary()['committed_usd'] == pytest.approx(.4)


@pytest.mark.asyncio
async def test_authorized_skip_keeps_unknown_reservation_and_never_replays(tmp_path):
    task, configs, contract = skipped_collection_fixture(tmp_path)
    runtime = Runtime()
    for _ in range(2):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert runtime.calls == []
    events = cli.read_events(tmp_path / 'development-outcomes.jsonl')
    assert len([e for e in events if e['event'] == 'collection_abandoned']) == 1
    assert cli.effective_completions(events) == []
    observed = cli.effective_outcomes(events)
    assert len(observed) == 1 and observed[0].cost_usd is None and observed[0].success is False
    assert cli.configuration_score([task], observed, {task.episode_id: 'baseline'}) is None
    budget = cli.ProductStudyBudget(tmp_path / 'budget')
    assert budget.summary()['outstanding_reserved_cost_usd'] == .2
    assert budget.summary()['realized_cost_usd'] == 0
    # A different collection may reserve its own bound; the old bound remains.
    budget.reserve('development', 'different-row', .1, calls=1)
    assert budget.summary()['committed_usd'] == pytest.approx(.3)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['pending', 'tampered', 'unapproved'])
async def test_skip_rejects_pending_tampered_or_unapproved_evidence(tmp_path, failure):
    task, configs, contract = skipped_collection_fixture(tmp_path, pending=failure == 'pending')
    if failure == 'tampered':
        cli.append_event(tmp_path / 'attempts.jsonl', {'event': 'extra',
            'episode_id': task.episode_id, 'policy_id': 'luna'})
    if failure == 'unapproved':
        path = tmp_path / 'authorized-collection-skips.json'
        value = cli.read_json(path); value['approved'] = False; atomic_json(path, value)
    runtime = Runtime()
    with pytest.raises(ValueError):
        await cli._collect_row(task, configs, stage='development', runtime=runtime,
                               directory=tmp_path, contract=contract)
    assert runtime.calls == []
    assert cli.ProductStudyBudget(tmp_path / 'budget').summary()['outstanding_reserved_cost_usd'] == .2


@pytest.mark.asyncio
async def test_collection_resumes_without_repeating_paid_calls(tmp_path):
    runtime = Runtime()
    kwargs = dict(stage="pilot", runtime=runtime, directory=tmp_path, contract={"version": "test"})
    configurations = {
        "baseline": {"kind": "current"},
        "challenger": {"kind": "model", "model": "test"},
    }
    await cli.collect_row(row(split="pilot"), configurations, **kwargs)
    await cli.collect_row(row(split="pilot"), configurations, **kwargs)
    assert len(runtime.calls) == 2
    assert cli.ProductStudyBudget(tmp_path / "budget").summary()["realized_cost_usd"] == 0.02


@pytest.mark.asyncio
async def test_unknown_collection_charge_keeps_whole_reservation(tmp_path):
    runtime = Runtime(unknown=True)
    kwargs = dict(stage="pilot", runtime=runtime, directory=tmp_path, contract={})
    configurations = {
        "baseline": {"kind": "current"},
        "challenger": {"kind": "model", "model": "test"},
    }
    for _ in range(2):
        with pytest.raises(StudyBudgetExceeded):
            await cli.collect_row(row(split="pilot"), configurations, **kwargs)
    assert len(runtime.calls) == 1
    assert (
        cli.ProductStudyBudget(tmp_path / "budget").summary()["outstanding_reserved_cost_usd"]
        == 0.2
    )


@pytest.mark.asyncio
@pytest.mark.parametrize('collector', ['development', 'paired'])
async def test_real_runtime_timeout_persists_outcome_before_collector_stops(tmp_path, monkeypatch, collector):
    import asyncio
    import time
    from theriad.routing.episode_runtime import EpisodeRuntime
    from theriad.routing.episode_study import ProductStudyBudget, execute_pair, read_events
    monkeypatch.setattr('theriad.settings.secrets.get_active_assistant_provider', lambda: 'theriad')
    task = replace(row(split='pilot'), max_seconds=1, max_calls=1)
    calls = []
    class SlowProvider:
        name = 'openrouter'
        async def complete(self, request):
            calls.append(request.model)
            await asyncio.Event().wait()
    metadata = {'test/slow': {'provider': 'openrouter', 'input_per_1m': 1.,
                            'output_per_1m': 2., 'capabilities': ['text']}}
    runtime = EpisodeRuntime(directory=tmp_path, metadata=metadata,
        baseline_plans={task.episode_id: {'kind': 'current', 'assistant_provider': 'theriad',
            'chain': [['openrouter', 'test/slow']], 'system_prompt': 'Test only',
            'request_extras': {}}}, provider_factory=lambda *a, **kw: SlowProvider())
    policies = {'baseline': {'kind': 'current'}, 'challenger': {'kind': 'model', 'model': 'test/slow'}}
    output = tmp_path / 'pilot-outcomes.jsonl'
    started = time.monotonic()
    for _ in range(2):
        with pytest.raises(StudyBudgetExceeded):
            if collector == 'development':
                await cli.collect_row(task, policies, stage='pilot', runtime=runtime,
                                      directory=tmp_path, contract={})
            else:
                await execute_pair(row=task, policies=policies, order=['baseline', 'challenger'],
                    runner=runtime, maximum_costs={p: runtime.maximum_cost(task, v) for p, v in policies.items()},
                    budget=ProductStudyBudget(tmp_path / 'budget'), stage='pilot', output=output,
                    lock_hash='synthetic-no-paid-timeout-regression')
    # Grace is for cancellation/accounting, not extra generation time.
    assert time.monotonic() - started < 3
    assert len(calls) == 1
    terminal = [e['outcome'] for e in read_events(output) if e.get('event') == 'episode_completed']
    assert len(terminal) == 1
    assert terminal[0]['status'] == 'timeout'
    assert terminal[0]['success'] is False and terminal[0]['cost_usd'] is None
    attempts = [e for e in read_events(tmp_path / 'attempts.jsonl') if e.get('event') == 'attempt_finished' and e.get('kind') == 'model']
    assert len(attempts) == 1 and attempts[0]['status'] == 'cancelled'
    assert ProductStudyBudget(tmp_path / 'budget').summary()['outstanding_reserved_cost_usd'] > 0


def test_different_successful_models_get_distinct_cost_scores():
    rows = [row(family) for family in FAMILIES]
    observed = [
        result(r, policy, cost=cost) for r in rows for policy, cost in (("a", 0.1), ("b", 0.03))
    ]
    scores = [
        cli.configuration_score(rows, observed, {r.episode_id: policy for r in rows})
        for policy in ("a", "b")
    ]
    assert scores[0]["success"] == scores[1]["success"] == 1
    assert scores[0]["cost_usd"] > scores[1]["cost_usd"]
    observed[0] = replace(observed[0], fixture_hash="stale")
    with pytest.raises(ValueError, match="frozen fixture"):
        cli.configuration_score(rows, observed, {r.episode_id: "a" for r in rows})


def test_selection_reads_only_development_and_freezes_once(tmp_path):
    rows = [row(family, split) for split in ("train", "validation") for family in FAMILIES]
    split_values = {
        split: [asdict(r) for r in rows if r.split == split] for split in ("train", "validation")
    }
    contract = {
        "alternatives": ["candidate"],
        "created_at": "2026-09-07T00:00:00Z",
        "split_hashes": {split: digest(values) for split, values in split_values.items()},
    }
    atomic_json(tmp_path / "contract.json", contract)
    for split, values in split_values.items():
        atomic_json(tmp_path / "fixtures" / f"{split}.json", values)
    for r in rows:
        for policy, cost in (("baseline", 0.02), ("candidate", 0.01)):
            cli.append_event(
                tmp_path / "development-outcomes.jsonl",
                {"event": "episode_completed", "outcome": asdict(result(r, policy, cost=cost))},
            )
    catalog = {
        "candidate": {
            "capabilities": ["text"],
            "context_window": 100000,
            "input_per_1m": 1,
            "output_per_1m": 1,
        }
    }
    with pytest.raises(RuntimeError, match="separate train, calibration"):
        cli.select_challenger(tmp_path, Runtime(), contract, catalog)
    assert not (tmp_path / "challenger.json").exists()
    assert not (tmp_path / "fixtures" / "test.json").exists()


def test_changed_development_fixture_is_rejected(tmp_path):
    atomic_json(tmp_path / "contract.json", {"split_hashes": {"train": "old"}})
    atomic_json(tmp_path / "fixtures/train.json", [asdict(row(split="train"))])
    with pytest.raises(ValueError, match="frozen split"):
        cli.load_rows(tmp_path, ["train"])


@pytest.mark.asyncio
async def test_canonical_prepare_writes_frozen_artifacts_with_mocked_catalog(tmp_path, monkeypatch):
    """Exercise prepare without credentials, network, or provider dispatch."""
    import httpx

    from theriad.routing.episode_fixtures import build_four_way_seed_corpus
    from theriad.routing import episode_runtime

    rows = build_four_way_seed_corpus()
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text("\n".join(json.dumps(asdict(value)) for value in rows) + "\n")
    monkeypatch.setattr(cli, "registry_model_pool", lambda: {
        "test/model": {"model": "test/model", "provider": "openrouter", "input_per_1m": 0.1,
                        "output_per_1m": 0.1, "context_window": 100_000, "registry_tiers": [],
                        "registry_source": "test"}
    })
    monkeypatch.setattr(episode_runtime, "freeze_baseline", lambda row: {
        "kind": "current", "assistant_provider": "theriad",
        "chain": [["openrouter", "test/model"]], "request_extras": {},
    })

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {"data": [{"id": "test/model", "pricing": {"prompt": "0.0000001", "completion": "0.0000001"},
                               "architecture": {"input_modalities": ["text", "image"]},
                               "supported_parameters": ["tools"], "context_length": 100_000}]}

    class Client:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        async def get(self, url):
            assert url.endswith("/api/v1/models")
            return Response()

    monkeypatch.setattr(httpx, "AsyncClient", Client)
    await cli.prepare(tmp_path / "study", [], corpus)

    study = tmp_path / "study"
    assert (study / "contract.json").is_file()
    assert (study / "model-pool.json").is_file()
    assert (study / "budget-config.json").is_file()
    assert {path.stem for path in (study / "fixtures").glob("*.json")} == set(cli.ALL_SPLITS)
    contract, catalog, plans = cli.validate_contract(study)
    proposal = cli.read_json(study / "BUDGET_PROPOSAL.json")
    assert contract["budget_usd"] == proposal["budget_config"]["hard_cap_usd"]
    assert contract["budget_proposal_hash"] == digest(proposal)
    assert proposal["stages"]["test"]["configuration_count"] == 2 * len([r for r in rows if r.split == "test"])
    assert catalog["test/model"]["capabilities"] == ["text", "search", "vision", "tools"]
    assert len(plans) == len([r for r in rows if r.split in ("pilot", *cli.DEVELOPMENT_SPLITS)])
    assert json.loads((study / "model-pool.json").read_text())["exclusions_are_frozen"] is True


@pytest.mark.asyncio
async def test_terminal_unknown_can_keep_full_bound_without_repeating_or_imputing(tmp_path):
    runtime = Runtime(unknown=True)
    configurations = {'baseline': {'kind': 'current'}, 'challenger': {'kind': 'model', 'model': 'test'}}
    kwargs = dict(stage='pilot', runtime=runtime, directory=tmp_path, contract={}, allow_bounded_unknown=True)
    await cli.collect_row(row(split='pilot'), configurations, **kwargs)
    await cli.collect_row(row(split='pilot'), configurations, **kwargs)
    assert len(runtime.calls) == 2
    budget = cli.ProductStudyBudget(tmp_path / 'budget')
    assert budget.summary()['committed_usd'] == .2
    assert budget.summary()['realized_cost_usd'] == 0
    assert all(r.cost_usd is None for r in cli.outcomes(tmp_path / 'pilot-outcomes.jsonl'))
    budget.reserve('pilot', 'another-bounded-episode', .1, calls=1)
    with pytest.raises(StudyBudgetExceeded):
        budget.reserve('pilot', 'cannot-evade-cap', 10, calls=1)


@pytest.mark.asyncio
async def test_append_only_cost_reconciliation_resumes_completed_unknown_collection(tmp_path):
    runtime = Runtime(unknown=True)
    configurations = {'baseline': {'kind': 'current'}, 'challenger': {'kind': 'model', 'model': 'test'}}
    kwargs = dict(stage='pilot', runtime=runtime, directory=tmp_path, contract={}, allow_bounded_unknown=True)
    await cli.collect_row(row(split='pilot'), configurations, **kwargs)
    path = tmp_path / 'pilot-outcomes.jsonl'
    original_events = cli.read_events(path)
    for event in original_events:
        if event.get('event') == 'episode_completed':
            amended = {**event['outcome'], 'cost_usd': .01}
            cli.append_event(path, {'event': 'episode_cost_reconciled', 'collection_id': event['collection_id'],
                'original_outcome_hash': digest(event['outcome']), 'outcome': amended})
    terminal = next(e for e in original_events if e.get('event') == 'collection_completed')
    cli.append_event(path, {'event': 'collection_cost_reconciled', 'collection_id': terminal['collection_id'],
        'original_completion_hash': digest(terminal), 'completion': {**terminal, 'cost_usd': .02}})
    await cli.collect_row(row(split='pilot'), configurations, **kwargs)
    assert len(runtime.calls) == 2
    assert cli.ProductStudyBudget(tmp_path / 'budget').summary()['realized_cost_usd'] == .02
    assert [e for e in cli.read_events(path) if e.get('event') == 'collection_completed'][0]['cost_usd'] is None


def test_report_uses_real_paired_report_fields_and_observed_family_means(tmp_path):
    rows = [replace(row(family, 'test'), episode_id=f'{family}-test-{i}', group_id=f'{family}-group-{i}',
                    template_id=f'{family}-template-{i}', prompt=f'{family} synthetic task {i}') for family in FAMILIES for i in range(2)]
    values = [asdict(r) for r in rows]
    atomic_json(tmp_path / 'contract.json', {'split_hashes': {'test': digest(values)}})
    atomic_json(tmp_path / 'fixtures/test.json', values)
    lock = cli.freeze_test(tmp_path / 'test-lock.json', episodes=rows, baseline={}, challenger={'model': 'candidate'},
                           pricing={}, power={'required_independent_groups': 100, 'used_variance': .25})
    for r in rows:
        for policy, cost in [('baseline', .01), ('challenger', .005)]:
            observed = replace(result(r, policy, cost=cost), policy_hash=lock['policy_hashes'][policy])
            cli.append_event(tmp_path / 'test-outcomes.jsonl', {'event': 'episode_completed', 'outcome': asdict(observed)})
    report = cli.render_report(tmp_path)
    assert report['primary_metrics']['baseline_success'] == 1
    assert report['primary_metrics']['challenger_cost_usd'] == .005
    assert report['decision'] == 'inconclusive'
    assert report['protocol_verified'] is False
    text = (tmp_path / 'REPORT.md').read_text()
    assert 'Locked test observations' in text
    assert '100 independent groups' in text
    assert '$0.00500000' in text


@pytest.mark.asyncio
async def test_candidate_review_guard_precedes_reservation_and_dispatch(tmp_path):
    candidate = replace(row(split="pilot"), fixture={"source_metadata": {"independence_status": "screened_candidate_not_certified"}})
    runtime = Runtime()
    with pytest.raises(ValueError, match="quarantined"):
        await cli.collect_row(candidate, {"baseline": {"kind": "current"}}, stage="pilot", runtime=runtime,
                              directory=tmp_path, contract={"version": "test"})
    assert runtime.calls == []
    assert list(tmp_path.iterdir()) == []
