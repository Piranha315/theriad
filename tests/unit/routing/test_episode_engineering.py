"""No-paid engineering admission and budget/exposure regression tests."""
from __future__ import annotations

import importlib.util
import json
from dataclasses import replace
from pathlib import Path

import pytest

from theriad.routing.episode_engineering import (
    ALLOCATIONS, PROFILE, authored_provenance, engineering_price_bounds,
    prepare_admission, require_paid_authorization, verify_admission,
)
from theriad.routing.episode_fixtures import require_collection_ready
from theriad.routing.episode_study import atomic_json, digest

ROOT = Path(__file__).resolve().parents[3]
CORPUS = ROOT / "data/router-episode-corpus-20260911-v9/combined-episodes.jsonl"
_spec = importlib.util.spec_from_file_location("engineering_cli_tests", ROOT / "scripts/run_router_episode_study.py")
cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(cli)


@pytest.fixture
def admitted():
    return prepare_admission(CORPUS)


def test_pinned_selection_preserves_rows_and_removes_prior_overlap_from_smoke(admitted):
    rows, admission = admitted
    originals = {raw['episode_id']: raw for raw in map(json.loads, CORPUS.read_text().splitlines())}
    assert len(rows) == len({r.group_id for r in rows}) == 156
    assert {r.split for r in rows} == set(ALLOCATIONS)
    for row in rows:
        assert row.split == originals[row.episode_id]['split']
        assert row.group_id == originals[row.episode_id]['group_id']
        assert row.fixture['semantic_review']['collection_ready'] is False
    smoke = [r for r in rows if r.split == 'smoke']
    assert len(smoke) == 21
    assert sum(r.family == 'chat' for r in smoke) == 7
    assert all(r.fixture['semantic_review']['disposition'] == 'retained_research_only' for r in smoke)
    assert len({r['group_id'] for r in admission['diagnostic_only_prior_overlap']}) == 1
    assert prepare_admission(CORPUS)[1] == admission
    with pytest.raises(ValueError, match='quarantined'):
        require_collection_ready(rows)


def test_corpus_edit_cannot_get_engineering_admission(tmp_path):
    changed = tmp_path / 'corpus.jsonl'
    changed.write_bytes(CORPUS.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='exact reviewed'):
        prepare_admission(changed)


def test_admission_binds_fixture_and_rejects_heldout_or_excluded_row(tmp_path, admitted):
    rows, admission = admitted
    atomic_json(tmp_path / 'engineering-admission.json', admission)
    contract = {'profile': PROFILE, 'engineering_admission_hash': digest(admission)}
    verify_admission(tmp_path, rows, contract)
    for changed in (replace(rows[0], prompt='modified'), replace(rows[0], split='test')):
        with pytest.raises(ValueError):
            verify_admission(tmp_path, [changed], contract)
    excluded = replace(rows[0], fixture={**rows[0].fixture, 'semantic_review': {
        **rows[0].fixture['semantic_review'], 'disposition': 'excluded'}})
    with pytest.raises(ValueError, match='retained'):
        verify_admission(tmp_path, [excluded], contract)
    admission['research_collection_clearance'] = True
    atomic_json(tmp_path / 'engineering-admission.json', admission)
    with pytest.raises(ValueError, match='changed'):
        verify_admission(tmp_path, rows, contract)


def test_paid_authorization_is_explicit_and_bound_to_source_budget_and_phase(tmp_path):
    contract = {'budget_proposal_hash': 'budget', 'budget_config': {'hard_cap_usd': 10},
                'protocol_manifest_hash': 'protocol'}
    with pytest.raises(ValueError, match='missing'):
        require_paid_authorization(tmp_path, contract, phase='development')
    approval = {'phase': 'development', 'paid_calls_authorized': True,
                'contract_hash': digest(contract), 'user_authorization_reference': 'test-only authorization fixture',
                **contract}
    atomic_json(tmp_path / 'authorization-development.json', approval)
    require_paid_authorization(tmp_path, contract, phase='development')
    for key in ('contract_hash', 'protocol_manifest_hash', 'budget_proposal_hash', 'user_authorization_reference'):
        atomic_json(tmp_path / 'authorization-development.json', {**approval, key: ''})
        with pytest.raises(ValueError):
            require_paid_authorization(tmp_path, contract, phase='development')
    with pytest.raises(ValueError):
        require_paid_authorization(tmp_path, contract, phase='smoke')


def test_source_repair_needs_its_own_exact_approval_and_preserves_original_snapshot(tmp_path):
    from theriad.routing.episode_engineering import execution_protocol_record
    contract = {'protocol_manifest_hash': 'original', 'budget_config': {'stage_caps_usd': {'pilot': .21, 'development': 14.25}}}
    assert execution_protocol_record(tmp_path, contract, {'manifest_hash': 'original'}) == 'execution-protocol.json'
    with pytest.raises(ValueError, match='source/environment changed'):
        execution_protocol_record(tmp_path, contract, {'manifest_hash': 'replacement'})
    amendment = {'approved': True, 'contract_hash': digest(contract),
        'original_protocol_hash': 'original', 'replacement_protocol_hash': 'replacement',
        'approved_phase_a_cap_usd': 14.46, 'user_authorization_reference': 'TEST ONLY'}
    atomic_json(tmp_path / 'source-amendment.json', amendment)
    assert execution_protocol_record(tmp_path, contract, {'manifest_hash': 'replacement'}) == 'execution-protocol-replacement.json'
    for key in ['approved', 'contract_hash', 'replacement_protocol_hash', 'original_protocol_hash', 'user_authorization_reference', 'approved_phase_a_cap_usd']:
        atomic_json(tmp_path / 'source-amendment.json', {**amendment, key: None})
        with pytest.raises(ValueError, match='source amendment'):
            execution_protocol_record(tmp_path, contract, {'manifest_hash': 'replacement'})


def test_provider_billing_reconciliation_preserves_raw_unknown_and_rejects_tampering(tmp_path):
    import hashlib
    from theriad.routing.episode_engineering import apply_billing_reconciliation
    original = {'event': 'attempt_finished', 'attempt_id': 'attempt', 'cost_usd': None}
    metadata = {'id': 'generation', 'total_cost': .00007523}
    atomic_json(tmp_path / 'contract.json', {})
    atomic_json(tmp_path / 'cancelled-generation-metadata.json', metadata)
    record = {'contract_hash': digest({}), 'attempt_id': 'attempt',
        'original_attempt_terminal_hash': digest(original), 'generation_id': 'generation',
        'provider_metadata_sha256': hashlib.sha256((tmp_path / 'cancelled-generation-metadata.json').read_bytes()).hexdigest(),
        'cancelled_request_cost_usd': metadata['total_cost']}
    atomic_json(tmp_path / 'billing-reconciliation.json', record)
    attempts = {'attempt': dict(original)}
    apply_billing_reconciliation(tmp_path, attempts, [original])
    assert original['cost_usd'] is None
    assert attempts['attempt']['cost_usd'] == metadata['total_cost']
    assert attempts['attempt']['cost_basis'] == 'provider_generation_metadata'
    atomic_json(tmp_path / 'cancelled-generation-metadata.json', {**metadata, 'total_cost': 0})
    with pytest.raises(ValueError, match='immutable provider evidence'):
        apply_billing_reconciliation(tmp_path, attempts, [original])


def test_authored_provenance_points_to_the_real_pinned_source_file(admitted):
    rows, _ = admitted
    for row in rows:
        if row.source == 'newly_authored_20260911':
            meta = authored_provenance(row)
            path = meta['provenance_url'].split('#')[0].removeprefix('repo:')
            assert (ROOT / path).is_file()
            import hashlib
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == meta['source_file_sha256']


def test_reservation_covers_cache_write_and_price_tiers():
    bounds = engineering_price_bounds({'prompt': '0.0000002', 'completion': '0.0000012',
        'input_cache_write': '0.00000025', 'overrides': [
            {'min_prompt_tokens': 272000, 'prompt': '0.0000004', 'completion': '0.0000018',
             'input_cache_write': '0.0000005'}]})
    assert bounds['reservation_input_per_1m'] == pytest.approx(.9)
    assert bounds['reservation_output_per_1m'] == pytest.approx(1.8)
    assert bounds['require_provider_billing'] and bounds['forbid_paid_addons']
    for invalid in ('NaN', '-1', True):
        with pytest.raises(ValueError):
            engineering_price_bounds({'prompt': invalid, 'completion': '0'})


def fake_catalog_and_baseline(monkeypatch, tmp_path):
    from theriad.routing import episode_runtime
    models = ('test/cheap', 'test/strong', 'test/unselected')
    monkeypatch.setattr(cli, 'registry_model_pool', lambda: {m: {'model': m} for m in models})
    def baseline(row):
        if row.family in {'chat', 'coding'}:
            return {'kind': 'current', 'chain': [['openrouter', models[1]]]}
        return {'kind': 'current', 'models': {'agent_planner': models[1]}}
    monkeypatch.setattr(episode_runtime, 'freeze_baseline', baseline)
    catalog = tmp_path / 'public-catalog.json'
    atomic_json(catalog, {'data': [{'id': m, 'pricing': {'prompt': '0.000001', 'completion': '0.000002'},
                                  'context_length': 2000000, 'supported_parameters': ['tools']}
                                 for m in models]})
    return models, catalog


@pytest.mark.asyncio
async def test_preparation_restricts_grid_and_cannot_dispatch_without_approval(tmp_path, monkeypatch):
    models, catalog = fake_catalog_and_baseline(monkeypatch, tmp_path)
    directory = tmp_path / 'engineering'
    await cli.prepare(directory, models[:2], CORPUS, profile=PROFILE, catalog_path=catalog)
    contract = cli.read_json(directory / 'contract.json')
    assert contract['alternatives'] == list(models[:2])
    assert contract['selection_cost_ratio'] == .95
    assert contract['protocol_manifest_hash']
    assert not (directory / 'fixtures/test.json').exists()
    assert not (directory / 'test-lock.json').exists()
    proposal = cli.read_json(directory / 'BUDGET_PROPOSAL.json')
    assert proposal['stages']['development']['configuration_count'] == 390
    assert proposal['stages']['smoke']['configuration_count'] == 42
    assert proposal['stages']['test']['configuration_count'] == 0
    def forbidden():
        pytest.fail('Credentials accessed before explicit paid authorization')
    monkeypatch.setattr(cli, 'load_credentials', forbidden)
    with pytest.raises(ValueError, match='authorization is missing'):
        await cli.run_study(directory, stop_after='fit')
    assert not (directory / 'attempts.jsonl').exists()
    await cli.prepare(directory, models[:2], CORPUS, profile=PROFILE, catalog_path=catalog)
    with pytest.raises(ValueError, match='different frozen alternatives'):
        await cli.prepare(directory, (models[0], models[2]), CORPUS, profile=PROFILE, catalog_path=catalog)
    assert cli.read_json(directory / 'training-readiness.json')['router_input_aggregation_checked'] is True


@pytest.mark.asyncio
async def test_fit_stop_never_loads_test_even_with_complete_development(tmp_path, monkeypatch, admitted):
    models, catalog = fake_catalog_and_baseline(monkeypatch, tmp_path)
    directory = tmp_path / 'engineering'
    await cli.prepare(directory, models[:2], CORPUS, profile=PROFILE, catalog_path=catalog)
    contract = cli.read_json(directory / 'contract.json')
    atomic_json(directory / 'authorization-development.json', {
        'phase': 'development', 'paid_calls_authorized': True,
        'user_authorization_reference': 'synthetic orchestration test, no provider execution',
        'contract_hash': digest(contract), 'budget_proposal_hash': contract['budget_proposal_hash'],
        'budget_config': contract['budget_config'], 'protocol_manifest_hash': contract['protocol_manifest_hash']})
    loaded = []
    real_load = cli.load_rows
    def load(directory, splits):
        loaded.extend(splits)
        assert 'test' not in splits and 'smoke' not in splits
        return real_load(directory, splits)
    monkeypatch.setattr(cli, 'load_rows', load)
    monkeypatch.setattr(cli, 'load_credentials', lambda: None)
    monkeypatch.setattr(cli, 'preflight', lambda _: None)
    async def no_dispatch(*args, **kwargs):
        return []
    monkeypatch.setattr(cli, 'collect_row', no_dispatch)
    monkeypatch.setattr(cli, 'select_challenger', lambda *_: {'kind': 'surface'})
    monkeypatch.setattr(cli, 'require_surface_challenger', lambda _: None)
    monkeypatch.setattr(cli, 'render_report', lambda _: None)
    # This is a stage-boundary unit test. The canonical rehearsal separately
    # proves actual fitting and routing; this stub is never performance evidence.
    await cli.run_study(directory, stop_after='fit')
    assert loaded and not (directory / 'test-lock.json').exists()


def test_dispatch_caps_and_billing_are_enforced_without_mutating_request():
    from theriad.core.providers.interfaces import CompletionRequest, CompletionResponse, Message
    from theriad.routing.episode_hooks import EpisodeRoutingSession, EpisodeRoutingError
    quote = {'input_per_1m': 1., 'output_per_1m': 2., 'context_window': 20000,
             **engineering_price_bounds({'prompt': '0.000001', 'completion': '0.000002'})}
    session = EpisodeRoutingSession('test', model_metadata={'test/paid': quote}, max_input_tokens=2000, max_output_tokens=50)
    original = CompletionRequest(model='test/paid', messages=[Message('user', 'Hi')],
                                 extra={'provider': {'sort': 'price', 'max_price': {'prompt': .5}}})
    guarded = session.guard_request(original)
    assert guarded.extra['provider']['max_price'] == {'prompt': .5, 'completion': 2., 'request': 0., 'image': 0.}
    assert original.extra['provider']['max_price'] == {'prompt': .5}
    assert guarded.max_tokens == 50
    for extra in ({'plugins': []}, {'models': ['another/model']}, {'provider': {'max_price': {'prompt': float('nan')}}}):
        with pytest.raises(EpisodeRoutingError):
            session.guard_request(replace(original, extra=extra))
    response = CompletionResponse(model='test/paid', content='Hi', prompt_tokens=10, completion_tokens=10,
                                  latency_ms=1, provider='openrouter')
    assert session.price_response(response, None) is None  # no made-up settled cost
    assert session.price_response(replace(response, cost_usd=.004), None) == .004


def test_runtime_and_proposal_use_the_same_reservation_rates(tmp_path, admitted):
    from theriad.routing.episode_runtime import EpisodeRuntime
    from theriad.routing.episode_planning import propose_episode_budget
    row = next(r for r in admitted[0] if r.family == 'chat' and r.split == 'pilot')
    quote = {'input_per_1m': 1., 'output_per_1m': 2., 'context_window': 2000000, 'capabilities': ['text'],
             **engineering_price_bounds({'prompt': '0.000001', 'completion': '0.000002', 'input_cache_write': '0.0000005'})}
    catalog = {'test/paid': quote}
    plans = {row.episode_id: {'chain': [['openrouter', 'test/paid']]}}
    runtime = EpisodeRuntime(directory=tmp_path, metadata=catalog, baseline_plans=plans)
    maximum = runtime.maximum_cost(row, {'kind': 'model', 'model': 'test/paid'})
    proposal = propose_episode_budget([row], catalog, plans, {row.episode_id: {'test/paid': True}}, alternatives=['test/paid'])
    assert proposal['stages']['pilot']['model_reservation_usd'] == pytest.approx(2 * maximum)
    assert maximum == pytest.approx(row.max_calls * (row.max_input_tokens * 1.5 + row.max_output_tokens * 2.) / 1e6)


def test_unreachable_long_context_tier_is_excluded_only_with_an_enforced_bound():
    from theriad.core.providers.interfaces import CompletionRequest, Message
    from theriad.routing.episode_hooks import EpisodeRoutingSession, EpisodeRoutingError
    pricing = {'prompt': '0.000001', 'completion': '0.000002', 'overrides': [
        {'min_prompt_tokens': 272000, 'prompt': '0.000005', 'completion': '0.00001'}]}
    bounded = engineering_price_bounds(pricing, max_input_tokens=24000)
    assert bounded['reservation_input_per_1m'] == 1.
    assert bounded['reservation_output_per_1m'] == 2.
    assert engineering_price_bounds(pricing)['reservation_output_per_1m'] == 10.
    session = EpisodeRoutingSession('test', model_metadata={'test/model': bounded})
    request = CompletionRequest(model='test/model', messages=[Message('user', 'x' * 24000)])
    with pytest.raises(EpisodeRoutingError, match='pricing-tier'):
        session.guard_request(request)


@pytest.mark.asyncio
@pytest.mark.parametrize('failure', ['unknown_charge', 'budget_exhausted', 'changed_source'])
async def test_engineering_dispatch_stops_and_keeps_reservations(tmp_path, monkeypatch, failure):
    from theriad.routing import episode_protocol
    from theriad.routing.episode_study import EpisodeOutcome, ProductStudyBudget
    from theriad.routing.study_budget import StudyBudgetExceeded
    models, catalog = fake_catalog_and_baseline(monkeypatch, tmp_path)
    directory = tmp_path / 'engineering'
    await cli.prepare(directory, models[:2], CORPUS, profile=PROFILE, catalog_path=catalog)
    contract = cli.read_json(directory / 'contract.json')
    atomic_json(directory / 'authorization-development.json', {
        'phase': 'development', 'paid_calls_authorized': True,
        'user_authorization_reference': 'TEST ONLY: no actual provider execution',
        'contract_hash': digest(contract), 'budget_proposal_hash': contract['budget_proposal_hash'],
        'budget_config': contract['budget_config'], 'protocol_manifest_hash': contract['protocol_manifest_hash']})
    if failure == 'changed_source':
        snapshot = cli.read_json(directory / 'execution-protocol.json')
        monkeypatch.setattr(episode_protocol, 'capture_protocol', lambda _: {**snapshot, 'manifest_hash': 'changed'})
        monkeypatch.setattr(cli, 'load_credentials', lambda: pytest.fail('Credentials accessed after source change'))
        with pytest.raises(ValueError, match='source/environment changed'):
            await cli.run_study(directory, stop_after='fit')
        return
    calls = []
    maximum = .01 if failure == 'unknown_charge' else contract['budget_config']['hard_cap_usd'] + 1
    class Runtime:
        def maximum_cost(self, row, policy):
            return maximum
        async def __call__(self, row, policy_id, policy):
            calls.append(policy_id)
            return EpisodeOutcome(row.episode_id, row.group_id, row.family, policy_id,
                row.content_hash, True, 'completed', None, 1., 1,
                evidence={'accounting': {'pending_ids': []}})
    row = cli.load_rows(directory, ['pilot'])[0]
    configs = {'baseline': {'kind': 'current'}, 'challenger': {'kind': 'model', 'model': models[0]}}
    for _ in range(2):
        with pytest.raises(StudyBudgetExceeded):
            await cli.collect_row(row, configs, stage='pilot', runtime=Runtime(), directory=directory,
                                  contract=contract, allow_bounded_unknown=True)
    assert len(calls) == (1 if failure == 'unknown_charge' else 0)
    summary = ProductStudyBudget(directory / 'budget').summary()
    assert summary['realized_cost_usd'] == 0
    assert summary['outstanding_reserved_cost_usd'] == (.02 if failure == 'unknown_charge' else 0)
