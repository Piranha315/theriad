"""Census omissions, stale identities, exposure and dependence must fail closed."""
from dataclasses import asdict
import importlib.util
from pathlib import Path

import pytest

from theriad.routing.episode_fixtures import build_four_way_seed_corpus, require_collection_ready
from theriad.routing.episode_study import Episode

path = Path(__file__).resolve().parents[3] / 'scripts/finalize_public_episode_semantic_census.py'
spec = importlib.util.spec_from_file_location('semantic_census', path)
census = importlib.util.module_from_spec(spec)
spec.loader.exec_module(census)


def inputs():
    rows = [asdict(r) for r in build_four_way_seed_corpus()]
    ledger = [{'inventory': 'seed', 'index': i, 'episode_id': r['episode_id'],
               'row_sha256': census.canonical_sha(r), 'mechanism': f'unique_{i}', 'reviewer': 'test reviewer'}
              for i, r in enumerate(rows)]
    policy = {'version': 'test', 'reviewer': 'test reviewer', 'aliases': {}, 'exclusions': [],
              'resolved_flags': [], 'additional_edges': []}
    return rows, [], ledger, {'clusters': []}, policy


@pytest.mark.parametrize('failure', ['missing', 'duplicate', 'hash', 'identity', 'unknown_index'])
def test_census_rejects_incomplete_or_stale_row_decisions(failure):
    rows, legacy, ledger, prior, policy = inputs()
    if failure == 'missing': ledger.pop()
    if failure == 'duplicate': ledger.append(ledger[0])
    if failure == 'hash': rows[0]['prompt'] += ' changed'
    if failure == 'identity': ledger[0]['episode_id'] = ledger[1]['episode_id']
    if failure == 'unknown_index': ledger[0]['index'] = 99999
    with pytest.raises(ValueError):
        census.finalize(rows, legacy, ledger, prior, policy)


def test_legacy_overlap_closes_transitively_and_never_moves_exposure_later():
    rows, legacy, ledger, prior, policy = inputs()
    ids = ['coding.rolling_average.trial-1', 'coding.decimal_money.trial-1']
    for d in ledger:
        if d['episode_id'] in ids: d['mechanism'] = 'shared'
    old = {'id': 'old-task', 'prompt': 'previously used task'}
    legacy.append(old)
    ledger.append({'inventory': 'legacy', 'index': 0, 'episode_id': old['id'],
                   'row_sha256': census.canonical_sha(old), 'mechanism': 'shared', 'reviewer': 'test reviewer'})
    kept, excluded, decisions, graph = census.finalize(rows, legacy, ledger, prior, policy)
    changed = [r for r in kept if r['episode_id'] in ids]
    assert len({r['group_id'] for r in changed}) == 1
    assert {r['split'] for r in changed} == {'train'}
    assert all(d['legacy_matches'] == ['legacy:old-task'] for d in decisions if d['episode_id'] in ids)
    assert next(r for r in rows if r['episode_id'] == ids[1])['split'] == 'test'
    assert not excluded


def test_cross_family_component_is_excluded_instead_of_counted_twice():
    rows, legacy, ledger, prior, policy = inputs()
    ids = ['tools.csv_join.trial-1', 'agents.invoice_reconciliation_project.trial-1']
    policy['additional_edges'] = [{'members': ids, 'rationale': 'shared join', 'kind': 'cross_family'}]
    kept, excluded, decisions, graph = census.finalize(rows, legacy, ledger, prior, policy)
    assert {r['episode_id'] for r in excluded} == set(ids)
    assert not set(ids) & {r['episode_id'] for r in kept}
    assert len(kept) + len(excluded) == len(rows)
    assert all(d['decision'] == 'excluded' for d in decisions if d['episode_id'] in ids)


def test_prior_components_survive_and_exclusion_does_not_drop_accounting():
    rows, legacy, ledger, prior, policy = inputs()
    ids = ['coding.merge_intervals.trial-1', 'coding.rolling_average.trial-1', 'coding.decimal_money.trial-1']
    prior['clusters'] = [{'members': ids[:2], 'rationale': 'old edge', 'decision': 'group_conservatively'}]
    policy['additional_edges'] = [{'members': ids[1:], 'rationale': 'new edge', 'kind': 'within_family'}]
    policy['exclusions'] = [{'episode_id': ids[1], 'reason': 'contradictory prompt'}]
    kept, excluded, decisions, graph = census.finalize(rows, legacy, ledger, prior, policy)
    related = [r for r in kept if r['episode_id'] in ids]
    assert len({r['group_id'] for r in related}) == 1
    assert {r['split'] for r in related} == {'train'}
    assert len(excluded) == 1 and len(decisions) == len(rows)


def test_completed_review_is_still_blocked_without_collection_clearance():
    kept, *_ = census.finalize(*inputs())
    with pytest.raises(ValueError, match='review complete but collection clearance missing'):
        require_collection_ready([Episode.from_dict(r) for r in kept])


def test_unknown_edges_and_conflicting_exclusions_fail_closed():
    rows, legacy, ledger, prior, policy = inputs()
    policy['additional_edges'] = [{'members': ['unknown'], 'rationale': 'test', 'kind': 'test'}]
    with pytest.raises(ValueError, match='unknown edge'):
        census.finalize(rows, legacy, ledger, prior, policy)
    policy['additional_edges'] = []
    key = rows[0]['episode_id']
    policy['exclusions'] = [{'episode_id': key, 'reason': 'bad'}]
    policy['resolved_flags'] = [{'episode_id': key, 'reason': 'good'}]
    with pytest.raises(ValueError, match='conflicting flag'):
        census.finalize(rows, legacy, ledger, prior, policy)
