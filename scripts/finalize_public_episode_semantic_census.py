#!/usr/bin/env python3
"""Bind a complete qualitative census to immutable inputs and apply its decisions.

No distance threshold clears a row. The manually authored ledger and policy are
inputs; this program checks completeness, closes dependence edges, and excludes
cross-family components unsupported by the existing stratified inference design.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import Episode, validate_episodes

STAGES = ('pilot', 'smoke', 'train', 'calibration', 'threshold_selection', 'test')
FAMILIES = ('chat', 'coding', 'search', 'tools', 'agents')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def canonical_sha(row):
    return sha(json.dumps(row, sort_keys=True).encode())


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def write_json(path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + '\n')


def write_rows(path, rows):
    path.write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in rows))


def finalize(rows, legacy, ledger, prior, policy):
    """Return retained rows, excluded rows, complete decisions and audit graph."""
    inventories = {name: [r for r in rows if r['source'] == name]
                   for name in ('gsm8k', 'humaneval', 'hotpotqa')}
    inventories['seed'] = [r for r in rows if r['source'] not in inventories]
    inventories['legacy'] = legacy
    expected = {(name, index): row for name, values in inventories.items()
                for index, row in enumerate(values)}
    actual = {}
    for decision in ledger:
        key = (decision['inventory'], decision['index'])
        if key in actual:
            raise ValueError('duplicate census decision')
        if key not in expected:
            raise ValueError('unknown census row')
        row = expected[key]
        if decision['episode_id'] != row.get('episode_id', row.get('id')) or decision['row_sha256'] != canonical_sha(row):
            raise ValueError('census identity or row hash mismatch')
        if not decision.get('mechanism') or not decision.get('reviewer'):
            raise ValueError('qualitative mechanism and reviewer required')
        actual[key] = decision
    if set(expected) != set(actual):
        raise ValueError('incomplete row census')
    by_id = {r['episode_id']: r for r in rows}
    legacy_nodes = {'legacy:' + r['id'] for r in legacy}
    if len(by_id) != len(rows) or len(legacy_nodes) != len(legacy) or set(by_id) & legacy_nodes:
        raise ValueError('duplicate corpus identifiers')
    parent = {key: key for key in list(by_id) + sorted(legacy_nodes)}
    edges = []

    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    def union(members, reason, kind):
        members = sorted(set(members))
        if not members or any(k not in parent for k in members) or not reason:
            raise ValueError('unknown edge member or missing rationale')
        if len(members) > 1:
            edges.append({'members': members, 'rationale': reason, 'kind': kind})
        for member in members[1:]:
            parent[root(member)] = root(members[0])

    existing = defaultdict(list)
    for row in rows:
        existing[row['group_id']].append(row['episode_id'])
    for members in existing.values():
        union(members, 'Preserve original declared group.', 'original_group')
    for edge in prior['clusters']:
        if edge.get('decision') != 'group_conservatively':
            raise ValueError('unsupported prior decision')
        union(edge['members'], edge['rationale'], 'prior_review')
    aliases = policy['aliases']
    pools = defaultdict(list)
    legacy_pools = defaultdict(list)
    mechanisms = {}
    for decision in ledger:
        mechanism = aliases.get(decision['mechanism'], decision['mechanism'])
        node = decision['episode_id']
        if decision['inventory'] == 'legacy':
            legacy_pools[mechanism].append('legacy:' + node)
        else:
            mechanisms[node] = mechanism
            pools[by_id[node]['family'], mechanism].append(node)
    for (family, mechanism), members in sorted(pools.items()):
        union(members + legacy_pools[mechanism],
              f'Manually classified {mechanism} solution structure; conservative pool, not asserted equivalence or measured error correlation.',
              'mechanism_and_legacy_pool')
    for edge in policy['additional_edges']:
        union(edge['members'], edge['rationale'], edge['kind'])
    groups = defaultdict(list)
    for node in parent:
        groups[root(node)].append(node)
    exclusions = {}
    for item in policy['exclusions']:
        key = item['episode_id']
        if key not in by_id or key in exclusions or not item.get('reason'):
            raise ValueError('unknown, duplicate or unreasoned exclusion')
        exclusions[key] = [item['reason']]
    resolved = {x['episode_id']: x['reason'] for x in policy['resolved_flags']}
    if any(key not in by_id for key in resolved) or set(resolved) & set(exclusions):
        raise ValueError('conflicting flag decision')
    graph, allocation = [], {}
    for members in sorted(groups.values(), key=lambda x: sorted(x)):
        members = sorted(members)
        current = [key for key in members if key in by_id]
        if not current:
            continue
        families = sorted({by_id[key]['family'] for key in current})
        prior_ids = sorted(set(members) & legacy_nodes)
        group_id = 'census:' + sha(json.dumps(members).encode())[:24]
        split = min((by_id[key]['split'] for key in current), key=STAGES.index)
        if prior_ids and STAGES.index(split) > STAGES.index('train'):
            split = 'train'
        cross_family = len(families) > 1
        if cross_family:
            for key in current:
                exclusions.setdefault(key, []).append('Conservative dependence component crosses workload families; excluded because the current bootstrap assumes independence across family strata.')
        graph.append({'group_id': group_id, 'members': current, 'legacy_members': prior_ids,
                      'families': families, 'split': split, 'cross_family_excluded': cross_family})
        for key in current:
            allocation[key] = (group_id, split, prior_ids, cross_family)
    review_hash = canonical_sha({'policy': policy, 'prior': prior, 'ledger': ledger})
    kept, removed, final_decisions = [], [], []
    for decision in ledger:
        item = copy.deepcopy(decision)
        key = item['episode_id']
        if item['inventory'] == 'legacy':
            item.update(decision='comparison_only', review_complete=True,
                        rationale='Prior benchmark exposure reference; never imported as new evidence.')
            final_decisions.append(item)
            continue
        group, split, prior_ids, cross_family = allocation[key]
        reasons = exclusions.get(key, [])
        disposition = 'excluded' if reasons else 'development_only_prior_overlap' if prior_ids else 'retained_research_only'
        item.update(decision=disposition, review_complete=True, normalized_mechanism=mechanisms[key],
                    review_group=group, original_split=by_id[key]['split'], assigned_split=None if reasons else split,
                    legacy_matches=prior_ids, cross_family_component=cross_family,
                    rationale=reasons or [resolved.get(key, f'Pool by reviewed {mechanisms[key]} solution structure and explicit dependence edges; absence of another edge is not certification of independence.')])
        final_decisions.append(item)
        row = copy.deepcopy(by_id[key])
        row['fixture']['semantic_review'] = {
            'version': policy['version'], 'reviewer': policy['reviewer'], 'review_hash': review_hash,
            'status': 'reviewed_excluded' if reasons else 'reviewed_research_only',
            'review_complete': True, 'collection_ready': False, 'disposition': disposition,
            'rationale': item['rationale'],
            'collection_blockers': ['No independent reviewer or uncontaminated provenance',
                                    'Family coverage and confirmatory power not established']}
        row['fixture_version'] += '-semantic-census-v2'
        if reasons:
            removed.append(row)
        else:
            row['group_id'] = group
            row['split'] = split
            row['fixture']['lineage_id'] = group
            row['fixture']['semantic_lineage_ids'] = [group]
            kept.append(row)
    # Empty cells are a finding, never silently repaired by recycling old rows.
    validate_episodes([Episode.from_dict(r) for r in kept], require_all_families=False)
    if len(kept) + len(removed) != len(rows):
        raise ValueError('unaccounted corpus row')
    return kept, removed, final_decisions, {'review_hash': review_hash, 'components': graph, 'edges': edges}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('input', 'legacy', 'ledger', 'prior', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output must be a new directory')
    policy = json.loads(args.policy.read_text())
    prior = json.loads(args.prior.read_text())
    source = args.input / 'combined-episodes.jsonl'
    for path, expected in ((source, policy['input_sha256']), (args.legacy, policy['legacy_sha256']),
                           (args.ledger, policy['ledger_sha256'])):
        if sha(path.read_bytes()) != expected:
            raise ValueError('review input file hash mismatch: ' + str(path))
    if prior['input_sha256'] != policy['input_sha256']:
        raise ValueError('prior review input differs')
    rows, legacy, ledger = read_rows(source), read_rows(args.legacy), read_rows(args.ledger)
    kept, removed, decisions, graph = finalize(rows, legacy, ledger, prior, policy)
    args.output.mkdir(parents=True)
    new_ids = {r['episode_id'] for r in read_rows(args.input / 'new-episodes.jsonl')}
    write_rows(args.output / 'combined-episodes.jsonl', kept)
    write_rows(args.output / 'new-episodes.jsonl', [r for r in kept if r['episode_id'] in new_ids])
    write_rows(args.output / 'excluded-episodes.jsonl', removed)
    write_rows(args.output / 'row-decisions.jsonl', decisions)
    write_json(args.output / 'dependence-graph.json', graph)
    write_json(args.output / 'semantic-review.json', policy)
    shutil.copyfile(args.ledger, args.output / 'classification-ledger.jsonl')
    shutil.copyfile(args.legacy, args.output / 'legacy-comparison.jsonl')
    shutil.copyfile(args.prior, args.output / 'prior-review.json')
    manifest = json.loads((args.input / 'source-manifest.json').read_text())
    for entry in manifest:
        entry['path'] = os.path.relpath(args.input / entry['path'], args.output)
    write_json(args.output / 'source-manifest.json', manifest)
    for name in ('reference-answers.jsonl', 'ATTRIBUTION.md', 'adapter-needed.json'):
        shutil.copyfile(args.input / name, args.output / name)
    counts = {family: {split: len({r['group_id'] for r in kept if r['family'] == family and r['split'] == split})
                       for split in STAGES} for family in FAMILIES}
    original = {r['episode_id']: r for r in rows}
    empty = [f'{family}/{split}' for family in FAMILIES for split in STAGES if not counts[family][split]]
    report = {
        'review_complete': True, 'classification_records': len(decisions), 'candidate_rows_reviewed': len(rows),
        'legacy_rows_reviewed': len(legacy), 'unadjudicated_rows': 0,
        'retained_rows': len(kept), 'excluded_rows': len(removed),
        'explicit_prompt_or_key_exclusions': len(policy['exclusions']),
        'cross_family_excluded_rows': sum(d.get('cross_family_component', False) for d in decisions),
        'development_only_prior_overlap_rows': sum(d['decision'] == 'development_only_prior_overlap' for d in decisions),
        'retained_groups': len({r['group_id'] for r in kept}), 'groups_by_family_split': counts,
        'split_moves': sum(r['split'] != original[r['episode_id']]['split'] for r in kept),
        'test_rows_moved_earlier': sum(original[r['episode_id']]['split'] == 'test' and r['split'] != 'test' for r in kept),
        'test_rows_excluded': sum(original[r['episode_id']]['split'] == 'test' for r in removed),
        'test_rows': sum(r['split'] == 'test' for r in kept),
        'test_review_groups': sum(c['test'] for c in counts.values()),
        'empty_family_split_cells': empty, 'all_family_schema_eligible': not empty,
        'source_level_test_stress_pools': {family: len({r['source'] for r in kept if r['family'] == family and r['split'] == 'test'}) for family in FAMILIES},
        'source_stress_interpretation': 'Extreme sensitivity with all test rows from the same source pooled. Not an estimate of effective sample size or a claim of perfect dependence.',
        'review_hash': graph['review_hash'], 'input_sha256': policy['input_sha256'],
        'legacy_sha256': policy['legacy_sha256'], 'output_sha256': sha((args.output / 'combined-episodes.jsonl').read_bytes()),
        'collection_ready': False, 'semantic_independence_certified': False,
        'model_calls': 0, 'limitations': policy['limitations']}
    write_json(args.output / 'review-report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
