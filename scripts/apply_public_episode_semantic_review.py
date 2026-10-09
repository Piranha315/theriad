#!/usr/bin/env python3
"""Apply explicit qualitative decisions; never infer independence from distance."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import Episode, validate_episodes, atomic_json, digest


def apply_review(rows, review):
    rows = copy.deepcopy(rows)
    by_id = {r['episode_id']: r for r in rows}
    if len(by_id) != len(rows): raise ValueError('duplicate episode ids')
    parent = {key:key for key in by_id}
    def root(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]; key = parent[key]
        return key
    def union(ids):
        if not ids or any(x not in by_id for x in ids): raise ValueError('review references unknown episode')
        for x in ids[1:]: parent[root(x)] = root(ids[0])
    # Existing declared groups cannot be split by a partial new review.
    existing = defaultdict(list)
    for row in rows: existing[row['group_id']].append(row['episode_id'])
    for ids in existing.values(): union(ids)
    for cluster in review['clusters']:
        if cluster.get('decision') != 'group_conservatively' or not cluster.get('rationale'): raise ValueError('explicit decision and rationale required')
        union(cluster['members'])
    groups = defaultdict(list)
    for key in by_id: groups[root(key)].append(key)
    # Keep every connected component in its earliest exposure stage. In
    # particular, no development item can be moved into a final test split.
    priority = {s:i for i,s in enumerate(('pilot','smoke','train','calibration','threshold_selection','test'))}
    assignments = []
    reviewed = {x for c in review['clusters'] for x in c['members']}
    for members in sorted(groups.values(), key=lambda ids:sorted(ids)):
        related = [by_id[key] for key in members]
        if len({r['family'] for r in related}) != 1: raise ValueError('cross-family dependence needs removal or a different inference design')
        if not reviewed.intersection(members): continue
        split = min((r['split'] for r in related), key=priority.get)
        group = 'semantic:'+digest(sorted(members))[:24]
        for row in related:
            assignments.append({'episode_id':row['episode_id'],'old_group':row['group_id'],'new_group':group,'old_split':row['split'],'new_split':split})
            row['group_id'] = group
            row['fixture']['lineage_id'] = group
            row['fixture']['semantic_lineage_ids'] = [group]
            row['split'] = split
    for row in rows:
        row['fixture']['semantic_review'] = {'version':review['version'],
            'review_hash':digest(review), 'status':'targeted_grouping_reviewed' if row['episode_id'] in reviewed else 'not_reviewed',
            'collection_ready':False}
        row['fixture_version'] += '-semantic-review-v1'
    validate_episodes([Episode.from_dict(r) for r in rows])
    return rows, assignments


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists(): raise ValueError('output must be a new directory')
    original=(a.input/'combined-episodes.jsonl').read_bytes(); review=json.loads(a.review.read_text())
    if hashlib.sha256(original).hexdigest()!=review['input_sha256']: raise ValueError('review input hash mismatch')
    rows,assignments=apply_review([json.loads(x) for x in original.splitlines()],review)
    new_ids={json.loads(x)['episode_id'] for x in (a.input/'new-episodes.jsonl').read_text().splitlines()}
    a.output.mkdir(parents=True)
    for name,values in [('combined-episodes.jsonl',rows),('new-episodes.jsonl',[r for r in rows if r['episode_id'] in new_ids])]:
        (a.output/name).write_text(''.join(json.dumps(r,sort_keys=True,ensure_ascii=True)+'\n' for r in values))
    # Package references the unchanged pinned source bundle by relative path;
    # checksums still verify every file. No second 98 MB source copy is needed.
    import os
    manifest=json.loads((a.input/'source-manifest.json').read_text())
    for entry in manifest: entry['path']=os.path.relpath(a.input/entry['path'],a.output)
    atomic_json(a.output/'source-manifest.json',manifest)
    for name in ('reference-answers.jsonl','ATTRIBUTION.md','adapter-needed.json'):
        shutil.copyfile(a.input/name,a.output/name)
    atomic_json(a.output/'semantic-review.json',review)
    atomic_json(a.output/'assignments.json',assignments)
    report={'input_sha256':review['input_sha256'],'output_sha256':hashlib.sha256((a.output/'combined-episodes.jsonl').read_bytes()).hexdigest(),
        'episodes':len(rows),'declared_groups_before':len({json.loads(x)['group_id'] for x in original.splitlines()}),
        'declared_groups_after':len({r['group_id'] for r in rows}),
        'reviewed_related_rows':len(assignments),'split_moves':sum(x['old_split']!=x['new_split'] for x in assignments),
        'test_rows_moved_to_earlier_split':sum(x['old_split']=='test' and x['new_split']!='test' for x in assignments),
        'groups_by_family_split':{f:{s:len({r['group_id'] for r in rows if r['family']==f and r['split']==s}) for s in ('train','calibration','threshold_selection','test','pilot','smoke')} for f in ('chat','coding','search','tools','agents')},
        'collection_ready':False,'semantic_independence_certified':False,'model_calls':0,'limitations':review['limitations']}
    atomic_json(a.output/'review-report.json',report)
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
