#!/usr/bin/env python3
"""Append a reviewed batch using the existing census closure; preserve foundation bytes.

This is a research snapshot, never a collection clearance or confirmatory lock.
All original rows (including prior exclusions) and legacy rows remain comparison
inputs. Any added component touching one of them is ineligible as a new group.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.finalize_public_episode_semantic_census import (finalize,read_rows,write_rows,write_json,canonical_sha,STAGES,FAMILIES)
from theriad.routing.episode_study import Episode,validate_episodes,MIN_INFERENCE_GROUPS_PER_FAMILY
from theriad.routing.episode_fixtures import require_collection_ready


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def extend(foundation,original,candidates,review_path,output):
    if output.exists():raise ValueError('output must be a new directory')
    manual=json.loads(review_path.read_text())
    inputs={'foundation':foundation/'combined-episodes.jsonl','original':original/'combined-episodes.jsonl',
        'candidates':candidates/'candidates.jsonl','legacy':foundation/'legacy-comparison.jsonl',
        'foundation_policy':foundation/'semantic-review.json','foundation_ledger':foundation/'classification-ledger.jsonl'}
    for key,path in inputs.items():
        if digest(path)!=manual['input_hashes'][key]:raise ValueError('input hash mismatch: '+key)
    sources=[]
    for folder in (foundation,candidates):
        for entry in json.loads((folder/'source-manifest.json').read_text()):
            path=(folder/entry['path']).resolve()
            if digest(path)!=entry['sha256']:raise ValueError('source hash mismatch: '+str(path))
            sources.append(dict(entry,path=os.path.relpath(path,output)))
    old=read_rows(inputs['original']);base=read_rows(inputs['foundation']);new=read_rows(inputs['candidates']);legacy=read_rows(inputs['legacy'])
    ledger=read_rows(inputs['foundation_ledger']);prior=json.loads((foundation/'prior-review.json').read_text())
    policy=json.loads(inputs['foundation_policy'].read_text())
    foundation_report=json.loads((foundation/'review-report.json').read_text())
    if digest(inputs['foundation'])!=foundation_report['output_sha256']:raise ValueError('foundation report mismatch')
    if policy['input_sha256']!=digest(inputs['original']) or prior['input_sha256']!=digest(inputs['original']):raise ValueError('original review input mismatch')
    if policy['legacy_sha256']!=digest(inputs['legacy']) or policy['ledger_sha256']!=digest(inputs['foundation_ledger']):raise ValueError('foundation census mismatch')
    # First reproduce the completed review exactly, before analyzing additions.
    foundation_recomputed=finalize(old,legacy,ledger,prior,policy)
    if foundation_recomputed[0]!=base:raise ValueError('foundation census does not reproduce')
    if foundation_recomputed[1]!=read_rows(foundation/'excluded-episodes.jsonl'):raise ValueError('foundation exclusions differ')
    if foundation_recomputed[3]!=json.loads((foundation/'dependence-graph.json').read_text()):raise ValueError('foundation graph differs')
    reviews={r['episode_id']:r for r in manual['row_reviews']}
    if len(manual['row_reviews'])!=len(new) or len(reviews)!=len(new) or set(reviews)!={r['episode_id'] for r in new}:raise ValueError('incomplete or duplicate addition review')
    # Mechanisms are bound by the candidate row hash, not an unbound sidecar.
    mechanisms={r['episode_id']:r['template_id'].removeprefix('authored-20260911:') for r in new}
    index=sum(r['source'] not in ('gsm8k','humaneval','hotpotqa') for r in old)
    for i,row in enumerate(new,index):
        ledger.append(dict(inventory='seed',index=i,episode_id=row['episode_id'],row_sha256=canonical_sha(row),
            mechanism=mechanisms[row['episode_id']],reviewer=manual['reviewer'],decision='semantic_classified',scope=reviews[row['episode_id']]['scope']))
    combined_policy=copy.deepcopy(policy)
    combined_policy['version']=manual['version'];combined_policy['reviewer']=manual['reviewer']
    combined_policy['extension_review_sha256']=digest(review_path)
    combined_policy['additional_edges']+=manual['additional_edges']
    combined_policy['exclusions']+=manual['exclusions']
    excluded_ids={r['episode_id'] for r in manual['exclusions']}
    combined_policy['resolved_flags'] += [{'episode_id':r['episode_id'],'reason':r['rationale']} for r in manual['row_reviews'] if r['episode_id'] not in excluded_ids]
    kept,excluded,decisions,graph=finalize(old+new,legacy,ledger,prior,combined_policy)
    new_ids={r['episode_id'] for r in new};old_ids={r['episode_id'] for r in old}
    related={key for component in graph['components'] if set(component['members'])&old_ids or component['legacy_members'] or component['cross_family_excluded'] for key in component['members'] if key in new_ids}
    # Even a same-family old/new relation cannot manufacture another independent group.
    unadjudicated=related-excluded_ids
    if unadjudicated:raise ValueError('related additions need explicit exclusions: '+str(sorted(unadjudicated)))
    additions=[r for r in kept if r['episode_id'] in new_ids]
    removed=[r for r in excluded if r['episode_id'] in new_ids]
    if len(additions)+len(removed)!=len(new):raise ValueError('unaccounted addition')
    raw={r['episode_id']:r for r in new}
    for row in additions:
        if STAGES.index(row['split'])>STAGES.index(raw[row['episode_id']]['split']):raise ValueError('later exposure move')
        row['fixture']['semantic_review']['collection_blockers']=['Same author/reviewer; independence and pretraining novelty not certified','Joint confirmatory power and representative workload sampling not established']
    validate_episodes([Episode.from_dict(r) for r in base+additions])
    try:require_collection_ready([Episode.from_dict(r) for r in base+additions])
    except ValueError as exc:quarantine=str(exc)
    else:raise ValueError('collection quarantine unexpectedly bypassed')
    output.mkdir(parents=True)
    # Never reserialize, reassign, relabel or rewrite old accepted rows.
    added_bytes=''.join(json.dumps(r,sort_keys=True)+'\n' for r in additions).encode()
    (output/'combined-episodes.jsonl').write_bytes(inputs['foundation'].read_bytes()+added_bytes)
    (output/'new-episodes.jsonl').write_bytes((foundation/'new-episodes.jsonl').read_bytes()+added_bytes)
    write_rows(output/'added-episodes.jsonl',additions);write_rows(output/'excluded-additions.jsonl',removed)
    write_rows(output/'addition-decisions.jsonl',[r for r in decisions if r['episode_id'] in new_ids])
    write_rows(output/'comparison-decisions.jsonl',decisions)
    write_rows(output/'classification-ledger.jsonl',ledger)
    write_rows(output/'comparison-corpus.jsonl',old+new)
    write_json(output/'dependence-comparison-graph.json',graph)
    write_json(output/'comparison-policy.json',combined_policy)
    (output/'extension-review.json').write_bytes(review_path.read_bytes())
    write_json(output/'source-manifest.json',sources)
    (output/'reference-answers.jsonl').write_bytes((foundation/'reference-answers.jsonl').read_bytes()+ (candidates/'reference-answers.jsonl').read_bytes())
    counts={f:{s:len({r['group_id'] for r in base+additions if r['family']==f and r['split']==s}) for s in STAGES} for f in FAMILIES}
    report={'version':manual['version'],'foundation_rows_unchanged':len(base),'foundation_bytes_sha256':digest(inputs['foundation']),
        'original_comparison_rows':len(old),'legacy_comparison_rows':len(legacy),'new_candidates':len(new),'unadjudicated_additions':0,
        'retained_addition_rows':len(additions),'retained_addition_groups':len({r['group_id'] for r in additions}),
        'excluded_additions':len(removed),'combined_rows':len(base)+len(additions),'combined_review_groups':len({r['group_id'] for r in base+additions}),
        'groups_by_family_split':counts,'remaining_structural_deficits':[{'family':f,'split':s,'deficit':max(0,(MIN_INFERENCE_GROUPS_PER_FAMILY if s=='test' else 1)-counts[f][s])} for f in FAMILIES for s in STAGES if counts[f][s]<(MIN_INFERENCE_GROUPS_PER_FAMILY if s=='test' else 1)],
        'foundation_split_changes':0,'new_later_split_moves':0,'collection_ready':False,'confirmatory_lock_created':False,
        'collection_guard':quarantine,'source_checksums_verified':len(sources),'joint_power_ready':False,
        'limitations':manual['limitations'],'output_sha256':digest(output/'combined-episodes.jsonl'),'comparison_review_hash':graph['review_hash'],
        'input_paths':{k:os.path.relpath(v,output) for k,v in inputs.items()},'input_hashes':manual['input_hashes'],'model_calls':0}
    write_json(output/'review-report.json',report)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('foundation','original','candidates','review','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();print(json.dumps(extend(a.foundation,a.original,a.candidates,a.review,a.output),indent=2))
if __name__=='__main__':main()
