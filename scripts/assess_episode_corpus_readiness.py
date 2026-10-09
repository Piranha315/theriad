#!/usr/bin/env python3
"""Recompute coverage and scenario diagnostics with existing protocol/simulation.

No provider calls, assumed outcomes, or count floors can clear collection.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import FAMILIES,MIN_INFERENCE_GROUPS_PER_FAMILY,simulate_joint_power
from theriad.routing.v2_training import featurise
from scripts.finalize_public_episode_semantic_census import STAGES,read_rows


def assess(foundation,extended,output,repetitions=200,bootstrap_samples=400):
    if output.exists():raise ValueError('output must be new')
    groups={name:{f:{s:len({r['group_id'] for r in read_rows(path) if r['family']==f and r['split']==s}) for s in STAGES} for f in FAMILIES} for name,path in [('before',foundation),('after',extended)]}
    requirements=[{'family':f,'split':s,'existing_review_groups':groups['before'][f][s],'after_review_groups':groups['after'][f][s],
        'required_structural_groups':MIN_INFERENCE_GROUPS_PER_FAMILY if s=='test' else 1,
        'before_deficit':max(0,(MIN_INFERENCE_GROUPS_PER_FAMILY if s=='test' else 1)-groups['before'][f][s]),
        'after_deficit':max(0,(MIN_INFERENCE_GROUPS_PER_FAMILY if s=='test' else 1)-groups['after'][f][s])} for f in FAMILIES for s in STAGES]
    simulations={}
    for name in ('before','after'):
        simulations[name]=simulate_joint_power(group_counts_by_family={f:groups[name][f]['test'] for f in FAMILIES},repetitions=repetitions,bootstrap_samples=bootstrap_samples)
    result={'input_sha256':{'before':hashlib.sha256(foundation.read_bytes()).hexdigest(),'after':hashlib.sha256(extended.read_bytes()).hexdigest()},
        'cells':requirements,'minimum_additional_structural_groups':sum(r['before_deficit'] for r in requirements),'remaining_structural_deficit':sum(r['after_deficit'] for r in requirements),
        'groups':groups,'train_fit_floor_per_eligible_model':max(20,len(featurise('x'))+2),'calibration_floor_per_eligible_model':10,
        'training_note':'Pooled per eligible model, not per family. Current catalog/capability eligibility and paid outcome completeness have not been established.',
        'joint_power_required_groups':None,'reason_no_powered_minimum':'No empirical pilot or approved joint-alternative population assumptions. Existing default scenarios are sensitivity diagnostics only.',
        'simulations':simulations,'readiness':{'structural_coverage':all(r['after_deficit']==0 for r in requirements),'certified_independence':False,'representative_workload_sampling':False,'joint_power':False,'paid_collection':False,'confirmatory_freeze':False},
        'real_provider_calls':0,'real_provider_spend_usd':0}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('cells','groups','simulations')},indent=2))
    for name,result in simulations.items():print(name,[(s['scenario']['name'],s['rates']['joint_pass']) for s in result['results']])
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('foundation','extended','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument('--repetitions',type=int,default=200);p.add_argument('--bootstrap-samples',type=int,default=400)
    a=p.parse_args();assess(a.foundation,a.extended,a.output,a.repetitions,a.bootstrap_samples)
