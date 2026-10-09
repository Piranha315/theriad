#!/usr/bin/env python3
"""Reproduce an appended census package and cross-check its executable keys offline."""
import argparse
import hashlib
import json
from pathlib import Path
import runpy
import sys
import tempfile
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.extend_reviewed_episode_corpus import extend,digest,read_rows
from scripts.build_authored_episode_extension import build
from theriad.routing.episode_study import Episode
from theriad.routing.episode_runtime import _json_equal,grade_workspace_state
from theriad.routing.sandbox import run_python_tests,CODE_GRADER_RUNTIME_ENV
import os


def verify(package,oracles,output):
    if output.exists():raise ValueError('output must be new')
    report=json.loads((package/'review-report.json').read_text())
    paths={k:(package/v).resolve() for k,v in report['input_paths'].items()}
    with tempfile.TemporaryDirectory(prefix='verify-extension-') as tmp:
        rebuilt=Path(tmp)/'reproduced'
        candidate_dir=paths['candidates'].parent
        source_manifest=json.loads((candidate_dir/'source-manifest.json').read_text())
        source_dirs={(candidate_dir/r['path']).resolve().parent for r in source_manifest}
        if len(source_dirs)!=1:raise ValueError('expected one authored source directory')
        generated=Path(tmp)/'generated';build(next(iter(source_dirs)),generated)
        for name in ('candidates.jsonl','reference-answers.jsonl','review-input.jsonl'):
            if (generated/name).read_bytes()!=(candidate_dir/name).read_bytes():raise ValueError('candidate build mismatch: '+name)
        # Relative path metadata depends on output directory; semantic data must match.
        extend(paths['foundation'].parent,paths['original'].parent,paths['candidates'].parent,package/'extension-review.json',rebuilt)
        for name in ('combined-episodes.jsonl','added-episodes.jsonl','excluded-additions.jsonl','addition-decisions.jsonl','comparison-decisions.jsonl','classification-ledger.jsonl','comparison-corpus.jsonl','dependence-comparison-graph.json','comparison-policy.json'):
            if (package/name).read_bytes()!=(rebuilt/name).read_bytes():raise ValueError('reproduction mismatch: '+name)
    if not (package/'combined-episodes.jsonl').read_bytes().startswith(paths['foundation'].read_bytes()):raise ValueError('foundation bytes changed')
    solve=runpy.run_path(str(oracles))['answer']
    refs={r['episode_id']:r['reference'] for r in read_rows(package/'reference-answers.jsonl')}
    os.environ[CODE_GRADER_RUNTIME_ENV]=sys.executable
    records=[]
    for raw in read_rows(package/'added-episodes.jsonl'):
        row=Episode.from_dict(raw)
        if row.family=='coding':
            good=run_python_tests(refs[row.episode_id],row.fixture['tests'],entry_point=row.fixture['entry_point'])
            bad=run_python_tests('def solve(*args): return None',row.fixture['tests'],entry_point=row.fixture['entry_point'])
            checks={'reference_pass':good.passed,'constant_none_rejected':not bad.passed};details={'reference_execution':good.to_dict(),'negative_execution':bad.to_dict()}
        else:
            files=row.fixture['files']
            data=json.loads(files['input.json']) if row.family=='tools' else {Path(k).stem:json.loads(v) for k,v in files.items() if k.startswith('inputs/')}
            actual=solve(row.episode_id.split(':')[1],data);expected=next(iter(row.fixture['expected_json'].values()))
            checks={'separately_computed_key_matches':_json_equal(actual,expected)};details={'computed_result':actual}
            with tempfile.TemporaryDirectory(prefix='verify-state-') as tmp:
                workspace=Path(tmp)
                for name,value in files.items():
                    path=workspace/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
                target=workspace/next(iter(row.fixture['expected_json']));target.parent.mkdir(parents=True,exist_ok=True)
                checks['missing_rejected']=grade_workspace_state(row,workspace,'completed')[0] is False
                target.write_text(json.dumps(actual));checks['computed_state_passes']=grade_workspace_state(row,workspace,'completed')[0] is True
                checks['unfinished_rejected']=grade_workspace_state(row,workspace,'running')[0] is False
                target.write_text('{invalid');checks['malformed_rejected']=grade_workspace_state(row,workspace,'completed')[0] is False
                target.write_text(json.dumps({'deliberately_wrong':True}));checks['wrong_rejected']=grade_workspace_state(row,workspace,'completed')[0] is False
                target.write_text(json.dumps(actual));(workspace/next(iter(files))).write_text('changed input')
                checks['changed_input_rejected']=grade_workspace_state(row,workspace,'completed')[0] is False
        records.append({'episode_id':row.episode_id,'content_hash':row.content_hash,'checks':checks,**details})
    result={'package_sha256':digest(package/'combined-episodes.jsonl'),'addition_sha256':digest(package/'added-episodes.jsonl'),
        'oracle_sha256':digest(oracles),'runtime_sha256':digest(Path('theriad/routing/episode_runtime.py')),
        'sandbox_sha256':digest(Path('theriad/routing/sandbox.py')),'verifier_sha256':digest(Path(__file__)),
        'candidate_source_reproduced':True,'census_reproduced':True,'foundation_bytes_unchanged':True,'collection_quarantine_verified':True,
        'code_references_executed':sum('reference_pass' in r['checks'] for r in records),
        'workspace_keys_recomputed':sum('separately_computed_key_matches' in r['checks'] for r in records),
        'all_passed':all(all(r['checks'].values()) for r in records),'rows':records,'model_calls':0,'real_provider_spend_usd':0,
        'limitations':['Same author supplied fixtures and executable cross-checks; no independent reviewer.','Workspace state validation does not establish a model can plan or solve the task.','Code tests cover fixed cases, not all legal inputs.']}
    output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='rows'},indent=2))
    if not result['all_passed']:raise ValueError('one or more checks failed; inspect report')
    return result
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('package','oracles','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();verify(a.package,a.oracles,a.output)
