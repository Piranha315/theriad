#!/usr/bin/env python3
"""Build fresh authored candidates in the existing Episode format; never clear them."""
import argparse
import ast
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import Episode,validate_episodes

def build(source,output):
    if output.exists():raise ValueError('output must be new')
    code=runpy.run_path(str(source/'coding_tasks.py'))['TASKS']
    work=runpy.run_path(str(source/'workspace_tasks.py'))['TASKS']
    rows=[];references=[];decisions=[]
    version='authored-episode-candidates-20260911-v2'
    def make(name,family,split,prompt,fixture,grader,mechanism,reason):
        identity='authored-20260911:'+name
        group='authored-20260911:'+mechanism
        metadata={'dataset':'newly_authored_20260911','author':'Codex, same study author/reviewer',
            'authored_date':'2026-09-11','source_id':name,'independence_status':'candidate_review_required',
            'public_benchmark_derivation':False,'pretraining_novelty_certified':False,
            'provenance':'Original bounded problem and data authored locally; familiar algorithms do not imply unseen concepts.',
            'source_file_sha256':hashlib.sha256((source/('coding_tasks.py' if family=='coding' else 'workspace_tasks.py')).read_bytes()).hexdigest()}
        row=Episode(identity,group,family,split,prompt,dict(fixture,lineage_id=group,source_metadata=metadata),grader,
            source='newly_authored_20260911',template_id=group,fixture_version=version,grader_version=version,
            max_calls=24 if family=='agents' else 8,max_input_tokens=24000,max_output_tokens=4096,max_seconds=180,
            tags=('authored_research_candidate','not_production_weighted'),capabilities=('text',))
        rows.append(asdict(row));decisions.append({'episode_id':identity,'family':family,'split':split,'mechanism':mechanism,'initial_rationale':reason})
    for name,contract,reference,cases,reason in code:
        assertions=['def check(candidate):','    import copy', '    def equal(a,b):', '        if type(a) is not type(b): return False', '        if isinstance(a,dict): return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)', '        if isinstance(a,list): return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))', '        return a==b']
        for args,expected in cases:
            assertions.extend([f'    args={args!r}', '    original=copy.deepcopy(args)',f'    assert equal(candidate(*args),{expected!r})', '    assert args==original'])
        signature=ast.unparse(ast.parse(reference).body[0].args)
        make(name,'coding','test','Implement Python function solve('+signature+'). Return only Python code.\n'+contract,
            {'entry_point':'solve','tests':'\n'.join(assertions)+'\n'}, {'kind':'code'},name,reason)
        references.append({'episode_id':'authored-20260911:'+name,'reference':reference,'cases':cases})
    for task in work:
        name,family=task['name'],task['family']
        if family=='agents':
            files={f'inputs/{key}.json':json.dumps(value,ensure_ascii=True,indent=2)+'\n' for key,value in task['data'].items()}
            files['policy.md']=task['contract']+'\n'
            prompt='Complete the local '+name.replace('_',' ')+' project. Read policy.md and the inputs/ JSON files, reconcile the supplied state under that policy, and write deliverables/result.json containing the required result. Each input filename names its field in the project data. Preserve all input files. Finish only after checking the complete output. The policy is frozen for this exercise; use it even if a real service would have other rules.'
            output_name='deliverables/result.json'
        else:
            files={'input.json':json.dumps(task['data'],ensure_ascii=True,indent=2)+'\n'}
            prompt='Read input.json and write output.json containing the JSON result specified here. Preserve input.json.\n'+task['contract']
            output_name='output.json'
        make(name,family,task['split'],prompt,{'files':files,'expected_json':{output_name:task['answer']}},
            {'kind':'state'},task['mechanism'],'New fixed scenario with operation '+task['mechanism']+'; requires full semantic comparison before counting.')
    validate_episodes([Episode.from_dict(x) for x in rows],require_all_families=False)
    output.mkdir(parents=True)
    for filename,data in [('candidates.jsonl',rows),('reference-answers.jsonl',references),('review-input.jsonl',decisions)]:
        (output/filename).write_text(''.join(json.dumps(x,sort_keys=True,ensure_ascii=True)+'\n' for x in data))
    (output/'source-manifest.json').write_text(json.dumps([{ 'path':os.path.relpath((source/name).resolve(),output.resolve()),'sha256':hashlib.sha256((source/name).read_bytes()).hexdigest()} for name in ('coding_tasks.py','workspace_tasks.py')],indent=2)+'\n')
    return rows
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();print('Built',len(build(a.source,a.output)),'uncleared candidate rows')
