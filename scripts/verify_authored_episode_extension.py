#!/usr/bin/env python3
"""No-provider checks of authored coding references and production state graders."""
import argparse,json,os,sys,tempfile
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import Episode
from theriad.routing.episode_runtime import grade_workspace_state
from theriad.routing.sandbox import run_python_tests,CODE_GRADER_RUNTIME_ENV

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--directory',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('output must be new')
    rows=[Episode.from_dict(json.loads(x)) for x in (a.directory/'candidates.jsonl').read_text().splitlines()]
    refs={r['episode_id']:r['reference'] for r in map(json.loads,(a.directory/'reference-answers.jsonl').read_text().splitlines())}
    os.environ[CODE_GRADER_RUNTIME_ENV]=sys.executable
    records=[]
    for row in rows:
        checks={}
        if row.family=='coding':
            good=run_python_tests(refs[row.episode_id],row.fixture['tests'],entry_point=row.fixture['entry_point'])
            bad=run_python_tests('def solve(*args): return None',row.fixture['tests'],entry_point=row.fixture['entry_point'])
            checks={'real_reference_pass':good.passed,'constant_none_rejected':not bad.passed}
            details={'reference':good.to_dict(),'negative':bad.to_dict()}
        else:
            details={}
            with tempfile.TemporaryDirectory(prefix='authored-state-check-') as tmp:
                root=Path(tmp)
                for name,value in row.fixture['files'].items():
                    path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(value)
                name,expected=next(iter(row.fixture['expected_json'].items()));target=root/name
                checks['missing_output_rejected']=grade_workspace_state(row,root,'completed')[0] is False
                target.parent.mkdir(parents=True,exist_ok=True);target.write_text(json.dumps(expected))
                checks['specified_reference_state_pass']=grade_workspace_state(row,root,'completed')[0] is True
                checks['unfinished_task_rejected']=grade_workspace_state(row,root,'running')[0] is False
                target.write_text('{invalid')
                checks['malformed_output_rejected']=grade_workspace_state(row,root,'completed')[0] is False
                target.write_text(json.dumps({'wrong':'deliberately incorrect'}))
                checks['wrong_output_rejected']=grade_workspace_state(row,root,'completed')[0] is False
                target.write_text(json.dumps(expected))
                input_name=next(iter(row.fixture['files']));(root/input_name).write_text('mutated')
                checks['changed_input_rejected']=grade_workspace_state(row,root,'completed')[0] is False
        records.append({'episode_id':row.episode_id,'checks':checks,**details})
    report={'model_calls':0,'real_provider_spend_usd':0,'real_code_grader':True,
            'workspace_checks':'Production final-state grader, materialized specified oracle and negative states; not an independent solution or end-to-end agent rollout.',
            'rows':records,'all_passed':all(all(x['checks'].values()) for x in records)}
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print('rows',len(rows),'all_passed',report['all_passed'])
    for r in records:
        if not all(r['checks'].values()):print(json.dumps(r,indent=2))
    if not report['all_passed']:raise SystemExit(1)
if __name__=='__main__':main()
