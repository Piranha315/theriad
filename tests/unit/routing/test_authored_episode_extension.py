"""Regression gates for research extension checks and final-state grading."""
import json
import pytest
from theriad.routing.episode_study import Episode
from theriad.routing.episode_runtime import grade_workspace_state
from theriad.routing import episode_study as study


def row(expected):
    return Episode('test','test','tools','test','Fixture grader regression',
        {'files':{'input.txt':'keep'},'expected_json':{'result.json':expected}}, {'kind':'state'})


@pytest.mark.parametrize('content',[None,'{invalid','NaN','{"x":1,"x":2}'])
def test_invalid_or_missing_file_cannot_equal_null(tmp_path,content):
    (tmp_path/'input.txt').write_text('keep')
    if content is not None:(tmp_path/'result.json').write_text(content)
    success,evidence=grade_workspace_state(row(None),tmp_path,'completed')
    assert success is False
    assert evidence['invalid_json_outputs']==['result.json']


def test_null_is_valid_but_requires_completed_preserved_workspace(tmp_path):
    (tmp_path/'input.txt').write_text('keep');(tmp_path/'result.json').write_text('null')
    assert grade_workspace_state(row(None),tmp_path,'completed')[0] is True
    assert grade_workspace_state(row(None),tmp_path,'running')[0] is False
    (tmp_path/'input.txt').write_text('changed')
    assert grade_workspace_state(row(None),tmp_path,'completed')[0] is False


def test_boolean_is_not_an_integer_answer(tmp_path):
    (tmp_path/'input.txt').write_text('keep');(tmp_path/'result.json').write_text('true')
    assert grade_workspace_state(row(1),tmp_path,'completed')[0] is False


def test_nonuniform_simulation_uses_actual_family_counts(monkeypatch):
    counts=dict(zip(study.FAMILIES,[21,20,44,20,20]));seen=[]
    original=study._paired_bootstrap_decision
    def capture(groups,**kwargs):
        seen.append({k:len(v) for k,v in groups.items()})
        return original(groups,**kwargs)
    monkeypatch.setattr(study,'_paired_bootstrap_decision',capture)
    result=study.simulate_joint_power(group_counts_by_family=counts,repetitions=2,bootstrap_samples=100,scenarios=[study.JointPowerScenario('probe')])
    assert seen==[counts,counts]
    assert result['independent_groups']==sum(counts.values())
    assert result['group_counts_by_family']==counts
    assert result['groups_per_family'] is None


def test_equal_count_mapping_preserves_simulation():
    kwargs=dict(repetitions=2,bootstrap_samples=100,scenarios=[study.JointPowerScenario('equal')])
    old=study.simulate_joint_power(groups_per_family=20,**kwargs)
    mapped=study.simulate_joint_power(group_counts_by_family=dict.fromkeys(study.FAMILIES,20),**kwargs)
    assert old['results']==mapped['results']


@pytest.mark.parametrize('counts',[{'chat':20},dict.fromkeys(study.FAMILIES,0),dict.fromkeys(study.FAMILIES,True)])
def test_invalid_family_counts_rejected(counts):
    with pytest.raises(ValueError):study.simulate_joint_power(group_counts_by_family=counts)


def test_extension_preserves_foundation_and_reproduces_adjudications(tmp_path):
    from pathlib import Path
    from scripts.extend_reviewed_episode_corpus import extend,read_rows
    root=Path(__file__).resolve().parents[3]
    foundation=root/'data/public-episode-corpus-20260910-v6'
    original=root/'data/public-episode-corpus-20260910-v4'
    candidate=root/'data/router-v2-authored-20260911/candidates-v5'
    policy=root/'data/router-v2-authored-20260911/extension-review-v3.json'
    output=tmp_path/'extension'
    result=extend(foundation,original,candidate,policy,output)
    assert result['remaining_structural_deficits']==[]
    assert result['retained_addition_rows']==result['retained_addition_groups']==51
    assert result['excluded_additions']==15
    assert result['collection_ready'] is False
    assert (output/'combined-episodes.jsonl').read_bytes().startswith((foundation/'combined-episodes.jsonl').read_bytes())
    # Raw policy bytes matter to the audit hash; JSON reformatting broke replay.
    assert (output/'extension-review.json').read_bytes()==policy.read_bytes()
    original_stages={r['episode_id']:r['split'] for r in read_rows(candidate/'candidates.jsonl')}
    assert all(r['split']==original_stages[r['episode_id']] for r in read_rows(output/'added-episodes.jsonl'))
    repeat=tmp_path/'repeat'
    extend(foundation,original,candidate,output/'extension-review.json',repeat)
    assert (output/'combined-episodes.jsonl').read_bytes()==(repeat/'combined-episodes.jsonl').read_bytes()


@pytest.mark.parametrize('change',['hash','missing_review','duplicate_review'])
def test_extension_rejects_unbound_or_incomplete_review(tmp_path,change):
    from pathlib import Path
    from scripts.extend_reviewed_episode_corpus import extend
    root=Path(__file__).resolve().parents[3]
    policy=json.loads((root/'data/router-v2-authored-20260911/extension-review-v3.json').read_text())
    if change=='hash':policy['input_hashes']['candidates']='0'*64
    elif change=='missing_review':policy['row_reviews'].pop()
    else:policy['row_reviews'].append(policy['row_reviews'][0])
    path=tmp_path/'review.json';path.write_text(json.dumps(policy))
    with pytest.raises(ValueError):
        extend(root/'data/public-episode-corpus-20260910-v6',root/'data/public-episode-corpus-20260910-v4',root/'data/router-v2-authored-20260911/candidates-v5',path,tmp_path/'out')
    assert not (tmp_path/'out').exists()


def test_semantic_retrieval_includes_delivered_policy_but_not_hidden_key():
    from scripts.retrieve_semantic_census_evidence import text
    row={'episode_id':'authored:test','prompt':'Read the workspace','fixture':{'files':{'policy.md':'Unique causal-vector policy'},'expected_json':{'result.json':'HIDDEN_KEY'},'tests':'HIDDEN_TEST'}}
    value=text(row)
    assert 'Unique causal-vector policy' in value
    assert 'HIDDEN_KEY' not in value and 'HIDDEN_TEST' not in value


def test_state_grader_survives_instrumented_runtime_factory(tmp_path,monkeypatch):
    from theriad.routing import episode_runtime
    monkeypatch.setattr(episode_runtime,'EpisodeRuntime',lambda **kwargs:None)
    (tmp_path/'input.txt').write_text('keep');(tmp_path/'result.json').write_text('null')
    assert grade_workspace_state(row(None),tmp_path,'completed')[0] is True


def test_state_grader_cannot_follow_output_symlink_outside_workspace(tmp_path):
    from dataclasses import replace
    workspace=tmp_path/'workspace';workspace.mkdir()
    outside=tmp_path/'outside.json';outside.write_text('null')
    (workspace/'result.json').symlink_to(outside)
    with pytest.raises(ValueError,match='escapes its workspace'):
        grade_workspace_state(row(None),workspace,'completed')
