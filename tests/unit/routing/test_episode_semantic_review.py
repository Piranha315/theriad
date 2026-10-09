"""Semantic decisions control splits and collection; scores cannot clear review."""
from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from theriad.routing.episode_fixtures import build_four_way_seed_corpus, require_collection_ready
from theriad.routing.episode_study import validate_episodes
from theriad.routing import sandbox

ROOT=Path(__file__).resolve().parents[3]
def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result
reviewer=module('semantic_review','apply_public_episode_semantic_review.py')


def test_transitive_dependencies_move_to_earliest_split_without_mutating_input():
    rows=[asdict(r) for r in build_four_way_seed_corpus()]
    ids=['coding.merge_intervals.trial-1','coding.run_length_codec.trial-1','coding.nested_lookup.trial-1']
    review={'version':'test','clusters':[{'members':ids[:2],'decision':'group_conservatively','rationale':'fixture edge one'},
                                        {'members':ids[1:],'decision':'group_conservatively','rationale':'fixture edge two'}]}
    changed,assignments=reviewer.apply_review(rows,review)
    affected=[r for r in changed if r['episode_id'] in ids]
    assert len({r['group_id'] for r in affected})==1
    assert {r['split'] for r in affected}=={'train'}
    assert next(r for r in rows if r['episode_id']==ids[-1])['split']=='calibration'
    assert len(assignments)==3
    assert all(r['fixture']['semantic_review']['collection_ready'] is False for r in changed)


def test_unknown_and_cross_family_review_edges_fail_closed():
    rows=[asdict(r) for r in build_four_way_seed_corpus()]
    for members, message in [(['unknown'],'unknown episode'),
        (['coding.merge_intervals.trial-1','chat.conversation_revision.trial-1'],'cross-family')]:
        with pytest.raises(ValueError,match=message):
            reviewer.apply_review(rows,{'version':'test','clusters':[{'members':members,'decision':'group_conservatively','rationale':'test'}]})


def test_semantic_lineage_cannot_be_hidden_by_renaming_group_or_template():
    rows=build_four_way_seed_corpus()[:2]
    rows=[replace(row,split='train',fixture={**row.fixture,'semantic_lineage_ids':['shared-semantic-template']}) for row in rows]
    with pytest.raises(ValueError,match='related templates'):
        validate_episodes(rows,require_all_families=False)


@pytest.mark.parametrize('fixture',[
    {'source_metadata':{'independence_status':'screened_candidate_not_certified'}},
    {'semantic_review':{'collection_ready':False}},
    {'semantic_review':{'collection_ready':True}},
])
def test_unreviewed_candidates_cannot_enter_collection(fixture):
    row=replace(build_four_way_seed_corpus()[0],fixture=fixture)
    with pytest.raises(ValueError,match='quarantined'):require_collection_ready([row])


def test_availability_probe_permits_loader_reads_but_uses_fixed_binary_and_empty_env(monkeypatch):
    seen=[]
    def run(command,**kwargs):seen.append((command,kwargs));return SimpleNamespace(returncode=0)
    sandbox.sandbox_available.cache_clear()
    monkeypatch.setattr(sandbox,'_sandbox_exec_path',lambda:'/usr/bin/sandbox-exec')
    monkeypatch.setattr(sandbox.sys,'platform','darwin')
    monkeypatch.setattr(sandbox.subprocess,'run',run)
    try:
        assert sandbox.sandbox_available()
        command,kwargs=seen[0]
        assert command[-1]=='/usr/bin/true'
        assert '(allow file-read*)' in command[2]
        assert '(deny default)' in command[2]
        assert kwargs['env']=={}
    finally:sandbox.sandbox_available.cache_clear()


def test_real_boundary_blocks_network_host_files_and_inherited_environment():
    if not sandbox.sandbox_available():pytest.skip('macOS sandbox-exec required')
    verifier=module('sandbox_boundary_verifier','verify_public_coding_sandbox.py')
    import sys
    result=verifier.verify_boundary(Path(sys.executable))
    assert result['passed'],result
