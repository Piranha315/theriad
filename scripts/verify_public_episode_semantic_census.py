#!/usr/bin/env python3
"""Offline verification of a census package, including intentionally empty cells."""
import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from theriad.routing.episode_study import Episode, validate_episodes
from theriad.routing.episode_fixtures import require_collection_ready
from theriad.routing.episode_runtime import _text_grade
from finalize_public_episode_semantic_census import finalize, read_rows, canonical_sha


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--original',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.output.exists():raise ValueError('output must be new')
    original=read_rows(a.original/'combined-episodes.jsonl')
    kept=read_rows(a.directory/'combined-episodes.jsonl')
    removed=read_rows(a.directory/'excluded-episodes.jsonl')
    decisions=read_rows(a.directory/'row-decisions.jsonl')
    report=json.loads((a.directory/'review-report.json').read_text())
    policy=json.loads((a.directory/'semantic-review.json').read_text())
    prior=json.loads((a.directory/'prior-review.json').read_text())
    ledger=read_rows(a.directory/'classification-ledger.jsonl')
    legacy=read_rows(a.directory/'legacy-comparison.jsonl')
    assert hashlib.sha256((a.original/'combined-episodes.jsonl').read_bytes()).hexdigest()==policy['input_sha256']
    assert hashlib.sha256((a.directory/'legacy-comparison.jsonl').read_bytes()).hexdigest()==policy['legacy_sha256']
    assert hashlib.sha256((a.directory/'classification-ledger.jsonl').read_bytes()).hexdigest()==policy['ledger_sha256']
    recomputed=finalize(original,legacy,ledger,prior,policy)
    assert recomputed[:3]==(kept,removed,decisions)
    assert recomputed[3]==json.loads((a.directory/'dependence-graph.json').read_text())
    assert report['review_hash']==recomputed[3]['review_hash']
    assert hashlib.sha256((a.directory/'combined-episodes.jsonl').read_bytes()).hexdigest()==report['output_sha256']
    ids=lambda rows:{r['episode_id'] for r in rows}
    assert not ids(kept)&ids(removed) and ids(kept)|ids(removed)==ids(original)
    assert len(decisions)==len(original)+len(legacy)
    assert all(d['review_complete'] for d in decisions)
    old={r['episode_id']:r for r in original}
    stage=('pilot','smoke','train','calibration','threshold_selection','test')
    payloads=0
    for row in kept+removed:
        before=old[row['episode_id']]
        # No input, grader, source, capability or execution-bound rewrites.
        for key in ('prompt','grader','grader_version','capabilities','source','template_id','max_calls','max_input_tokens','max_output_tokens','max_seconds'):
            assert row[key]==before[key],(row['episode_id'],key)
        for key,value in before['fixture'].items():
            if key not in ('lineage_id','semantic_lineage_ids','semantic_review'):
                assert row['fixture'][key]==value,(row['episode_id'],key)
        if row['family']=='coding':payloads+=1
    assert all(stage.index(r['split'])<=stage.index(old[r['episode_id']]['split']) for r in kept)
    assert all(d['assigned_split'] not in ('test','calibration','threshold_selection') for d in decisions if d['decision']=='development_only_prior_overlap')
    validate_episodes([Episode.from_dict(r) for r in kept],require_all_families=False)
    try:validate_episodes([Episode.from_dict(r) for r in kept])
    except ValueError as exc:coverage_error=str(exc)
    else:coverage_error=None
    assert bool(coverage_error)==bool(report['empty_family_split_cells'])
    quarantine=[]
    for subset in (kept,removed):
        try:require_collection_ready([Episode.from_dict(r) for r in subset])
        except ValueError as exc:quarantine.append(str(exc))
        else:raise AssertionError('quarantine bypassed')
    sources=json.loads((a.directory/'source-manifest.json').read_text())
    for item in sources:
        assert hashlib.sha256((a.directory/item['path']).read_bytes()).hexdigest()==item['sha256']
    refs={r['episode_id']:r['reference'] for r in read_rows(a.directory/'reference-answers.jsonl')}
    new_ids=ids(read_rows(a.directory/'new-episodes.jsonl'))
    positive_negative=code_ast=0
    for row in kept:
        if row['episode_id'] not in new_ids:continue
        episode=Episode.from_dict(row)
        if row['family']=='coding':
            ast.parse(refs[row['episode_id']]);ast.parse(row['fixture']['tests']);code_ast+=1
        else:
            assert _text_grade(episode,str(row['grader']['answer']))[0] is True
            assert _text_grade(episode,'THIS IS A DELIBERATELY WRONG ANSWER')[0] is False
            positive_negative+=1
    result={'complete_census_reproduced':True,'candidate_rows_accounted_for':len(kept)+len(removed),
            'legacy_comparison_rows':len(legacy),'source_file_checksums':len(sources),
            'all_original_task_and_grader_payloads_unchanged':True,'coding_payloads_unchanged':payloads,
            'new_retained_code_ast_checks':code_ast,'new_retained_text_grader_positive_negative_checks':positive_negative,
            'collection_quarantine_messages':quarantine,'expected_full_family_validation_failure':coverage_error,
            'no_exposure_moved_later':True,'no_prior_related_heldout_rows':True,
            'source_sha256':report['input_sha256'],'output_sha256':report['output_sha256'],
            'review_hash':report['review_hash'],'code_executed_in_this_verification':False,'model_calls':0}
    a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
