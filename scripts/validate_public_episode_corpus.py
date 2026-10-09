#!/usr/bin/env python3
"""Offline whole-corpus integrity checks; never executes reference code."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import ast, json, hashlib, asyncio
from pathlib import Path
from collections import Counter
from theriad.routing.episode_study import Episode, validate_episodes, atomic_json
from theriad.routing.episode_runtime import _text_grade, FrozenEvidenceService
from theriad.routing.sandbox import sandbox_available
import argparse
parser=argparse.ArgumentParser(description="Validate a downloaded public Episode corpus without model calls or code execution")
parser.add_argument('--directory',type=Path,required=True)
args=parser.parse_args()
package=args.directory.resolve()
rows=[Episode.from_dict(json.loads(x)) for x in (package/'new-episodes.jsonl').read_text().splitlines()]
combined=[Episode.from_dict(json.loads(x)) for x in (package/'combined-episodes.jsonl').read_text().splitlines()]
refs={x['episode_id']:x['reference'] for x in [json.loads(x) for x in (package/'reference-answers.jsonl').read_text().splitlines()]}
source_files=json.loads((package/'source-manifest.json').read_text())
for info in source_files:
 assert hashlib.sha256((package/info['path']).read_bytes()).hexdigest()==info['sha256']
validate_episodes(combined)
for row in rows:
 assert row.fixture['source_metadata']['license_name']
 assert row.fixture['source_metadata']['source_file_sha256']
 assert row.fixture['source_metadata']['provenance_url'].startswith('https://')
 assert row.fixture['source_metadata']['source_revision']
 if row.family=='coding':
  assert 'canonical_solution' not in row.fixture and 'reference' not in row.fixture
  tree=ast.parse(refs[row.episode_id]);tests=ast.parse(row.fixture['tests'])
  assert any(isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name==row.fixture['entry_point'] for n in ast.walk(tree))
  assert any(isinstance(n,ast.FunctionDef) and n.name=='check' for n in ast.walk(tests))
  assert any(isinstance(n,ast.Assert) for n in ast.walk(tests))
 else:
  expected=row.grader['answer'];good,_=_text_grade(row,str(expected));bad,_=_text_grade(row,'THIS IS A DELIBERATELY WRONG ANSWER')
  assert good is True,(row.episode_id,expected)
  assert bad is False,(row.episode_id,bad)
used=set()
async def evidence():
 for row in rows:
  if row.family!='search':continue
  titles={x.casefold() for x in row.fixture['original_context_titles']}
  assert not used & titles
  used.update(titles)
  docs=row.fixture['documents'];assert 1<=len(docs)<=4 and all(len(d['text'])<=1700 for d in docs)
  service=FrozenEvidenceService(docs)
  results=await service.search(row.prompt)
  assert len(results)==len(docs)
  for result in results: assert (await service.fetch(result.url))['text']
asyncio.run(evidence())
report={'schema_valid':True,'source_file_checksums_verified':len(source_files),'rows_checked':len(rows),
 'text_grader_correct_and_wrong_answer_checks':sum(r.family!='coding' for r in rows),
 'code_reference_and_test_ast_checks':sum(r.family=='coding' for r in rows),
 'code_executed':False,'real_sandbox_available':sandbox_available(),
 'search_fixture_service_checks':sum(r.family=='search' for r in rows),
 'all_original_search_document_titles_disjoint':True,'source_provenance_present':True,
 'model_calls':0,'production_changed':False,'semantic_independence_certified':False}
atomic_json(package/'validation.json',report)
print(json.dumps(report,indent=2))
