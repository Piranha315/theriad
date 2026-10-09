#!/usr/bin/env python3
"""Offline, deterministic public-task acquisition audit and Episode conversion.

No provider calls or reference-code execution. IDs identify records; conservative
structural screens reduce known dependence but never certify IID sampling.
"""
from __future__ import annotations
import argparse
import ast
import copy
import gzip
import hashlib
import json
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from theriad.routing.episode_study import Episode, atomic_json, digest, validate_episodes
from theriad.routing.episode_fixtures import build_four_way_seed_corpus

VERSION = 'public-episode-acquisition-20260910-v4'
LICENSES = {'humaneval':'MIT','gsm8k':'MIT','hotpotqa':'CC-BY-SA-4.0','tau2':'MIT'}


def normalized(text):
    return ' '.join(re.findall(r'[a-z0-9]+', text.casefold()))


def previous_prompt_texts(rows):
    values=[]
    for row in rows:
        prompt=str(row.get('prompt',row.get('question','')))
        values.append(prompt)
        if prompt.startswith('Answer the following question with a short, precise response.') and '\n\n' in prompt:
            values.append(prompt.split('\n\n',1)[1])
    return values


def json_record(value):
    # The existing runner uses splitlines(), including Unicode separators.
    return json.dumps(value,sort_keys=True,ensure_ascii=True)+'\n'


def lexical_signature(text):
    # Remove numbers and proper names: cosmetic variants must remain neighbors.
    text = re.sub(r'\b[A-Z][a-z]+\b', 'PERSON', text)
    words = re.findall(r'[a-z]+|NUMBER', re.sub(r'\d+(?:\.\d+)?', 'NUMBER', text).lower())
    return frozenset(tuple(words[i:i+4]) for i in range(max(0,len(words)-3)))


def near_duplicate(left, right):
    if not left or not right: return False
    return len(left & right) / min(len(left), len(right)) >= .70


def arithmetic_shape(answer):
    terms = re.findall(r'<<([^<>]+)>>', answer)
    return tuple(re.sub(r'\d+(?:\.\d+)?', 'N', term.split('=')[0].replace(' ', '')) for term in terms)


def code_shape(code):
    class Names(ast.NodeTransformer):
        def visit_Name(self, node):
            node.id = 'name'; return node
        def visit_arg(self, node):
            node.arg = 'arg'; return node
        def visit_FunctionDef(self, node):
            node.name = 'function'; self.generic_visit(node)
            node.body = [x for x in node.body if not (isinstance(x,ast.Expr) and isinstance(x.value,ast.Constant) and isinstance(x.value.value,str))]
            return node
        def visit_Constant(self, node):
            return ast.copy_location(ast.Constant(value=type(node.value).__name__),node)
    return ast.dump(Names().visit(ast.parse(code)),include_attributes=False)


def assigned_split(group):
    value = int(hashlib.sha256(('public-corpus-split-v1:'+group).encode()).hexdigest()[:12],16) % 100
    for upper, split in [(45,'train'),(60,'calibration'),(75,'threshold_selection'),(95,'test'),(98,'pilot'),(100,'smoke')]:
        if value < upper: return split


def lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(cache, output, previous, *, math_limit=500, search_limit=500):
    if output.exists(): raise ValueError('output must be a new directory; previous corpus packages are immutable')
    manifest = json.loads((cache/'manifest.json').read_text())
    files = {}
    for item in manifest:
        if 'error' in item: continue
        source = Path(item['path'])
        if not source.resolve().is_relative_to(cache.resolve()): raise ValueError('source path escapes cache')
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']: raise ValueError('source checksum mismatch')
        files[str(source.relative_to(cache))] = item
    def origin(source, path):
        item = files[source+'/'+path]
        return {'dataset':source, 'source_revision':item['revision'], 'source_file_sha256':item['sha256'],
                'license_name':LICENSES[source], 'license_status':'approved', 'provenance_url':item['url'],
                'public_benchmark_contamination_possible':True, 'independence_status':'screened_candidate_not_certified',
                'adaptation_version':VERSION}
    prior_external_rows = [row for path in previous for row in lines(path)]
    prior_seed_rows = [asdict(row) for row in build_four_way_seed_corpus()]
    old_rows = prior_external_rows + prior_seed_rows
    old_text = previous_prompt_texts(old_rows)
    old_normal = {normalized(value) for value in old_text}
    old_sigs = [lexical_signature(value) for value in old_text]
    accepted, rejected, signatures, seen_shapes, seen_prompts = [], [], [], set(), set(old_normal)
    raw_counts = {}; references = []; counts = Counter()
    def reject(source, identity, reason):
        rejected.append({'source':source,'source_id':identity,'reason':reason})
    def admit(row, source, identity, text, shape, reference=None):
        sig = lexical_signature(text)
        if normalized(text) in seen_prompts or any(near_duplicate(sig,x) for x in old_sigs):
            reject(source,identity,'previous_local_prompt_overlap'); return False
        if shape in seen_shapes:
            reject(source,identity,'same_structural_template'); return False
        if any(near_duplicate(sig,x) for x in signatures):
            reject(source,identity,'near_duplicate_wording'); return False
        seen_shapes.add(shape); seen_prompts.add(normalized(text)); signatures.append(sig)
        accepted.append(row); counts[source]+=1
        if reference is not None: references.append({'episode_id':row.episode_id,'reference':reference})
        return True
    def episode(source, identity, family, prompt, fixture, grader, shape, metadata, **bounds):
        group = source+':'+digest(shape)[:24]
        return Episode(f'{source}:{identity}', group, family, assigned_split(group), prompt,
            {**fixture,'lineage_id':group,'source_metadata':{**metadata,'source_id':identity}},grader,
            source=source,template_id=group,fixture_version=VERSION,grader_version=VERSION,
            tags=('public_source','candidate_independence','not_production_weighted'),
            capabilities=('text','search') if family=='search' else ('text',),**bounds)

    path=cache/'humaneval/data/HumanEval.jsonl.gz'
    raw=[json.loads(line) for line in gzip.decompress(path.read_bytes()).splitlines()]
    raw_counts['humaneval']=len(raw)
    for item in sorted(raw,key=lambda row:digest(row['task_id'])):
        identity=item['task_id']; full=item['prompt']+item['canonical_solution']
        try: shape=('humaneval',code_shape(full)); ast.parse(item['test'])
        except SyntaxError: reject('humaneval',identity,'invalid_reference_or_test_syntax'); continue
        prompt='Implement the function specified below. Return the complete Python code, including necessary imports and the original function name.\n\n'+item['prompt']
        row=episode('humaneval',identity,'coding',prompt,{'tests':item['test'],'entry_point':item['entry_point']},
            {'kind':'code'},shape,origin('humaneval','data/HumanEval.jsonl.gz'),max_calls=4,max_output_tokens=2048)
        admit(row,'humaneval',identity,item['prompt'],shape,full)

    raw=[]
    for split in ('train','test'):
        filename=f'grade_school_math/data/{split}.jsonl'
        raw += [(split,index,item) for index,item in enumerate(lines(cache/'gsm8k'/filename))]
    raw_counts['gsm8k']=len(raw)
    # Upstream test labels have already been used in the old local grid; use new
    # router splits only as public-benchmark holdouts, never as unseen pretraining data.
    old_math_shapes={arithmetic_shape(item['answer']) for _,_,item in raw if normalized(item['question']) in old_normal}
    for upstream_split,index,item in sorted(raw,key=lambda value:digest(value[2]['question'])):
        identity=f'{upstream_split}/{index}'
        shape=('gsm8k',arithmetic_shape(item['answer']))
        if not shape[1]: reject('gsm8k',identity,'missing_arithmetic_trace'); continue
        if shape[1] in old_math_shapes: reject('gsm8k',identity,'structural_template_in_previous_local_grid'); continue
        if counts['gsm8k']>=math_limit: reject('gsm8k',identity,'bounded_sample_limit'); continue
        answer=item['answer'].rsplit('####',1)[-1].strip().replace(',','')
        if not re.fullmatch(r'-?\d+(?:\.\d+)?',answer): reject('gsm8k',identity,'unsupported_numeric_answer'); continue
        prompt=item['question']+'\nReturn only the final numeric answer, without units or explanation.'
        row=episode('gsm8k',identity,'chat',prompt,{}, {'kind':'numeric','answer':answer},shape,
                    {**origin('gsm8k',f'grade_school_math/data/{upstream_split}.jsonl'),'upstream_split':upstream_split,
                     'workload_scope':'short_form_arithmetic_reasoning_not_general_conversation'},max_calls=4,max_output_tokens=1024)
        admit(row,'gsm8k',identity,item['question'],shape,item['answer'])

    raw=lines(cache/'hotpotqa/distractor/validation.jsonl');raw_counts['hotpotqa']=len(raw)
    used_titles=set()
    for item in sorted(raw,key=lambda row:digest(row['id'])):
        identity=item['id']
        if counts['hotpotqa']>=search_limit: reject('hotpotqa',identity,'bounded_sample_limit'); continue
        titles=item['context']['title'];sentences=item['context']['sentences']
        title_set={normalized(title) for title in titles}
        if used_titles & title_set: reject('hotpotqa',identity,'shared_context_document'); continue
        supporting=set(item['supporting_facts']['title'])
        docs=[{'title':t,'text':''.join(s),'url':'https://en.wikipedia.org/wiki/'+quote(t.replace(' ','_'))} for t,s in zip(titles,sentences,strict=True)]
        # Four full paragraphs fit the existing six-result synthesis adapter.
        chosen=[d for d in docs if d['title'] in supporting]
        if not chosen or len(chosen)>4 or any(len(d['text'])>1700 for d in chosen):
            reject('hotpotqa',identity,'support_exceeds_frozen_evidence_adapter');continue
        chosen += sorted([d for d in docs if d['title'] not in supporting and len(d['text'])<=1700],key=lambda d:digest(d['title']))[:4-len(chosen)]
        shape=('hotpotqa',tuple(sorted(title_set)))
        prompt=item['question']+'\nUse only the frozen evidence. Return only the short answer, preserving its spelling in the evidence; do not include explanation or citations.'
        row=episode('hotpotqa',identity,'search',prompt,{'documents':chosen,'network_required':False,
            'original_context_titles':titles,'snapshot_kind':'public_benchmark_four_paragraph_adaptation'},
            {'kind':'exact','answer':item['answer']},shape,
            {**origin('hotpotqa','distractor/validation.jsonl'),'original_source_url':'https://hotpotqa.github.io/',
             'adaptation':'all supporting paragraphs plus up to two bounded distractors; strict answer grader, not official HotpotQA score'},max_calls=2,max_output_tokens=1024)
        if admit(row,'hotpotqa',identity,item['question'],shape,item['answer']): used_titles.update(title_set)

    validate_episodes(accepted,require_all_families=False)
    combined=build_four_way_seed_corpus()+accepted
    validate_episodes(combined)
    output.mkdir(parents=True)
    source_manifest=[]
    for relative,item in sorted(files.items()):
        target=output/'sources'/relative;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(cache/relative,target)
        source_manifest.append({**item,'path':'sources/'+relative})
    atomic_json(output/'source-manifest.json',source_manifest)
    for name,rows in [('new-episodes.jsonl',[asdict(r) for r in accepted]),('combined-episodes.jsonl',[asdict(r) for r in combined]),('rejections.jsonl',rejected),('reference-answers.jsonl',references)]:
        (output/name).write_text(''.join(json_record(row) for row in rows))
    unsupported=[]
    for domain in ('airline','retail','telecom'):
        rows=json.loads((cache/f'tau2/data/tau2/domains/{domain}/tasks.json').read_text())
        raw_counts['tau2_'+domain]=len(rows)
        unsupported.append({'source':'tau2','domain':domain,'raw_tasks':len(rows),'usable_episode_rows':0,
            'independent_groups':None,'reason':'native domain environment, user simulator, official state/action grader and lineage adaptation required; generated scenarios are not counted as independent',
            'revision':origin('tau2',f'data/tau2/domains/{domain}/tasks.json')['source_revision']})
    atomic_json(output/'adapter-needed.json',unsupported)
    report={'version':VERSION,'raw_counts':raw_counts,'new_episode_rows':len(accepted),
        'screened_candidate_groups':len({r.group_id for r in accepted}),'independence_certified':False,
        'new_by_source':dict(counts),'new_by_family_split':{f:dict(Counter(r.split for r in accepted if r.family==f)) for f in ('chat','coding','search','tools','agents')},
        'combined_manifest':validate_episodes(combined),'combined_sha256':hashlib.sha256((output/'combined-episodes.jsonl').read_bytes()).hexdigest(),
        'prior_corpora':[{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()} for p in previous],
        'previous_local_rows_screened':len(old_rows),'previous_builtin_seed_rows_screened':len(prior_seed_rows),'previous_builtin_seed_hash':digest(prior_seed_rows),'previous_arithmetic_templates_excluded':len(old_math_shapes),
        'rejections_by_reason':dict(Counter(r['reason'] for r in rejected)),
        'overlap_checks':{'normalized_prompts':True,'near_duplicate_wording':True,'code_ast_renaming_and_constants':True,
                          'gsm_arithmetic_trace_templates':True,'hotpot_all_original_context_titles_disjoint':True},
        'schema_valid':True,'code_reference_execution':'not_run_real_sandbox_unavailable','model_outcomes_collected':0,
        'paid_calls':0,'paid_collection_ready':False,'production_enabled':False,
        'limitations':['Candidate groups pass mechanical screens; semantic independence requires review.',
                      'Public benchmarks may already appear in model pretraining or old evaluations outside the available local corpus.',
                      'Math is a narrow chat proxy; four-paragraph QA is frozen evidence synthesis, not live search.',
                      'No new runnable tools/agents tasks; tau2 raw counts do not increase sample size.',
                      'Code syntax checked only; real grader and reference correctness remain unverified.',
                      'No empirical quality, price, achieved power, or production weighting is inferred.']}
    atomic_json(output/'acquisition-report.json',report)
    (output/'ATTRIBUTION.md').write_text('''# Source attribution and adaptations\n\n- HumanEval: OpenAI, MIT; https://github.com/openai/human-eval . Original prompts/tests preserved; complete-code instruction added.\n- GSM8K: OpenAI, MIT; https://github.com/openai/grade-school-math . Original questions preserved; final-number instruction added.\n- HotpotQA: Yang et al. (EMNLP 2018), CC BY-SA 4.0; https://hotpotqa.github.io/ . Mirror: https://huggingface.co/datasets/hotpotqa/hotpot_qa . Derived HotpotQA rows/passages remain CC BY-SA 4.0: https://creativecommons.org/licenses/by-sa/4.0/ . Four bounded source paragraphs are retained and strict-answer instructions added. Wikipedia article links occur in each fixture; these are historical corpus snapshots, not today's pages. No endorsement is implied.\n- tau2-bench: Sierra Research (2025), MIT; https://github.com/sierra-research/tau2-bench . Native task/policy files retained unchanged for adapter development; no independence count assigned.\n\nExact revisions, retrieval URLs, SHA-256 values, original READMEs and MIT license files are in sources/ and source-manifest.json. The initial CMU HotpotQA endpoint timed out; its pinned public HF mirror was used. No blanket repository license overrides these source licenses.\n''')
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--previous',type=Path,nargs='+',required=True)
    args=p.parse_args(); print(json.dumps(build(args.cache,args.output,args.previous),indent=2))
if __name__=='__main__': main()
