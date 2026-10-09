#!/usr/bin/env python3
"""Reproducible lexical support for a qualitative census; no automatic clearance."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def rows(path):return [json.loads(x) for x in path.read_text().splitlines()]
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def text(row):
    value=row['prompt'].replace('Implement the function specified below. Return the complete Python code, including necessary imports and the original function name.','')
    value=value.split('Return only the final numeric')[0].split('Use only the frozen evidence. Return only')[0]
    if 'episode_id' not in row and '\n\n' in value:value=value.split('\n\n',1)[1]
    # Delivered workspace policies and evidence are part of the task.
    # Hidden answers, grading tests and reference implementations are excluded.
    files=row.get("fixture",{}).get("files",{})
    return value + "\n" + "\n".join(str(v) for _,v in sorted(files.items()))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('corpus','legacy','classifications','output'):p.add_argument('--'+key,type=Path,required=True)
    p.add_argument("--addition-prefix",help="Also retain per-addition nearest original/legacy/addition comparisons")
    a=p.parse_args()
    if a.output.exists():raise ValueError('output must be new')
    current,legacy=rows(a.corpus),rows(a.legacy)
    allrows=current+legacy
    classes={r['episode_id']:r['mechanism'] for r in rows(a.classifications)}
    identify=lambda i:allrows[i].get('episode_id',allrows[i].get('id'))
    matrix=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit_transform([text(r) for r in allrows])
    similarity=cosine_similarity(matrix)
    report={'corpus_sha256':digest(a.corpus),'legacy_sha256':digest(a.legacy),
            'classifications_sha256':digest(a.classifications),'automatic_clearance':False,
            'method':'Delivered prompts and fixture files, excluding hidden graders and references. TF-IDF English-stop-word-filtered word unigrams/bigrams, sublinear TF, cosine. Retain top 100 pairs and every row nearest pair per comparison category. This does not detect every semantic relationship.',
            'pair_candidates':{},'pair_counts_scanned':{}}
    for kind in ('legacy','cross_family','within_family'):
        pairs=[]
        for i in range(len(current)):
            for j in range(i+1,len(allrows)):
                if (kind=='legacy') != (j>=len(current)):continue
                if kind=='cross_family' and current[i]['family']==current[j]['family']:continue
                if kind=='within_family' and current[i]['family']!=current[j]['family']:continue
                pairs.append((float(similarity[i,j]),i,j))
        pairs.sort(reverse=True)
        nearest={}
        for value in pairs:
            for k in value[1:]:nearest.setdefault(k,value)
        chosen=sorted(set(pairs[:100]+list(nearest.values())),reverse=True)
        report['pair_counts_scanned'][kind]=len(pairs)
        report['pair_candidates'][kind]=[{'score':s,'left':identify(i),'right':identify(j),
            'left_mechanism':classes[identify(i)],'right_mechanism':classes[identify(j)]} for s,i,j in chosen]
    if a.addition_prefix:
        additions=[i for i,r in enumerate(current) if r['episode_id'].startswith(a.addition_prefix)]
        originals=[i for i,r in enumerate(current) if not r['episode_id'].startswith(a.addition_prefix)]
        report['focused_addition_neighbors']=[]
        for i in additions:
            groups={}
            for label,indices in [('original',originals),('legacy',range(len(current),len(allrows))),('addition',additions)]:
                indices=[j for j in indices if j!=i]
                nearest=sorted(indices,key=lambda j:-similarity[i,j])[:5]
                groups[label]={'comparisons_scanned':len(indices),'neighbors':[{'episode_id':identify(j),'mechanism':classes[identify(j)],'score':float(similarity[i,j])} for j in nearest]}
            report['focused_addition_neighbors'].append({'episode_id':identify(i),'comparisons':groups})
    docs=[(r['episode_id'],d['title'],d['text']) for r in current for d in r['fixture'].get('documents',[])]
    dm=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit_transform([d[2] for d in docs])
    ds=cosine_similarity(dm)
    report['document_neighbors_above_032']=sorted([{'score':float(ds[i,j]),'left':docs[i],'right':docs[j]}
        for i in range(len(docs)) for j in range(i) if docs[i][0]!=docs[j][0] and ds[i,j]>=.32],key=lambda d:-d['score'])
    phrases,paragraphs,titles=defaultdict(set),defaultdict(set),defaultdict(set)
    for key,title,value in docs:
        tokens=re.findall(r'\w+',value.casefold());paragraphs[' '.join(tokens)].add(key)
        for i in range(len(tokens)-15):phrases[' '.join(tokens[i:i+16])].add(key)
    for r in current:
        for title in r['fixture'].get('original_context_titles',[]):titles[title.casefold()].add(r['episode_id'])
    report.update(documents=len(docs),original_context_titles=len(titles),
        repeated_original_titles=[{'title':k,'members':sorted(v)} for k,v in titles.items() if len(v)>1],
        repeated_normalized_paragraphs=[{'text':k,'members':sorted(v)} for k,v in paragraphs.items() if len(v)>1],
        shared_16_word_phrases=[{'phrase':k,'members':sorted(v)} for k,v in phrases.items() if len(v)>1])
    a.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('pair_candidates','document_neighbors_above_032','shared_16_word_phrases','focused_addition_neighbors')},indent=2))
if __name__=='__main__':main()
