#!/usr/bin/env python3
"""Retrieve candidates for qualitative review, never certify independence."""
import argparse
import hashlib
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--limit',type=int,default=200)
    a=p.parse_args()
    if a.output.exists():raise ValueError('output exists')
    if a.limit<=0:raise ValueError('limit must be positive')
    data=a.corpus.read_bytes();rows=[json.loads(x) for x in data.splitlines()]
    def prompt(row):
        return row['prompt'].replace('Implement the function specified below. Return the complete Python code, including necessary imports and the original function name.','').split('Return only the final numeric')[0].split('Use only the frozen evidence. Return only')[0]
    matrix=TfidfVectorizer(stop_words='english',ngram_range=(1,2),sublinear_tf=True).fit_transform([prompt(r) for r in rows])
    scores=cosine_similarity(matrix)
    pairs=sorted([(float(scores[i,j]),i,j) for i in range(len(rows)) for j in range(i) if rows[i]['family']==rows[j]['family']],reverse=True)
    report={'corpus_sha256':hashlib.sha256(data).hexdigest(),'method':'TF-IDF word unigrams/bigrams, English stop words, sublinear TF, cosine; within-family only',
        'automatic_clearance':False,'limitations':['Shared boilerplate can inflate scores.','Low lexical similarity cannot rule out semantic overlap.','Cross-family and legacy corpus review is separate.'],
        'pairs':[{'similarity':s,'left':rows[i]['episode_id'],'right':rows[j]['episode_id'],'left_split':rows[i]['split'],'right_split':rows[j]['split'],'left_prompt':prompt(rows[i]),'right_prompt':prompt(rows[j])} for s,i,j in pairs[:a.limit]]}
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
