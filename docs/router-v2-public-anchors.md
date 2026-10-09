# Router V2 public outcome anchors

The primary Router V2 experiment remains the exact 2,000-prompt Theriad study:
1,200 train, 400 validation, and 400 untouched test prompts. Public benchmark
anchors supplement and warm-start that experiment; they do not replace it. A
statistically sized Theriad-specific calibration study is an optional,
separately named follow-up, not the primary corpus.

## BrowseComp-Plus audit

The first audit target is the official [BrowseComp-Plus repository](https://github.com/texttron/BrowseComp-Plus).
As of the current audit, its public artifacts separate into:

- the [benchmark dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus),
  with 830 queries and encrypted/obfuscated query content;
- the public [corpus dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus-corpus),
  with 100,195 documents;
- the public [run dataset](https://huggingface.co/datasets/Tevatron/browsecomp-plus-runs),
  with four large trajectory configurations: BM25/Qwen3-Embedding-8B crossed
  with GPT-5/o3;
- the [aggregate result table](https://huggingface.co/datasets/Tevatron/BrowseComp-Plus-results),
  which reports leaderboard-level rows, not prompt-by-model correctness.

The raw run rows contain `query_id`, `status`, `search_counts`, and a
trajectory `result`. A `completed` status is not a correctness label. The
importer therefore leaves `outcome: null` until an explicit per-query
evaluation is supplied. The official repository's helper can decrypt locally
downloaded files, but the raw source files remain the immutable provenance
boundary; the normalized index stores source path, line, and record hash rather
than duplicating potentially large model outputs.

The third-party [bcp-eval-logs collection](https://huggingface.co/datasets/ZhuofengLi/bcp-eval-logs)
contains 100-query log configurations but is manually gated and is not treated
as an official usable source without access and schema/licensing review.

## Evaluated log collections

The [i-DeepSearch observation-masking evaluation logs](https://huggingface.co/datasets/i-DeepSearch/observation-masking-eval-logs)
are directly usable for this study. The Apache-2.0 dataset contains 113
`evaluated.jsonl` files covering BrowseComp-Plus, xBench, GAIA-text, and
BrowseComp-ZH. The evaluated rows include `qid`, question text, reference
answer, generated answer, and an explicit boolean `correct`; the much larger
trajectory shards are not required for the first difficulty pass. `CM`,
retrieval configuration, benchmark family, and ablation variants are retained
as separate protocol metadata rather than pooled together.

The pinned replay command is:

```bash
.venv/bin/python scripts/reproduce_observation_masking_import.py \
  --manifest reports/router-v2/observation-masking-manifest.json \
  --cache-dir /tmp/theriad-router-v2-observation-masking \
  --output data/router-v2/observation-masking-anchors.jsonl \
  --audit reports/router-v2/observation-masking-audit.json \
  --rejections reports/router-v2/observation-masking-rejections.jsonl
```

The manifest pins revision
`19342195a5c5c9f8ae016cac6a388d937371df8e`, Apache-2.0 metadata, and the
113-file evaluated-log tree. The current replay accepts 1,322 prompts and
77,369 observations, with 77,368 explicit outcomes and one missing outcome.
The raw cache is outside the repository; the normalized JSONL and raw cache
are explicitly gitignored. A normal replay verifies the recorded file hashes;
use `--refresh-manifest` only when intentionally re-auditing the file list at
the already pinned revision.

The manifest must explicitly provide model/provider/family, protocol, source
revision, provenance URL, and license metadata for each local evaluated file.
The importer can recover a missing question only from another file in the same
benchmark family; it records any remaining error rows as rejections.

The compact offline verification command is:

```bash
.venv/bin/python scripts/verify_observation_masking_import.py \
  --manifest reports/router-v2/observation-masking-manifest.json \
  --cache-dir /tmp/theriad-router-v2-observation-masking \
  --anchors data/router-v2/observation-masking-anchors.jsonl \
  --audit reports/router-v2/observation-masking-audit.json \
  --rejections reports/router-v2/observation-masking-rejections.jsonl
```

It verifies the 113 cached file hashes and the exact 1,322/77,369/77,368,
9-model/27-protocol/zero-rejection audit boundary without contacting a model
provider.

The [ZhuofengLi BCP-100 logs](https://huggingface.co/datasets/ZhuofengLi/bcp-eval-logs)
use the same generic importer once the manual Hub gate is approved and the
files are downloaded locally. The Hub metadata does not publish a license, so
the manifest must not invent one; use an explicit access/licensing status and
keep those observations separate from the Apache-2.0 collection until review.

Run the metadata-only audit with:

```bash
.venv/bin/python scripts/audit_browsecomp_plus.py \
  --output reports/router-v2/browsecomp-plus-audit.json
```

`--count-run-rows` is available when an operator explicitly accepts streaming
the large public trajectory files. It does not call a model or spend provider
money.

## Normalized anchor representation

Use an explicit manifest for each run configuration. Every entry must identify
the exact model, provider, protocol, retrieval/scaffold conditions, source
revision, provenance URL, and license status:

```json
[
  {
    "path": "raw/browsecomp-plus-runs/bm25/gpt5.jsonl",
    "model_id": "gpt-5",
    "provider": "openai",
    "model_family": "openai-frontier",
    "protocol_id": "browsecomp-plus-bm25-v1",
    "retrieval_config": "bm25",
    "source_revision": "Tevatron/browsecomp-plus-runs@<sha>",
    "provenance_url": "https://huggingface.co/datasets/Tevatron/browsecomp-plus-runs",
    "license_name": "MIT",
    "license_status": "public",
    "evaluation_path": "evaluations/bm25/gpt5.jsonl",
    "evaluation_method": "qwen3-32b-judge-v1"
  }
]
```

Import locally decrypted queries and runs with:

```bash
.venv/bin/python scripts/import_browsecomp_plus.py \
  --queries raw/browsecomp-plus/queries.jsonl \
  --manifest raw/browsecomp-plus/runs.json \
  --output data/router-v2/public-anchors.jsonl \
  --audit reports/router-v2/public-anchor-audit.json \
  --rejections reports/router-v2/public-anchor-rejections.jsonl
```

The normalized record preserves one observation per
`prompt_id × model_id × protocol_id`, nullable correctness, raw source
references, tool/search counts, trajectory-step counts, evaluation method, and
available latency/token/cost metadata. It derives, without collapsing the raw
observations:

- unweighted model-equal success and failure rate;
- provider/model-family-balanced success and failure rate;
- normalized binary-entropy disagreement;
- explicit capability-order violations, upset rate, and specialization proxy.

Price is never used as capability. A capability profile must be supplied or
estimated from a caller-selected train-only/out-of-fold set.

## Held-out analysis

`theriad/routing/anchor_evaluation.py` implements three falsifiable feature
sets:

1. empirical difficulty;
2. difficulty plus disagreement;
3. difficulty, disagreement, specialization/upsets, prompt features, and
   optional anchor-neighbor features.

Each model head uses leave-one-model-out anchor features when predicting that
model's outcome. This prevents the target model's own public result from
leaking into its feature vector. The report includes quality, cost,
cost-per-success proxies, oracle regret, unnecessary/missed escalation,
selection accuracy, bootstrap intervals, paired bootstrap deltas, and an
approximately equal-difficulty low/high-specialization slice.

Semantic anchor features require a pinned local embedding file through
`AnchorFeatureIndex.from_embeddings`. The TF-IDF index is only a labeled
dependency-light baseline. Neighbors provide evidence features; they never
copy a neighbor's route.

IID splits may contain the same benchmark family while keeping prompt/group
keys together. OOD splits hold out entire benchmark families. IRT/Rasch is
reported only after enough raw outcomes exist and is compared against a
cell-held-out empirical baseline.

## Other public-source investigation

The follow-up audit was deliberately limited to official benchmark artifacts
and read-only metadata:

- [BFCL's official repository](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard)
  and its [Hugging Face dataset](https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard)
  publish test categories, function documentation, possible answers, and a
  checker that produces category/aggregate scores. They do not provide a common public
  `prompt_id x model_id x outcome` table. It is importable after obtaining
  model result files and running the official checker locally, with category
  and execution protocol recorded as separate protocol IDs.
- [GAIA](https://huggingface.co/datasets/gaia-benchmark/GAIA) is gated, and
  [public submissions](https://huggingface.co/datasets/gaia-benchmark/submissions_public)
  are also gated. The public [results table](https://huggingface.co/datasets/gaia-benchmark/results_public)
  currently exposes submission-level model/system-prompt/level aggregates
  (96 validation rows and 3,602 test rows in the audited metadata), not
  task-level outcomes. It is therefore an aggregate comparison source, not a
  direct anchor source. A permitted per-task submission export could be
  normalized with the existing schema after its access and license are
  verified.
- [SWE-bench Verified](https://huggingface.co/datasets/SWE-bench/SWE-bench_Verified)
  provides 500 human-validated issue instances and the [official evaluation
  harness](https://www.swebench.com/SWE-bench/guides/evaluation/) produces
  per-instance reports. The dataset itself contains no model outcome; a local
  model run plus its `report.json` files would be needed. The source maps well
  to anchors, but its outcome is patch-test resolution, not interchangeable
  with answer accuracy.

None of these sources is silently coerced into correctness labels. The next
importers should accept only explicit per-item results, preserve benchmark
family and protocol boundaries, and leave aggregate-only rows out of router
training.

## Optional calibration follow-up

After public anchors are usable, size a smaller calibration set from an
explicit confidence and precision target:

```bash
.venv/bin/python scripts/plan_router_v2_calibration.py \
  --anchors data/router-v2/public-anchors.jsonl \
  --calibration-corpus data/router-v2/theriad-calibration.jsonl \
  --models <model-a> <model-b> <model-c> \
  --report reports/router-v2/calibration-sizing.json \
  --plan artifacts/router-v2/calibration-plan.json
```

The planner uses the existing offline pricing, budget ledger, idempotency, and
fail-closed collector. It does not call a provider. Execution remains a
separate explicit operation and is not authorized by this document.
