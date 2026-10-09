# Router V2

Router V2 predicts each eligible model's calibrated success probability,
lower confidence bound, expected output size, cost, latency, and uncertainty.
The runtime selects the cheapest model whose lower bound clears a validation-
frozen knee threshold. It never asks callers to price the value of correctness.

## Safe rollout

The runtime is disabled by default:

```text
THERIAD_ROUTER_V2_MODE=off|shadow|canary|on
THERIAD_ROUTER_V2_ARTIFACT=/absolute/path/to/router-v2.json
THERIAD_ROUTER_V2_CANARY_PCT=5
THERIAD_ROUTER_V2_SHADOW_BUDGET_FRACTION=0.05
THERIAD_ROUTER_V2_ENABLE_EMBEDDINGS=false
THERIAD_ROUTING_OBSERVATION_STORE=auto
```

`shadow` computes and stores the counterfactual route but does not expose or
silently buy a challenger answer. `canary` applies V2 to a deterministic prompt
sample. A missing, incompatible, embedding-dependent, or corrupt artifact
falls back to the existing resolver.

The stronger semantic encoder must be trained offline and exported into the
same artifact contract. The live server deliberately does not import PyTorch or
Sentence Transformers. An artifact declaring `embedding_model` is rejected
unless the caller supplies a backend or explicitly enables embeddings and the
artifact records a runtime profile at or below 120 MB incremental RSS and 20 ms
p95. The profile is a promotion artifact, not a self-reported model metric.

## Study input

`scripts/train_router_v2.py` consumes one JSON object per prompt:

```json
{
  "prompt": "...",
  "split": "train",
  "source": "gaia",
  "domain": "research",
  "template_family": "gaia-level-2",
  "collected_at": "2026-08-24",
  "lane": "general",
  "outcomes": {
    "openai/gpt-5.6-luna": {
      "success": 1,
      "cost_usd": 0.0021,
      "output_tokens": 310,
      "latency_ms": 1450
    }
  }
}
```

Training refuses normalized-prompt or frozen leakage-group overlap across
splits. Source and benchmark families may appear in more than one split only
when their pinned row-level leakage groups are disjoint. The test split is
accepted by the contract but is never read during artifact fitting or knee
selection.

```bash
python scripts/train_router_v2.py \
  --input data/router-v2/study.jsonl \
  --output artifacts/router-v2/surface.json
```

The trainer defaults to the approved production candidates in the plan. Use
repeated `--general-model` or `--code-model` arguments only when a challenger
has complete holdout coverage and has passed the promotion gates. Models not
listed for a lane remain measurable but are ineligible at runtime.

The shipped trainer is the auditable surface-only baseline. RouterDC,
RouteLLM/BERT, semantic kNN, and hybrid trainers should emit the same JSON
artifact fields, which keeps experimental frameworks out of production.
Profile a semantic artifact on production-class hardware before enabling it:

```bash
python scripts/profile_router_embedding.py \
  --artifact artifacts/router-v2/hybrid.json \
  --prompts data/router-v2/profile-prompts.txt \
  --write-profile
```

The current read-only candidate assembler is reproducible from pinned local
source caches. It writes exactly 2,000 candidate rows and returns exit code 2
with a machine-readable gap report when promotion requirements are not met:

```bash
.venv/bin/python scripts/assemble_router_v2_corpus.py \
  --hf-cache-dir /tmp/theriad-router-v2-source-cache/hf-rows-pinned \
  --bfcl-cache-dir /tmp/theriad-router-v2-source-cache/bfcl-6ea57973c7a6097fd7c5915698c54c17c5b1b6c8 \
  --tau3-cache-dir /tmp/theriad-router-v2-source-cache/tau3-a2c024725189473d2d7cea3a5cfdbcc67478e41f \
  --anchor-jsonl data/router-v2/observation-masking-anchors.jsonl \
  --output data/router-v2/router-v2-candidates.jsonl \
  --audit reports/router-v2/router-v2-corpus-audit.json \
  --gap-report reports/router-v2/router-v2-corpus-gap-report.json
```

The assembler is allowed to emit a frozen 1,200/400/400 corpus only after
those quotas pass. In the current local checkout it fails closed: BFCL and
tau-bench rows are excluded because their real executors are unavailable, and
no licensed pinned executable code source is present. Its gap report therefore
shows the missing 240/80/80 code rows instead of relabeling tool selection as
code. It assigns `ood` only when a test row is outside the training
source/domain/benchmark/template values; test membership alone is not evidence
of OOD. `representative` is the complementary test slice.

Collection uses task-family adapters. Plain text rows use `complete`; BFCL
requires provider-native tool calls, preserved schemas/order/parallel groups,
and the pinned official checker; tau-bench requires an agent inside the
pinned environment with persisted state and the official evaluator. Missing
executor or tool capability is ineligible, not a failed model outcome.

`scripts/collect_router_v2_study.py` writes four separate plans: core-model
training, outcome-based Grok selection/training, validation-only router
selection, and a final-manifest-gated untouched test. The pre-test manifest is
broader than the corpus checksum and test unlock remains false until all
adapter, grading, validation, and configuration gates pass.

`disagreement-candidate` is a preregistered public-model proxy tag only. It is
never reported as observed candidate-model disagreement. Authentic
`model-disagreement` is not a pre-collection gate; it is a post-collection
slice computed from explicit core-model outcomes and is excluded from training
and tuning. The Grok informative-selection algorithm is frozen before any
outcomes exist, uses only train rows, and can be re-applied after the three
core-model outcomes exist.

Every frozen row carries a normalized SHA-256 prompt hash. `load_jsonl` fails
closed on a missing or mismatched hash, and the audit/freeze manifest records
the complete ordered corpus checksum. `scripts/prepare_router_v2_study.py`
should be called with `--freeze-manifest` after the audit passes.

## Storage and privacy

Each run can persist normalized candidate rows in `routing_observations`.
Storage contains a normalized SHA-256 prompt hash, predictions, propensity,
realized tokens/cost/latency, and evaluator version. Raw prompts are absent by
default; embeddings are opt-in. Apply the Supabase migration
`20260824000000_routing_observations.sql` before enabling shared storage.
`auto` mirrors rows when the existing server-side Supabase URL and secret are
configured; local SQLite remains the request-path fallback during an outage.

## Promotion gates

- Shadow: at least 500 eligible decisions and 100 objectively labeled dual runs.
- Canary: 5% for at least seven days and 500 requests.
- Untouched-test ECE at most 0.05.
- No major slice loses more than three percentage points.
- Non-dominated under both free- and paid-Nemotron sensitivity prices.
- Incremental memory at most 120 MB and p95 routing latency at most 20 ms.

Dual execution and contextual-bandit updates remain deliberately disabled until
the budget ledger, objective evaluator, and inverse-propensity audit pass these
gates.

## Study collection safety

The paid study is a separate, explicit workflow. Prepare a locally supplied
JSONL corpus first; every row must carry a source, benchmark family, domain,
template family, collection vintage, licence status, pinned source revision,
provenance URL, lane, and executable grading metadata. The validator rejects
cross-split normalized-hash/leakage-group overlap and anything other than
exactly 1,200 train, 400 validation, and 400 test rows. Same-split repeated
visible inputs are accepted only for the pinned τ2
`router-input-group-mean-v1` contract: raw environment trials remain separate
for collection and grading, while training/evaluation aggregate outcomes by
visible normalized prompt and bootstrap by that effective group. Hidden state
is never appended to the router input. Gated or unapproved source metadata is
rejected rather than inferred.

```bash
python scripts/prepare_router_v2_study.py \
  --input data/router-v2/router-v2-candidates.jsonl \
  --output data/router-v2/study.jsonl \
  --audit data/router-v2/corpus-audit.json \
  --rejections data/router-v2/rejections.jsonl \
  --freeze-manifest reports/router-v2/router-v2-freeze-manifest.json
```

BFCL and τ2 tool/agent rows are eligible only when their pinned official
executor is locally invocable. BFCL receives task-specific schemas and
structured tool calls; τ2 runs an interactive agent/environment/state loop.
A grader name is not a grade: trusted-result envelopes must identify the
executor/version and include a SHA-256 evidence hash. Pairwise rows freeze
judge identity/version, criteria, sampling, and position randomization. Run
the provider-free fixture smoke test before any approval review:

```bash
python scripts/smoke_router_v2_graders.py \
  --output reports/router-v2/router-v2-grader-smoke.json
```

The collector command below writes the exact four-phase schedule but
materializes zero calls. It uses format-specific token/turn limits, the
offline price snapshot, reserves the $10 untouched-test budget, and binds the
schedule to a SHA-256 plan hash. Review the printed maximum before any provider
call:

```bash
python scripts/collect_router_v2_study.py \
  --corpus data/router-v2/study.jsonl \
  --plan reports/router-v2/router-v2-phased-plan-repeated-contract.json \
  --budget-ledger reports/router-v2/router-v2-budget-ledger-repeated-contract.json \
  --report reports/router-v2/router-v2-phased-plan-repeated-contract-report.json
```

Phase-specific preflight is a separate command and is also provider-free by
default. It validates every planned adapter call before any reservation, reads
the public OpenRouter model catalog (metadata only), and writes an immutable
single-phase executable plan. The executable hash covers the exact calls,
status, price snapshot, executor manifest, anchor boundary, corpus, parent
phase/phased-plan hashes, and metadata response hash:

```bash
python scripts/execute_router_v2_phase.py \
  --phase A_core_train \
  --corpus data/router-v2/study.jsonl \
  --plan reports/router-v2/router-v2-phased-plan-repeated-contract.json \
  --execution-plan reports/router-v2/router-v2-phase-A-executable-plan.json \
  --state reports/router-v2/router-v2-study-state-repeated-contract.json \
  --budget-ledger reports/router-v2/router-v2-budget-ledger-repeated-contract.json \
  --anchor-overlap-manifest reports/router-v2/router-v2-public-anchor-overlap-manifest.json \
  --executor-manifest reports/router-v2/router-v2-executor-environment-manifest.json \
  --report reports/router-v2/router-v2-phase-A-preflight-repeated-contract.json
```

The printed `execution_plan_hash`—not the locked parent phase hash—is the only
hash accepted by `--execute --reviewed-plan-hash HASH`; execution reloads that
file without materializing or changing its schedule and verifies it again just
before provider construction. Paid calls additionally require
`--allow-paid-calls`. Phase A has no test calls, Phase B cannot run before
complete Phase A outcomes, Phase C cannot run before A and B, and Phase D
remains locked until the complete pre-test configuration freeze. No command
silently substitutes a model, executor, or price.
Unknown pricing, an ambiguous provider failure, an unsettled reservation, or a
worst-case plan above $25 stops the run. The state file stores exact provider
model, token counts, realized cost, latency, finish reason, evaluator version,
retry status, output hash/preview, and deterministic grading status; it does not
store full model outputs by default.

`scripts/finalize_router_v2_study.py` refuses to emit trainable JSONL while any
required outcome is missing, truncated, pending tool state, pending a pairwise
judge, or errored. Deterministic evidence is authoritative when a later judge
disagrees.

Public anchors supplement the primary exact 2,000-prompt study; they do not
replace it. Start with
`docs/router-v2-public-anchors.md`, `scripts/audit_browsecomp_plus.py`, and
`scripts/import_browsecomp_plus.py`. Use the old collector plan only for a
deliberately approved legacy experiment; public anchor trajectories with no
correctness label must remain nullable and must not be counted as failures.

Verify the pinned i-DeepSearch import locally without network access:

```bash
python scripts/verify_observation_masking_import.py \
  --manifest reports/router-v2/observation-masking-manifest.json \
  --cache-dir /tmp/theriad-router-v2-observation-masking \
  --anchors data/router-v2/observation-masking-anchors.jsonl \
  --audit reports/router-v2/observation-masking-audit.json \
  --rejections reports/router-v2/observation-masking-rejections.jsonl
```

The command checks the pinned revision, all 113 file hashes, 1,322 prompts,
77,369 observations, 77,368 explicit outcomes, 9 models, 27 separate
protocols, and zero rejected rows. Those public outcomes remain protocol-
isolated anchors and are never treated as Theriad candidate-model outcomes.

## Research family artifacts

`scripts/train_router_v2_family.py` emits the same artifact schema for the
surface logistic baseline, an embedding kNN baseline, a shared-representation
adapter, and the hybrid surface-plus-embedding adapter. Embeddings are supplied
as a pinned local file; `scripts/encode_router_v2_embeddings.py` can create that
file for `all-MiniLM-L6-v2` or `gte-modernbert-base` only after an explicit model
download acknowledgement. The RouteLLM/RouterDC-named modes are research
adapters, not claims that a BERT or RouterDC reproduction exists without the
corresponding pinned encoder and training data. All embedding artifacts remain
ineligible for the normal request path until a measured runtime profile records
at most 120 MB incremental RSS and 20 ms p95 latency.
