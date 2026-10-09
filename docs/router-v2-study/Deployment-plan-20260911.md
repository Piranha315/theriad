# Router V2: implementation plan for engineering evidence and a reversible canary

Planning checkpoint: 2026-09-11; final consistency review: 2026-09-12. **This document proposes changes; none of them are implemented or authorized for paid execution or deployment.** Only this plan and the Roadmap were written. No commit, push, provider inference, production configuration change, or deployment was performed.

## Decision

Separate the engineering deployment decision from the existing confirmatory study. Start with an explicitly authorized, allowlisted internal/opt-in ordinary-text-chat pilot, preceded by a frozen paid engineering benchmark. Do not require a 95% bound above −1pp before that tightly bounded pilot. Require stronger evidence before broad rollout. A feature flag and fast rollback reduce exposure; they do not establish that answers are correct or undo answers already delivered.

Keep the strict research criterion unchanged: one-sided 95% quality-difference lower bound > −0.01 AND one-sided 95% cost-per-success-ratio upper bound < 0.90, with the existing dependence/family/inference guards. Neither passing an engineering gate nor a production rollout changes the research result.

The shortest path has two spending boundaries. There is no verified deployable fitted artifact from this study: its successful rehearsal used synthetic providers. First freeze a bounded development collection/training contract and authorize its spend. Then fit/calibrate/select the actual Router V2, freeze the resulting policy, and separately authorize a small engineering evaluation. Freezing a learned artifact before collecting its training outcomes is impossible here. An older fixed-model challenger or proxy artifact is not a substitute.

Do not expand the corpus for this path. Do not use the 125 research test groups. Use the existing development splits for fitting and the existing smoke split once for engineering acceptance. This sacrifices neither research-test exposure nor the five-family research design. Smoke evidence is limited and potentially benchmark-contaminated; it supports an internal pilot, not a claim about general production quality.

## Verified repository state and handoff

| Item | Location / observed state |
|---|---|
| Current product checkout | `/Users/puranvs/Cost-Optimization`, HEAD `fe14bdf518462cb056a77cf49780e5e779d1dced` |
| Repaired study implementation | `/Users/puranvs/.codex/worktrees/fd92/Cost-Optimization`, detached base `6aa36d5884213ad5819e0465bd3bb357dcae6393` |
| Remote | `https://github.com/Piranha315/Cost-Optimization.git` |
| Study checkpoint | `docs/router-v2-study/Roadmap.md` in both checkouts |
| Current research corpus | Study worktree: `data/router-episode-corpus-20260911-v9/` |
| Final review | Study worktree: `benchmark_results/router_corpus_extension_20260911/REVIEW.md` |
| Accepted rehearsal | `benchmark_results/router_corpus_extension_20260911/offline-infrastructure-rehearsal-v2/` |
| Interpreter used for inspection | `/Users/puranvs/Cost-Optimization/.venv/bin/python` |

Both checkouts are dirty. Much of the repaired study is untracked. The product checkout contains newer authentication, hosted storage, accounting, UI and agent changes. **Do not deploy the old study worktree wholesale or copy its server over main.** Recheck HEAD/status/diffs before implementation and integrate scoped changes onto the then-current product base, preserving unrelated changes. The base commit alone cannot restore the study implementation or corpus; retain the worktree and referenced source packages.

This inspection revalidated all **53 research-manifest file hashes**, with zero mismatches, and counted **1,201 rows /652 reviewed groups**, including the unchanged 1,150-row foundation and 51 additions. Test counts are chat21/coding20/search44/tools20/agents20. Smoke has **22 groups**: chat8/coding4/search8/tools1/agents1. All corpus collection flags remain false; no v9 confirmatory lock exists.

The 76-test pass and complete synthetic/real-sandbox rehearsal are retained historical evidence, not a fresh integrated-main test result. The power figures (8% and20% joint pass under two favorable scenarios) are simulation diagnostics, not estimated power from real provider outcomes. This planning pass did not inspect held-out provider outcomes or rerun paid collection. Deployed SHA, runtime flags, traffic volume, telemetry backend and on-call capacity were not inspected; repo defaults do not prove live production state.

## Current architecture and concrete gaps

Paths in the following tables are repository-relative. Study-specific entries refer to the repaired worktree; production entries refer to the current product checkout. Line numbers may drift in these dirty checkouts; function names are the authoritative anchors.

| Component | Existing implementation | Reuse / gap |
|---|---|---|
| Router | `theriad/routing/v2.py`: `RouterV2Artifact.from_dict`, `RouterV2.predict`, `resolve_from_environment`, `observation_rows` | Actual surface routing, model eligibility, uncertainty, OOD, off/shadow/canary/on flags and research-only rejection exist. Shadow computes predictions; it does not call a second provider or measure counterfactual quality. |
| Production policy | `theriad/models/policy.py`: `PolicyResolver.resolve` | Resolves legacy policy first, filters V2 candidates by constraints/context/circuits, can prepend legacy selection as fallback. No endpoint-scoped experiment context is supplied. A global mode can affect other callers. |
| Cohorts | Main `resolve_from_environment`; repaired worktree extends decision metadata | Main hashes router version + prompt + lane. Worktree can accept episode/experiment IDs, but `PolicyResolver.resolve` does not pass them. Neither gives the required stable authenticated-account production assignment end to end. |
| Artifact safety | `_load_router`, `_env_percentage`, `clear_router_cache` in `v2.py` | Cache is keyed by path. Bind the hash to the bytes actually loaded; reject nonfinite percentages and fail to legacy on load/hash/routing errors. Worktree reads artifact bytes again for a hash, outside the prediction exception boundary. |
| OOD/fallback | `RouterV2.predict`; runner `select_challenger` | OOD selects the artifact's configured fallback. The trainer wrapper sets both lane fallbacks to the first sorted alternative, not the real legacy policy. Deploy through a shared admission/fallback wrapper that uses the already-resolved legacy route on OOD or no candidate clearing the knee. Evaluate that same wrapper in the engineering benchmark. |
| Real UI route | `theriad-ui/src/Chat.jsx` calls `/api/chat/send`; `theriad/api/server.py`: `chat_send` | This is the initial canary target. Chat does pass `observation_rows` to `save_run`, and aggregates attempt cost/tokens. Its prediction observation currently receives final-response cost/tokens, not total request cost. `/api/run` also has observations. Do not conflate either path with the study adapter or assume other endpoints are covered. |
| Usage | `theriad/agents/model_telemetry.py`: `complete_with_telemetry`, `record_model_usage`, `usage_totals`; `server.py`: `_record_routing_usage`; `theriad/billing/service.py` | Reuse actual attempt collection and truthful nullable costs. Preserve newer main accounting. Empty/missing attempt data is not evidence of a free request; unknown failed-attempt costs must remain unknown. |
| Persistence | `theriad/api/projects.py`: `save_run`, `_save_routing_observations`; `theriad/storage/database.py`; `theriad/routing/observation_store.py`: `mirror_to_supabase` | Local success-path observations exist. **Hosted `save_run` returns immediately after inserting a run and ignores routing observations.** Existing mirror is best effort and generates new IDs. Cohort fields added in the worktree are not columns in main's schemas. Neither per-candidate rows nor success-only run records are a complete request denominator. |
| Hosted data | `theriad/storage/hosted.py`: `HostedRepository`, `RunRepository`; `supabase/migrations/20260824000000_routing_observations.sql` | Extend the real hosted path and RLS. Do not assume a local SQLite mirror is authoritative for hosted users. |
| Quality | `theriad/quality/objective_verifier.py`: `verify_objective_response`; `theriad/quality/loop_engine.py`: `run_quality_loop` | JSON/schema and citation-presence checks are narrow signals. HTTP success, valid JSON, a link, or model confidence is not general answer correctness. No adequate general-chat canary-quality gate is wired. |
| Study runner | `scripts/run_router_episode_study.py`: `prepare`, `training_readiness`, `run_study`, `select_challenger`, `render_report` | Existing candidate grid, actual training, staged collection and reporting. `--alternatives` does not currently restrict the effective grid: prepare installs the registry pool. Selection uses a hard-coded0.90 ceiling in both trainer arguments and outer feasibility check. `stop_after='selection'` still loads and freezes test before returning. |
| Training | `theriad/routing/v2_training.py`: `StudyRow`, `aggregate_router_input_groups`, `train_surface_artifact` | Separate train/calibration/threshold sets; effective router-input groups; current support floor26train/10calibration per model. Threshold coverage must support both lanes. Keep these guards. Runner recognizes only two older authored-source names; new `newly_authored_20260911` needs explicit provenance mapping to its real source manifest, not invented license metadata. |
| Corpus admission | `theriad/routing/episode_fixtures.py`: `require_collection_ready`; semantic census/extension verifiers | v9 deliberately fails paid admission. Add an explicit, narrow engineering-use admission record; never toggle v9 flags or add a generic quarantine bypass. Exclusions, graders, identities and exposure constraints remain binding. |
| Execution | `theriad/routing/episode_runtime.py`: `freeze_baseline`, `EpisodeRuntime`, `models_for`, `_surface_resolver`, existing graders | Reuse adapters and actual predictions. Verify that the evaluated routing prompt, candidate pool and fallback semantics equal the eligible production path; ordinary chat is narrower than these five-family adapters. |
| Locking | `theriad/routing/episode_protocol.py`: `capture_protocol`, `verify_protocol`; `episode_study.py`: `freeze_test`, `verify_lock`, `execute_pair` | Source/environment snapshot, exclusive writes, hashes, paired schedule, durable execution and tamper guards exist. `freeze_test` explicitly accepts only test rows and embeds the research decision. Extract shared internals while retaining that strict public wrapper; do not relabel smoke as test. |
| Budget | `theriad/routing/episode_planning.py`: `propose_episode_budget`; `episode_study.py`: `ProductStudyBudget`; `study_budget.py` | Conservative per-call token/model-chain bounds, staged reservations and unresolved-charge retention exist. Extend the exact schedule/profile inputs, not the ledger architecture. |
| Rollback/deploy | Worktree `v2.py`: `canary_rollback_required`; `scripts/deploy.sh` | Helper checks serious incidents or two mature100-per-arm windows with failure/latency regressions; no production caller was found. Deploy script has SHA rollback and health probes. Its dry-run is not a purely read-only plan command; do not run it during planning. |

## Exact recommended gates

These are proposed risk-management defaults to freeze before outcomes, not guarantees of a particular population error rate. They deliberately distinguish permission for a small experiment from evidence for wide deployment.

### Engineering benchmark → internal/opt-in pilot

Return `deployable_with_caution`, scope `allowlisted_text_chat_only`, only when all conditions hold:

1. Integrated product route, artifact validation, hosted telemetry, cost accounting, tenant isolation, fallback and kill-switch tests pass; rollback drill succeeds. All mandatory evidence artifacts verify. An identified operator and quality reviewer are available.
2. One genuine surface Router V2 was fitted on admitted train groups, calibrated only on calibration, and selected only on threshold-selection. No fixed-policy substitution, smoke tuning or research-test use. Freeze the production fallback/admission wrapper with the artifact.
3. Evaluate all22 smoke groups, one preselected representative per group, baseline and frozen V2 wrapper, in the locked randomized order. Preserve all original split/group IDs. Every scheduled pair is complete with a valid grader and settled known cost. No unresolved attempt/reservation or missing outcome is silently dropped.
4. On the eight chat smoke groups: no baseline-success/V2-failure group; at least one success for each policy; V2 cost per successful task ≤0.95×baseline. Evaluate cost including all bounded attempts, retries and fallbacks. The zero-loss condition is deliberately conservative for this small engineering screen; it is **not** 95% non-inferiority.
5. Report all other families and all operational/critical failures. A critical shared-runtime safety failure blocks admission. Ordinary task-quality losses in an excluded family do not grant that family deployment eligibility and do not get erased from the report.
6. Explicit user authorization accepts the limited evidence and the internal pilot's exposure/cost cap. The offline result alone never enables a public1% cohort.

Status precedence: invalid/incomplete evidence → `insufficient_data`; valid evidence with an in-scope quality or safety failure → `not_deployable_quality`; otherwise no positive practical savings → `not_deployable_cost`; all conditions above → `deployable_with_caution`. Also emit every failed predicate, counts and uncertainty, so the single status cannot hide an adverse finding. `deployable` is reserved for the broader production gate below, never this tiny smoke set.

A zero-discordance eight-group result has substantial uncertainty and benchmark exposure limitations. Never output a degenerate all-ties bootstrap interval as proof of zero risk. Show counts and mark uncertainty uninformative where appropriate. The public-reference chat tasks are not representative of ordinary customer conversations; production quality review is essential.

### Production expansion

Before the first internal request, freeze the rollout profile, rubric, assignment unit, evaluation windows, minimum counts, cost definition and guardrails. Proposed stages:

| Stage | Exposure and hard cap | Gate to enter next stage |
|---|---|---|
| Instrumentation / prediction shadow | 0% changed answers; enable V2 prediction only after artifact admission | A no-paid integration rehearsal and an authorized production telemetry/rollback check. Shadow is not quality evidence. |
| Internal/opt-in | Only explicitly enrolled accounts; 50:50 assignment within this separate allowlist experiment; ≤100 treatment requests and ≤100 evaluated controls, ≤7days, reviewer covers all. Normal production remains baseline. | At the fixed review:100 mature labeled requests per arm, ≥20 assignment units per arm, no critical incident, quality point difference≥0, CPS ratio≤0.95, operational guardrails pass and complete costs/labels. If the enrollment cannot support the cluster floor, stop at internal status rather than loosening it. |
| Public1% of eligible accounts | Separate public experiment; ≤100 treatment requests total in7days; controls contemporaneous; review every treatment and a prespecified random control sample | Explicit deployment approval; internal gate above. End-of-stage promotion needs100 labeled requests per arm, ≥20 units per arm, the same point gates and operational checks. No formal−1pp claim. |
| Public5% | ≤500 treatment requests total in14days; same narrow eligibility; fixed artifact and enrollment rules | Enter only after public1% passes. To consider broader exposure:≥500 labeled requests per arm, ≥50 units per arm, point gates pass, and one-sided95% cluster-bootstrap quality lower bound >−0.01 on the fixed-stage sample. Cost point target remains5%; do not require its entire CI below0.90. An uninformative interval is insufficient. |
| 25% →50% →100% of this eligible chat scope | Separate approved stages; retain contemporaneous control through50%; each stage gets an explicit dollar/request cap before enabling | Each promotion requires a fresh fixed-stage report with at least the same counts, the−1pp quality bound, point CPS ratio≤0.95, complete telemetry, and no guardrail breach. Failure to obtain precision pauses promotion. Keep100% out of automatic progression. |

The counts and caps above are operational minimums, not power calculations. At low traffic, percentage ladders are slow: use internal enrollment to debug, then stop safely if evidence is insufficient. Do not add10% merely to fill out a ladder. If the proposal's100/500-request caps prevent sufficient independent users, report that fact; changing a cap requires prospective approval and a new fixed review window. No extension based on watching a nearly passing confidence interval.

For public stages, control requests need not be capped operationally; randomly select their audit sample independently of outcomes and record its sampling probability. Wait for all prescheduled labels before promotion. Use within-arm assignment-unit cluster resampling for randomized production, **not paired-task or five-family research resampling**. Aggregate each unit’s request/cost/success totals, resample whole units with replacement independently within each arm (4,000 draws, seed731), and recompute the ratio-of-totals estimands in every draw. The quality lower bound is the fifth percentile of treatment-minus-control success-rate draws. Preserve within-unit dependence and the prespecified audit weights; do not average per-user CPS ratios or resample candidate observations. Undefined or degenerate draws cannot establish non-inferiority. Add explicit estimator tests with repeated requests and unequal cluster sizes. Confidence intervals from successive stages are operational diagnostics; do not advertise a sequentially controlled confirmatory claim. No automatic promotion is allowed, even if the monitor reports green. A later externally claimed fixed-horizon production study must specify its analysis before collection.

## Minimal implementation, in execution order

### 1. Integrate the repaired infrastructure into the current product base

Files: scoped diff of `scripts/run_router_episode_study.py`, `scripts/rehearse_router_episode_study.py`, the `theriad/routing/episode_*.py`, relevant `v2*.py`, `study_budget.py`, sandbox and study hooks, and their tests. Incorporate only necessary call-site changes in `theriad/agents/model_telemetry.py`, planner/runtime files and `theriad/models/policy.py`. Retain reviewed corpus/evidence as immutable inputs and preserve referenced v4/v6/source packages.

Create a file inventory with source hashes and an integration checklist before changing code. Compare both working copies, not just commits. Carry the newer main chat accounting, auth and hosted behavior forward. A future integration branch may use `codex/` after implementation is authorized; no branch/commit/push is authorized by this planning request.

Tests: existing study guards/rehearsal plus main `tests/unit/api/test_chat.py`, `test_fallback_accounting.py`, `test_hosted_auth_boundary.py`, `tests/unit/agents/test_model_telemetry.py`. Confirm off-mode product behavior is unchanged and no study context leaks into normal requests.

### 2. Add one engineering profile to the existing runner, with bounded sampling and admission

Files: `scripts/run_router_episode_study.py` (`prepare`, `run_study`, `select_challenger`, CLI), `theriad/routing/episode_fixtures.py`, `episode_planning.py`; provenance adapter adjacent to the current `StudyRow` construction.

Add a strict enum profile `research` (default, unchanged) / `engineering_canary`. Engineering preparation loads only pilot/train/calibration/threshold-selection/smoke rows. It may validate the full research manifest by hashes, but must not load research-test outcomes or create a test lock. Make the after-fit return occur before all test loading, power estimation and locking.

Freeze an engineering admission record with exact corpus/review hashes, explicit retained row IDs, original stages/groups, intended use, acknowledged same-author/representativeness limits and authorization reference. Accept only retained rows covered by the completed review and validated graders. Deny prior-exposed rows in smoke/later stages, excluded candidates, missing provenance and unknown review states. Keep v9 `collection_ready=false` and the research `require_collection_ready` behavior intact. This is a documented use-specific engineering exception, never `--ignore-quarantine` or a fabricated independent-review signoff. Record all subsequent real-provider exposure in an append-only ledger.

Implement `--alternatives` as the exact nonempty registry subset, while cataloging every additional baseline/fallback model needed for cost bounds. Start with two distinct currently available, appropriately capable paid candidate models. Select and document them from registry/provider metadata before any outcomes; do not choose from smoke results or stale proxy rankings. Missing live identifiers/prices means preparation fails. Do not silently evaluate the whole registry. No automatic model search after a disappointing run.

Recommended bounded initial group budget, using one representative per reviewed group and an outcome-blind seed731 hash order within each family/split:

| Split | Chat | Coding | Search | Tools | Agents | Total |
|---|---:|---:|---:|---:|---:|---:|
| Pilot | 1 | 1 | 1 | 1 | 1 | 5 |
| Train | 20 | 18 | 18 | 2 | 2 | 60 |
| Calibration | 10 | 9 | 9 | 1 | 1 | 30 |
| Threshold selection | 12 | 12 | 13 | 1 | 2 | 40 |
| Smoke | 8 | 4 | 8 | 1 | 1 | 22 |

These are economical fitting/screening allocations, not new research quotas or evidence of sufficient prediction quality. Run `aggregate_router_input_groups` and `training_readiness` on the selected manifest: both intended model heads must meet the current26/10 floors after all aggregation/eligibility filtering, and both threshold lanes must be scoreable. Reject unsupported fitting; do not relax floors or fill gaps from smoke/test. If a different outcome-blind development selection is needed, settle it before authorization. Further fitting runs or larger grids require a new budget/contract, while the consumed smoke set becomes diagnostic-only for any retuned policy.

Use `selection_cost_ratio=0.95` only in this engineering profile, sourced once from its contract in both trainer and outer feasibility calculation. Preserve0.90 for research, actual surface fitting, and fixed-policy rejection. Map authored-source provenance to the actual pinned generator/spec manifest; never grant all unfamiliar sources authored status.

Tests: extend `test_episode_cli.py`, `test_episode_planning.py`, `test_episode_selection_guards.py`, `test_episode_fixtures.py`; monkeypatch test-row loading and credential loading to raise in the no-paid/after-fit paths; exact candidate/schedule cardinality; excluded-row rejection; source-prefix spoof rejection; original research defaults unchanged.

### 3. Reuse protocol and paired execution for a smoke engineering lock

Files: `theriad/routing/episode_study.py` (`freeze_test`, `verify_lock`, `execute_pair`), `episode_protocol.py`, `scripts/run_router_episode_study.py`.

Factor the existing schedule/hash/exclusive-write/verification internals into shared helpers. Keep `freeze_test` and research verification strict and backwards compatible. Add an engineering wrapper requiring `split='smoke'`, `evidence_kind='engineering_benchmark'`, exact selected groups, profile and admission hashes, and a distinct `engineering-lock.json`. Never rename smoke fixtures or give an engineering lock a research-test version/acceptance field.

Bind baseline plans plus baseline code/config, artifact bytes, deployment admission/fallback policy, candidate/provider identifiers, provider routing options, exact prices, corpus/grader/source hashes, features, knees, call/token/time limits, retries, schedule seed, selected split identities, expected input serialization and evaluation policy. Snapshot source/environment before any paid phase and verify on every run/resume. The main runner is already included in `capture_protocol`; explicitly include any additional executable script introduced by this plan. No credentials or environment values in snapshots.

The paid development phase gets a frozen contract/protocol before its first call; the engineering phase gets a new lock referencing that development contract and the learned artifact. Export the inner `surface-candidate.json` only through an explicit admission command: the `challenger.json` family-wrapper envelope is not directly a production `RouterV2Artifact`. Preserve the research-only source artifact, write a separately hashed deployment copy, and compare all model/feature/calibration/threshold fields for identity. Bind deployment eligibility to that exact exported hash. This is an auditable authorization boundary, not flipping a flag in place.

Tests: extend `test_episode_protocol.py`, `test_episode_study.py`, `test_episode_rehearsal.py`; both lock kinds reject changed source, artifact, sample, grader, baseline, candidate pool, prices, eligibility or bounds. Research rejects smoke; engineering rejects test; interrupted/resumed pairs are idempotent; no paid call can precede admission and budget approval.

### 4. Add an engineering report without changing confirmatory inference

Files: `theriad/routing/episode_study.py` for reusable validated group metrics; one small new `theriad/routing/deployment.py` with proposed `evaluate_engineering_gate`, `deployment_choice`, `assign_cohort`, and `evaluate_canary_window` pure functions; runner `render_report` profile dispatch.

Preserve `paired_report` and its research decision defaults. Share identity/completeness/group calculations, not the research pass bit. Produce `engineering-report.json/.md` with an explicit evidence kind, scope, gate version, pass/fail predicates, source/lock hashes, group/row counts, per-family success, win/loss/tie counts, total and per-success cost, latency, retries/fallbacks, selected-model distribution, missing labels/costs and limitations. Equal-family aggregate and chat-only gate results must be separate. Multiple rows in a group never increase independent n.

CPS is group-weighted mean all-attempt cost divided by group-weighted success probability, with the frozen family weights when aggregating five families. Zero success makes CPS undefined and blocks the economic gate; zero-priced observed calls require actual evidence. Report uncertainty descriptively; do not force the research20-group minimum onto the eight-chat engineering screen or imply precision it lacks.

Tests: new `tests/unit/routing/test_deployment.py` covering gate truth table, undefined/unknown cost, missing/duplicate pairs, success ties, treatment fallbacks retained under treatment, research-vs-engineering label isolation and family masking prevention.

### 5. Scope runtime admission and stable assignment to the real chat endpoint

Files: `theriad/routing/deployment.py`, `v2.py`, `theriad/models/policy.py`, `theriad/api/server.py` (`chat_send`), `.env.example` documentation only until separately authorized deployment.

Add an optional explicit rollout context to `PolicyResolver.resolve` and `resolve_from_environment`. No context means baseline/off for deployment mode, including preview, `/api/run`, agent calls and API compatibility endpoints. Compute context server-side before any dispatch, using the authenticated principal, actual endpoint, request ID and frozen experiment config. Client input cannot select its arm or forge eligibility.

Initial eligibility: explicitly enrolled accounts at internal stage; later authenticated ordinary text chat using automatic Theriad model selection, no explicit model/provider override, no free-only path, no attachments/vision, retrieval/search/browser/tools/agents, or extra quality-loop/model calls. If needed, narrow to a named ordinary-chat tier so the candidate pool is predictable. Do not change excluded requests' existing behavior to make them eligible. Sensitive/high-impact exclusions are an additional screen, not a reliable safety classifier; internal enrollment must acknowledge this limitation. Preserve existing privacy, memory permissions and billing preferences.

Choose organization ID as assignment unit where accounts share organization state; otherwise authenticated user ID. Use `HMAC(server experiment secret, experiment_id + unit_id)` to derive a uniform stable bucket. The secret is an opaque secret-manager reference, never an artifact/log value. Keep assignment stable across conversations, retries, app processes and artifact loads. Changing an artifact or baseline starts a new approved experiment version; do not silently reuse its evidence. Log arm probability at decision time. Separate the internal and public experiment IDs, and exclude previously treated internal accounts from public control enrollment. Within a public experiment use nested buckets as exposure rises so a treated unit never becomes control. Preserve conversation treatment history; start newly exposed measurement conversations prospectively and label carryover limitations. At percentage changes record stage/version; analyze stages separately. Record eligible assignment-unit counts before outcomes and check the observed allocation against its configured probability; an unexplained sample-ratio mismatch or systematic missing assignment IDs blocks promotion. Do not expect request counts to match account-assignment percentages when activity differs.

Compute and preserve the exact legacy resolution first. Only apply V2 when endpoint eligibility, candidate constraints, capability/context limits, circuit status and artifact admission pass. Use the legacy resolution for OOD, no candidate above the knee, invalid/stale artifact, guardrail stop or denied capacity. Keep treatment assignment in ITT telemetry even when V2 falls back. The engineering runtime must invoke the same pure admission/fallback function; otherwise its benchmark is for a different policy. Do not disable baseline fallback or add unbounded retries. Treat capability/model/price drift as a pause condition.

Tests: `test_v2.py`, `test_chat.py` and a new focused deployment-context test file. Validate cohort stability/cross-worker consistency; no prompt/artifact-version hashing; no caller-forged arm; all excluded endpoints baseline;0% off and100% explicit behavior; finite percent parsing; immutable-byte cache/hash match; missing/corrupt artifact baseline; OOD and no-qualified candidate legacy; exact evaluated-vs-served decision parity.

### 6. Persist request-level experiment outcomes through both storage paths

Files: `theriad/api/server.py`, `theriad/api/projects.py`, `theriad/storage/database.py`, `theriad/storage/hosted.py`, `theriad/routing/observation_store.py`, `theriad/agents/model_telemetry.py`; new additive migration `supabase/migrations/20260912000000_router_v2_canary.sql` (choose a later unused timestamp if occupied).

Use one small `router_experiment_requests` table in both backends, keyed by request ID, with indexed experiment/stage/unit/arm/time fields and bounded JSON for the existing attempt summaries. This is justified because existing runs require success-path entities and observations are per candidate. Do not repurpose candidate rows as requests or broaden `runs` into a new general event system. Add service-owned insert/update helpers in the existing observation store; reuse hosted ownership/RLS conventions. Add cohort/artifact fields to prediction-observation schemas and fix hosted `save_run` observation handling with idempotent IDs. No new analytics service or queue is needed.

Write a durable admitted/assigned record before dispatch for both arms, then idempotently finalize it on success, error, cancellation or timeout. A collector sweeps expired pending requests into an explicit unresolved state; it must not infer cost zero. Request record includes:

- experiment/stage/config/artifact/baseline hashes; pseudonymous assignment unit, arm, assignment probability, eligibility/family/tier and reason;
- request/decision/run correlation IDs; proposed, applied and actual model/provider; baseline recommendation; OOD/admission/fallback reason;
- start/end/timeouts, routing/provider/total latency, terminal status and bounded error class;
- every attempted provider/model, token/usage availability, known/unknown cost, retries/fallbacks, plus request totals;
- quality-label status, pass/fail, evaluator/rubric version, audit sampling probability, maturity and missingness.

No raw prompts, outputs, embeddings, secrets or stack traces in these new events. For assignment/analysis use experiment-keyed pseudonyms; unsalted prompt hashes are neither assignment units nor anonymization. Authorized reviewers access existing scoped conversation content without duplicating it into telemetry. Permit only trusted server/reviewer label writes; model self-reports and client success flags are never labels.

Separate operational completion from task-quality success. A failed delivered request counts as operational failure and unsuccessful for an all-request task-success estimand. General chat needs an arm-blind human rubric: correct/relevant, follows requested constraints, sufficiently complete, no material fabricated claims or unsafe content; overall pass requires every applicable essential criterion. Store criterion decisions and adjudication status. Reviewers must not see model/arm/cost before labeling. Objective checks are supplemental or define a clearly named narrow subset, not substitutes for that rubric.

For internal traffic label every included request. Public audits use a frozen outcome-independent random sample within each arm, with known inclusion probability; cost includes all eligible assigned requests and quality estimates use representative sampled labels. Report missingness and sampling weights. Until that implementation is tested, keep public analysis on the completely audited capped sample and label its scope; never combine all-traffic cost with selectively reviewed success labels without a valid sampling estimator.

Preserve `actual_cost_usd=null` when any attempt is unknown, retain known subtotal separately and never display a missing charge as$0. Include V2-to-baseline fallback costs in treatment. Do not charge prediction rows as additional calls. A telemetry write failure returns a delivered answer normally if already obtained but disables subsequent treatment admission. Do not turn a telemetry outage into repeated provider calls.

Tests: `test_observation_store.py`, `tests/unit/storage/test_routing_observations.py`, `test_hosted_storage.py`, `tests/unit/api/test_real_observation.py`, `test_fallback_accounting.py`, `test_hosted_auth_boundary.py`, `test_model_telemetry.py`. Cover hosted early return, pre-dispatch failure, cancellation, partial paid failures, mirror outage, idempotent retries, unknown costs, cross-tenant denial, missing labels and absence of prompt/secret data. Verify a whole mocked `/api/chat/send` request writes the assigned cohort and every attempt into the authoritative hosted path.

### 7. Wire a small fail-closed canary controller and report command

Files: `theriad/routing/deployment.py`, worktree `v2.py` rollback helper (retain its existing semantics), existing observation store, one new `scripts/check_router_v2_canary.py`, `.env.example`, deployment runbook next to this plan. Include this script in protocol source snapshots.

Use one service-owned experiment control record alongside the request table: approved immutable config hash, active stage, remaining request/dollar allowance, enabled bit, stop reason and monitor heartbeat. Atomically reserve treatment capacity before dispatch across workers; backend failure means baseline. Implement this as one transaction/RPC in the additive Supabase migration (service-only execution, check enabled/config hash/heartbeat/request and dollar allowance, insert request ID and reserve its worst-case charge), and a matching SQLite transaction using the same invariants. Use request ID for idempotence. Finalization settles known cost once; unknown cost keeps its reservation. Never emulate this with separate HTTP read/decrement calls or a process-local lock. Include controller bootstrap with a short initial lease only after all startup checks; thereafter the monitor must renew it. Do not rely on process-local counters or environment hot-reload. A monitor started by the existing service manager polls once/minute, reads matured telemetry, writes heartbeat/stop state and never promotes. A stale heartbeat (>120seconds), disabled/missing control record, bad config hash or exhausted cap denies treatment on the next request. An authenticated operator can set disabled via an administrative CLI; do not add a public API.

Hard stops: serious privacy/safety incident; invalid routing/capability/authorization; unknown artifact; hard budget/request cap; telemetry outage/staleness; any unexplained duplicate paid dispatch. Do not interrupt an already delivered answer or silently retry it. Two treatment terminal failures in the latest10treatment requests pause for review. One confirmed materially harmful quality failure in an in-scope treatment answer pauses immediately. These are conservative operational circuit breakers, not statistically attributed regressions.

After counts permit, also use the existing rollback helper: two consecutive nonoverlapping windows, each with≥100matured requests per arm, with treatment operational-failure rate >baseline+2pp OR p95latency >1.20×baseline. Freeze24-hour windows; insufficient counts leave the comparative check unavailable, not green. Apply the request/time caps even when windows cannot mature. Any missing labels beyond the24-hour review SLA pauses further treatment pending review. No automatic recovery after a stop.

Maintain the existing environment off switch as an independent emergency stop requiring service restart. `scripts/deploy.sh` remains the code rollback mechanism. A canary rollback normally changes only treatment admission, leaving telemetry and legacy routing active. Keep the prior executable/artifact available. Drill both control-record stop and env-off restart on the actual deployment topology before public exposure; require next-request stop for the former and a documented≤60-second operational rollback for the latter. If this cannot be demonstrated, remain internal/off.

Tests: new `test_canary_controller.py`, including concurrent reservations, stale monitor, unavailable backend, all guardrails, insufficient windows, false auto-recovery and no automatic promotion; no-paid ASGI endpoint drill plus hosted-store integration. Reporting must retain randomized assigned arms, stage boundaries and fallback traffic.

## Benchmark execution procedure and cost authorization

The following command names/options are **interfaces to implement**, not commands that exist today:

1. `prepare --profile engineering_canary`: read verified immutable inputs, select the frozen group subset and exact two-model pool, build baseline plans and eligibility, capture source/environment, write admission proposal and development contract. Refresh the public provider catalog without inference; provide an offline catalog fixture for rehearsal. Do not load provider credentials in offline preparation.
2. `rehearse --profile engineering_canary`: network-denied fake providers, real training/calibration/threshold/predict path and real code sandbox. Exercise both success and rejection, source tamper, budget exhaustion, unknown charges, interrupted resume, both lock types and the real mocked chat endpoint. The existing canonical research rehearsal must still pass. No substitution of fixed-model output for actual V2 routing.
3. `preview --phase development`: emit exact row/group/model/configuration counts, token/call bounds, stage reservations, contingency and total hard cap using `propose_episode_budget` and the actual planned schedule. With the recommended complete two-candidate grid:5pilot×2 +130development×3 = **400 policy trajectories** before engineering evaluation. This is not400provider calls; agent/tool/retry bounds can multiply calls. If eligibility differs, print the actual smaller schedule and verify fitting support.
4. **STOP A:** report conservative maximum and an expected-spend range only if grounded in measured comparable usage. Otherwise expected spend is explicitly unavailable. Show known incurred, unresolved reserved and remaining separately. Ask the user to authorize the exact development contract hash, model pool, stage caps and total dollars. An earlier historical$100 preference is not authorization. Do not copy stale prices or invent a dollar estimate in this plan.
5. After explicit authorization, run the bounded pilot and development collection through `ProductStudyBudget` and existing durable events. Every network preflight that can incur cost is included and authorized. Reserve worst-case per-attempt chains/token limits before dispatch; stop on unresolved/over-cap state. Fit/calibrate/select only the designated splits. Stop immediately after fitting; no test lock and no smoke inspection for tuning.
6. Write the inner candidate and approved deployment-wrapper configuration; freeze an `engineering-lock.json` for22smoke pairs. Preview Phase B: **44 policy trajectories**, with exact wrapper fallback/model limits and current validated pricing. If source/config/prices changed, create a new frozen phase and disclose the change; never amend an already executed lock. An outcome-driven policy change needs a new independent engineering evaluation allocation, not repeated smoke acceptance.
7. **STOP B:** show the frozen artifact/baseline/lock hashes, Phase B maximum, cumulative Phase A actual+reserved and total possible spend. Obtain explicit authorization for Phase B. A single earlier authorization may cover both only if it explicitly named both phases and their conservative combined caps before spending; this plan recommends separate approvals.
8. Run randomized paired smoke evaluation once, settle/reconcile all attempts, validate evidence, and emit the engineering decision. Maximum default schedule across both phases is **444 trajectories**, before any separately approved rerun, not a promised dollar budget. Any paid judge is out of scope; existing deterministic graders cost no provider inference.
9. **STOP C:** present the engineering report, limitation statement, deployment artifact, precise internal eligibility/assignment rules, exposure/dollar cap, monitor/rollback drill and operator/reviewer ownership. Ask for deployment authorization. Only then deploy with behavior initially off, verify live SHA/config/backend/health, then activate the authorized internal stage. No public cohort follows automatically.

A dollar cap protects against expense only if provider calls, retries, input/output limits and pricing charges are actually bounded. Existing planning rejects unbounded per-request/image charges; retain that behavior. Unknown completed charges retain full reservations. Exclude unsupported charge types/providers until bounded, rather than claiming an exact ceiling from token prices alone.

## No-paid acceptance checklist for the coding model

Run focused tests after each change; once integrated, run the following relevant suite using the project's interpreter and `PYTHONPATH=.`. Extend the named existing files and add the proposed tests above:

- `tests/unit/routing/test_v2.py`, `test_v2_training.py`, `test_episode_cli.py`, `test_episode_protocol.py`, `test_episode_planning.py`, `test_episode_selection_guards.py`, `test_episode_study.py`, `test_episode_runtime.py`, `test_episode_rehearsal.py`, `test_study_budget.py`, `test_authored_episode_extension.py`;
- `tests/unit/routing/test_observation_store.py`, new deployment/controller tests;
- `tests/unit/api/test_chat.py`, `test_real_observation.py`, `test_fallback_accounting.py`, `test_hosted_auth_boundary.py`, plus storage and model-telemetry tests identified above.

Run `scripts/rehearse_router_episode_study.py` in a new output directory with `--real-code-sandbox`, and the new engineering-profile rehearsal. Deny provider/network dispatch for all no-paid acceptance runs and demonstrate the deny hook was active. Verify the locked research manifest again without regenerating its contents. Run additive SQLite and disposable local/test Supabase migrations, including ownership/RLS and failure cases. Local mocks alone are not hosted-schema acceptance. If a disposable hosted backend is unavailable, mark that test blocked and leave production canary admission disabled.

Verify actual UI→`/api/chat/send` request behavior with mocked provider responses, including failed first attempt, legacy fallback and complete telemetry. Later, the explicitly authorized live pilot must verify real providers and the deployed route; neither a unit test nor the study adapter proves that end to end.

## Evidence and external claim policy

| Evidence obtained | Allowed statement | Not supported |
|---|---|---|
| Current synthetic rehearsal | The infrastructure completed a no-paid rehearsal with actual router training/routing and local sandbox checks. | Any model-quality or savings result; production readiness. |
| Frozen engineering smoke benchmark | On22smoke groups from this reviewed public/authored corpus, the frozen policies had stated success/cost results; report all family counts, exposure limits and eight-chat scope. | Production savings, benchmark-pretraining independence, general−1pp NI, or the strict research acceptance. |
| Internal/initial public canary | Descriptive randomized results for the named eligible cohort, dates, request and assignment-unit counts, audit coverage, costs and uncertainty. Internal opt-in population is identified explicitly. | All users/all workloads, semantic quality from HTTP status, causal claims from shadow predictions or unrandomized before/after comparisons. |
| Larger fixed production sample | “Across N eligible requests from K randomized assignment units, during [dates], inference cost per [defined successful task] changed X%, while audited task success changed Ypp.” Include frozen versions, arm sizes, audit method, clustered uncertainty, missingness and excluded scope. | A global claim while agents/tools/search remain excluded; pooling stage versions opportunistically; sequentially valid inference from repeatedly inspected ordinary intervals. |
| Valid confirmatory study | The prespecified five-family benchmark met/did not meet its original joint criterion, with confidence bounds and exact frozen scope. | Production representativeness or deployment success merely because the benchmark passed. |

Always distinguish observed task-success rate from satisfaction, factuality, operational completion and proxy format checks. Report absolute costs as well as ratios, baseline identity and model-selection distribution. Inference cost excludes infrastructure and human-review labor unless separately measured. Do not call a tiny or inconclusive result “quality preserved”; state observed differences and what uncertainty still permits. Production100% loses a contemporaneous control, so it is monitoring, not a continuing randomized benchmark without a separately retained holdback.

## Statistical rationale checked against primary sources

Treating requests inside one randomized user/tenant as independent can understate uncertainty; tenant randomization also reduces effective sample size. Use the smallest assignment unit that avoids shared-state interference, and report assignment-unit counts. These cautions are supported by [Microsoft Experimentation Platform’s tenant-randomization analysis](https://www.microsoft.com/en-us/research/articles/why-tenant-randomized-a-b-test-is-challenging-and-tenant-pairing-may-not-work/). They do not establish that this plan’s proposed operational counts are statistically powered.

Repeated interim evaluation requires methods that account for peeking if formal error-rate guarantees are claimed. This plan therefore allows early stopping for harm but forbids repeated ordinary-interval checks for a confirmatory success claim; see [Microsoft’s during-experiment guidance](https://www.microsoft.com/en-us/research/articles/patterns-of-trustworthy-experimentation-during-experiment-stage/). The canary thresholds and caps are engineering recommendations, not thresholds endorsed by those sources.

## Non-goals and authorization boundaries

Postpone corpus expansion, independent external semantic review, representative tools/agents sampling and powered confirmatory design to the research track. Do not rewrite their review status. Do not build native browser/agent simulators, train embeddings, download models, add paid LLM judges, run a registry-wide search, build a dashboard/experimentation service, change pricing/billing product semantics or auto-promote to100% for this first canary. Do not promise investor-ready efficacy numbers before collecting valid evidence.

Implementation of this plan is a subsequent user instruction. Explicit authorization is still needed for each paid scope, deployment/config changes, internal/public exposure and caps, expanded spending, promotion and any later commit/push. Ordinary reversible code/test work can proceed once the user instructs implementation; it should reach a concrete reviewable state before asking for the final spending/deployment approval. Do not ask redundant approvals for already authorized work.

Immediate blockers are integration, the engineering admission/phase boundary, a fitted real artifact, complete hosted telemetry and a working controller/reviewer workflow. Independent corpus review and high confirmatory power are not prerequisites for the limited internal pilot proposed here. Missing quality labels, unknown costs, absent rollback or unverified tenant isolation remain blockers regardless of urgency.

## Ordered implementation checklist

1. [ ] Recheck both dirty checkouts; preserve source/data packages and update Roadmap; integrate scoped study changes onto current product code.
2. [ ] Add engineering profile, outcome-blind group manifest, exact two-model pool, provenance mapping and explicit admission; keep research flags/criterion unchanged.
3. [ ] Implement a true after-fit stop before test access; parameterize engineering selection economics in both existing checks.
4. [ ] Factor existing lock internals; add smoke-only engineering lock and artifact-export admission without modifying the research lock contract.
5. [ ] Implement pure engineering gates and shared evaluated/served fallback semantics; no fixed-model substitution.
6. [ ] Scope chat eligibility and authenticated stable assignment; validate immutable artifacts and baseline fallback.
7. [ ] Persist request/attempt/cohort/quality records in local and hosted storage; fix hosted prediction-observation persistence and retain truthful costs.
8. [ ] Wire atomic exposure/budget admission, heartbeat monitor, conservative stop rules and operator kill switch; no automatic promotion.
9. [ ] Pass no-paid research/engineering rehearsals, real sandbox, actual endpoint mocks, migrations/RLS and rollback tests; verify original manifest unchanged.
10. [ ] Freeze development contract and conservative budget; STOP for paid authorization A.
11. [ ] Collect bounded development outcomes, fit actual V2, freeze wrapper/artifact and engineering smoke lock; STOP for authorization B.
12. [ ] Complete the one-shot smoke benchmark and report every gate/limitation; STOP for deployment authorization C.
13. [ ] Deploy off, verify actual host/route/telemetry/rollback, run the capped internal pilot, then seek each prospective public-stage approval.
14. [ ] Preserve the independent five-family confirmatory track and untouched research test for later work.
