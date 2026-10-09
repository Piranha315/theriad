# Router V2 episode study

This is a public, benchmark-weighted evaluation of the existing product routing path. It is research evidence only: it does not imply production improvement, live-user quality, or authorization to spend.

## Frozen design

- Five families (chat, coding, tools, search, agents) receive equal weight; groups are the independent unit and repeated trials remain grouped.
- The corpus records source/license/version/lineage, fixture and grader hashes, and four disjoint group-level splits: training, probability calibration, threshold selection, and locked final test. The seed corpus is explicitly underpowered until expanded.
- Collection snapshots source bytes, dependency versions, settings, fixtures, graders, prices, provider metadata, model eligibility/exclusions, and the randomized schedule. A fresh process verifies the snapshot before any paid dispatch and on resume. Hosted model implementations cannot be frozen by a local hash.
- Baseline and every eligible deployable-registry candidate are collected on all development splits. Failures, unavailable models, retries, fallbacks, missing grades, and unknown charges remain observations.
- Router V2 uses the existing surface-feature/general and code lanes. Training, calibration, and threshold selection never read the locked test. The final runner calls `RouterV2.predict()` through the product execution path and includes routing overhead in episode latency.

## Gates and accounting

Primary quality is successful episode completion under the frozen rubric; operational failure counts against success. Report correctness, completion, operational failures, missing grades, fallback rates, p95 latency, family results, and cost completeness separately. Cost per success is weighted mean total episode cost divided by weighted success rate; unknown charges never become zero.

The preregistered one-sided gates are a quality difference lower bound greater than `-0.01` and a cost-per-success ratio upper bound below `0.90`, with complete accounting and sufficient independent groups required. Inference is paired, family-stratified, and group-clustered. No interim efficacy decisions or post-test tuning are allowed.

## No-call workflow

```sh
python scripts/run_router_episode_study.py prepare --directory benchmark_results/product_episode_study_v2 --stop-after smoke
python scripts/run_router_episode_study.py report --directory benchmark_results/product_episode_study_v2
```

`prepare` freezes the protocol and writes the no-call budget/sample-size proposal. It must complete before any later paid pilot authorization. The current implementation does not authorize paid pilot or final-test dispatch.

## Disabled canary

Router V2 remains fail-closed by default (`THERIAD_ROUTER_V2_MODE=off`). The existing `shadow` and `canary` modes use deterministic episode assignment, preserve the baseline for non-treatment traffic, and expose propensity/decision IDs for analysis. A later live run must add an experiment ID, assigned arm, artifact hash, applied decision, override/fallback reason, and matured objective outcome. Start at 5% treatment only after offline passage, a verified user-route smoke test, and separate authorization. Roll back for a serious attributable incident, treatment operational-failure increase above 2 percentage points, or p95 latency increase above 20% in two consecutive windows with at least 100 matured episodes per arm. Broad chat and unmeasured workflows stay outside the initial live quality claim.

Historical `product_episode_study_v1` outputs are preserved. New study versions must use a new output directory; changed sources, environment, prices, fixtures, artifacts, or locks invalidate resume and require a new version.

## Repair workflow (2026-09-10)

Use `docs/router-episode-study-acceptance.md` for current acceptance and limits. The default four-way seed contains only 50 semantic groups (15 train, 10 calibration, 10 threshold selection, 5 test, 5 pilot, 5 smoke) and cannot satisfy the trainer's minimum support. Renamed variants are no longer generated. `prepare` writes `training-readiness.json` and derives `BUDGET_PROPOSAL.json`/Markdown and budget caps from actual frozen inputs. It does not authorize those caps for spending.

For the bounded synthetic rehearsal, choose a **new empty output directory**:

```sh
PYTHONPATH=. /Users/puranvs/Cost-Optimization/.venv/bin/python scripts/rehearse_router_episode_study.py --directory benchmark_results/my_new_offline_rehearsal
```

The script runs the production orchestration and actual `RouterV2.predict()` with network-blocked provider fixtures. The coding sandbox is explicitly mocked. Reports are marked `rehearsal_only`; provider ledger amounts are synthetic. Resume and policy-tampering checks are included.

For explicit joint-power and boundary-null diagnostics:

```sh
PYTHONPATH=. /Users/puranvs/Cost-Optimization/.venv/bin/python scripts/run_router_episode_study.py simulate-power --directory benchmark_results/my_new_power_scenarios --groups-per-family 20 --simulation-repetitions 100 --bootstrap-samples 400
```

The simulator uses the same joint bootstrap decision as the evaluator and reports assumptions and Monte Carlo intervals. It does not turn a quality-only approximation into joint power, estimate production weights, or justify paid sample size without reviewed assumptions and prospective pilot data. Initial baseline/source/corpus/catalog inputs freeze before development dispatch; the trained challenger freezes after development and before any test outcome.
