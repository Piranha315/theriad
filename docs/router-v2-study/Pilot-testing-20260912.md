# Router V2 pilot testing — 2026-09-12

## Decision

Phase A engineering collection is prepared for exact-budget approval. No paid calls have occurred. The actual Router V2 has no empirical benchmark or Pareto-frontier position from this work. Deployment and the later engineering evaluation are not ready. Production remains unchanged.

Implementation is in `/Users/puranvs/.codex/worktrees/router-v2-pilot-20260912/Cost-Optimization`, detached product base `fe14bdf518462cb056a77cf49780e5e779d1dced`, with documented snapshots of the newer dirty product sources and repaired study sources. No commit or push was made. Preserve the whole worktree; its untracked files are not recoverable from HEAD.

## Completed verification

- 201 selected tests passed with Internet connections and DNS denied before test collection: `benchmark_results/router_v2_pilot_20260912/combined-no-paid-tests-v3.txt`. They cover engineering admission/budgets, actual chat endpoint tests with mocked providers, fallback accounting, observations, study runtime/protocol, and the trainer/router. This is not a full-repository acceptance run.
- Engineering profile: complete offline rehearsal passed using actual training, calibration, threshold selection and RouterV2 predictions, real product task adapters, real code sandbox, and explicitly synthetic providers, billing and corpus/review identities. 290 simulated provider requests, 505 actual router predictions, zero network attempts. Resume produced no repeat dispatch; fitting never loaded smoke or research test; smoke dispatch was rejected. See `engineering-real-sandbox-rehearsal/engineering-rehearsal-verification.json` and its `REPORT.md` under the same output directory.
- The canonical research-profile rehearsal also passed with the real code sandbox: `research-real-sandbox-rehearsal-v2/REPORT.md`. Both rehearsal source/environment manifests match the current preparation exactly. Synthetic success and cost figures are plumbing checks, not model performance.
- All 53 original research-manifest file hashes revalidated. The v9 corpus, original review, quarantine and 125 research-test groups remain unchanged. No original exposure stage was moved.
- Negative tests verify missing/mismatched authorization, changed source, exhausted budget, unknown-charge stop with retained reservation and no repeated dispatch, changed frozen candidate pool, request price/add-on guards and original research quarantine.
- The first engineering rehearsal reached fitting but failed a test that incorrectly expected locked-research protocol evidence. Reporting now separately exposes verified development source snapshots while leaving the locked-test evidence false. The failed log is retained.
- Nine unrelated browser/OpenSandbox baseline failures remain outside the selected acceptance suite. `git diff --check` still reports the inherited extra blank line at EOF in copied `theriad/routing/study_budget.py:249`; this was not changed after freezing source. Intermediate and prior baseline logs remain available.

OpenRouter documents automatic billing details in non-streaming responses; no extra request flag is necessary. Missing paid-model billing is treated as unknown and stops the engineering collection, never as zero or an estimated settled cost. [Official usage accounting documentation](https://openrouter.ai/docs/cookbook/administration/usage-accounting).

## Frozen Phase A proposal

Current directory: `benchmark_results/router_v2_pilot_20260912/preparation-v4/`. v1/v2/v3 preparations are superseded by this source revision and must not be authorized.

- Contract digest: `c098fd657b176e64f0ccd6dfd846d60aa843b1bedf114a39ea7a5b44183bbb60`
- Source/environment manifest: `7d498f98e4d24f5f7f24f6a5f9aeef19a991ce039d4cb246ead066184068f05f`
- Budget proposal digest: `11dd1f51c7c4d15106fded74d827596bfb100cfe3706b0b4e0237a457f5cd9c5`
- Two candidate models: `deepseek/deepseek-v4-flash-0731`, `openai/gpt-5.6-luna`, compared with the frozen current baseline plans.
- Public model catalog: read-only snapshot fetched 2026-09-12T06:00:05.991691+00:00; SHA256 `5df3ca95a59ebaf647dafbedcd09ab4344bff7ff8428866dd9525b6fbbbcd142`. No inference was used for pricing.

| Stage | Original groups | Policy trajectories | Budget cap including 20% contingency |
|---|---:|---:|---:|
| Pilot | 5 | 10 | $0.21 |
| Train / calibration / threshold selection | 60 / 30 / 40 | 390 | $14.25 |
| **Phase A requested** | **135** | **400** | **$14.46** |
| Later smoke, separate approval and lock required | 21 | 42 | $2.42 |

Phase A fits the five-family development corpus; the contemplated later live pilot is ordinary text chat only. Policy trajectories may contain multiple provider calls. Expected spend is unavailable. The full proposal's $16.88 includes the provisional later smoke allocation, which is not authorized or executable now. Current real committed, incurred and unresolved reserved amounts are each $0.

Reservation bounds include published reachable token/cache tiers and enforce input/output, call and provider price limits. Paid add-ons and hidden model routing are forbidden. Both arms receive the same provider price constraints; the result would describe that constrained baseline, not unrestricted production routing. As with any published-tariff reservation, this is not an unconditional guarantee against a provider changing its pricing or misbilling. Provider-reported charges remain authoritative and unresolved charges stop the run.

Both candidates have outcome-blind effective support of 60 train, 30 calibration and 40 threshold groups after the trainer's input aggregation. These clear the necessary support floors; successful actual collection and fitting are still required. A failed fit must not borrow later data or substitute a fixed model.

## Methodological correction to the historical plan

One of the proposed eight chat smoke groups contains 17 rows marked development-only prior overlap. Those rows stay diagnostic. The engineering subset therefore contains seven usable chat smoke groups, 21 smoke groups overall (chat7/coding4/search8/tools1/agents1), and 156 selected groups including development. This reduces the historical proposed 444 trajectories to 442 across both phases. Optional extra fixed-model comparison arms are not included. No raw-count reshuffling or relaxed semantic clearance was used.

The small smoke set can support only an engineering screen, not a claim of preserved quality or a powered five-family experiment. Same-author review, possible public-benchmark contamination, short deterministic tasks and limited tools/agents diversity remain. The confirmatory study still needs its independent review, representative sampling and joint-power prerequisites.

## Resume and remaining work

1. Present this exact Phase A contract and $14.46 cap for user approval. No real `authorization-development.json` exists. The records in synthetic rehearsal directories are TEST ONLY and cannot authorize the actual corpus/models.
2. After explicit exact-budget approval, record its real conversation reference with the contract/budget/protocol bindings required by `require_paid_authorization`. Revalidate source and prices; if anything changes, create a new preparation and obtain approval for that revised scope. Then run the existing runner with `--stop-after fit`, stopping on unknown charges or insufficient fitting support. No smoke/test collection follows automatically.
3. Implement the separate engineering smoke lock, paired evaluation report and artifact admission/export. Keep the research test-lock semantics unchanged. Freeze the learned policy before separately requesting the $2.42 Phase B scope; recalculate if wrapper/config/prices alter its schedule.
4. Complete ordinary-text-chat eligibility and legacy/OOD fallback, hosted request-level telemetry, assignment and rollout controls, and no-paid UI-to-HTTP acceptance. The current passing endpoint tests do not implement or prove those new live controls.
5. Before deployment, present the concrete artifact/configuration, allowlist, exposure/cost cap and monitor/rollback evidence for separate approval. No live or public cohort is authorized by this report.

Rehearsal reproduction (new empty output directory required):

```sh
PYTHONPATH=. /Users/puranvs/Cost-Optimization/.venv/bin/python scripts/rehearse_router_episode_study.py --profile engineering_canary --real-code-sandbox --directory benchmark_results/router_v2_pilot_20260912/engineering-rehearsal-next
PYTHONPATH=. /Users/puranvs/Cost-Optimization/.venv/bin/python scripts/rehearse_router_episode_study.py --profile research --real-code-sandbox --directory benchmark_results/router_v2_pilot_20260912/research-rehearsal-next
```

Evidence index: `benchmark_results/router_v2_pilot_20260912/resume-verification.json`. The public-catalog preparation command is saved in the prior handoff and `preparation-v4-command.txt` retains its output. The durable network-denial plugin is `benchmark_results/router_v2_pilot_20260912/deny_network.py`; no reliance on credentials or a paid model is needed for local verification.
