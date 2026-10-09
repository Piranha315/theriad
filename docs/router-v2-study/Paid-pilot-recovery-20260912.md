# Phase A paid pilot recovery — 2026-09-12

The approved Phase A pilot started and stopped on its first tools-group pair. Billing is now fully reconciled: **$0.00050787 incurred and settled; $0 outstanding**. There were 11 actual model requests plus six locally accounted operations. No development, smoke or research-test collection ran; no fitted Router V2 or deployment was produced.

The baseline ended `task_failed`. The challenger hit the 180-second episode deadline. Its cancelled request was charged **$0.00007523**. The local response never returned a generation ID; after the user signed in to OpenRouter, the activity dashboard identified `gen-1789225509-vPS8fylkFCJndfmVS9kV`. A read-only authenticated generation lookup verified the cancellation, matching start time, application, provider and exact charge. The analytics schema request had returned HTTP403 with the existing key; no new key or permissions were created.

Evidence remains in the original `benchmark_results/router_v2_pilot_20260912/preparation-v4/` directory: `cancelled-generation-metadata.json`, `billing-reconciliation.json`, `attempts.jsonl`, `pilot-outcomes.jsonl`, `execution-history.jsonl`, and the updated `report.json`/`REPORT.md`. Original raw attempt/outcome events were not overwritten. Recovery appended an accounting-only terminal challenger record with **success=null**, because its final task state and grade were not retained. The original reservation is settled to the verified total. A direct resume check with dispatch forbidden proved that this recovered pair will be skipped without another request or reservation.

These are pilot operational observations, not a Router V2 benchmark: the learned policy has not been fitted. One incomplete/failed tools pair cannot locate the router on a Pareto frontier.

## Repairs and validation

The collector and runtime used identical deadlines. The outer collector cancelled before the inner runtime could preserve its timeout outcome. The repair gives the collector five seconds for outcome finalization; the runtime's existing task/provider deadline, output caps, call limits and prices are unchanged. Both development and paired collectors use it. Actual-runtime cancellation tests demonstrate a durable timeout outcome, retained unknown-charge reservation, no repeat dispatch and no extra generation time.

The trace also reveals repeated reasoning-only completions at the product planner's 700-token cap. That is retained as observed behavior. The repair does not increase token limits or tune the baseline/candidate after seeing pilot outcomes.

The reporting path now validates a provider charge reconciliation against original terminal-event hashes and the pinned provider metadata, without rewriting unknown raw records or treating them as free. A source-amendment guard requires explicit approval tied to the original contract and the exact replacement source snapshot.

**228 selected network-denied tests passed** (`recovery-final-tests.txt`). Both final current-source rehearsals passed with the real code sandbox (`recovery-final-engineering-rehearsal.txt`, `recovery-final-research-rehearsal.txt`). Their source manifests match the proposed amendment. All 53 original research-manifest files revalidated unchanged. Exact source diff: `preparation-v4/source-amendment.diff`. No tests made paid calls.

## Concrete source-only amendment for review

- Original approved contract: `c098fd657b176e64f0ccd6dfd846d60aa843b1bedf114a39ea7a5b44183bbb60`
- Original frozen source/environment: `7d498f98e4d24f5f7f24f6a5f9aeef19a991ce039d4cb246ead066184068f05f`
- Proposed replacement source/environment: `c734fd82eaeb9b1cc92dee7550b8e2b05454018ed596b4d973a33ee7e43ee8ae`
- Proposal: `benchmark_results/router_v2_pilot_20260912/preparation-v4/source-amendment-proposal.json`; approved=false, no real `source-amendment.json` exists.
- Only three source files differ: `scripts/run_router_episode_study.py`, `theriad/routing/episode_study.py`, `theriad/routing/episode_engineering.py`. Tests and handoff documents are additional local changes.
- Original contract, baseline plans, corpus identities/exposure stages, candidate models, token/call/task-work limits and stage caps remain bound and unchanged. The original source snapshot is retained permanently.
- **No added budget:** original Phase A cap $14.46; already spent $0.00050787; remaining $14.45949213. Original ledger carries prior expense forward. Remaining schedule: eight pilot trajectories and 390 development trajectories (398 total); no repeat of the first pair.
- After explicit source-amendment approval, write `source-amendment.json` from the proposal with approved=true and the actual user approval reference. Validate its hash binding, then use the existing launcher `benchmark_results/router_v2_pilot_20260912/run-approved-phase-a.py pilot`; inspect the remaining pilot, then resume with `fit`. The launcher and runner both reject unapproved source drift. New executions retain their own source snapshot alongside the original one.
- Unknown future charges still halt paid work. No charge reconciliation infers missing quality, releases expense as zero, changes exposure assignments or supplies missing training outcomes.

## Handoff

Implementation: `/Users/puranvs/.codex/worktrees/router-v2-pilot-20260912/Cost-Optimization`. Interpreter: `/Users/puranvs/Cost-Optimization/.venv/bin/python`. Keep the complete worktree and all snapshots; no commit/push occurred. No paid process remains running.

The historical testing report's zero-spend and no-authorization paragraphs are superseded by this recovery report for Phase A only. The initial $14.46 approval remains recorded. Before further dispatch, the source-only repair needs the separately bound approval specified by the frozen study protocol. This is not a request for a larger budget or deployment permission.

Longer-term remaining work: actual development/fitting, a separately locked and authorized engineering smoke comparison, and ordinary-text-chat admission/telemetry/assignment/rollout controls. Research independence, representative sampling and joint-power limitations remain unchanged. Production is untouched.
