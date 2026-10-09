# Development collection stop — 2026-09-13

## Current status

Paid Phase A is stopped, not running or fitted. Session 13734 exited 1 at 04:13:36 UTC. Pilot has 10 outcomes: 6 successes, 3 task failures and one timeout with unknown grade. Both pilot arms passed 3/5. Development has 5/390 outcomes: four successes and one operational execution error. Only the first development group has all three configurations. This is neither a fitted Router V2 benchmark nor Pareto-frontier evidence.

Known incurred provider spend is **$0.002823465**. Settled ledger spend is **$0.002779185**. Outstanding reservation is **$0.06686976**, covering the entire partially collected three-configuration group, including its known $0.00004428 DeepSeek call. Ledger committed amount is **$0.069648945**. Do not add known incurred spend to the reservation: that double counts the known call inside the reserved group. One terminal provider charge remains unknown. The original $14.46 cap and paid authorization remain valid; this is an accounting/request-compatibility stop, not a lack of user approval.

## Exact failure and diagnosis

- Episode `gsm8k:train/2997`; group `census:e3c8edeb2b53e9a4f46c4f1b`; model/policy `openai/gpt-5.6-luna`.
- Attempt `85731d90d7654b6d8856afa7b17d22e5`; ProviderError, HTTP 404; terminal 2026-09-13T04:13:36.463372+00:00 (September 13 12:13:36 a.m. Eastern); latency 122.054 ms. Neither provider request ID nor response body persisted. Billing remains null; do not impute zero from 404.
- Collection `52f6ee19c2907aa4b43cd56dba8524b72371ee8c7d513254f20782164b942514`; reservation `fc3ec165-1e62-47aa-9b9e-5605340985fc`. DeepSeek completed and Luna failed; baseline was never dispatched. Never replay the two observed arms or fabricate the missing baseline.
- User-authenticated OpenRouter Generations and Upstream Requests logs were inspected/refreshed after the stop. Both showed the surrounding successful requests, but no matching Luna failure/generation entry. Absence is not provider-certified zero billing. A generation API lookup needs a generation ID, which is unavailable for this failure.
- Offline exact request reconstruction with the actual runtime/provider serializer, MockTransport, dummy credential and socket-connect denial confirmed `temperature=0.0` plus `provider.require_parameters=true`. All seven current Luna endpoint entries omit temperature from supported parameters. Request max_tokens=1024 and price caps prompt=.45/completion=1.2/request=0/image=0 were retained. This identifies a request-compatibility defect and a probable reason for the 404, not recovery of its original response body.
- [OpenRouter parameter routing documentation](https://openrouter.ai/docs/guides/routing/provider-selection#requiring-providers-to-support-all-parameters) says require_parameters excludes providers that cannot support every supplied parameter. Search synthesis had succeeded earlier because it uses a different product request path. Model-level text/tools eligibility and unchanged catalog prices do not establish request-level compatibility.

## Evidence and preserved boundaries

Active implementation remains `/Users/puranvs/.codex/worktrees/router-v2-pilot-20260912/Cost-Optimization`. Relative to it:

- `benchmark_results/router_v2_pilot_20260912/preparation-v4/development-stop-status-20260913.json`
- `benchmark_results/router_v2_pilot_20260912/preparation-v4/development-404-offline-diagnostic.json`
- `benchmark_results/router_v2_pilot_20260912/preparation-v4/development-luna-endpoints-20260913.json`
- `benchmark_results/router_v2_pilot_20260912/diagnose-development-404.py` reproduces the request offline. Its synthetic HTTP response is explicitly labeled and never enters paid outcomes or billing.
- `benchmark_results/router_v2_pilot_20260912/paid-development-log.txt`, plus original append-only attempts/development outcomes and budget ledger.

The current approved source was reverified unchanged as `c734fd82eaeb9b1cc92dee7550b8e2b05454018ed596b4d973a33ee7e43ee8ae`. No frozen-source edits, replacement models, limit relaxations, ledger rewrites, retries, held-out exposure, commit, push or deployment occurred. Historical 228 selected passing tests and two passing rehearsals apply to this unchanged source; the new offline reconstruction passes its request/cost assertions, but reveals a provider-compatibility gap those tests did not cover.

## Recovery path

1. Obtain provider evidence for the exact failed Luna request above, including whether it reached inference and its billed amount. The dashboard is signed in; no further login request is needed. If no record becomes available, a provider-support inquiry is required; none has been sent (user has not authorized messaging support). The current generation lookup cannot reconcile without an ID.
2. Repair request compatibility offline with an explicit capability-aware policy. Do not disable require_parameters or silently alter temperature semantics just to pass. Preserve the fixed candidate list, price limits and split boundaries. Add exact-request eligibility checks covering every candidate/product path and whitelisted provider error diagnostics (no prompts/credentials in errors).
3. Any source change needs a concrete new source amendment under the existing approval binding. Do not overwrite the approved amendment or original source snapshots. Predefine how pre/post repair development data are handled; never merge changed request semantics without provenance or silently rerun unfavorable outcomes.
4. Build append-only, evidence-bound recovery for the partial group: reconcile Luna's charge, retain its execution failure, and dispatch only the unobserved baseline under the original reservation. The current collector intentionally blocks any partial group; a blind canonical rerun will stop again. Generalize reconciliation without overwriting the first pilot's cancelled-generation evidence.
5. Reverify source/admission and offline recovery tests; resume remaining development and fit only when the billing and amended-source gates pass. Smoke/test/deployment remain unauthorized. Pilot quality and current five development outcomes are insufficient for quality or Pareto claims.
