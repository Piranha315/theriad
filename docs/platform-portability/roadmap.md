# Theriad portability and automatic adoption roadmap

Updated: 2026-10-08, America/New_York.
Status: local implementation and live SDK/Hermes/Codex CLI fixture pilots verified; a private Codex profile now completes a real Responses request and local shell-tool round trip. Desktop chat acceptance, full qualification, container and Google Cloud acceptance remain pending.
Production defaults, global harness settings, existing UI, and unrelated dirty changes are preserved. No commit, push, production restart, or cloud resource creation has occurred.

## Implemented locally

- Standalone `theriad.gateway.app`, scoped credentials, Chat Completions and real upstream SSE, native/translated Responses, namespaced function tools, principal-scoped continuation and compaction. No UI instructions or hidden compression. No hosted tools or WebSockets.
- Exact deployment records, endpoint/region/origin restrictions, conservative full-context checks, fresh-price and output checks, independent credentials on fallback, explicit errors, and no retry after output. Deployment/release affinity during tool cycles and opaque reasoning replay.
- Concrete OpenRouter host pinning and serving variants; baseline `calibrate` command fits the already-approved roster from actual measurements. Importing old heads requires exact deployment provenance, never an assumed host conversion.
- Content-addressed signed release bundles, independent fitted replacement surfaces, immutable router cache identities, local hash-verified ONNX assets, atomic activation, rollback, outcome reports, and disconnected import/export including assets.
- Daily-worker implementation: paginated free Artificial Analysis and actual OpenRouter endpoint catalogs; exact verified identity/configuration joins; timestamped snapshots; full methodology version alongside the API's major.minor version; headline/pricing shortlist; ambiguous records stay pending.
- Transactional $25 NY-calendar-month evaluation ledger, $15 pilot/training and $10 qualification, per-attempt reservations, retained unknown/failed charges, cross-month reconciliation, one worker lease, pause, strict paired/group-aware simultaneous dominance bounds, and minimum 100 held-out groups per affected family.
- Qualification uses independently measured responses, real streaming/native protocol probes supplied in a versioned suite, and controlled multi-turn file-tool fixtures with the existing workspace-state oracle. It never executes model-generated shell commands inside the privileged updater. Training/validation/qualification remain disjoint.
- Automatic 5% canary, minimum 24h and 50 completed candidate/control requests, charge reconciliation, immediate unsafe-event rollback, early operational regression rollback, configurable cost/failure/latency limits, atomic promotion and evidence reports. Pricing changes terminate the canary and require requalification.
- Google ADC/service identity adapter, regional Vertex OpenAI endpoint mapping, refreshable loopback IAM bridge, Firestore transactions/TTL and Cloud Storage immutable records/assets/reports. External providers retain their own charges.
- Minimal backend container build context; Terraform Cloud Run gateway (us-central1, 2 CPU, 2 GiB, min0/max2, concurrency8, timeout600), updater Job, Scheduler03:00 NY, Secret Manager references, private storage, IAM identities, optional existing VPC/subnet, budget alerts, and read-only deployment preflight.
- Opt-in Hermes/Codex profiles and genuine isolated SDK/Hermes/Codex fixture runner. Legacy server duplicate `/v1` decorators removed; the old UI backend remains separate. The new gateway owns its own canonical implementation.
- Codex-specific `/v1/models` catalog while preserving the standard OpenAI model-list shape for other clients. The local profile exposes function tools, disables hosted tools and parallel calls, and selects translated Responses over the private loopback gateway.

## Verification evidence

| Evidence | Result |
|---|---|
| Gateway + existing harness + Router V2 regression | 101 passed: 49 gateway, 24 legacy harness, 28 Router V2 |
| Static checks / packaging | Ruff, compileall and diff checks passed; standalone wheel built. Linux container not built. |
| Terraform Google provider schema validation | Passed September30 with Google6.50.0 / Terraform1.14.7; Terraform source unchanged since validation |
| Local ONNX assets | September30: two hash-verified assets; real local encoder returned384 dimensions. Temporary staged assets expired; regenerate/package approved assets for deployment. |
| Codex CLI0.145.0 request shape | Captured from installed CLI; sanitized fixture covers namespaces/client metadata. Capture server performed no inference. |
| Codex CLI0.145.0 live local profile | October8: model catalog loaded, a Responses turn completed, then Codex ran `pwd` in `/private/tmp` through the local shell tool and returned the observed output. Gateway listener was loopback-only on127.0.0.1:8010. |
| Existing OpenAI-style SDK through real gateway | October1: succeeded on pinned Venice/fp8; one completed request, $0.0000087 known cost. Earlier September30 authentication failure remains separate. |
| Hermes fixture execution | Pinned and virtual `theriad/agent` passed: observed read, patch, pytest and final answer. Five completed requests each. |
| Codex CLI fixture execution | Pinned and virtual `theriad/agent` passed using translated Responses; observed read, edit, pytest and final answer. Three completed requests each. Persisted CLI resume also passed. Forced6000-token compaction looped and was stopped;10000-token session passed without a remote compact request. Successful compaction remains pending. |
| Codex desktop | Not exercised; the named profile is opt-in, and a separate new desktop chat still needs an observed tool round trip. |
| Container Linux build/runtime/profile | Not exercised; Docker daemon unavailable |
| Cloud IAM/inference/refresh/state/rollback/private network | Not exercised; deployment inputs absent |

September's three authentication-failed attempts retain $0.0002868 in the September ledger. October has $0.033936208 settled and $0.612655355 retained across39 unresolved failed/diagnostic attempts; $24.353408437 remains available. Reservations are committed allowance, not claimed actual charges. The calendar rollover did not erase September reservations. No model has been automatically promoted.

Reviewed, shareable metadata and artifact hashes are in `verification.json`. Private raw logs/captures/evidence are under `output/gateway-evidence/2026-10-01/`, ignored by Git. September temporary SDK evidence expired, so do not link it as a current artifact. The current credential check returned200. Live failures exposed Qwen host parameter incompatibility, GLM mandatory reasoning, and provider rate limits; bootstrap no longer assumes native Responses, and catalog-declared parallel-call requirements are checked before dispatch.

| Successful pilot | Completion | Known task cost USD | Wall time seconds | Run failures |
|---|---|---:|---:|---:|
| Hermes pinned / Venice fp8 | passed | 0.011797550 | 310.111 | 0/1 |
| Hermes virtual / same deployment | passed | 0.009869110 | 310.869 | 0/1 |
| Codex CLI pinned / DeepSeek Inceptron fp4 | passed | 0.001044982 | 12.786 | 0/1 |
| Codex CLI virtual / same deployment | passed | 0.000965858 | 16.017 | 0/1 |

Each is one task group in an explicitly uncalibrated single-model pilot pool. These are descriptive comparisons, not evidence of multi-model Router V2 quality or statistical dominance. Five earlier Codex runs and one earlier SDK run failed before completed inference; their reservations remain unresolved. All successful completed calls reported charges; failed-call costs remain unknown.

## Inputs still required

1. Maintain valid server-side provider credentials in environment/secret storage, never chat. October1 OpenRouter authentication and actual inference succeeded; availability/rate limits still require runtime handling.
2. Free `ARTIFICIAL_ANALYSIS_API_KEY`; none available. No subscription purchased.
3. A versioned, source-pinned representative evaluation suite, including critical protocol cases and at least100 genuinely independent held-out groups per affected family. The pilot fixture is one group and cannot establish qualification. Existing synthetic variants are not100 independent tasks.
4. Verified identity mappings for initially paired benchmark/provider configurations. The free AA API does not establish OpenRouter hosting revision/quantization or physical region; names alone cannot justify those claims. Unverified new releases remain pending.
5. Google project ID, billing account, actual credit balance/expiry/service eligibility evidence, authorized identities, private-network policy, and approved regional Google model/prices.
6. A working Linux container builder. Cloud performance must be measured on the deployed architecture; a Mac ONNX profile is not cloud proof.

## Remaining work / acceptance

- SDK, Hermes and Codex CLI fixture pilots are complete, including pinned/virtual trials. Repeat with a provenance-bound, calibrated multi-model release for a meaningful pinned-versus-routed task quality/cost/latency/failure comparison. Native upstream Responses remains explicit opt-in and requires real compatibility evidence.
- CLI resume and a local shell-tool round trip are verified. Verify desktop chats, cancellation, successful compaction, and protected cloud bridge separately. Do not substitute protocol fixtures for these executions.
- Build the Linux container and measure actual cold start, RAM, routing latency, and streaming under Cloud Run limits. Correct dependencies/runtime as needed before deployment.
- Obtain and verify actual billing/credit evidence in Google's billing system. The preflight checks a fresh evidence record and project/billing attachment; it cannot independently query a universal credit-balance API.
- Bootstrap a signed GCP release and approved Google deployment, run cloud IAM/token-refresh/restart/TTL/rollback acceptance, and prove a permitted legacy SDK path on the target private network.
- Populate an adequate reviewed suite and exact identities; run small-budget training then qualification across months. Keep the Scheduler paused until initial cloud acceptance passes, then enable automatic daily qualification/promotion without per-promotion approval.
- The controlled file-tool evaluator covers multi-turn agent-style state changes. Full Hermes/Codex task-quality collectors across the reviewed qualification corpus are still required before labeling that evidence actual external-harness qualification.
- Add an authoritative Vertex charge reconciler and exact Google catalog/tariff importer before enabling automated adoption on Google deployments. The current managed-inference adapter performs token-based estimates; unknown actual charges cannot qualify.
- Native AWS/Azure adapters, GPU hosting, hosted Codex Cloud, WebSockets, provider-hosted tools, and military operational authorization remain outside this pilot.

## Handoff instructions

Read `runbook.md`, then `theriad/gateway/{contracts,engine,releases,adoption,worker,evaluation}.py`, the gateway tests, and `infra/gateway-gcp/main.tf`.

Preserve all unrelated dirty files. New work is confined to the gateway, gateway tests/scripts, portability docs, infrastructure, two optional extras/CLI entrypoint in `pyproject.toml`, and removal of the server's two duplicate route decorators. Do not stage the entire checkout.

Use `THERIAD_DISABLE_LIVE_PRICING=1` for test/registry inspection so unrelated legacy price caches are not refreshed. A pilot release DB must be private to its output directory; its evaluation ledger must point to the one shared authority. In GCP, all environments must use the same Firestore authority, not separate local allowances.

Latest account usage check: five-hour24%, weekly45%; no immediate token-limit block. Update this roadmap with new tests, exact live evidence, errors, and next steps before changing harness/provider.

## Exact local verification command

```sh
THERIAD_DISABLE_LIVE_PRICING=1 .venv/bin/python -m pytest tests/unit/gateway tests/unit/api/test_harness.py tests/unit/routing/test_v2.py -q
.venv/bin/python -m ruff check theriad/gateway scripts/gateway infra/gateway-gcp/*.py tests/unit/gateway
```

Gate evaluation/cache identity now binds implementation hashes, so new protocol/evaluation code cannot reuse old qualification results silently. The approved source provenance and independent held-out review remain prerequisites before any paid qualification. Unknown vendor origin is ineligible when an origin exclusion policy is active.

The explicit6000-token compaction stress trial repeatedly reread without advancing; it was stopped and is recorded as failed. A10000-token trial completed the initial task and resumed read/test/final response, costing $0.002865656 across8 completed calls. No remote compaction endpoint was used. Keep this separate from a successful compaction claim.

Before cloud evaluation cutover, stop/drain local evaluation dispatch, preserve all month ledgers and unresolved generation reservations, and transfer them into the one authoritative Firestore project/bucket before enabling evaluations there. Configure every environment to that same authority. Never reset the allowance by moving to a new release DB, bucket, project, or monthly window. The migration and its budget-total reconciliation are outstanding deployment work.

## Private input setup — October3

Created a Git-ignored `.env.gateway.local` for optional API keys and references to the private deployment/credit JSON files in `~/.theriad/gateway/cloud-inputs/`. New files are0600 and the directory0700; no credentials were copied, printed or supplied. See `private-setup.md`. Filling these templates is not verification of credit facts or authorization to bypass preflight. Existing defaults and files were preserved.

## Read-only Google metadata discovery — October4

Used agent-reach for Google's billing documentation and the user's authenticated Google Cloud Console session for actual account metadata. The project number currently in `.env` maps to an existing project linked to the configured billing account. The separately named Theriad project has no billing account. The linked account shows "Paid account" and its Credits page shows "No credits to display"; positive balance, expiry and service eligibility remain unverified, so the credit-funded deployment preflight remains blocked. No cloud resources, IAM roles or billing links were changed.

Full identities and observed URLs are recorded only in the0600 private report `~/.theriad/gateway/cloud-inputs/metadata-discovery-2026-10-04.json`. Existing `.env` and preflight JSON inputs were preserved. Handoff: use the report's project ID rather than its numeric project number, fix the invoker field to a JSON principal list, identify an actual eligible credit source before deploying, and install/authenticate the local Google CLI. Local `gcloud` is absent from PATH and no ADC file was found. The existing `.env` has0644 permissions; the private setup file remains the preferable location for secrets. The signed-in identity is observed, but deployment IAM permissions and network policy have not been verified.

October4 account usage check: five-hour6%, weekly12%; no immediate usage-limit block.

## Current readiness check — October7

Re-ran the documented gateway/legacy-harness/Router V2 test selection:101 passed in7.35s; Ruff checks on gateway, scripts, infrastructure and gateway tests passed. These are source/protocol checks, not fresh paid harness executions. `.env` now has a plausible project ID and a nonempty valid JSON invoker list. No credentials were printed or changed.

Neither the existing backend on8000 nor standalone gateway on8010 responded to local health probes. Google CLI is still absent from PATH, no local ADC file exists, and no cloud gateway URL is configured. Private deployment inputs and the credit record still contain placeholders; credit validation rejects the record because no positive verified balance is supplied. The October4 console credit observation remains historical and was not rechecked today. Existing `.env` fields do not automatically populate the structured preflight files.

Next: supply verified credit evidence and complete private deployment inputs, establish local Google authentication, build/pin the container, preserve/reconcile the canonical evaluation ledger before cloud cutover, and execute protected-cloud acceptance. Desktop, successful remote compaction, full qualification and Google charge reconciliation remain outstanding. Keep Scheduler paused and do not claim cloud readiness from passing local tests. No cloud resource was created and no paid inference was dispatched during this check. Usage: five-hour5%, weekly4%; no immediate usage-limit block.

## Student-plan funding clarification — October7

The user confirmed the Google AI Pro student trial (12 months free), not a separate credit grant. Public Google documentation currently lists $10/month of Google Cloud credits among Google AI Pro's Developer Program benefits. The user's individual credit entitlement is not verified. Google requires an active Pro membership linked to a Developer profile; check My Benefits under the same account, then apply an available credit to the intended billing account before recording balance/expiry/scope. "No credits to display" establishes no applied credits in the checked account; it does not prove no unclaimed subscription benefit.

Subscription quotas and Google One AI credits are separate from Cloud billing and direct API access. A $10 credit benefit cannot be treated as unlimited inference or as funding the entire $25 evaluation allowance plus hosting. Verify actual eligibility and restrictions from the credit card and billing evidence; do not change the ledger or manufacture a qualifying balance. Sources: https://support.google.com/googleone/answer/14534406 , https://support.google.com/googleone/answer/17422238 , https://developers.google.com/profile/help/benefits , https://ai.google.dev/gemini-api/docs/google-ai-plans .

## Credit redemption and authentication verified — October7, evening

After the user applied September and October student-plan benefits, the Developer benefits page initially kept showing Apply. A single October Apply retry followed by a page reload showed both cards claimed, each with $10 remaining and a Cloud Billing link. The authenticated Cloud Billing Credits ledger then independently showed two Available,100%-remaining $10 credits on the intended account. Scope is all Google Cloud Platform and Google Maps APIs, at list pricing; expiry dates are September1,2027 and October1,2027. This supersedes the October4 "No credits to display" observation. No additional September application was performed.

Google CLI and the local ADC file are now present. An authenticated Google CLI lookup confirmed that the intended project has billing enabled on the intended account. Recorded the account evidence privately in `~/.theriad/gateway/cloud-inputs/credit-evidence-2026-10-07.json`, populated the private credit record ($20, using the conservative earliest expiry at UTC day start because the displayed dates omit timezone), and populated only verified project/billing/authorized-invoker inputs while preserving remaining deployment placeholders. Credit evidence must still be refreshed within24h of deployment. No cloud resource was deployed, no paid inference was dispatched, and the canonical evaluation budget was not changed. Continue with container/artifact, remaining input, IAM/network, accounting cutover and cloud acceptance gates.

## Linux container preparation — October7, evening

Verified that the selected CLI project and billing attachment match the private inputs and that ADC refresh succeeds with the intended quota project. The active CLI identity has a direct project Owner binding; this does not establish every billing-account permission or organizational policy. Docker Desktop is now available. No further login command is needed for these checks.

Built `theriad-gateway:pilot-2026-10-07` for Linux/amd64 from a minimal private context with no environment files or state databases. Packaged only two immutable ONNX/tokenizer blobs from the existing artifact; no router heads were remapped, no routing release was activated, and the source artifact was preserved. The first container could start but its service user could not traverse the privately packaged asset directories. Corrected the Dockerfile asset copy to assign UID/GID10001, then rebuilt and verified actual encoder execution as UID10001.

Final local image identity: `sha256:eb7066b2eaf3fa6ef21a972a83a6c4d2a9ed5f6a6a91b2be8bbf3d9440c6db74`. This image has not been pushed to Artifact Registry; do not substitute a local image identity for a verified registry digest in deployment inputs. Local container checks: `/health`200, unauthenticated `/v1/models`401, and `/ready`503 explicitly because no release signing key was supplied. Both asset hashes verified and the encoder returned384 finite dimensions. The single51.05ms encode observation was under local amd64 emulation, not a Cloud Run p95/resource qualification. No calibrated routing or managed inference is established by this smoke test.

Terraform1.14.7 ran in the official container. Validation initially failed because the lockfile lacked the cached Linux provider checksum. Retrieved signed HashiCorp checksums for both darwin_arm64 and linux_amd64 without changing Google provider6.50.0; validation then passed. Targeted gateway operations tests:16 passed. No Terraform plan or apply ran. Private preparation and smoke evidence are in `~/.theriad/gateway/cloud-build/2026-10-07-czrbj5a5/`; the latest preparation pointer is `~/.theriad/gateway/cloud-build/latest-preparation.json`.

Handoff: inspect the private evidence, push/pin the reviewed image, provision the required secret names without putting values in Terraform state, and register/sign an exact regional Google deployment with verified capabilities and current prices. Resolve existing Firestore/resource ownership before planning. The private inputs still have image/bucket/suite-hash/identity-hash placeholders; named secrets and object paths are not proof those resources exist. Keep Scheduler paused while the reviewed evaluation suite and accounting cutover remain incomplete. Preserve all current local ledger totals and unresolved charges; no new evaluation allowance was created. Run protected-cloud IAM/bridge, streaming, restart, persistence/TTL, resource-gate and rollback checks before claiming deployment readiness. The temporary smoke container was removed; no cloud resources or paid inference were created during this turn.

Account usage check: five-hour59%, weekly13%; no current usage-limit block. This roadmap preserves the remaining work for a harness/provider handoff.

## Browser deployment preparation — October7, later evening

Opened the intended project's Cloud Run and Artifact Registry pages in the user's existing local Chrome session. No login or MFA was requested. Artifact Registry redirected to its API activation page and displayed Enable plus a statement that using the product agrees to Google Cloud terms. Requested action-time approval through the browser confirmation policy; that request remains pending. Left Chrome tab494335736 marked for handoff. No API was enabled, no repository/image was published, and no cloud resource or paid request was created. Read-only CLI inventory could not establish Firestore/Artifact Registry contents because their APIs are disabled; inspect again after authorized activation rather than assuming resources are absent.

Added explicit `enable_updates:false` deployment support, retaining the existing true default. Gateway-only mode omits the update Job, Scheduler, updater/scheduler accounts and their IAM grants, and does not require Artificial Analysis/evaluation-suite inputs. Actual worker deployment still requires real object/hash inputs and initial paused scheduling. Terraform moved blocks preserve previously managed resource addresses. Also supplied the worker's missing managed-inference role when updates are enabled. Gateway-only source tests:17 operations tests passed; Ruff passed. Terraform validation passed after the final optional-secret default adjustment; the authenticated preview plan passed with23 creates, zero changes/deletions, no worker resources/accounts and no public IAM members. This is a review preview, not an apply-ready deployment.

Private preview files are `~/.theriad/gateway/cloud-inputs/gateway-only.preview.{tfvars.json,tfplan,summary.json}` and are0600. The preview image points at the intended future registry location using the locally built image digest; publication is still pending. Bucket naming is provisional and must be checked. Regenerate the plan after publishing/verifying the registry digest, provisioning actual secrets and verifying existing Firestore/resource ownership. Do not apply the saved preview directly. No evaluation ledger was reset or migrated.

Current official Google docs introduce a further adapter/endpoint check: Gemini2.5 Flash-Lite's card says Chat Completions is unsupported and retires October20,2026. The current OpenAI-compatible supported-model list names Gemini3.x models. Gemini3.1 Flash-Lite and3.5 Flash-Lite support Chat Completions but their cards list global and `us`/`eu` multi-region locations, whereas the current adapter validator accepts only individual regions. Cloud Run can remain in us-central1; managed-inference location handling needs an explicit declared boundary and matching policy, never a silent global fallback. Gemini tool thought signatures, reasoning parameters and supported schemas must pass genuine continuation/streaming checks before signing an agent-capable release. This is implementation work, not a reason to ask the user for more credentials.

Sources checked today: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate/openai/overview , https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/2-5-flash-lite , https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-1-flash-lite , https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-5-flash-lite , https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing . Public source snapshots are temporary under `/tmp/theriad-google-*-20261007.md`; refresh them before implementing exact catalog/pricing assumptions.

Next: obtain the pending Artifact Registry activation approval or observe the user's own Enable click; provision the private repository and publish the image; continue the Google protocol/location work, then signed-release/bootstrap and protected-cloud acceptance. Account usage: five-hour73%, weekly15%; remaining work and evidence are preserved here for handoff.

## Artifact Registry publication — October8, morning

After the user clicked Enable, verified that Artifact Registry is active in the intended project. Created the private `theriad-pilot` Docker repository in us-central1 with immutable tags and published the tested Linux container. Both the registry API and Console version table confirm `sha256:eb7066b2eaf3fa6ef21a972a83a6c4d2a9ed5f6a6a91b2be8bbf3d9440c6db74`, tagged `pilot-2026-10-07`. The private deployment inputs now reference this actual registry digest; other input fields were preserved. Publication evidence is `~/.theriad/gateway/cloud-inputs/registry-publication-2026-10-08.json`, mode0600. Repository IAM has no direct bindings; inherited project IAM still applies. No public binding was added. Existing scanning configuration remains disabled; paid scanning was not enabled.

Credit/project preflight passed again against the existing, still-fresh billing evidence. The service-account inventory returned no accounts. Cloud Run, Firestore, Secret Manager, IAM Credentials and Billing Budgets still need activation; prepared their single Console Enable batch in the actual foreground Chrome tab. Activation accepts Google's terms/billing responsibility, so action-time browser confirmation is pending. No login details are needed currently. Terraform now includes IAM Credentials for the bridge and enables the Scheduler API only when updates are enabled; configuration validation passed. The previous preview is superseded and must be regenerated.

Publication is not gateway deployment or inference readiness. No Cloud Run service, persistent state, secrets or signed Google routing release has been provisioned in this stage, and no paid model request was dispatched. Automatic updates remain disabled. Continue by verifying authorized batch activation and existing Firestore/secrets/bucket resources, resolving the Google model location/reasoning/thought-signature compatibility gaps noted above, signing a tested exact deployment release, then regenerating/reviewing the infrastructure plan and executing protected-cloud acceptance. Track/import the manually bootstrapped repository or preserve its explicit bootstrap instructions; do not recreate it. Keep the existing evaluation authority and unresolved reservations intact before any cloud cutover.

Account usage: five-hour7%, weekly19%; no immediate usage-limit block.

Added explicit `us`/`eu` managed-inference locations to deployment validation while retaining exact project/host/path matching and rejecting global. A multi-region is a separate allowlist permission: allowing us-central1 does not allow `us`. Eight boundary cases plus existing operations checks passed (25 tests total); Ruff passed. These are source/security tests, not live availability or agent compatibility evidence. The already-published October7 container predates this source change: rebuild under a new immutable tag/digest after the remaining Google reasoning/signature work; never overwrite the published tag or deploy it while claiming the new location support. Fresh October8 model documentation confirms the multi-region locations; it does not qualify a model or fit router heads.

## Google protocol execution and protected-cloud preparation — October8

Verified the user's five-API activation through the actual enabled-service inventory: Cloud Run, Firestore, Secret Manager, IAM Credentials and Billing Budgets are active. Firestore databases, project secrets and storage buckets were empty at inspection. No login or MFA is currently needed.

The first paid synthetic probe revealed that constructing `us-aiplatform.googleapis.com` is wrong: Google returned400 INVALID_ARGUMENT, Invalid hostname. The US/EU multi-region endpoints use `aiplatform.us.rep.googleapis.com` and `aiplatform.eu.rep.googleapis.com`. Corrected the constructor and exact endpoint validator; global remains prohibited. A subsequent real ADC-authenticated Gemini3.1 Flash-Lite request returned OK through the declared US endpoint. All attempts, including failures, reserved charges in the preexisting canonical local monthly ledger; unresolved charges remain committed. Initial probe/report paths are private under `~/.theriad/gateway/cloud-inputs/google-compat-2026-10-08/`.

Implemented Google reasoning-effort translation, conservative recursive-schema rejection, developer-to-system role translation, and gateway enforcement of `parallel_tool_calls:false`. Preserved streamed tool signatures, grouped flat Responses parallel calls into one assistant batch, and added authenticated encrypted provider-context items so typed Codex clients can replay signatures with `store:false`. Context binds principal, exact call name/arguments, deployment and release; it expires after24h and uses a separate encryption domain from model-generated compaction summaries. Stored continuations retain the same metadata. Old cycles remain on their release; fresh user turns can select the new release and omit foreign deployment signatures. No tool is executed by the gateway.

New security/protocol cases passed, alongside the prior suite (119 tests before the final accounting case). Real Google probes passed Chat Completions tool/result continuation, streamed Responses argument/lifecycle plus stateless encrypted continuation, and structured JSON output. `scripts/gateway/google_compat.py` is the reproducible synthetic probe runner and uses the existing evaluation ledger. Its first run completed two calls but hit a reporting-method typo; fixed the reporter and preserved those attempts. The successful `protocol-2/report.json` is compatibility evidence only, not qualification or fitted heads. `protocol-3/report.json` also passed after accounting changes.

Live usage revealed that Google reports visible completion tokens separately from reasoning tokens. Normalized completion usage to include reasoning when total-token arithmetic proves they are separate; reject ambiguous accounting. Worst-case evaluation reservations include reasoning budgets, with the full documented output ceiling when configuration is unspecified. Earlier estimates were incomplete for reasoning; do not use them for qualification. Provider-reported monetary charges are still unavailable, so no estimated amount settles a reservation or qualifies a candidate. Current canonical October totals after these probes: settled$0.033936208, unresolved committed$0.636578155, available$24.329485637 (before subsequent cloud/harness probes).

Created only two regional Secret Manager secrets for release signing and scoped gateway credential hashes; raw values are private and never enter Terraform state or terminal output. The Google-only pilot does not copy an unused OpenRouter credential. Optional external-provider secret handling preserves existing configurations when supplied; workers still require their normal external/evaluation inputs. The bridge now receives the narrower OpenID-token creator role. Private inputs set `enable_updates:false`, paused scheduling and a verified registry digest; remaining work is the updated image build/publish, concrete Terraform plan, cloud resource application, signed-release installation, bridge/auth/state/restart/rollback and actual harness execution. No Cloud Run service is deployed yet. Ledger migration and authoritative Google charge reconciliation remain prerequisites for enabling the worker.

Source references: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/priority-paygo , https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/thinking/thought-signatures , https://docs.cloud.google.com/docs/authentication/get-id-token . Agent-reach's Exa free quota is exhausted; read official public pages through its Jina reader and use primary-source web search as fallback. Do not purchase a subscription or transmit private Console URLs to public readers.

Account usage check: five-hour35%, weekly24%; handoff evidence and remaining work are recorded here.


## Storage blocker and exact handoff — October8, 12:27 EDT

Final source verification passed120 gateway/legacy-harness/Router V2 tests plus Ruff. Terraform validation passed. Restored strict parallel-parameter catalog eligibility after a regression check; the Google deployment explicitly declares the gateway-enforced single-call constraint.

Rebuilt Linux/amd64 image theriad-gateway:pilot-2026-10-08, local manifest-list digest sha256:a304b8f234450e4000d9e20fd413f25e37c3aefddaa9428082d317115d2ab9c5. Context: ~/.theriad/gateway/cloud-build/2026-10-08-dzr860ic/context. A container executed non-root UID10001, correct multi-region endpoint, reasoning and token accounting assertions, but Docker failed during cleanup with metadata.db I/O error (exit125). Do not label this a clean container smoke success.

The Mac data volume has only117–137MiB available (100% full); Docker Desktop stopped and cannot start. Publication failed before creating the October8 registry tag; the updated digest is absent from Artifact Registry. Inputs still reference the published October7 image. Do not deploy that image while claiming the new Google support. No Cloud Run service, Firestore database or bucket has been created; only two regional secrets and the authorized APIs exist. The proposed bucket returned404. A new Terraform plan/apply remains pending.

Asked the user to free at least5GB, or choose a Cloud Build fallback needing another API. Do not prune Docker globally or delete Docker.raw/personal files. To preserve this handoff, removed only two verified duplicate encoder blobs (23,566,316 bytes) from this task October7 build context; the October8 assets remain intact and match immutable hashes. Earlier build evidence stays intact; that old context now needs its asset blobs recopied before rebuilding.

After space is freed: start the stopped Docker Desktop; verify the local image and final source hashes; publish the October8 tag; verify the actual registry digest; update private inputs; rerun fresh credit/project/secret/resource checks; generate a new private Terraform plan/state; apply the reviewed gateway-only resources. Keep updates disabled and scheduling paused. Install a signed uncalibrated single-deployment pilot release into GCS/Firestore using vertex-e4b6a115b25c5b213aec3c42 (the tested low-reasoning deployment), then execute real IAM/bridge, continuation/restart/TTL/rollback and harness acceptance. Every paid evaluation dispatch must reserve against the preexisting canonical authority; never create a new allowance.

Google charge reconciliation, ledger migration, reviewed evaluation inputs, fitted heads and qualification still gate the worker. No model has qualified or promoted. Desktop and actual private-network integration remain unverified. Current usage check: five-hour35%, weekly24%.

Final source-hash inspection found engine.py had lost an optional debug-print branch after the build context was captured. Preserve the current source removal, refresh the context and rebuild after Docker recovery before publication; the local a304 image is not an exact copy of the latest source. Hashes/drift and failed-publication evidence are in the private build-evidence.json.


## Obsidian dataset cleanup — October 8

User explicitly authorized deletion of Polymarket datasets and migration of Kalshi datasets outside Obsidian. Full bsdtar scan (exit 0) of /Applications/Obsidian/AI Conversations/Nexus/Claude/Projects/Kalshi API/data.tar.zst found 41,502 Polymarket files and 7,983 Kalshi files. The mixed 34.434 GiB archive must not be deleted until a Kalshi-only archive is created and verified. An iCloud vault copy is compressed/dataless, same size and modification timestamp; content identity is not yet verified. Do not hydrate it onto the nearly full disk before making room. No source dataset was deleted or moved. Destination preparation at /Users/puranvs/ResearchData/Obsidian-market-data-migration-2026-10-08 failed with ENOSPC. Asked user for separate authorization to run uv cache clean (30.3 GiB cache); approval remains pending. Preserve notes, source code and original active research checkouts. After space is available, stream-filter the archive to Kalshi only outside the vault, verify per-file hashes/counts, then remove the Polymarket-containing source; migrate exported Kalshi datasets with a manifest and leave a locator note. Moving within the same volume does not free space. The GCP deployment remains blocked pending disk recovery and exact-source container rebuild.
