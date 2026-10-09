# Gateway pilot runbook

This gateway is opt-in and independent of the existing UI backend. Source tests are not proof of live harness or cloud execution; current evidence and blockers are in `roadmap.md`.

## Local initialization

From the repository with its Python environment:

```sh
.venv/bin/python -m theriad.gateway.cli init
source "$HOME/.theriad/gateway/gateway.env"
export OPENROUTER_API_KEY='set locally through your secret mechanism'
THERIAD_DISABLE_LIVE_PRICING=1 .venv/bin/python -m theriad.gateway.cli bootstrap
.venv/bin/python -m theriad.gateway.cli calibrate --suite /approved/suite.json
.venv/bin/python -m uvicorn theriad.gateway.app:app --host 127.0.0.1 --port 8010
```

`init` writes a0600 environment file and refuses to overwrite it. Keep its scoped Theriad credential separate from upstream provider credentials. Bootstrap pins priced agent-capable hosts with verified serving variants. `calibrate` fits incumbent heads using actual deployment measurements from training/validation only. If importing an existing artifact with `bootstrap --router /approved/artifact.json`, its heads must attest the exact revision, hosting endpoint, variant and reasoning profile; model-level labels cannot be assigned to a new host. The command packages local ONNX assets. If no approved measured roster survives catalog/eligibility checks, resolve that explicitly. Bootstrapping without `--router` is an explicit uncalibrated pilot pool, not automated adoption readiness.

The existing backend's local BYOK facade is a legacy application path; it has not gained the new gateway's capabilities. Use port8010 for the standalone gateway. Standalone compatibility is available only with explicit `THERIAD_GATEWAY_LOOPBACK_BYOK=true` and a loopback client address: its bearer can authorize OpenRouter deployments only. It cannot authenticate remote clients or become a credential for Google/other providers. The default remains scoped gateway authentication.

Gateway credential mapping is `THERIAD_GATEWAY_TOKENS`, a JSON map from SHA256(token) to either a principal string or `{"principal":"team-user","deployments":["approved-id"],"models":["approved-model"]}`. Principal ownership isolates continuation, tool bindings and outcomes. Optional deployment/model scopes apply again on fallback. Upstream references live in each deployment's `credential_env`; Google deployments use ADC/service identity.

## Harness trials

```sh
.venv/bin/python scripts/gateway/pilot.py --harness sdk --output /tmp/theriad-sdk-trial
.venv/bin/python scripts/gateway/pilot.py --harness hermes --output /tmp/theriad-hermes-trial
.venv/bin/python scripts/gateway/pilot.py --harness codex --output /tmp/theriad-codex-trial
```

Each run uses a private gateway release DB and isolated harness home/profile. Every model attempt uses the shared `~/.theriad/gateway/state.sqlite` evaluation allowance unless `--ledger` selects the already-authoritative ledger. Never create one allowance per environment. For cloud-backed local tests set `THERIAD_EVALUATION_STORE=gcp`, `THERIAD_EVALUATION_PROJECT` and `THERIAD_EVALUATION_BUCKET` to the authoritative project/bucket; ADC must allow reservations there.

The runner preserves existing defaults, writes private request captures, fixture/logs/evidence under `--output`, rechecks the caller's original fixture tests after the agent exits, and returns nonzero on failure. `--virtual` tests the virtual `theriad/agent` pool; compare with pinned runs on completion, known/unknown total costs, wall time and failures. This single-model pilot pool verifies protocol, not Router V2's multi-model routing quality. Run a separate trial with a qualified multi-model release for that comparison.

Use `hermes.yaml` and `theriad.config.toml` as opt-in configuration examples. Codex uses Responses, function tools, no hosted tools/apps/plugins/WebSockets. Upstream native Responses is opt-in (`--native-responses` for a pilot); ordinary OpenRouter bootstrap uses the implemented Chat-to-Responses adapter. The Codex-specific model catalog advertises sequential tool calls; the gateway omits an unsupported `parallel_tool_calls:false` parameter upstream. A catalog advertising tools does not prove native Responses. Select a host whose declared parameters match the client, and use `--reasoning-effort low` for endpoints requiring reasoning. Keep the requested reasoning semantics intact. A desktop test requires a separate opted-in chat and observed tool/test completion; CLI evidence does not establish desktop or hosted Cloud compatibility.

## Automatic updating

```sh
.venv/bin/python -m theriad.gateway.cli update --suite /approved/suite.json --identities /approved/identities.json
.venv/bin/python -m theriad.gateway.cli status
.venv/bin/python -m theriad.gateway.cli pause
.venv/bin/python -m theriad.gateway.cli resume
.venv/bin/python -m theriad.gateway.cli rollback
```

The single leased updater imports the free AA API and actual provider endpoint data, shortlists within existing tiers, measures paired tasks, fits new heads, qualifies, canaries and automatically promotes. It does not copy a predecessor's labels. Unknown identities/configuration, changed benchmark versions, missing critical probes, missing heads, insufficient100 independent groups, stale/unpriced endpoints, unknown charges and exhausted allowance produce pending states.

Suite shape: top-level `version`, `source_revision`, `tasks`; every task has `id`, `group`, `split` (`train`, `validation`, `qualification`), `family`, `request`, and deterministic `grader` (`exact`, `json`, `tool_calls`, `fixture`). Each source also requires `provenance` with `license_name`, approved `license_status` (`approved`, `permitted`, `public`, `public-domain`, or `internal`), `provenance_url`, and `collection_vintage`; a task can override suite provenance. This is validated before paid dispatch. Exact/JSON/tool graders have `expected`. Fixture tasks have `fixture.files` plus `expected_files` and/or `expected_json`, and bounded `max_calls`1–20. Qualification also requires a suite-level `independence_review` containing `reviewer`, the same `source_revision`, and every held-out `reviewed_groups` identity. Declare related task `lineage_ids`; related groups must stay in the same split. This records an actual review, not inferred independence from different wording.

The fixture harness owns only list/read/write tools in a fresh directory; no generated code or shell executes in the updater. It reuses Theriad's existing workspace-state grader and sums all step charges/latency.

Mark separate compatibility tasks with `critical:true` and `compatibility_case`. Required cases are `text`, `streaming`; tools add `tool_only`, `parallel_tools`, `nullable_history`, `stream_arguments`, `tool_continuation`; schemas add `json_schema`; native Responses adds `native_responses`. Real streaming probes must set `stream:true`; native probes use `wire_api:"responses"`. Critical tasks do not contribute to the100 group count or fitting. Qualification groups must also fall in the frozen incumbent-selection stratum for the affected family; unrelated baseline traffic cannot establish replacement evidence. Reusable measurements are bound to hashes of the evaluator and protocol implementation, so changing that code invalidates their cache identity. Supply representative real probes, not renamed copies of one task. Unknown charges retain reservations and are reconciled against provider generation identity when possible.

The monthly cap is25USD, NY calendar, shared across worker and pilot calls. Pilot/training allocation15USD, qualification10USD. Ordinary inference, hosting and subscriptions are separate. Every failed attempt and any model grading must go through the same reservation authority. Expensive qualification can take months; the worker reports pending rather than lowering the rule.

The5% operational canary holds for at least24h and50 candidate/control completions, with known charges and latency samples. Unsafe protocol/routing/budget events roll back immediately; measurable regression can roll back earlier. Low traffic and unresolved charges extend it. Reports include identities, confidence bounds, protocol/evidence/pricing/release hashes, spend and rollback target. Price changes require requalification. Pause prevents new paid evaluations and immediately checks rollback of a current canary. Daily workers also honor the shared pause flag.

## Google Cloud deployment

Bootstrap Artifact Registry once, before publishing the image, after any required API activation consent. The pilot's manually created private Docker repository uses immutable tags. Equivalent bootstrap commands (use the verified project and check existing repositories first):

```sh
gcloud artifacts repositories describe theriad-pilot --location=us-central1 --project="$THERIAD_GCP_PROJECT"
# Run create only when describe establishes that the repository does not exist.
gcloud artifacts repositories create theriad-pilot --repository-format=docker --location=us-central1 --immutable-tags --project="$THERIAD_GCP_PROJECT"
gcloud auth configure-docker us-central1-docker.pkg.dev
```

The repository currently has a separate bootstrap lifecycle from Terraform. Preserve it when redeploying; do not assume Terraform owns it. Push each new build under a new tag, verify the registry digest, and use the digest in deployment inputs. Repository-level IAM inherits project IAM unless explicitly constrained; absence of direct repository bindings does not prove isolation from project members. Gateway-only Terraform enables IAM Credentials for the bridge and omits Scheduler activation when updates are disabled.

Install the optional cloud/ONNX runtime dependencies and authenticated `gcloud`/Terraform locally. Supply actual project/billing/credit/identity inputs. `inputs.example.json` and `credit-record.example.json` are templates, not verified facts.

```sh
.venv/bin/python infra/gateway-gcp/build_context.py /tmp/theriad-container --router-assets "$HOME/.theriad/gateway/router-assets"
docker build -t YOUR_REGISTRY/theriad-gateway:PILOT /tmp/theriad-container
# Push the reviewed image and use its immutable sha256 digest in deployment inputs.
.venv/bin/python infra/gateway-gcp/preflight.py --project ACTUAL_PROJECT --credit-record /private/credit-record.json
.venv/bin/python infra/gateway-gcp/deploy.py --inputs /private/inputs.json --credit-record /private/credit-record.json
# After reviewing the concrete Terraform plan, the same command with --apply creates resources.
```

The wrapper validates credit evidence freshness/balance/expiry/eligibility, project/billing attachment, secret metadata, explicit identities, image digest, network pair and initial paused Scheduler before planning. Actual credit facts must be verified from Google billing evidence: supplied JSON alone does not prove the real balance. Existing default Firestore database/resources may require import into Terraform state; do not recreate them. Secret values never enter Terraform inputs.

For initial gateway acceptance, set `enable_updates:false` in the private deployment inputs. This omits the update Job, its invocation binding and Scheduler, and does not require evaluation-suite/identity objects or their hashes. Existing deployments retain `enable_updates:true` by default; Terraform moved blocks preserve existing job/Scheduler state addresses. Enable the worker only with reviewed objects and real SHA-256 pins, after accounting cutover and cloud acceptance. Keep `scheduler_paused:true` during that transition. A gateway-only deployment is not qualification or automatic-adoption readiness.

Defaults: IAM-protected Cloud Run, us-central1,2vCPU/2GiB,min0/max2,concurrency8,timeout600; immutable GCS records; Firestore lease/budget/active pointers; service identities; Secret Manager; Job/Scheduler03:00 America/New_York. Alerts cover project spend, including ordinary hosting/inference, and supplement the hard evaluation ledger. Configure an existing network/subnet, approved region/endpoints and ingress before private-network trials; no prohibited fallback is allowed.

Bootstrap the signed routing release using `THERIAD_GATEWAY_STORE=gcp`, `GOOGLE_CLOUD_PROJECT`, `THERIAD_GATEWAY_BUCKET` and the signing secret. Add an approved regional Google deployment with `vertex_deployment(...)`, documented limits/prices and matching policy. Vertex uses refreshed access tokens and native Chat Completions/SSE. It does not borrow OpenRouter billing/credits, and estimated Vertex task cost does not qualify as a settled bill. A Vertex catalog/tariff importer and authoritative per-generation charge reconciliation are not supplied by this pilot: register an approved exact deployment explicitly, and leave evaluation/promotion pending when charges cannot be reconciled.

Local IAM bridge:

```sh
gcloud auth application-default login
export THERIAD_GATEWAY_URL='https://YOUR_PROTECTED_SERVICE.run.app'
export THERIAD_BRIDGE_SERVICE_ACCOUNT='bridge-account@ACTUAL_PROJECT.iam.gserviceaccount.com'
.venv/bin/python -m uvicorn theriad.gateway.bridge:create_bridge --factory --host 127.0.0.1 --port 8010
```

The bridge refreshes Google identity tokens in `X-Serverless-Authorization` and keeps the caller's separate Theriad bearer in `Authorization`. Authorized operators need Token Creator on the bridge account; it has Cloud Run Invoker. Internal-only ingress additionally needs approved network connectivity; an ID token alone does not provide a network path.

Cloud rollback is two independent actions: route traffic to the previous approved container revision (`gcloud run services update-traffic ... --to-revisions PRIOR_REVISION=100`), and run the gateway's routing-release `rollback` against the shared store. Record the container digest/revision, active routing hash and prior targets before deployment. Recheck readiness and a genuine task after both actions. Do not enable Scheduler until cloud acceptance is recorded.

## Disconnected / legacy integration

```sh
.venv/bin/python -m theriad.gateway.cli export /approved/release.tar.gz
export THERIAD_GATEWAY_DISCONNECTED=true
.venv/bin/python -m theriad.gateway.cli install /approved/release.tar.gz
```

Archive import verifies HMAC signature, release hash, all asset hashes and paths before activation. Exchange the signing key through the organization's existing secret mechanism. The target's release must permit only internal endpoints/regions/credentials; importing a public endpoint catalog does not authorize external egress. Disconnected mode disables the update worker's external refresh; it does not rewrite an approved endpoint policy for you.

The SDK trial demonstrates the unchanged OpenAI-style client shape when credentials permit: existing SDK, gateway base URL, scoped key and virtual model. Integration with an actual legacy network or military workload still requires that organization's identity, network and operational authorization.

## Current local acceptance and rerun

On October 1, the existing OpenAI SDK and pinned Hermes/Codex CLI fixture executions succeeded through the actual gateway. On October 8, an opt-in local Codex 0.145 profile loaded the gateway's Codex model catalog, completed a Responses request, and executed a read-only `pwd` shell tool round trip. Start the loopback gateway, then launch Codex with `codex --profile theriad`; the named profile leaves the default provider unchanged. The desktop app still needs a separate new-chat acceptance pass. The private fixture evidence directory is `output/gateway-evidence/2026-10-01/`; it is ignored by Git because its databases contain local signing keys and its request captures contain client instructions. Publish only a reviewed summary and file hashes. Do not transfer those databases as public evidence.

Known working pinned commands from this run:

```sh
.venv/bin/python scripts/gateway/pilot.py --harness hermes --hosting-provider venice/fp8 --output /new/private/hermes-trial
.venv/bin/python scripts/gateway/pilot.py --harness codex --model deepseek/deepseek-v4-flash-0731 --hosting-provider inceptron/fp4 --reasoning-effort low --output /new/private/codex-trial
```

Append `--virtual` to exercise `theriad/agent`. These fixtures use an explicitly uncalibrated single-deployment pilot release: their cost/latency comparisons are descriptive and do not prove Router V2 quality or qualify a replacement. Actual pinned revisions, serving variants, costs and release hashes are recorded in each run's evidence. Availability may change; do not substitute an unknown host. CLI model discovery now returns Codex metadata; desktop and resumed-session acceptance remain separate.

For persisted CLI resume, append `--resume`. `--compact-at N` changes only the isolated trial's threshold. The6000-token stress run looped;10000 completed and resumed but did not exercise remote compaction. Never count a merely configured threshold as successful compaction evidence.

Before enabling cloud evaluation, drain/stop local evaluation dispatch and preserve the existing calendar-month ledgers, unresolved charges and generation identities in the canonical Firestore authority. Reconcile the imported totals, then point all environments at that exact project/bucket. A new project must not start another $25 allowance alongside local spending. This accounting cutover remains pending until a real Google project is supplied.
