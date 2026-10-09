# Local model delegation MVP

Objective: let the local TaskEngine orchestrator submit one bounded depth-1
task graph, route each child to the least-capable suitable model tier, run
ready children with bounded concurrency, and resume the parent without a
model-driven permission or polling loop.

## Implementation

- [x] Add durable parent/child task metadata and deterministic graph admission.
- [x] Add the batched `task.plan` tool and child-role model routing.
- [x] Run dependency-ready children and resume the parent with compact results.
- [x] Add local task-tree API visibility, cancellation, and usage accounting.
- [x] Add focused unit/integration coverage and run the relevant test suite.

## Verification

- 236 focused agent, task API, chat-agent, and episode-routing tests passed.
- A fresh local FastAPI process accepted an authenticated two-node
  `task.plan`, returned `approval_required: false`, persisted the dependency,
  and exposed the graph through `GET /api/agent/tasks/{id}/tree`.
- The smoke did not invoke a model provider. Unit coverage drives the complete
  root-plan, dependency-ordered workers, automatic join, and final-synthesis
  trajectory with a deterministic planner double.
- The repository-wide suite finished with 2,469 passed, 8 skipped, 11 xfailed,
  and two failures outside delegation: the existing analytics owner/fail-closed
  status expectation and an authored-study import requiring unavailable
  `scikit-learn`.
- No OpenRouter credential is configured on this host, so a paid/live provider
  trajectory was not dispatched. The local API and provider-free execution
  path were verified independently.

## Guardrails

- Depth 1, at most 3 children, at most 2 running concurrently.
- Children cannot broaden workspace, permissions, capabilities, domains, apps,
  or the root model-call budget.
- Delegation itself never asks for approval. Existing state-changing tool
  approvals remain authoritative.
- Child outputs are treated as untrusted result data and returned through a
  compact contract; hidden reasoning is never requested or persisted.
