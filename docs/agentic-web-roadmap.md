# Theriad agentic web audit and roadmap

## Executive finding

Theriad did not have a single web capability. It had several partially
overlapping execution paths:

- `theriad.search.SearchService` for SearXNG, Brave, Scrapling, DuckDuckGo,
  and an optional browser launch page;
- `theriad.agents.searxng_tool` as a second SearXNG-only client;
- `ToolBroker.web.search`, which used only that second client;
- `/api/chat/agent`, which special-cased search and direct URL fetching;
- the browser agent, which could navigate a search engine;
- the unified task engine, which had no web tool at all.

This allowed component tests to pass while the autonomous planner still could
not search. Browser settings also influenced API search selection, and an
installed Scrapling extra could silently put a stealth-browser escalation in
the default path.

## Target contract

One planner-visible contract should own public internet reads:

```text
planner -> web.search -> SearchService -> configured API -> keyless fallback
        -> web.fetch  -> bounded public-page evidence
        -> browser.*  -> only when stateful interaction is actually required
```

The contract must preserve these distinctions:

- installed/configured is not the same as live-verified;
- a search snippet is discovery, not verified page evidence;
- public search/fetch is read-only; browser interaction has separate policy;
- provider retries are bounded; CAPTCHA/auth/policy/no-progress states stop or
  replan rather than loop;
- tool results include the effective provider and structured failure state.

## Consolidation completed in this change

- Removed the second SearXNG client and routed legacy `ToolBroker.web.search`
  through `SearchService`.
- Added `web.search` and `web.fetch` to the unified autonomous task engine and
  planner. Web reads are available without opting into a stateful browser.
- Separated the browser navigation-engine preference from API search provider
  selection.
- Made Scrapling/stealth acquisition explicit opt-in instead of install-time
  default behavior.
- Added an initial SSRF boundary for literal local/private URL targets.
- Added a truthful capability manifest: the code path is installed, while live
  provider/network availability is reported per call.
- Made `/api/chat/agent` create a canonical `TaskEngine` task and use that task
  ID as its public run ID. Direct search, fetch, and terminal work now use the
  same `ToolCall`/`ToolResult`, event, approval, and policy boundary as the task
  API; `ChatAgentRun` remains only a reconstructible UI compatibility view.
- Added atomic task and browser-session checkpoints. Pending approvals, budgets,
  trace, and final state survive restart; an interrupted `running` state is
  recovered as resumable rather than falsely reported as still executing.
- Added bounded model/search provider circuit breakers plus secret-free
  configured-vs-live readiness reporting and an explicit provider probe API.

## What should remain

- `SearchService`: the single search provider/fallback abstraction.
- `search.evidence`: public source acquisition and text extraction.
- The browser runtime: necessary for authenticated, JavaScript-heavy, or
  interactive work, but not for ordinary discovery.
- MCP adapters: useful for optional user-selected capabilities, not as a hard
  startup dependency for basic web search.

## What should be retired next

The special raw network clients in `/api/chat/agent` are gone; its remaining
branching is presentation and specialist-runtime selection. The next retirement
target is duplicate lifecycle state inside `BrowserAgentSession`. Keep its
browser planner as a specialist executor, but project all lifecycle, approval,
budget, and terminal state into the canonical task event stream before removing
the compatibility endpoints.

## Delivery plan

### Phase 1: reliable web capability

1. Add an async-native tool execution boundary so task tools do not bridge
   coroutines through worker-thread `asyncio.run` calls.
2. Add a live readiness endpoint with per-provider latency/error class and a
   short cache. Never claim `available=true` solely because code is installed.
3. Add production egress protection for DNS resolution and redirects, not just
   literal private IPs.
4. Select one production-grade default: managed Brave Search for hosted Theriad
   or a monitored SearXNG deployment. Keep DuckDuckGo HTML as best-effort only;
   it is not an SLA-backed primary.
5. Add deterministic contract tests plus a scheduled live canary for search,
   fetch, citation grounding, provider fallback, timeout, rate limit, and empty
   results.

Exit criterion: a task created with `requested_capabilities=["web"]` can search,
fetch at least two sources, cite them, and terminate with either a grounded
answer or a classified failure without opening a browser.

### Phase 2: one agent loop

1. **Completed:** route chat agent requests into `TaskEngine`; `ChatAgentRun`
   is now a durable-task-backed compatibility projection, not an authority.
2. **Implemented, activation pending:** local mode persists atomic mode-0600
   JSON snapshots; shared mode persists the same task, event, approval, context,
   and browser-specialist checkpoints in Supabase Postgres. Database leases
   prevent two TaskEngine supervisors from advancing one planner loop. Apply
   the migration and server-only environment configuration to activate it.
3. Replace keyword-only intent routing with planner tool choice constrained by
   capability and policy. Keep deterministic hard safety gates.
4. **Partially completed:** bounded circuits, recovery, approval, completed,
   failed, stopped, and budget outcomes are explicit. Add first-class retry,
   replan, and user-input events to finish the state machine.

Exit criterion: chat, API clients, and the Agent page show the same task ID,
event stream, approval state, and final result.

### Phase 3: optional MCP/plugin ecosystem

1. Add disabled-by-default MCP server configuration with lazy process startup,
   tool discovery, per-server allowlists, timeouts, and redacted status.
2. Normalize discovered MCP tools into the existing `ToolCall`/`ToolResult`
   schema so the planner has one tool namespace and one approval policy.
3. Ship a curated search MCP preset only as a fallback or enterprise override.
   Do not require end users to install an MCP merely to get ordinary web
   search in hosted Theriad.
4. Reuse the completed shared circuit-breaker registry and same-state/no-progress
   policy around every external tool. Never rotate blindly across plugins after
   auth, CAPTCHA, or policy failures.

Exit criterion: an optional MCP can be enabled without affecting startup when
absent or invalid, and its failures are distinguishable from core web failures.

### Phase 4: autonomous task quality

1. Add goal decomposition, dependency-aware subtasks, and verification steps.
2. Enforce budgets for model calls, tool calls, time, bytes fetched, and cost.
3. Add idempotency keys and side-effect receipts for mutations.
4. Build an autonomy evaluation suite that scores task completion, factual
   grounding, unnecessary tool calls, unsafe actions, recovery quality, and
   honest failure reporting.

Exit criterion: release gates are based on end-to-end completion and evidence,
not on isolated tool or planner tests.

## Product decision

Superseded (2026-09-26): chat now gives the model web tools and lets it decide
(`theriad/agents/web_tool_loop.py`). Search is OpenRouter's `web_search` server
tool on the user's key; page reads use `search.evidence`; JavaScript pages use
an OpenSandbox browser (see `infra/opensandbox/`). The notes below describe the
earlier classifier-driven design.

Earlier decision: for hosted Theriad, use a monitored search API or monitored SearXNG deployment
as the primary. A browser, stealth scraper, MCP, or user-installed plugin should
not be the default answer to basic search. Those are escalation paths for tasks
that genuinely need them.
