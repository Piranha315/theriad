# Browser Agent

The browser agent is an additive feature layered on top of Theriad's existing
routing infrastructure. It lets Theriad complete multi-step browser tasks through
a structured observe → plan → validate → execute loop.

---

## Architecture

```
User → POST /api/agent/browser/start
         │
         ▼
   BrowserSession (in-memory, BrowserSessionStore)
         │
         ▼
POST /api/agent/browser/step  ──►  run_step()
                                       │
                              ┌────────┴─────────┐
                              ▼                   ▼
                        observe_page()     CapabilityDetector
                        (DOM or screenshot)       │
                              │            route_subtask()
                              ▼                   │
                       _mock_plan_action()  ◄──────┘
                              │
                              ▼
                       validate_action()   ← browser_policy.py
                              │
                    ┌─────────┴──────────┐
                    ▼                    ▼
             Approval gate         _mock_execute_action()
          (waiting_for_approval)         │
                    │                    ▼
                    └──────────► ActionLogEntry + StepDiagnostic
```

All browser tool calls are dispatched through `ChromeMCP`, which wraps
`MCPBroker` → `ChromeDevToolsAdapter`. In v1, actions are **mocked** by
default; real Playwright or browser-use execution requires explicit opt-in.

---

## Key modules

| Module | Responsibility |
|---|---|
| `theriad/agents/browser_runtime.py` | 10-stage step pipeline (observe → plan → validate → approve → execute → log) |
| `theriad/agents/browser_policy.py` | Deterministic action validator and prompt-injection guard |
| `theriad/agents/browser_observation.py` | DOM / screenshot observation (real Playwright or mock) |
| `theriad/browser/bridge.py` | Shared policy-gated browser contract used by runtimes and brokers |
| `theriad/browser/adapters.py` | Mock, async Playwright, optional Chrome MCP, and extension-stub adapters |
| `theriad/agents/browser_router.py` | Per-subtask tier routing (cheap_reasoning / vision / strong) |
| `theriad/agents/browser_session_store.py` | In-memory session registry (async-safe) |
| `theriad/agents/browser_schemas.py` | Pydantic schemas for all request/response types |
| `theriad/agents/browser_events.py` | `ObservationEvent`, `ActionProposal`, `ActionResult` dataclasses |
| `theriad/tools/chrome_mcp.py` | High-level Chrome facade (navigate, click, type, read_dom, …) |
| `theriad/tools/mcp_broker.py` | 5-stage MCP call pipeline (allowlist → risk → budget → adapter → execute) |
| `theriad/tools/mcp_registry.py` | Tool registry with risk levels and per-subtask allowlists |
| `theriad/tools/chrome_devtools_adapter.py` | Env-var-configured adapter (stub until transport is wired) |
| `theriad/capabilities/detector.py` | Deterministic 14-flag capability detector; sets `execution_mode` |
| `theriad/runtime/profile.py` | `RuntimeConfig` / `RuntimeProfile` — consumer_secure vs personal_trusted |

---

## API endpoints

All endpoints are under `/api/agent/browser/` and are fully isolated from
`/api/execute`. Normal routing never auto-starts a browser session.

### `POST /api/agent/browser/start`

Start a new browser-agent session.

**Request body** (`BrowserAgentStartRequest`):

| Field | Type | Default | Notes |
|---|---|---|---|
| `goal` | str | required | Checked for prompt-injection phrases |
| `start_url` | str \| null | null | Initial URL |

`start_url` is an advanced override, not a requirement. Normal user prompts
are resolved by `theriad.search.resolve_initial_navigation()`. Set
`THERIAD_SEARCH_PROVIDER=google|bing`; the default is
`google`. Runtime navigation rejects `example.com` placeholders and
rewrites them to the configured search provider.

Browser-agent repair loop:

```bash
npm run test:agent
npm run test:agent:e2e
npm run debug:agent
npm run agent:selftest
```

`agent:selftest` is bounded (one attempt by default, maximum five via
`THERIAD_SELFTEST_ATTEMPTS`) and reports exact failing commands. It never
self-modifies source code.
| `policy` | str | `"balanced"` | Theriad routing policy |
| `max_steps` | int | 20 | Hard stop on step count |
| `max_runtime_seconds` | int | 300 | Hard stop on wall time |
| `max_model_calls` | int | 50 | Hard stop on model calls |
| `max_cost_usd` | float \| null | null | Hard stop on estimated cost |
| `allow_paid_calls` | bool | false | Must be true for live model calls |
| `approval_required_actions` | list[str] | (14 defaults) | Actions that require user approval |
| `real_observation_enabled` | bool | false | Use Playwright for real DOM observation |
| `capture_screenshot` | bool | false | Capture screenshots when DOM is insufficient |

**Response**: `BrowserAgentStatus` with a `session_id`.

Returns `400` if `goal` contains prompt-injection phrases.

---

### `POST /api/agent/browser/step`

Run N steps in an existing session.

```json
{"session_id": "<uuid>", "steps": 1}
```

Returns the updated `BrowserAgentStatus`. Steps stop early if the session
reaches a terminal state or a pending-approval gate.

---

### `POST /api/agent/browser/stop`

Stop a running session immediately.

```json
{"session_id": "<uuid>"}
```

---

### `GET /api/agent/browser/status?session_id=<uuid>`

Return the current `BrowserAgentStatus` for a session.

Returns `404` if the session does not exist.

---

### `POST /api/agent/browser/approve`

Approve or reject a pending action.

```json
{
  "session_id": "<uuid>",
  "approved": true,
  "reason": "optional rejection note"
}
```

- `approved: true` → clears `pending_approval`, sets status back to `running`.
- `approved: false` → records a `blocked` entry in `action_log`, clears `pending_approval`, sets status to `running`.

---

### `GET /api/agent/browser/capabilities`

Returns which optional dependencies are installed.

```json
{
  "playwright_available": false,
  "browser_use_available": false,
  "chrome_mcp_configured": false,
  "mock_mode_active": true
}
```

---

## Routing hook (`/api/route`)

The `/api/route` endpoint now includes two advisory fields:

```json
{
  "browser_agent_recommended": true,
  "capability": { "execution_mode": "browser_agent", ... }
}
```

`browser_agent_recommended` is `true` when `CapabilityDetector` sets
`execution_mode = "browser_agent"`. The router **never** auto-starts a session;
the caller must explicitly POST to `/api/agent/browser/start`.

---

## Configuration (`theriad.yaml`)

```yaml
browser:
  enabled: false          # Feature flag
  max_steps: 20
  headless: true
  approval_mode: strict   # "strict" | "minimal"
  timeout: 30.0
  allowed_domains: []     # Empty = no domain restriction
```

`BrowserConfig` is accessible via `TheriadConfig.browser`.

---

## Security

### Prompt-injection guard

All goals are scanned for injection phrases before a session is created
(`detect_injection_in_goal`). Page content is also scanned during action
validation (`validate_action(..., page_content=...)`). A match blocks the
action with `blocked_by_policy=True`.

### Approval gate

The following actions always require user approval regardless of session
configuration (`DEFAULT_APPROVAL_REQUIRED`):

`login`, `submit_form`, `send_message`, `purchase`, `checkout`, `delete`,
`edit_account_data`, `upload_file`, `download_file`, `enter_sensitive_info`,
`grant_permission`, `access_financial_portal`, `access_medical_portal`,
`access_legal_portal`

### Runtime profiles

| Profile | Browser mode | Approval threshold |
|---|---|---|
| `consumer_secure` (default) | `ephemeral` | Extended approval set |
| `personal_trusted` | `dedicated_persistent` | Hard-stop set only |

Set via `THERIAD_RUNTIME_PROFILE` environment variable.

---

## ChromeMCP facade

`theriad/tools/chrome_mcp.py` exposes a clean method-per-action interface:

```python
from theriad.tools.chrome_mcp import build_chrome_mcp

mcp = build_chrome_mcp(session_id="...", subtask_type="agent_safe_navigation")

mcp.navigate("https://example.com")
mcp.click("#submit-button")
mcp.type("#search", "LLM pricing")
dom = mcp.read_dom()          # → dict with visible_text, links, buttons, forms
img = mcp.screenshot()        # → dict with screenshot bytes/path
url = mcp.current_url()       # → str
mcp.back()
mcp.forward()
mcp.wait(timeout_ms=1000)
val = mcp.evaluate_js("document.title")
tabs = mcp.list_tabs()        # → list[{id, url, title}]
mcp.switch_tab(tab_id=1)
```

When Chrome MCP is not configured (or `broker=None` is passed), the facade
runs in **stub mode** — all calls return deterministic placeholder data.
This keeps the agent loop fully testable without a real browser.

---

## Testing

```bash
# Browser agent tests only
pytest tests/unit/api/test_browser_agent.py -v

# Full suite (excludes CLI collection error, pre-existing)
pytest --ignore=tests/unit/cli -q
```

All browser agent tests run without Playwright or Chrome MCP installed.
Mock mode is the default for all `run_step()` calls in the test suite.

---

## Deferred / not in v1

- Real transport wiring for `ChromeDevToolsAdapter` (raises `NotImplementedError`)
- Persistent session storage across restarts
- Multi-tab orchestration
- Streaming step events via WebSocket
- `personal_trusted` profile UI toggle
