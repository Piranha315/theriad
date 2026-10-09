# Unified local agent harness

The unified harness exposes one task, event, approval, and tool contract for
public web search/fetch, project files, guarded host processes, authenticated
Chrome tabs, optional Chrome DevTools MCP, and opt-in macOS computer access.

Web access is a core read-only capability, not a browser setting. The planner
uses `web.search` for discovery and `web.fetch` to inspect selected public
sources. Both delegate to `theriad.search`; stateful browser automation is
reserved for pages that actually require interaction. A configured SearXNG or
Brave failure gets one bounded keyless fallback and then returns a structured
failure instead of silently looping through browser backends.
`BrowserSearchProvider` and `THERIAD_SEARCH_FALLBACK_TO_BROWSER` remain only as
deprecated compatibility hooks; `/api/chat/agent` does not invoke them for
ordinary read-only searches.

Chat agent runs and unified tasks now share one durable ID. Task events and
snapshots live under `THERIAD_TASK_EVENT_DIR` (default `data/agent-tasks`), and
browser-specialist checkpoints live under `THERIAD_BROWSER_SESSION_DIR`
(default `data/browser-sessions`). Files are written atomically with mode 0600.
On restart, an interrupted task/browser run is marked ready to resume; it is
never reported as actively running when no process owns it.

`TaskEngine` is the canonical supervisor. `POST /api/agent/tasks/{id}/run`
claims an expiring storage lease before entering the planner loop, heartbeats
that lease between tool turns, and releases it on every exit path. Browser and
terminal runtimes are specialist executors whose status and checkpoints roll
up into the same task rather than independent lifecycle authorities.

For a shared deployment, apply
`supabase/migrations/20260821000000_agent_task_supervisor.sql`, configure the
server-only `SUPABASE_SECRET_KEY`, and set `THERIAD_TASK_STORE=supabase`. New
`sb_secret_` values are sent only through the Supabase `apikey` header; legacy
JWT-shaped service-role keys also use `Authorization`. Explicit shared mode
fails closed rather than silently writing a divergent local history. Inspect
the secret-free state at `GET /api/agent/storage`.
Apply `supabase/migrations/20260918000000_agent_task_delegation.sql` as well
before enabling delegated tasks against the shared store; it adds the durable
`waiting_for_children` state.

`GET /api/agent/providers/readiness` separates credential configuration from
observed live readiness. `POST /api/agent/providers/{provider}/probe` performs
an authenticated live check. Repeated transient failures open a short circuit
instead of making every planner turn wait on the same unavailable dependency.

## Hosted boundary

In `THERIAD_DEPLOYMENT_MODE=hosted`, private APIs require a Theriad bearer
session. Real terminal, browser, and chat-agent execution returns
`503 hosted_agent_execution_disabled` until an organization-isolated
OS/container sandbox, secret boundary, budgets, cancellation, and audit path
are deployed. `GET /api/agent/capabilities` reports
`hosted_execution.available=false`, `sandboxed=false`, and disabled backends;
the server does not label guarded host execution as a hosted product feature.
Blocked attempts are append-only owner-scoped audit events at
`GET /api/audit/events`.

The task examples below are therefore local/development examples. They are not
instructions for enabling hosted execution.

`TaskEngine` is the canonical lifecycle supervisor for unified tasks and chat
agent runs. It owns the durable task ID, approval state, event history, budgets,
specialist checkpoints, completion, and failure state. Browser and terminal
runtimes attach as specialist executors instead of maintaining independent
authoritative lifecycles. A supervisor must claim an expiring lease before it
advances a task, preventing two workers from executing the same task at once.

Local mode uses atomic JSON snapshots under `THERIAD_TASK_EVENT_DIR`. Shared
deployments can use the Supabase adapter after applying
`supabase/migrations/20260821000000_agent_task_supervisor.sql` and setting
`THERIAD_TASK_STORE=supabase`. Explicit shared mode fails closed if its
server-only credentials or schema are unavailable. `GET /api/agent/storage`
reports the selected backend without exposing credentials.

Public web discovery is exposed to supervised tasks as `web.search` and
`web.fetch`. Stateful browser automation remains a separate specialist for
pages that require interaction.

## OpenRouter credential and Codex

Theriad does not mint its own inference credentials. It uses an OpenRouter API
key for upstream model calls. Create a key while signed in at
<https://openrouter.ai/settings/keys>, give it a recognizable name such as
`theriad-codex`, and set a spend limit in OpenRouter. Do not paste the key into
this repository or commit it in a Codex configuration file.

There are two supported ways to give the key to Theriad:

1. Start the application and choose **Settings → Providers → OpenRouter →
   Connect**. This uses OpenRouter's PKCE flow and stores the resulting key in
   the current account's scoped settings file with user-only file permissions
   (`~/.theriad/secrets.json` for anonymous local use; a user-specific file
   under `~/.theriad/secrets-users/` for authenticated use).
2. Set `OPENROUTER_API_KEY` in the environment before starting the backend.
   The environment value takes precedence over the locally saved value.

Verify the local OpenAI-compatible facade without printing the credential:

```bash
curl -s http://127.0.0.1:8000/v1/models | jq '.data[].id'

curl -s http://127.0.0.1:8000/v1/chat/completions \
  -H 'content-type: application/json' \
  -H "authorization: Bearer $OPENROUTER_API_KEY" \
  -d '{
    "model":"theriad",
    "messages":[{"role":"user","content":"Reply with: ready"}],
    "stream":false
  }' | jq '{model, content: .choices[0].message.content, route: .x_theriad}'
```

The `/v1/chat/completions` facade is suitable for non-streaming,
OpenAI-compatible clients. It is **not yet a complete model backend for the
Codex CLI/IDE agent loop**: the facade currently rejects streaming and does not
implement the Responses API or round-trip model tool calls. Pointing Codex's
`model_provider` at it would therefore allow model discovery but would fail once
Codex requests streaming or tool use. A production Codex integration needs all
three capabilities before it should be advertised as supported:

- `POST /v1/responses`, including Codex-compatible server-sent events;
- function/tool-call input and output round-tripping without flattening the
  structured items to text; and
- cancellation plus usage/error propagation through the streaming response.

Until that adapter is implemented, use Codex as the development harness around
this repository and exercise Theriad through the curl request above or another
non-streaming Chat Completions client. The OpenRouter key is still the correct
credential; there is no separate Theriad key to generate.

## Start and inspect a task

```bash
curl -s http://127.0.0.1:8000/api/agent/capabilities | jq

# The server creates this file with mode 0600 unless an environment token is set.
AGENT_TOKEN=${THERIAD_LOCAL_API_TOKEN:-$(cat ~/.theriad/local-api-token)}

TASK=$(curl -s -X POST http://127.0.0.1:8000/api/agent/tasks \
  -H "authorization: Bearer $AGENT_TOKEN" \
  -H 'content-type: application/json' \
  -d '{
    "goal":"Run the focused tests and inspect the result",
    "permission_profile":"autonomous",
    "requested_capabilities":["terminal","project"]
  }' | jq -r .task_id)

# Let the configured OpenRouter planner run the bounded observe-plan-act loop.
curl -s -X POST "http://127.0.0.1:8000/api/agent/tasks/$TASK/run" \
  -H "authorization: Bearer $AGENT_TOKEN" | jq

curl -s -X POST "http://127.0.0.1:8000/api/agent/tasks/$TASK/tools" \
  -H "authorization: Bearer $AGENT_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name":"terminal.start","arguments":{"command":"python -m pytest tests/unit/agents/test_task_engine.py"}}' | jq
```

## Bounded local model delegation

Local root tasks expose one `task.plan` tool by default. The configured root
planner can submit a complete graph of two or three depth-1 children in one
turn. Admission, model-tier selection, dependency scheduling, and joining are
deterministic code paths and do not consume another model turn or request a
delegation approval. Up to two dependency-ready children run concurrently.

Workers inherit the root workspace, owner, permission profile, domain and app
allowlists. They can only request a subset of the root capabilities and share
its model-call and cost ceilings. Existing approval rules still apply if a
worker proposes a state-changing tool call in strict mode. After all ready
children finish, the root planner receives one compact `task.children` result
and performs final synthesis.

Inspect the graph without polling individual workers:

```bash
curl -s "http://127.0.0.1:8000/api/agent/tasks/$TASK/tree" \
  -H "authorization: Bearer $AGENT_TOKEN" | jq
```

Set `THERIAD_AGENT_DELEGATION_MODE=off` before starting the backend to hide
`task.plan`. `THERIAD_AGENT_MAX_CHILD_CONCURRENCY` may lower concurrency to
one; the local MVP caps it at two even when a larger value is supplied.

Terminal execution is guarded host execution, not a sandbox. It uses argv
execution without a shell, workspace path checks, a reduced environment, PTYs,
streaming output, process-group cancellation, redaction, and approval gates.

## Pair the current Chrome tab

1. Create a task with the `browser` capability.
2. Authenticated `POST /api/agent/tasks/{task_id}/extension/pair` and copy the one-time code.
3. Build and load `apps/theriad-extension/dist` from `chrome://extensions`.
4. Open the extension settings, enter the code, and choose **Pair current tab**.
5. The first pairing requests explicit browser-site access. Only the assigned
   tab and tabs created by that task are accepted by the backend.

The WebSocket accepts only `chrome-extension://` origins, optionally pins the
extension ID, consumes each pairing code once, rotates a session token, enforces
monotonic command sequences, limits payload size, and binds commands to a task's
assigned tab IDs. The extension has no cookies, history, bookmarks, or password
manager permission.

Example observation:

```bash
curl -s -X POST "http://127.0.0.1:8000/api/agent/tasks/$TASK/tools" \
  -H "authorization: Bearer $AGENT_TOKEN" \
  -H 'content-type: application/json' \
  -d '{"name":"browser.extension.observe","arguments":{}}' | jq
```

CAPTCHAs and unusual-traffic challenges pause for manual completion. There is
no automated challenge solving, fingerprint spoofing, or mock-success fallback.

## Chrome DevTools fallback

Set:

```dotenv
THERIAD_CHROME_MCP_ENABLED=true
THERIAD_CHROME_MCP_ATTACH_MODE=auto_connect
```

Then enable remote debugging through Chrome's user-facing
`chrome://inspect/#remote-debugging` flow. Active-profile attach may access all
open windows in that Chrome profile, so use it only for a task that explicitly
needs DevTools-level inspection. Network headers are redacted and optional MCP
usage statistics are disabled.

## Electron and desktop access

Electron RPC binds to loopback and requires a bearer token shared through
`THERIAD_ELECTRON_RPC_TOKEN` or `~/.theriad/electron-rpc-token`. Remote
renderers are sandboxed, browser permission requests default to denied, and IPC
senders are origin-validated.

macOS computer access is unavailable unless the optional `desktop` dependency,
the feature flag, an application allowlist, and Accessibility/Screen Recording
permissions are all present. Input is refused when the assigned application is
not frontmost, and `desktop.stop` is the emergency stop.
