# Theriad browser architecture

The supported user surface is the React application rooted at `index.html` and
`frontend/src/main.jsx`. Its Browser Agent panel calls the FastAPI session
contract in `theriad.api.server`:

- `POST /api/browser/session/start`
- `GET /api/browser/session/{session_id}`
- `POST /api/browser/session/{session_id}/{approve|deny|stop|use-api-search}`

That contract wraps `theriad_browser.browser_agent.BrowserAgentStore`, the retained
policy-gated mock session adapter. It is explicitly a mock: it does not claim
to control a live browser. Approval gates, CAPTCHA handoff, API-search escape
hatch, and resource limits remain in that adapter.

The old `theriad_browser.server` routes under `/api/agent/browser/*` are compatibility
aliases for local scripts only. `static/index.html` is available only at
`/dev-fixtures/browser-agent` from that legacy server and is a test fixture,
not a product surface. The former chat-only `browser_runtime` now imports the
canonical store rather than creating independent runs.

Theriad Browser remains an extension adapter: normal page-aware chat goes through
`/api/chat/send`; its direct Gemini Computer Use implementation is legacy,
extension-local functionality and is not the supported Theriad browser
session path. It must not report those actions as Theriad session execution.
