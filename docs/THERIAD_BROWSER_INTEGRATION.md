# Theriad Browser Integration

## What was forked

The browser shell is based on [ComposioHQ/open-chatgpt-atlas](https://github.com/ComposioHQ/open-chatgpt-atlas), imported under:

```txt
apps/theriad-extension/
```

The fork preserves the upstream Chrome extension structure:

- `manifest.json`
- `background.ts`
- `content.ts`
- `sidepanel.tsx`
- `settings.tsx`
- `tools.ts`
- `electron-browser/`

Theriad-specific changes are isolated in:

- `theriadClient.ts`
- `browserContext.ts`
- `safety.ts`
- small settings/sidepanel integration patches

## Architecture

```txt
User in Chrome sidepanel
  -> Theriad Browser extension
  -> POST /api/chat/send on Theriad backend
  -> Theriad router resolves provider/model/policy
  -> provider execution/fallback layer
  -> response + route/cost metadata
  -> sidepanel assistant message + optional route trace
```

The extension does not choose final models for normal chat. It sends the active conversation and capped page context to the backend. The backend owns routing, provider choice, fallback policy, cost metadata, and safety warnings.

## Backend endpoints

The extension prefers:

```txt
POST /api/chat/send
```

Fallbacks are implemented in the client for:

```txt
POST /api/execute
POST /api/route
```

Connection/status endpoints:

```txt
GET /api/health
GET /api/tools/status
GET /api/provider/status
```

The backend enables permissive CORS for local extension development.

## Running the backend

From the repo root:

```bash
uvicorn theriad.api.server:app --reload --port 8000
```

Quick check:

```bash
curl http://localhost:8000/api/health
```

Example extension-style request:

```bash
curl -X POST http://localhost:8000/api/chat/send \
  -H 'Content-Type: application/json' \
  -d '{
    "messages": [{"role": "user", "content": "Summarize this page."}],
    "mode": "chat",
    "browserContext": {
      "url": "https://example.com",
      "title": "Example Domain",
      "pageText": "Example Domain is a page used for examples."
    },
    "runtimeProfile": "consumer_secure",
    "client": {"name": "theriad-extension", "version": "1.0.0"}
  }'
```

## Building and loading the extension

From `apps/theriad-extension/`:

```bash
npm install
npm run build
```

Then:

1. Open Chrome.
2. Go to `chrome://extensions`.
3. Enable Developer Mode.
4. Click “Load unpacked”.
5. Select `apps/theriad-extension/dist`.
6. Open any normal web page.
7. Click the extension icon to open the side panel.

## Settings

First-run setup is Theriad-first:

- Backend URL: default `http://localhost:8000`
- Mode: `chat` by default
- Runtime profile: `consumer_secure` by default
- Include page context: on by default
- Include browser memory: off by default
- Show route trace: on by default

The “Test Theriad Connection” button checks `/api/health` and supported fallbacks.

## Optional Browser Tools

Browser Tools are still the upstream Gemini Computer Use loop.

- Normal chat does not require a Google key.
- Google API key is only used after the Browser Tools toggle is enabled.
- Browser Tools still requests confirmation for sensitive/unsafe operations.

## Optional Composio

Composio is now feature-flagged:

- Add the Composio API key under “Tool Integrations”.
- Enable “Composio Tool Router”.
- It is not initialized on every settings refresh unless the feature flag and key are both present.

The current upstream Composio path still relies on a Google model client in the sidepanel. If Composio is enabled without a Google key, normal chat continues to use Theriad instead.

## Browser context shaping

Normal chat sends only the current active page context:

- URL and title
- selected text, if present
- page text capped to roughly 12k characters
- simplified links/images/forms/metadata/viewport
- browser memory only if explicitly enabled
- browser history is not sent
- screenshots are not sent in normal chat
- restricted pages such as `chrome://` degrade gracefully

## Safety model

Normal Theriad chat does not execute page actions directly.

Sensitive instructions include checkout, purchase, payment, delete, outbound messages, login, OAuth/account connection, form submission, and financial transactions.

- `consumer_secure`: confirms a broad set of sensitive actions.
- `personal_trusted`: still confirms payments, purchases, deletions, credentials, and outbound messages.
- Backend also returns a warning if browser action requests arrive without Browser Tools capability enabled.

## Electron path

`electron-browser/` exists upstream and has its own package and Vite/Electron scripts. Root convenience scripts were added:

```bash
npm run electron:dev
npm run electron
npm run build:electron
```

Current status: Chrome extension mode is the supported path for this integration. The Electron TypeScript path has upstream dependency/type issues unless its own dependencies are installed and its stricter type errors are resolved. Do not treat Electron as validated yet.

## Manual smoke test checklist

1. Start backend with `uvicorn theriad.api.server:app --reload --port 8000`.
2. Confirm `curl http://localhost:8000/api/health` returns `ok: true`.
3. Build extension with `npm run build` in `apps/theriad-extension/`.
4. Load `apps/theriad-extension/dist` in Chrome extensions.
5. Open a normal webpage.
6. Open Theriad Browser sidepanel.
7. Ask: “Summarize this page.”
8. Confirm response comes from backend and no Google API key is required.
9. Confirm route trace appears if enabled.
10. Ask: “Which model handled that?” and check route metadata.
11. Toggle Browser Tools without Google key; it should show a setup warning.
12. Add Google key if available and verify Browser Tools still uses Gemini Computer Use.
13. Turn Browser Tools off; normal chat should route through Theriad again.

## Known limitations

- The backend provider implementation is still the local deterministic provider shim unless production provider integrations are configured.
- `/api/route` is only a client fallback target; `/api/chat/send` and `/api/execute` are the working backend paths in this repo.
- Browser Tools and Composio are preserved but not deeply redesigned.
- Electron mode is documented but not validated end-to-end.
- The upstream extension build reports npm audit findings; they were present at baseline and did not block build.
