# Theriad

**Cost-first AI inference optimization engine.**

Theriad minimizes LLM inference cost while maintaining acceptable output quality.
It is **not** an AI gateway. It is a routing engine that selects the cheapest model that can do the job.

Customer-hosted Theriad has no mandatory annual or per-token fee. Provider and
compute costs remain customer-paid; optional support, defense assurance, and
implementation work are scoped separately in [the pricing policy](docs/pricing.md).

---

## Core Principles

1. Never spend more computation deciding than executing.
2. Terminate routing as early as possible.
3. Premium models should be the exception, never the default.
4. Every routing decision must be explainable.
5. Every optimization must produce measurable value.
6. Architecture is modular and provider-agnostic.
7. Simplicity beats cleverness.

---

## Local Development (Theriad UI)

One command starts both the FastAPI backend and the Vite frontend:

```bash
# 1. Install root dev tools (concurrently)
npm install

# 2. Install frontend dependencies
cd theriad-ui && npm install && cd ..

# 3. Start everything from the project virtual environment
source .venv/bin/activate
SEARXNG_URL=http://127.0.0.1:8081 npm run dev
```

This launches:

- **FastAPI backend** → http://localhost:8000
- **Vite frontend**   → http://localhost:5173

Open http://localhost:5173 in your browser. The sidebar shows **Backend: Connected** when both services are up.

To start services individually:

```bash
npm run dev:backend    # uvicorn only
npm run dev:frontend   # vite only
```

### Deployment mode and account scope

Local development keeps the CLI-friendly anonymous scope by default. A public
service must set `THERIAD_DEPLOYMENT_MODE=hosted`. In hosted mode, health,
model catalogs, public pricing, authentication, and anonymous landing-page
analytics remain public; private workspace, provider, memory, chat, usage, and
agent APIs return `401` until a valid Theriad bearer session is present.

Registered accounts receive one personal organization with an `owner` role.
For the hosted cutover, set `THERIAD_STORAGE_BACKEND=supabase`,
`THERIAD_HOSTED_STORAGE_REQUIRED=true`, and inject the server-only
`SUPABASE_SERVICE_ROLE_KEY` through the secret manager. The Supabase migration
owns projects, conversations, files, runs, memory, provider metadata, usage,
and audit rows; the request boundary fails closed if Postgres/Storage cannot
be reached. SQLite remains the local/development adapter and a guest's
short-lived staging rows are migrated only when the account is upgraded.
Legacy rows with a NULL owner are never silently assigned to the first hosted
account. Set `THERIAD_SECRET_ENCRYPTION_KEY` from the managed secret store for
a hosted, multi-worker deployment; the local fallback key is only a
single-machine development convenience.

Theriad is currently BYOK-only: users add an OpenRouter API key in Account and
OpenRouter bills that account directly. `/api/pricing/plans` remains as a
provider-usage policy surface, but there is no Theriad checkout, subscription,
included allowance, or platform usage fee. Hosted
terminal/browser execution remains disabled until Browserbase, the separate
container worker, and the reviewed remote task adapter all pass startup checks.
`/api/agent/capabilities` reports that state and blocked attempts are recorded
in the owner-scoped audit log.

Set your OpenRouter API key in **Settings → OpenRouter API Key** — no shell restart needed.

### Live web research

Models do not receive internet access from a prompt alone. Theriad performs
web research server-side, then gives the model the returned evidence. Ordinary
web questions work automatically with a built-in public search fallback; users
do not need to choose a browser, connect personal Chrome, or configure a
search provider. If a local SearXNG instance is available, Theriad prefers it:

```bash
docker compose -f docker-compose.searxng.yml up -d
curl 'http://127.0.0.1:8081/search?q=theriad&format=json'
```

`SEARXNG_URL` is optional. If it is set but unavailable, Theriad automatically
falls back to the built-in search path. For a split production deployment, do
not use `localhost` unless the search service shares the backend's network
namespace. Stop SearXNG with:
`docker compose -f docker-compose.searxng.yml down`.

When the Scrapling fetcher extra is installed, challenged public pages are
automatically retried through Scrapling's stealth/Cloudflare path using a
temporary Theriad-owned browser profile. No Scrapling setting is required. If
the challenge still cannot be cleared, Theriad opens that isolated task browser
for a one-click human handoff or offers API search; it never uses personal
Chrome tabs or cookies.

Theriad enables Headroom context optimization by default at the provider
boundary. It compresses eligible accumulated context while preserving the
latest user turn and fails open if the optimization layer is unavailable. The
deterministic pipeline is the default; set `THERIAD_HEADROOM_KOMPRESS_MODEL=default`
after warming and measuring the optional ML text compressor. Use
`THERIAD_HEADROOM_ENABLED=false` as an emergency bypass.

### Google sign-in and email verification

Theriad can use Supabase Auth for Google sign-in and sends a short-lived email
code before a Google session becomes an app session. Configure the SMTP values
in [`.env.example`](.env.example), copy the Supabase values, and run the
profile-table migration in [`supabase/`](supabase/). The full Google provider
setup, redirect URLs, SMTP requirements, and verification query are documented
in [`supabase/README.md`](supabase/README.md). The browser uses only the
Supabase publishable key; never expose a service-role key in `VITE_*` config.

To enable the private site-traffic dashboard, set
`THERIAD_ANALYTICS_OWNER_USERNAME` in the backend environment to the registered
Theriad username that should be allowed to open the private route
`/#/app/analytics`. It is intentionally not shown in the main site navigation.
As an account-recovery option, you may also set the backend-only
`THERIAD_ANALYTICS_ACCESS_CODE`; any registered account can enter that code on
the analytics page to receive access for seven days. Never put this value in a
`VITE_*` variable or frontend code.
The dashboard measures anonymous first-party visitors, sessions, page views,
and heartbeat-based duration; it does not store IP addresses or chat contents.
The owner view also provides an **Exclude my visits** control that marks the
current browser as opted out of future collection. This is intentionally
browser-based rather than IP-based, so it remains reliable when the owner's
network address changes; visits collected before opting out remain in history.

### Local agent harness

The backend also exposes a unified local task API with guarded host PTYs,
workspace tools, authenticated Chrome-tab pairing, optional DevTools fallback,
and opt-in macOS computer access. Check the truthful runtime manifest at
`GET /api/agent/capabilities`; production browser sessions fail visibly when no
live backend is available and never report mock work as completed.

See [docs/unified-agent-harness.md](docs/unified-agent-harness.md) for setup and API examples.
The guide also explains how to create and store a dedicated OpenRouter key for
Theriad, verify the OpenAI-compatible endpoint, and the remaining protocol work
required before using that endpoint as a Codex model provider.

---

## Quick Start

```bash
pip install theriad

# Route a prompt
theriad route "Write a Python function to reverse a string"
# → Model: poolside/laguna-xs-2.1:free  Confidence: 92%  Reason: code_detection: code-heavy prompt

# Explain why a routing decision was made
theriad explain "Analyze the competitive landscape of cloud providers"

# Show cost stats
theriad stats

# Benchmark routing speed
theriad benchmark "What is the capital of France?" --iterations 20
```

---

## Installation

```bash
# Basic install
pip install theriad

# With dev tools
pip install theriad[dev]
```

Requires Python 3.10+.

---

## Configuration

Create a `theriad.yaml` in your project root or `~/.theriad/config.yaml`:

```yaml
router:
  min_confidence: 0.75          # confidence threshold to accept a routing decision
  fallback_tier: cheap_bulk
  layers:
    - layer0                    # deterministic rules (always on)
    - layer1_features           # numerical features (fast, no model)
    # - layer2_embedding        # local embedding model (optional, requires sentence-transformers)
    # - layer3_classifier       # optional sklearn classifier (last resort)

providers:
  openrouter:
    name: openrouter
    base_url: https://openrouter.ai/api/v1
    api_key: ${OPENROUTER_API_KEY}  # or set directly
  ollama:
    name: ollama
    base_url: http://localhost:11434

logging:
  level: INFO
  format: json    # "json" | "text"

metrics_path: ~/.theriad/metrics.jsonl
```

---

## Routing Pipeline

```
Prompt
  │
  ▼
Layer 0 — Deterministic Rules     (regex, length, content type)  ~0.01ms
  │ low confidence?
  ▼
Layer 1 — Feature Extraction      (token count, code ratio, …)   ~0.5ms
  │ low confidence?
  ▼
Layer 2 — Embedding Router        (local BERT/MiniLM)             ~5ms
  │ low confidence?
  ▼
Layer 3 — Optional Classifier     (sklearn TF-IDF+LR, removable)  ~1ms
  │
  ▼
Execution Engine                  (retry, fallback, timeout)
  │
  ▼
Provider  (OpenRouter / Ollama)
  │
  ▼
Cost Accounting                   (JSONL persistence)
```

Each layer is independently testable and optional. Add/remove layers by editing `RouterConfig.layers`.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `theriad route <prompt>` | Route a prompt, show decision |
| `theriad route --json` | Machine-readable output |
| `theriad explain <prompt>` | Full per-rule explanation table |
| `theriad stats` | Aggregate cost/token metrics |
| `theriad config` | Show active configuration |
| `theriad benchmark <prompt>` | Measure per-layer routing latency |

---

## Python API

```python
import asyncio
from theriad.core.config.loader import load_config
from theriad.core.providers.openrouter import OpenRouterProvider
from theriad.core.providers.interfaces import Message
from theriad.engine.pipeline import TheriadPipeline
from theriad.layers.layer0.layer import build_default_layer
from theriad.layers.layer1_features.layer import FeatureRoutingLayer

cfg = load_config()
pipeline = TheriadPipeline(
    layers=[build_default_layer(), FeatureRoutingLayer()],
    providers={"openrouter": OpenRouterProvider(api_key="...")},
    fallback_model=cfg.router.fallback_model,
    fallback_provider="openrouter",
)

async def main():
    messages = [Message(role="user", content="Write a quicksort in Python")]
    response = await pipeline.execute(messages)
    print(response.content)

asyncio.run(main())
```

---

## Adding a Provider

```python
from theriad.core.providers.interfaces import (
    CompletionRequest, CompletionResponse, ProviderProtocol
)

class MyProvider(ProviderProtocol):
    @property
    def name(self): return "myprovider"

    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        ...

    async def health_check(self) -> bool:
        ...
```

---

## Adding a Routing Rule (Layer 0)

```python
from theriad.core.routing.interfaces import RoutingDecision, RoutingRule

class SqlDetectionRule(RoutingRule):
    @property
    def name(self): return "sql_detection"

    def evaluate(self, prompt: str) -> RoutingDecision | None:
        if "SELECT" in prompt.upper() and "FROM" in prompt.upper():
            return RoutingDecision(
                tier="cheap_code",
                confidence=0.90,
                reason="sql_detection: SQL statement detected",
            )
        return None
```

Register it by passing to `DeterministicLayer(rules=[..., SqlDetectionRule()])`.

---

## Constraints (v1)

The following are intentionally **out of scope** for v1:

- Reinforcement learning
- Automatic threshold tuning
- Fine-tuning
- Prompt optimization
- Semantic caching
- Multi-agent systems

---

## Development

```bash
git clone <repo>
pip install -e ".[dev]"
pytest          # run all tests
ruff check .    # lint
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [docs/architecture.md](theriad/docs/architecture.md).
