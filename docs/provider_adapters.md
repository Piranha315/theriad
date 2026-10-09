# Adding an inference provider

Theriad separates three decisions:

1. The capability router selects a tier.
2. The model registry selects a model for the requested speed/quality profile.
3. The provider adapter executes the request. Aggregators may also select an upstream endpoint.

## Fast path: OpenAI-compatible providers

The current live settings surface exposes OpenRouter, OpenAI, and Anthropic.
Groq and DeepInfra adapters remain only for benchmark/research compatibility;
they are not selectable assistant providers or accepted by the live settings
API.

Groq and DeepInfra use the generic `OpenAICompatibleProvider`. To add another
compatible provider, register its API base URL and environment variable in
`theriad/core/providers/factory.py`, add its provider name to
`theriad/settings/secrets.py`, then define its tier-to-model mapping in
`theriad/core/providers/runtime.py`. Add current per-token prices to
`theriad/accounting/pricing.py` and test one completion plus `/models` health.

For a provider with a different wire format, implement `ProviderProtocol`, then
register the adapter during application startup:

```python
from theriad.core.providers import register_provider_adapter
from my_package.providers.groq import GroqProvider

register_provider_adapter(
    "groq",
    lambda **kwargs: GroqProvider(**kwargs),
    env_key="GROQ_API_KEY",
    aliases=("groq-cloud",),
)
```

After registration, `build_provider("groq")` resolves `GROQ_API_KEY` and constructs the adapter. Pass an explicit `api_key` when credentials are managed outside the environment.

Add the provider's tier-to-model mapping in `theriad/core/providers/runtime.py` only when it is a user-selectable direct provider. OpenRouter upstream providers do not need direct adapters: configure their endpoint order in `theriad/core/providers/routing.py`.

Refresh current OpenRouter prices explicitly with:

```bash
python scripts/refresh_openrouter_prices.py
```

The application reads the checked-in snapshot and never makes a catalog request during startup.

## Credentials and OAuth

Do not put provider keys in the browser. Theriad supports two practical modes:

- Platform-managed: set a supported provider secret on the server once. Users
  do not enter a provider key. Benchmark-only adapters may still use their
  legacy environment variables outside the live app.
- Bring your own key: a user saves a key once in Settings. The local desktop
  anonymous scope stores it in `~/.theriad/secrets.json`; an authenticated
  scope stores it in a user-specific file under `~/.theriad/secrets-users/`.
  Provider values are encrypted individually and files use `0600` permissions.
  Hosted deployments should inject `THERIAD_SECRET_ENCRYPTION_KEY` from a
  managed secret store so multiple workers share the encryption boundary.

OAuth is only possible when the provider offers a delegated authorization flow.
OpenRouter does, so Theriad includes its S256 PKCE one-click connection and saves
the resulting user-controlled key on the backend. Set `THERIAD_PUBLIC_URL` on a
hosted deployment so OpenRouter receives the public callback origin. The local
callback works on any localhost port.

In a multi-worker hosted deployment, move OpenRouter's short-lived PKCE verifier
records to a shared TTL store such as Redis.

## Provider checklist

1. Add credentials and masked status support.
2. Register the adapter and its health check.
3. Map capability tiers to exact provider model IDs.
4. Add live input, cached-input, and output pricing when available.
5. Add unit tests for payload, errors, token usage, and provider isolation.
6. Add the connection card to `theriad-ui/src/ProviderSettings.jsx`.
