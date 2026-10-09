# Local OpenSandbox browser runtime

This is a development-only setup for running Theriad's isolated browser lane
on macOS. The supported local path is a Docker-backed OpenSandbox server
(`uvx opensandbox-server`) creating one task-owned `opensandbox/chrome`
container per browser session. Theriad connects to that container's Chromium
over CDP with Playwright; it never launches or reuses the host Chrome profile.

## First-time setup

From the repository root:

```bash
uv pip install -e ".[dev,browser]"
open -a Docker
docker info

mkdir -p work/opensandbox
uvx opensandbox-server init-config work/opensandbox/sandbox.toml --example docker
```

The `browser` extra installs the OpenSandbox Python SDK and Playwright's CDP
client. Chromium itself is supplied by `opensandbox/chrome:latest`; a host
`playwright install chromium` is not part of this path. `uvx` installs/runs
the separate OpenSandbox server command.

Edit `work/opensandbox/sandbox.toml` so the local server uses loopback, the
Docker bridge runtime, a server API key, and direct host-mapped endpoints:

```toml
[server]
host = "127.0.0.1"
port = 8080
api_key = "theriad-local-opensandbox-development-key"

[proxy]
resolve_internal = false

[runtime]
type = "docker"
execd_image = "opensandbox/execd:v1.0.22"

[docker]
network_mode = "bridge"
drop_capabilities = ["AUDIT_WRITE", "MKNOD", "NET_ADMIN", "NET_RAW", "SYS_ADMIN", "SYS_MODULE", "SYS_PTRACE", "SYS_TIME", "SYS_TTY_CONFIG"]
no_new_privileges = true
pids_limit = 4096

[ingress]
mode = "direct"
```

The checked-in `.env.example` and this checkout's ignored `.env` keep SearXNG
on `127.0.0.1:8081`, avoiding a port collision with OpenSandbox on 8080.
The `work/opensandbox/` configuration is local/ignored; do not commit its
development API key.

An exported shell variable takes precedence over `.env`. Before starting the
Theriad backend, check for and clear a stale value from an earlier setup:

```bash
env | grep '^SEARXNG_URL=' || true
unset SEARXNG_URL
SEARXNG_URL=http://127.0.0.1:8081 npm run dev
```

Do not point `SEARXNG_URL` at `127.0.0.1:8080`: that is the authenticated
OpenSandbox control API, not the search service, and it returns `401` to
SearXNG requests.

## Start and verify

```bash
open -a Docker
docker info
docker compose -f docker-compose.searxng.yml up -d
uvx opensandbox-server --config "$PWD/work/opensandbox/sandbox.toml"
```

The OpenSandbox command is intentionally foregrounded. If it reports
`address already in use` on `127.0.0.1:8080`, an OpenSandbox server is already
running; do not start a second one—check its health and reuse it.

In another terminal, verify the health gate and Docker state:

```bash
curl --fail --silent http://127.0.0.1:8080/health
docker ps
```

The expected health response contains `"status":"healthy"`. Keep the server
terminal open while running Theriad. The repository `.env` supplies the
matching `THERIAD_OPENSANDBOX_*` values; for a clean shell, export them
explicitly instead.

```bash
export THERIAD_OPENSANDBOX_DOMAIN=127.0.0.1:8080
export THERIAD_OPENSANDBOX_PROTOCOL=http
export THERIAD_OPENSANDBOX_API_KEY=theriad-local-opensandbox-development-key
export THERIAD_OPENSANDBOX_BROWSER_IMAGE=opensandbox/chrome:latest
export THERIAD_OPENSANDBOX_TIMEOUT_SECONDS=300
export THERIAD_OPENSANDBOX_REQUEST_TIMEOUT_SECONDS=30
export THERIAD_OPENSANDBOX_CDP_READY_TIMEOUT_SECONDS=20
export THERIAD_OPENSANDBOX_USE_SERVER_PROXY=false
```

Start the Theriad API from the same virtual environment so the backend, not a
global Python installation, loads the OpenSandbox SDK and Playwright:

```bash
source .venv/bin/activate
SEARXNG_URL=http://127.0.0.1:8081 npm run dev
```

If the shell has an older exported `SEARXNG_URL`, the explicit assignment above
must win; `env | grep '^SEARXNG_URL='` should never show port `8080` for search.

The runtime waits up to the CDP-ready timeout for Chromium's DevTools
endpoint and retries transient 502/503/504 responses. OpenSandbox's execd
health check alone does not guarantee that Chromium is ready yet.

If Docker Desktop is stuck in `starting` and its logs report a permission
error resizing `Docker.raw`, repair only that file's ownership after Docker
Desktop has stopped, then start it again:

```bash
sudo chown "$(id -un):$(id -gn)" "$HOME/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
sudo chmod u+rw "$HOME/Library/Containers/com.docker.docker/Data/vms/0/data/Docker.raw"
docker desktop start --detach
docker info
```

This is a host-level Docker Desktop repair, not a Theriad or OpenSandbox
configuration change. A factory reset is a last resort because it discards
Docker's local images, containers, volumes, and settings; after a reset, the
OpenSandbox and SearXNG images must be downloaded again.

If the next startup error is `service jfs failed: running: invalid database`,
the ownership issue has been resolved and Docker Desktop is now failing in its
internal VM/database layer. Run `docker desktop diagnose` locally (do not use
an upload option), save the diagnostic ID, and back up `Docker.raw` before
trying reset or reinstall. Docker documents that exact file as the Mac VM
backup/restore artifact. A factory reset is a last resort because it discards
Docker's local images, containers, volumes, and settings.

## E2E validation

```bash
source .venv/bin/activate
python scripts/opensandbox_e2e.py
```

The validation creates a real OpenSandbox sandbox, connects to Chromium over
CDP with Playwright, visits
`https://www.iana.org/domains/example` (which resolves to IANA's Example
Domains page), retrieves the title and visible text, and destroys the
sandbox in a `finally` block. It must report the sandbox ID and
`destroyed=true`; passing SDK/unit tests alone is not E2E proof.

The API route path was also exercised locally: `/api/browser/session/start`
created the OpenSandbox-backed task, `/api/browser/session/{id}` returned the
IANA observation, `/api/browser/session/stop` destroyed it, and
`/api/browser/session/{id}/use-api-search` returned SearXNG results.

## Shutdown and resource measurement

Stop the foreground server with `Ctrl-C`. Any active Theriad sessions should
be stopped by their owning request; inspect and clean leftovers with:

```bash
docker ps --filter 'label=opensandbox.io/id'
docker ps --filter 'label=theriad.component=browser'
docker stats --no-stream
```

For one live browser sandbox, record the container-specific CPU and RAM while
the E2E session is paused:

```bash
docker stats --no-stream --format 'table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}'
```

Prefer the owning Theriad stop/cleanup endpoint. For an orphan, first inspect
the exact container and then remove only that named `sandbox-*` container:

```bash
docker ps --filter 'label=theriad.component=browser'
docker rm -f <explicit-sandbox-container-name>
docker ps --filter 'label=opensandbox.io/id'
```

Do not remove the Docker VM disk as part of routine shutdown.

## Isolation boundary

The server binds only to loopback. Sandboxes use Docker bridge networking,
drop the dangerous capabilities listed above, enable `no_new_privileges`, and
have Docker's default seccomp profile and a PID limit. Host bind mounts are
rejected because `allowed_host_paths` is empty. `resolve_internal = false` is
an endpoint-routing setting needed for the macOS server process; it does not
switch to Docker host networking or expose the host Chrome profile.

With `[ingress].mode = "direct"`, Docker publishes each sandbox's mapped
endpoint on a random host port (the observed bindings were `0.0.0.0`), even
though the OpenSandbox control server itself is loopback-only. This is why
this configuration is local/development-only; do not expose the Mac to an
untrusted network while a sandbox is live.

On this Apple-silicon Mac, the server logs also report that the optional
OpenSandbox secure-runtime launcher archive for `linux/arm64` is unavailable.
The Docker isolation settings above are active, but the enhanced secure
runtime is not; production use needs a supported secure-runtime deployment
and a reviewed ingress/network policy.

Observed local sample on 2026-09-17 for one live IANA browser sandbox:

- container image: `opensandbox/chrome:latest`
- CPU: `0.00%` at an idle observation point
- memory: `175.5MiB / 2GiB` (`8.57%`)
- Docker Desktop VM allocation: approximately `7.75GiB`, 10 virtual CPUs

The container sample is workload- and timing-dependent; it is not a capacity
guarantee. Re-measure with `docker stats --no-stream` when changing pages,
parallelism, or the image.
