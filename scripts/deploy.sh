#!/usr/bin/env bash
#
# Deploy Theriad to the production host.
#
# Usage (on the server, as root):
#     /opt/theriad/scripts/deploy.sh
#     /opt/theriad/scripts/deploy.sh --dry-run     # show what would happen
#
# Replaces the manual sequence:
#     git pull --ff-only
#     cd theriad-ui && npm ci && npm run build
#     systemctl restart theriad
#     nginx -t && systemctl reload nginx
#     curl https://theriad.ai/api/health
#
# That sequence took the site down on 2026-08-01. Three gaps caused it, and each
# is closed below:
#
#   1. The rollback point was captured AFTER the pull, so it pointed at the
#      broken commit. Here it is captured first, before anything changes.
#   2. The health check ran AFTER the nginx reload, so the failure surfaced only
#      once the old process was already gone. Here the backend is verified on
#      127.0.0.1 while the previous state is still recoverable, and nginx is
#      only reloaded after that passes.
#   3. Nothing checked that the systemd unit's ASGI target still exists. The
#      fusion -> theriad rename moved it, uvicorn could not import
#      "fusion.api.server", and the service failed to start. Here the module
#      path is read out of the unit file and imported before any restart.
#
# On any failure after the pull, the working tree is restored to the pre-deploy
# commit and the service is restarted, so a failed deploy leaves the site up.

set -euo pipefail

APP_DIR="${THERIAD_APP_DIR:-/opt/theriad}"
UI_DIR="$APP_DIR/theriad-ui"
VENV="$APP_DIR/.venv"
SERVICE="${THERIAD_SERVICE:-theriad}"
UNIT_FILE="/etc/systemd/system/${SERVICE}.service"
HEALTH_LOCAL="http://127.0.0.1:8000/api/health"
HEALTH_PUBLIC="${THERIAD_HEALTH_URL:-https://theriad.ai/api/health}"
# The price cache is rewritten at runtime, so it is almost always dirty on a
# live host and would abort `git pull --ff-only`. It is a cache, not a record —
# see theriad/accounting/live_prices.py — so discarding local edits is correct.
CACHE_FILES=(
  "theriad/accounting/openrouter_prices.json"
  "fusion/accounting/openrouter_prices.json"
)

DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
warn() { printf '    \033[33m!\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }
run()  { if (( DRY_RUN )); then printf '    [dry-run] %s\n' "$*"; else eval "$@"; fi; }

cd "$APP_DIR" || die "no such directory: $APP_DIR"

# ── 0. Rollback point, captured BEFORE anything changes ──────────────────────
log "Recording rollback point"
ROLLBACK_SHA="$(git rev-parse HEAD)"
ok "current commit ${ROLLBACK_SHA:0:12}"
echo "$ROLLBACK_SHA" > /tmp/theriad-rollback.txt

rollback() {
  warn "deploy failed — rolling back to ${ROLLBACK_SHA:0:12}"
  git checkout -q "$ROLLBACK_SHA" -- . 2>/dev/null || true
  git checkout -q "$ROLLBACK_SHA" 2>/dev/null || true
  systemctl restart "$SERVICE" 2>/dev/null || true
  sleep 2
  if systemctl is-active --quiet "$SERVICE"; then
    warn "rolled back; service is up on the previous commit"
  else
    die "ROLLBACK ALSO FAILED — service is down. Check: journalctl -u $SERVICE -n 50"
  fi
  exit 1
}

# ── 1. Working tree ──────────────────────────────────────────────────────────
log "Checking working tree"
for f in "${CACHE_FILES[@]}"; do
  if [[ -f "$f" ]] && ! git diff --quiet -- "$f" 2>/dev/null; then
    warn "discarding runtime churn in $f (it is a cache)"
    run "git checkout -- '$f'"
  fi
done
if ! git diff --quiet || ! git diff --cached --quiet; then
  git status --short
  die "working tree has other local changes. Commit or stash them, then re-run."
fi
ok "clean"

# ── 2. Pull ──────────────────────────────────────────────────────────────────
log "Pulling"
run "git pull --ff-only"
NEW_SHA="$(git rev-parse HEAD)"
if [[ "$NEW_SHA" == "$ROLLBACK_SHA" ]]; then
  ok "already up to date"
else
  ok "${ROLLBACK_SHA:0:12} -> ${NEW_SHA:0:12}"
  git --no-pager log --oneline "$ROLLBACK_SHA..$NEW_SHA" | sed 's/^/      /'
fi

trap 'rollback' ERR

# ── 3. Python package ────────────────────────────────────────────────────────
# Editable installs read straight from the working tree, so this is usually a
# no-op — but it is what makes a renamed or newly added package importable, and
# skipping it is how a deploy silently keeps serving old code.
log "Syncing Python package"
run "$VENV/bin/pip install -e '.[browser]' --quiet"
if ! "$VENV/bin/python" -c "import opensandbox, playwright.async_api" 2>/dev/null; then
  die "OpenSandbox and Playwright are not importable in $VENV after installing the
    'browser' extra. The deployed browser runtime would be unavailable, so
    nothing has been restarted; the site is still up."
fi
ok "installed"

# ── 3b. Semantic intent classifier ───────────────────────────────────────────
# BrowserIntentDetector's embedding stage needs sentence-transformers. When the
# import fails it does not raise -- it returns "not browser" and the request
# silently falls back to substring keyword matching, so a question phrased
# without a trigger word ("is it going to rain tomorrow?") never reaches the
# web and gets answered from model training data instead. That degradation is
# invisible in the logs at INFO level, so this installs the extra and then
# asserts the import really works rather than trusting the exit code.
log "Syncing the semantic intent classifier"
run "$VENV/bin/pip install -e '.[embeddings]' --quiet"
if ! "$VENV/bin/python" -c "import sentence_transformers" 2>/dev/null; then
  die "sentence-transformers is not importable in $VENV after installing the
    'embeddings' extra. Browser-intent classification would silently degrade to
    keyword matching, so this deploy is stopping instead of shipping it.
    Nothing has been restarted; the site is still up."
fi
ok "package importable"

# The ~90 MB model is fetched on FIRST USE, not at install time, and that
# fetch is not covered by the ImportError fallback above -- it raises out of
# the request instead. Download it here so a blocked or failing fetch stops a
# deploy rather than surfacing on a user's first question.
#
# HF caches per-user, and this script runs as root while the service may not,
# so the cache is pinned to an explicit shared path. The systemd unit MUST set
# HF_HOME to the same directory or the service re-downloads it at runtime and
# this warm-up buys nothing.
HF_CACHE="${THERIAD_HF_HOME:-$APP_DIR/.cache/huggingface}"
log "Warming the intent model into $HF_CACHE"
run "mkdir -p '$HF_CACHE'"
if ! HF_HOME="$HF_CACHE" "$VENV/bin/python" -c \
    "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')" \
    2>/dev/null; then
  die "could not download the 'all-MiniLM-L6-v2' intent model into $HF_CACHE.
    The host may not reach huggingface.co. Browser-intent classification would
    raise on first use, so this deploy is stopping.
    Nothing has been restarted; the site is still up."
fi
if ! systemctl show "$SERVICE" -p Environment --value 2>/dev/null | grep -q "HF_HOME=$HF_CACHE"; then
  warn "the $SERVICE unit does not set HF_HOME=$HF_CACHE — the service will not
    reuse the model just warmed here. Add it to the unit's Environment."
fi
ok "intent model present"

# ── 4. Verify the ASGI target the unit file actually points at ───────────────
# This is the check whose absence caused the outage: the unit named a module
# that no longer existed, and nothing noticed until uvicorn failed to boot.
log "Verifying the service's ASGI target"
ASGI_TARGET="$(grep -oP '(?<=uvicorn )\S+(?=:app)' "$UNIT_FILE" || true)"
[[ -n "$ASGI_TARGET" ]] || die "could not parse the ASGI module from $UNIT_FILE"
ok "unit points at ${ASGI_TARGET}:app"
if ! "$VENV/bin/python" -c "import ${ASGI_TARGET}" 2>/dev/null; then
  die "$UNIT_FILE names '${ASGI_TARGET}', which this venv cannot import.
    The module was probably renamed. Fix with:
      sed -i 's|uvicorn ${ASGI_TARGET}:app|uvicorn <new.module.path>:app|' $UNIT_FILE
      systemctl daemon-reload
    Nothing has been restarted; the site is still up."
fi
ok "imports cleanly"

# ── 4b. Hosted configuration preflight ───────────────────────────────────────
# systemd owns the production environment, so inspect only the deployment-mode
# name here; never print the full Environment property (it can contain keys).
HOSTED_SERVICE=0
if systemctl show "$SERVICE" -p Environment --value 2>/dev/null | grep -q 'THERIAD_DEPLOYMENT_MODE=hosted'; then
  HOSTED_SERVICE=1
fi
if (( HOSTED_SERVICE )); then
  log "Checking hosted dependencies"
  PREFLIGHT_FLAGS="--expect-hosted"
  [[ "${THERIAD_DEPLOY_STRICT_AGENTS:-0}" == "1" ]] && PREFLIGHT_FLAGS="$PREFLIGHT_FLAGS --strict-agents"
  run "$VENV/bin/python '$APP_DIR/scripts/hosted_preflight.py' $PREFLIGHT_FLAGS"
  ok "hosted preflight passed"
fi

# ── 4c. Chat browsing sandbox (optional) ─────────────────────────────────────
# Web search runs through OpenRouter and needs nothing here. The browser tools
# are offered only when THERIAD_HOSTED_WEB_BROWSING=1 AND OpenSandbox reports
# healthy at request time, so an unhealthy sandbox degrades to search + page
# reads instead of failing chat. Report it; never block the deploy on it.
if systemctl show "$SERVICE" -p Environment --value 2>/dev/null | grep -q 'THERIAD_HOSTED_WEB_BROWSING=1' \
   || grep -qs '^THERIAD_HOSTED_WEB_BROWSING=1' /etc/theriad/opensandbox.env; then
  log "Checking the OpenSandbox browser runtime"
  if ! systemctl is-active --quiet opensandbox; then
    warn "opensandbox.service is not active — chat browsing will be unavailable.
    See infra/opensandbox/ for the unit, config, and egress firewall."
  elif [[ "$(curl -sS --max-time 5 http://127.0.0.1:8080/health 2>/dev/null)" == *healthy* ]]; then
    iptables -C INPUT -i br-theriad-sbx -j DROP 2>/dev/null \
      || warn "sandbox egress firewall is missing; run infra/opensandbox/sandbox-firewall.sh"
    ok "OpenSandbox healthy on 127.0.0.1:8080"
  else
    warn "OpenSandbox is running but not healthy — chat browsing will be unavailable."
  fi
fi

# ── 5. Frontend ──────────────────────────────────────────────────────────────
log "Building the frontend"
run "cd '$UI_DIR' && npm ci --silent && npm run build"
cd "$APP_DIR"
ok "built"

# ── 6. Restart, and verify BEFORE touching nginx ─────────────────────────────
log "Restarting $SERVICE"
run "systemctl restart '$SERVICE'"
if (( ! DRY_RUN )); then
  for i in $(seq 1 20); do
    sleep 1
    code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 "$HEALTH_LOCAL" 2>/dev/null || echo 000)"
    [[ "$code" == "200" ]] && break
    systemctl is-active --quiet "$SERVICE" || {
      warn "service died on start:"
      journalctl -u "$SERVICE" -n 20 --no-pager | sed 's/^/      /'
      false
    }
  done
  [[ "${code:-000}" == "200" ]] || {
    warn "backend never became healthy (last status: ${code:-000})"
    journalctl -u "$SERVICE" -n 20 --no-pager | sed 's/^/      /'
    false
  }
  ok "backend healthy on 127.0.0.1:8000 after ${i}s"

  # Health alone is not enough: a local-mode process can be healthy while
  # exposing every private API to anonymous callers. Production deploys must
  # prove that the explicit hosted auth boundary is active before nginx moves
  # traffic to the new process.
  private_code="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 3 \
    http://127.0.0.1:8000/api/projects 2>/dev/null || echo 000)"
  [[ "$private_code" == "401" ]] || {
    die "private API returned $private_code without auth; set THERIAD_DEPLOYMENT_MODE=hosted"
  }
  ok "hosted auth boundary active (anonymous private API -> 401)"
fi

# ── 7. nginx, only now that the backend is confirmed good ────────────────────
log "Reloading nginx"
run "nginx -t"
run "systemctl reload nginx"
ok "reloaded"

# ── 8. Public check ──────────────────────────────────────────────────────────
log "Public health check"
if (( ! DRY_RUN )); then
  public="$(curl -sS -o /dev/null -w '%{http_code}' --max-time 10 "$HEALTH_PUBLIC" || echo 000)"
  [[ "$public" == "200" ]] || die "public health check returned $public (backend is healthy locally, so look at nginx)"
  ok "$HEALTH_PUBLIC -> 200"

  # Verify that the public HTML references a current Vite fingerprint and that
  # at least the entry bundle bytes match the build just produced.
  PUBLIC_ROOT="${HEALTH_PUBLIC%/api/health}"
  public_html="$(curl -fsS --max-time 10 "$PUBLIC_ROOT/" || true)"
  grep -Eq 'assets/[^" ]+\.(js|css)' <<<"$public_html" || die "public HTML has no asset fingerprint"
  # Vite does not always prune historical fingerprinted assets. Resolve the
  # current entrypoint from the index produced by this build instead of taking
  # an arbitrary stale index-*.js file from dist/assets.
  asset_name="$(grep -oE 'assets/index-[^" ]+\.js' "$UI_DIR/dist/index.html" \
    | head -n 1 | sed 's|^assets/||')"
  [[ -n "$asset_name" ]] || die "frontend build has no index fingerprint"
  asset_path="$UI_DIR/dist/assets/$asset_name"
  [[ -f "$asset_path" ]] || die "frontend build references missing asset $asset_name"
  grep -Fq "$asset_name" <<<"$public_html" || die "public HTML does not reference $asset_name"
  local_asset_sha="$(sha256sum "$asset_path" | awk '{print $1}')"
  remote_asset_sha="$(curl -fsS --max-time 10 "$PUBLIC_ROOT/assets/$asset_name" | sha256sum | awk '{print $1}')"
  [[ "$local_asset_sha" == "$remote_asset_sha" ]] || die "public asset fingerprint differs from the new build"
  ok "public HTML and $asset_name match the built asset"
fi

trap - ERR
log "Deployed ${NEW_SHA:0:12}"
