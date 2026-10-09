#!/usr/bin/env bash
#
# Install and verify OpenSandbox for Theriad chat browsing on the production
# host. Idempotent; run as root from the app checkout:
#
#     sudo /opt/theriad/infra/opensandbox/install.sh            # install + verify
#     sudo /opt/theriad/infra/opensandbox/install.sh --restart  # ...and restart theriad
#
# Browsing is switched on (THERIAD_HOSTED_WEB_BROWSING=1) only after a real
# sandbox has started, loaded a public page, and been torn down behind the
# egress firewall. Any failure leaves chat on search + page reads.

set -euo pipefail

APP_DIR="${THERIAD_APP_DIR:-/opt/theriad}"
VENV="$APP_DIR/.venv"
SERVICE="${THERIAD_SERVICE:-theriad}"
CONF=/etc/theriad/opensandbox.toml
ENV_FILE=/etc/theriad/opensandbox.env
UNIT=/etc/systemd/system/opensandbox.service
DROPIN_DIR="/etc/systemd/system/${SERVICE}.service.d"
BRIDGE=br-theriad-sbx
RESTART=0
[[ "${1:-}" == "--restart" ]] && RESTART=1

log()  { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
ok()   { printf '    \033[32m✓\033[0m %s\n' "$*"; }
die()  { printf '\n\033[31m✗ %s\033[0m\n' "$*" >&2; exit 1; }

[[ $EUID -eq 0 ]] || die "run as root"
command -v docker >/dev/null && docker info >/dev/null 2>&1 || die "Docker is not running"
command -v iptables >/dev/null || die "iptables is required for the sandbox egress firewall"
UVX="$(command -v uvx || true)"
[[ -n "$UVX" ]] || die "uvx not found. Install uv first: https://docs.astral.sh/uv/getting-started/installation/"
[[ -x "$VENV/bin/python" ]] || die "no Theriad venv at $VENV"

log "Pulling sandbox images"
docker pull -q opensandbox/chrome:latest >/dev/null
docker pull -q opensandbox/execd:v1.0.22 >/dev/null
ok "images present"

log "Writing $CONF"
install -d -m 0700 /etc/theriad /var/lib/opensandbox
if [[ -f "$CONF" ]]; then
  API_KEY="$(sed -n 's/^api_key *= *"\(.*\)"/\1/p' "$CONF")"
  ok "kept existing config"
else
  API_KEY="$(openssl rand -hex 32)"
  sed "s/REPLACE_WITH_RANDOM_SECRET/$API_KEY/" "$APP_DIR/infra/opensandbox/sandbox.toml.example" > "$CONF"
  chmod 0600 "$CONF"
  ok "generated config with a new API key"
fi
[[ -n "$API_KEY" && "$API_KEY" != REPLACE_WITH_RANDOM_SECRET ]] || die "api_key missing in $CONF"

log "Installing opensandbox.service"
sed "s|/usr/local/bin/uvx|$UVX|" "$APP_DIR/infra/opensandbox/opensandbox.service" > "$UNIT"
systemctl daemon-reload
systemctl enable --now opensandbox >/dev/null
systemctl restart opensandbox
for i in $(seq 1 90); do
  [[ "$(curl -s --max-time 2 http://127.0.0.1:8080/health || true)" == *healthy* ]] && break
  sleep 1
done
[[ "$(curl -s --max-time 2 http://127.0.0.1:8080/health || true)" == *healthy* ]] \
  || die "OpenSandbox did not become healthy. Check: journalctl -u opensandbox -n 50"
ok "healthy on 127.0.0.1:8080 after ${i}s"

log "Checking the egress firewall"
iptables -C INPUT -i "$BRIDGE" -j DROP 2>/dev/null || die "sandbox INPUT rule missing"
iptables -C DOCKER-USER -o "$BRIDGE" -m conntrack --ctstate NEW -m addrtype ! --src-type LOCAL -j DROP 2>/dev/null \
  || die "inbound protection for published sandbox ports is missing"
iptables -C DOCKER-USER -i "$BRIDGE" -d 169.254.0.0/16 -j DROP 2>/dev/null || die "metadata block missing"
ok "firewall rules present"

cat > "$ENV_FILE" <<EOF
THERIAD_OPENSANDBOX_DOMAIN=127.0.0.1:8080
THERIAD_OPENSANDBOX_PROTOCOL=http
THERIAD_OPENSANDBOX_API_KEY=$API_KEY
THERIAD_OPENSANDBOX_BROWSER_IMAGE=opensandbox/chrome:latest
THERIAD_OPENSANDBOX_USE_SERVER_PROXY=false
THERIAD_OPENSANDBOX_TIMEOUT_SECONDS=300
EOF
chmod 0600 "$ENV_FILE"

log "Smoke test: one real sandbox browser session"
set -a; . "$ENV_FILE"; set +a
if ! "$VENV/bin/python" - <<'PY'
import asyncio, sys
from theriad.browser.opensandbox_runtime import get_opensandbox_browser_runtime

async def main() -> int:
    runtime = get_opensandbox_browser_runtime()
    snap = await runtime.start_session(start_url="https://www.wikipedia.org/", user_prompt="smoke test", owner_user_id="install-smoke")
    sid = snap.get("session_id")
    try:
        if not snap.get("ok", True) or not sid:
            print("    start failed:", snap.get("error")); return 1
        page = await runtime.observe(sid, screenshot=False, owner_user_id="install-smoke")
        title = page.get("page_title") or ""
        print("    loaded:", title[:80])
        return 0 if "wikipedia" in title.lower() else 1
    finally:
        if sid:
            await runtime.stop_session(sid, owner_user_id="install-smoke")

sys.exit(asyncio.run(main()))
PY
then
  die "sandbox smoke test failed; browsing stays disabled. Check: journalctl -u opensandbox -n 80"
fi
ok "sandbox started, loaded a public page, and was removed"

log "Enabling chat browsing for $SERVICE"
echo "THERIAD_HOSTED_WEB_BROWSING=1" >> "$ENV_FILE"
install -d "$DROPIN_DIR"
cat > "$DROPIN_DIR/opensandbox.conf" <<EOF
[Service]
EnvironmentFile=$ENV_FILE
EOF
systemctl daemon-reload
if (( RESTART )); then
  systemctl restart "$SERVICE"
  for i in $(seq 1 20); do
    [[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 3 http://127.0.0.1:8000/api/health || true)" == 200 ]] && break
    sleep 1
  done
  ok "$SERVICE restarted"
else
  ok "configured; applies on the next '$SERVICE' restart or deploy"
fi
