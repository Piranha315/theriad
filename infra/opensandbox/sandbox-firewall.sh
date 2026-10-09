#!/usr/bin/env bash
#
# Create the dedicated Docker network for OpenSandbox browser containers and
# restrict it to the public internet. Idempotent; run as root (the systemd
# unit runs it as ExecStartPre).
#
# Sandboxed pages are untrusted. Without these rules a page could make the
# browser reach cloud metadata (169.254.169.254), private services on the VPC,
# or this host's own ports (Theriad API, OpenSandbox control API, databases).
# OpenSandbox's egress sidecar filters by domain name only, which does not
# stop IP-literal URLs, so the boundary is enforced here at the kernel.

set -euo pipefail

NET="${THERIAD_SANDBOX_NETWORK:-theriad-sandbox}"
SUBNET="${THERIAD_SANDBOX_SUBNET:-172.31.240.0/24}"
BRIDGE="${THERIAD_SANDBOX_BRIDGE:-br-theriad-sbx}"
PORTS="${THERIAD_SANDBOX_PORTS:-40000:40300}"   # must match port_range_* in the config
BLOCKED=(10.0.0.0/8 172.16.0.0/12 192.168.0.0/16 169.254.0.0/16 100.64.0.0/10 127.0.0.0/8 0.0.0.0/8 224.0.0.0/4)

if ! docker network inspect "$NET" >/dev/null 2>&1; then
  docker network create --driver bridge --subnet "$SUBNET" \
    -o "com.docker.network.bridge.name=$BRIDGE" \
    -o "com.docker.network.bridge.enable_icc=false" "$NET" >/dev/null
fi

rule() {  # add an iptables rule once
  local table_chain=("$1" "$2"); shift 2
  iptables -t "${table_chain[0]}" -C "${table_chain[1]}" "$@" 2>/dev/null \
    || iptables -t "${table_chain[0]}" -I "${table_chain[1]}" "$@"
}

# Inbound: OpenSandbox 0.2.3 publishes every sandbox port (execd, Chromium CDP)
# on 0.0.0.0 with no authentication on the Docker runtime, and Docker's DNAT
# rules bypass ufw. Only the host itself may open connections into sandboxes.
rule filter DOCKER-USER -o "$BRIDGE" -m conntrack --ctstate NEW -m addrtype ! --src-type LOCAL -j DROP

# Forwarded traffic leaving the sandbox bridge: drop private/link-local ranges.
for cidr in "${BLOCKED[@]}"; do
  rule filter DOCKER-USER -i "$BRIDGE" -d "$cidr" -j DROP
done
# DNS: Docker's embedded resolver forwards queries from inside the sandbox
# network to the host's upstream resolver, which is usually a private or
# link-local address on cloud VMs. Allow port 53 only; page traffic to those
# ranges stays blocked above.
rule filter DOCKER-USER -i "$BRIDGE" -p udp --dport 53 -j RETURN
rule filter DOCKER-USER -i "$BRIDGE" -p tcp --dport 53 -j RETURN
# Replies to connections the host opened (Theriad -> Chromium CDP) stay allowed.
rule filter DOCKER-USER -i "$BRIDGE" -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN

# Published sandbox ports reached through docker-proxy arrive as INPUT on the
# host; only loopback clients (Theriad itself) may open them.
rule filter INPUT -p tcp --dport "$PORTS" ! -i lo -m conntrack --ctstate NEW -j DROP

# Traffic from sandboxes to the host itself (INPUT): only replies are allowed.
rule filter INPUT -i "$BRIDGE" -j DROP
rule filter INPUT -i "$BRIDGE" -m conntrack --ctstate ESTABLISHED,RELATED -j ACCEPT

# IPv6 is not routed for this network; make sure nothing leaks if it is enabled.
if command -v ip6tables >/dev/null 2>&1; then
  ip6tables -C FORWARD -i "$BRIDGE" -j DROP 2>/dev/null || ip6tables -I FORWARD -i "$BRIDGE" -j DROP
fi

echo "sandbox network $NET ($SUBNET on $BRIDGE) restricted to public egress"
