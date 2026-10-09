# OpenSandbox for chat browsing (production)

Chat web search needs nothing on this host: the model calls OpenRouter's
`openrouter:web_search` server tool, billed to the user's OpenRouter key.
This directory adds the optional browser tools (`browser_open`, `browser_click`,
…), which run each chat turn's browsing in an isolated, short-lived Chromium
container.

## Install (as root, once)

Prerequisites: Docker, iptables, and `uv` on the host.

```bash
sudo /opt/theriad/infra/opensandbox/install.sh --restart
```

The script is idempotent. It pulls the images, generates the config and API
key, installs `opensandbox.service` (which applies `sandbox-firewall.sh` before
every start), checks the firewall rules, and runs a real sandbox smoke test.
Only if all of that passes does it write `/etc/theriad/opensandbox.env`
(mode 0600, includes `THERIAD_HOSTED_WEB_BROWSING=1`) and attach it to the
`theriad` unit through a drop-in. To turn browsing off again, remove
`/etc/systemd/system/theriad.service.d/opensandbox.conf` and restart `theriad`.

Also restrict the provider firewall / security group to 22, 80, and 443:
the sandbox port range must never be reachable from outside.

## Safety boundary

- `sandbox-firewall.sh` (run by the unit before start) puts sandboxes on a
  dedicated bridge that can reach only public IPs (plus DNS): private ranges,
  cloud metadata, and the host's own ports are dropped.
- OpenSandbox 0.2.3 publishes every sandbox's execd and Chromium ports on
  0.0.0.0 without authentication on the Docker runtime, and Docker's port
  rules bypass ufw. The script drops any new connection into the sandbox
  network that does not originate on the host, including via docker-proxy.
- Verified on Docker 27 / iptables-nft: sandboxes reach public sites; private
  VPC services, the host API, and cloud metadata are blocked; outside clients
  cannot reach sandbox ports; Theriad on the host still can.
- Chat browsing is read-only: actions the browser policy marks as needing
  approval (submit, sign-in, purchase, download) are denied to the model.
- One sandbox per chat turn, one per user at a time, at most
  `THERIAD_OPENSANDBOX_MAX_SESSIONS` overall; 1 CPU / 1.5 GiB each by default.
- If the sandbox is down, chat keeps working with search and page reads.

`opensandbox-server` is pinned to 0.2.3: 1.1.0 fails to import
(`opensandbox_server.services.fast_sandbox.generated` is missing from the wheel).
