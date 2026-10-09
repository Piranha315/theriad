#!/usr/bin/env python3
"""Safe hosted deployment preflight.

The command reports configuration names and boolean readiness only. It never
prints environment values, Supabase keys, Stripe keys, provider keys, or
runtime data.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect-hosted", action="store_true")
    parser.add_argument("--strict-agents", action="store_true")
    args = parser.parse_args()

    from theriad.settings.deployment import get_deployment_mode
    from theriad.storage.hosted import (
        HostedStorageUnavailable,
        assert_hosted_storage_ready,
        storage_status,
    )
    from theriad.agents.hosted_runtime import hosted_agent_capability

    failures: list[str] = []
    mode = get_deployment_mode()
    if args.expect_hosted and mode != "hosted":
        failures.append("THERIAD_DEPLOYMENT_MODE=hosted")
    if mode == "hosted":
        if not os.environ.get("THERIAD_SECRET_ENCRYPTION_KEY", "").strip():
            failures.append("THERIAD_SECRET_ENCRYPTION_KEY")
        try:
            assert_hosted_storage_ready(probe=True)
        except HostedStorageUnavailable:
            failures.append("Supabase hosted Postgres/Storage connectivity")
        print(f"hosted storage: {storage_status(probe=False)['configured']} configured")

    print("subscription billing: disabled (BYOK; OpenRouter bills the supplied key)")

    agents = hosted_agent_capability(probe=True)
    print(f"hosted agents: {'ready' if agents['available'] else 'disabled'}")
    if args.strict_agents and not agents["available"]:
        failures.append("Browserbase, isolated runner, and hosted task adapter")

    from theriad.agents.chat_web_access import browsing_status

    browsing = browsing_status(probe=True)
    if browsing["enabled"]:
        state = "ready" if browsing["sandbox_healthy"] else "enabled but OpenSandbox is unhealthy (search + page reads only)"
    else:
        state = "disabled (search + page reads only)"
    print(f"chat web browsing: {state}")

    if failures:
        print("preflight failed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("preflight passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
