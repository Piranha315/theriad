"""Real local OpenSandbox -> Chromium/CDP -> Playwright smoke test.

This script intentionally uses Theriad's production runtime object rather than
calling the OpenSandbox SDK directly, so it validates the same factory,
health-gate, CDP connection, observation, and cleanup path used by the app.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

from theriad.browser.opensandbox_runtime import OpenSandboxBrowserRuntime


async def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    load_dotenv(project_root / ".env")

    runtime = OpenSandboxBrowserRuntime()
    health = await runtime.availability()
    if not health.get("available"):
        raise RuntimeError(f"OpenSandbox health gate failed: {health}")

    session_id = "theriad-local-opensandbox-e2e"
    started = await runtime.start_session(
        session_id=session_id,
        user_prompt="Open a public webpage for a local E2E smoke test.",
        start_url="https://www.iana.org/domains/example",
        allowed_domains=["iana.org"],
    )
    if not started.get("ok"):
        raise RuntimeError(f"OpenSandbox session creation failed: {started}")

    try:
        navigated = await runtime.navigate(session_id, "https://www.iana.org/domains/example")
        if not navigated.get("ok"):
            raise RuntimeError(f"OpenSandbox navigation failed: {navigated}")
        observed = await runtime.observe(session_id, screenshot=False)
        if not observed.get("ok") or not observed.get("page_title"):
            raise RuntimeError(f"OpenSandbox observation failed: {observed}")

        print(
            json.dumps(
                {
                    "health": health,
                    "sandbox_id": started.get("sandbox_id"),
                    "session_id": session_id,
                    "url": observed.get("current_url"),
                    "title": observed.get("page_title"),
                    "visible_text": (observed.get("visible_text_summary") or "")[:500],
                    "browser_backend": observed.get("browser_backend"),
                },
                indent=2,
                sort_keys=True,
            )
        )
    finally:
        stopped = await runtime.stop_session(session_id)
        if stopped.get("status") != "stopped":
            raise RuntimeError(f"OpenSandbox cleanup failed: {stopped}")
        print(json.dumps({"destroyed": True, "stop_status": stopped.get("status")}))


if __name__ == "__main__":
    asyncio.run(main())
