"""Start the app and verify a natural-language housing task uses a real URL."""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROMPT = "Find me a decent house in the Philly area"
BACKEND_PORT = int(os.environ.get("THERIAD_E2E_BACKEND_PORT", "18000"))
FRONTEND_PORT = int(os.environ.get("THERIAD_E2E_FRONTEND_PORT", "15175"))
BACKEND_ORIGIN = f"http://127.0.0.1:{BACKEND_PORT}"
FRONTEND = f"http://127.0.0.1:{FRONTEND_PORT}"
BACKEND = f"{BACKEND_ORIGIN}/api/health"


def wait_for(url: str, timeout: float = 25.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=1):
                return
        except Exception:
            time.sleep(0.25)
    raise AssertionError(f"Timed out waiting for {url}")


async def run() -> None:
    from playwright.async_api import async_playwright  # noqa: F401  (checked in main)

    env = {
        **os.environ,
        "THERIAD_SEARCH_PROVIDER": "google",
        "VITE_API_PROXY_TARGET": BACKEND_ORIGIN,
        # Keep the bounded local smoke deterministic and free of paid/provider
        # calls. The browser runtime falls back to its local search plan.
        "PYTEST_CURRENT_TEST": "agent_e2e_smoke",
    }
    processes = [
        subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "theriad.api.server:app",
             "--host", "127.0.0.1", "--port", str(BACKEND_PORT)],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        ),
        subprocess.Popen(
            ["npm", "--prefix", "theriad-ui", "run", "dev", "--",
             "--host", "127.0.0.1", "--port", str(FRONTEND_PORT), "--strictPort"],
            cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True,
        ),
    ]
    try:
        await asyncio.to_thread(wait_for, BACKEND)
        await asyncio.to_thread(wait_for, FRONTEND)
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True)
            page = await browser.new_page(viewport={"width": 1280, "height": 900})
            await page.goto(FRONTEND, wait_until="domcontentloaded")
            # Route-only keeps this smoke local and deterministic; it still
            # exercises real Playwright observation/actions in the agent leg.
            await page.get_by_text("Route only", exact=True).click()
            composer = page.get_by_test_id("chat-composer")
            await composer.fill(PROMPT)
            await composer.press("Enter")
            card = page.get_by_test_id("agent-run-card")
            await card.wait_for(state="visible", timeout=30_000)
            current_url = await card.get_attribute("data-current-url") or ""
            assert "example.com" not in current_url, (
                f"placeholder URL leaked into browser run: {current_url}"
            )
            assert any(domain in current_url for domain in (
                "bing.com", "google.com", "zillow.com",
                "redfin.com", "realtor.com", "apartments.com", "craigslist.org",
            )), f"expected real search/housing URL, got {current_url}"
            await card.get_by_text("Show details ▼").click()
            text = await card.inner_text()
            assert "navigate" in text and "executed" in text, text
            assert "Example Domain" not in text, text
            await browser.close()
            print(f"PASS: Local Chromium resolved housing task to {current_url}")
    finally:
        for process in processes:
            process.terminate()
        for process in processes:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        for name, process in zip(("backend", "frontend"), processes):
            output = process.stdout.read() if process.stdout else ""
            if process.returncode not in (0, -15):
                print(f"{name} log:\n{output}", file=sys.stderr)


def main() -> int:
    # Playwright ships in the optional "browser" extra. A missing optional
    # dependency is a skip, not a failure: reporting it as a traceback made
    # the bounded self-test look broken on an otherwise healthy checkout.
    try:
        import playwright.async_api  # noqa: F401
    except ImportError:
        print(
            "SKIP: browser e2e smoke needs the optional 'browser' extra.\n"
            "      Install it with:  pip install -e '.[browser]' "
            "&& playwright install chromium",
            file=sys.stderr,
        )
        return 0
    asyncio.run(run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
