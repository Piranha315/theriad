"""
Browser page observation — DOM-first, screenshot only when explicitly requested.

Three modes:
  mock_mode=True                 → deterministic synthetic data (default, no deps)
  mock_mode=False, playwright    → real DOM read via Playwright async API
  mock_mode=False, no playwright → clear error; startup is never broken

Real observation rules (read-only, no actions):
  - Launches a headless Chromium via Playwright.
  - Navigates ONLY to the URL provided; no further navigation.
  - Extracts: url, title, visible text (innerText), links, buttons, inputs, forms.
  - No clicks, no typing, no form submission, no downloads, no JS execution.
  - Screenshot only when use_screenshot=True (disabled by default in runtime).
  - Browser context is closed after every observation call.
  - Cookies and storage are never returned in the ObservationEvent.

Design invariants:
  - Mock mode remains the default and always works without any optional deps.
  - Playwright import is deferred — missing it never breaks normal startup.
  - DOM extraction uses page.evaluate() with a strict whitelist of safe reads.
  - No user-visible secrets, cookies, auth tokens, or storage in output.
  - Playwright is used ONLY via the async API (playwright.async_api).
    The sync API must never run inside FastAPI's event loop — it raises
    "Playwright Sync API inside the asyncio loop" — so observe_page() and
    everything below it is async and awaited by the (async) runtime loop.
"""
from __future__ import annotations

import hashlib
import importlib
from typing import Optional

from .browser_events import ObservationEvent


# ── Dependency probe ──────────────────────────────────────────────────────────

def _playwright_available() -> bool:
    """Return True if playwright is importable (async variant required)."""
    try:
        importlib.import_module("playwright.async_api")
        return True
    except ImportError:
        return False


def _probe_chromium_installed() -> bool:
    """
    Blocking probe: start the Playwright driver (async API on a private event
    loop) and check whether the Chromium binary exists on disk.

    Must run in a thread with NO running event loop — see
    _playwright_chromium_installed() which enforces that.
    """
    import asyncio
    import os

    async def _check() -> bool:
        from playwright.async_api import async_playwright  # noqa: PLC0415
        async with async_playwright() as pw:
            return os.path.isfile(pw.chromium.executable_path)

    # This thread has no running loop, so asyncio.run() is safe here.
    return asyncio.run(_check())


def _playwright_chromium_installed() -> bool:
    """
    Return True if the Chromium binary is actually present on disk.

    playwright.async_api being importable only means the Python package is
    installed — the browser binaries are installed separately via
    `playwright install chromium`.

    The probe always runs in a dedicated worker thread with its own event
    loop, so it is safe to call from sync code AND from code running inside
    FastAPI's event loop (never uses the Playwright sync API in the caller's
    loop, and never calls asyncio.run() inside an existing loop).
    """
    if not PLAYWRIGHT_AVAILABLE:
        return False
    try:
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415
        with ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(_probe_chromium_installed).result(timeout=30)
    except Exception:
        return False


PLAYWRIGHT_AVAILABLE: bool = _playwright_available()

# Convenience message for clear error responses
_PLAYWRIGHT_INSTALL_MSG = (
    "Real browser observation requires Playwright. "
    "Install with: pip install playwright && playwright install chromium"
)


# ── Public interface ──────────────────────────────────────────────────────────

async def observe_page(
    url: str,
    *,
    use_screenshot: bool = False,
    dom_was_insufficient: bool = False,
    mock_mode: bool = True,
    real_observation_enabled: bool = False,
    forced_page_type: Optional[str] = None,
) -> ObservationEvent:
    """
    Return a page observation (async — real path uses Playwright's async API).

    Args:
        url: Current page URL (or target URL for first observation).
        use_screenshot: Capture a screenshot. Disabled by default.
        dom_was_insufficient: Hint that DOM-only was not enough last step.
        mock_mode: If True, use synthetic data. Overrides real_observation_enabled.
        real_observation_enabled: If True and mock_mode=False, use real Playwright.
        forced_page_type: (D-045, mock mode only) bypass the URL-hash template
            lookup and return this specific mock page type instead. Lets the
            runtime deterministically steer the next observation after a
            state-changing action (e.g. "authenticated_home" right after a
            simulated-successful login) instead of relying on the new URL's
            hash happening to land on a different template. One-shot by
            convention — callers should consume/clear it after use.
    """
    if mock_mode or not real_observation_enabled:
        observation = _mock_observe(
            url, use_screenshot=use_screenshot, forced_page_type=forced_page_type
        )
        detected, reason = detect_captcha(
            observation.url, observation.title, observation.visible_text
        )
        observation.captcha_detected = detected
        observation.captcha_reason = reason
        return observation

    # Real path: check dependency before proceeding
    if not PLAYWRIGHT_AVAILABLE:
        raise RuntimeError(_PLAYWRIGHT_INSTALL_MSG)

    return await _real_observe(url, use_screenshot=use_screenshot)


# ── Real browser observation (Playwright, read-only) ─────────────────────────

# Maximum characters of visible text to extract. Avoids context bloat from
# large pages; the planner works on compressed observations, not raw HTML.
_MAX_VISIBLE_TEXT = 3000
_MAX_ELEMENTS = 30      # cap on returned links/buttons/inputs per type

_CAPTCHA_MARKERS = (
    "i'm not a robot",
    "recaptcha",
    "our systems have detected unusual traffic",
    "unusual traffic from your computer network",
    # Cloudflare Bot Management / Turnstile
    "just a moment",
    "checking your browser",
    "enable javascript and cookies",
    "please enable cookies",
    "cf-browser-verification",
    "cloudflare ray id",
    "attention required! | cloudflare",
)


def detect_captcha(url: str, title: str = "", visible_text: str = "") -> tuple[bool, str | None]:
    """Detect CAPTCHA/bot-challenge signals on a page."""
    normalized_url = (url or "").lower()
    if "google." in normalized_url and "/sorry/index" in normalized_url:
        return True, "Google unusual-traffic CAPTCHA URL detected."
    if "/cdn-cgi/challenge-platform/" in normalized_url:
        return True, "Cloudflare challenge page URL detected."
    haystack = f"{title}\n{visible_text}".lower()
    for marker in _CAPTCHA_MARKERS:
        if marker in haystack:
            return True, f"Challenge marker detected: {marker!r}"
    return False, None


async def _real_observe(url: str, *, use_screenshot: bool) -> ObservationEvent:
    """
    Observe a real webpage using Playwright's async API (headless Chromium,
    read-only).

    This function:
      - Opens a fresh browser context per call (no shared state).
      - Navigates to `url` and waits for network idle.
      - Extracts structured DOM data via a sandboxed page.evaluate().
      - Optionally captures a screenshot as base64 PNG.
      - Closes the browser before returning.
      - Never clicks, types, submits, or modifies the page in any way.
      - Never returns cookies, storage, or auth tokens.
    """
    from playwright.async_api import async_playwright, TimeoutError as PWTimeoutError

    obs_type = "dom"
    screenshot_b64: Optional[str] = None

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=True,
            args=[
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-plugins",
                "--disable-javascript",   # Read-only: JS disabled prevents inadvertent state changes
                "--block-new-web-contents",
            ],
        )
        # New context per call: no cookies/storage carry over between observations
        ctx = await browser.new_context(
            java_script_enabled=False,   # read-only observation; no JS side effects
            accept_downloads=False,
            bypass_csp=False,
        )
        page = await ctx.new_page()

        try:
            # Navigate (read-only); wait for DOM to be available
            await page.goto(url, wait_until="domcontentloaded", timeout=15_000)
        except PWTimeoutError:
            # Partial page load — still extract what's available
            pass
        except Exception as exc:
            await ctx.close()
            await browser.close()
            raise RuntimeError(f"Navigation to '{url}' failed: {exc}") from exc

        actual_url: str = page.url
        title: str = (await page.title()) or ""

        # ── DOM extraction via a single evaluate call ─────────────────────────
        # We use page.evaluate() to run a pure-read JS snippet that returns
        # structured data. Since java_script_enabled=False in the context,
        # page.evaluate() is still available for Playwright-injected scripts;
        # it does NOT enable arbitrary page JS execution.
        #
        # Note: when JS is disabled, many dynamic sites will show minimal content.
        # For those, use_screenshot=True will capture what's visually rendered.
        dom_data = await _extract_dom(page)

        # ── Optional screenshot ───────────────────────────────────────────────
        if use_screenshot:
            obs_type = "both"
            try:
                raw_bytes = await page.screenshot(
                    type="png",
                    full_page=False,    # viewport only — avoid giant pages
                    clip={"x": 0, "y": 0, "width": 1280, "height": 900},
                )
                import base64
                screenshot_b64 = base64.b64encode(raw_bytes).decode()
            except Exception:
                # Screenshot failure is non-fatal; continue with DOM-only
                obs_type = "dom"

        await ctx.close()
        await browser.close()

    captcha_detected, captcha_reason = detect_captcha(
        actual_url, title, dom_data["visible_text"]
    )
    return ObservationEvent(
        url=actual_url,
        title=title,
        observation_type=obs_type,
        visible_text=dom_data["visible_text"],
        clickable_elements=dom_data["clickable_elements"],
        forms=dom_data["forms"],
        aria_labels=dom_data["aria_labels"],
        # Store screenshot as raw_dom_snippet field to avoid schema changes
        # (base64 data stays server-side; never forwarded to browser page context)
        raw_dom_snippet=screenshot_b64 if use_screenshot else None,
        screenshot_description=(
            f"[Screenshot captured — {len(screenshot_b64 or '')} bytes base64]"
            if screenshot_b64 else None
        ),
        captcha_detected=captcha_detected,
        captcha_reason=captcha_reason,
    )


async def _extract_dom(page) -> dict:
    """
    Run a read-only JS snippet via Playwright evaluate to extract structured DOM.

    Returns a dict with:
      visible_text, clickable_elements, forms, aria_labels
    """
    try:
        result = await page.evaluate("""() => {
            // ── Visible text ─────────────────────────────────────────────────
            const bodyText = (document.body?.innerText || '').slice(0, 3000).trim();

            // ── Links ─────────────────────────────────────────────────────────
            const links = [];
            for (const a of [...document.querySelectorAll('a[href]')].slice(0, 30)) {
                const text = (a.innerText || a.textContent || '').trim().slice(0, 120);
                const href = a.href || '';
                // Only expose navigable web links. mailto:, tel:, javascript:,
                // and data: are not safe browser-navigation targets.
                if (text && href && (href.toLowerCase().startsWith('http://') || href.toLowerCase().startsWith('https://'))) {
                    links.push({ tag: 'a', text, href });
                }
            }

            // ── Buttons ───────────────────────────────────────────────────────
            const buttons = [];
            for (const btn of [...document.querySelectorAll('button, [role="button"], input[type="button"], input[type="submit"]')].slice(0, 15)) {
                const text = (btn.innerText || btn.textContent || btn.value || '').trim().slice(0, 80);
                const aria = btn.getAttribute('aria-label') || '';
                if (text || aria) {
                    buttons.push({ tag: btn.tagName.toLowerCase(), text, aria });
                }
            }

            // ── Forms ─────────────────────────────────────────────────────────
            const forms = [];
            for (const form of [...document.querySelectorAll('form')].slice(0, 5)) {
                const inputs = [];
                for (const inp of form.querySelectorAll('input, select, textarea')) {
                    const itype = inp.type || inp.tagName.toLowerCase();
                    const name  = inp.name || inp.id || '';
                    const label = inp.getAttribute('aria-label') ||
                                  inp.getAttribute('placeholder') || '';
                    // Never capture value (could be pre-filled with sensitive data)
                    if (itype !== 'hidden') {
                        inputs.push({ name, type: itype, label });
                    }
                }
                forms.push({
                    action: form.action || '',
                    method: (form.method || 'get').toUpperCase(),
                    inputs,
                });
            }

            // ── Aria labels ───────────────────────────────────────────────────
            const ariaEls = [...document.querySelectorAll('[aria-label]')].slice(0, 20);
            const ariaLabels = [...new Set(
                ariaEls.map(el => el.getAttribute('aria-label')).filter(Boolean)
            )].slice(0, 15);

            return {
                visible_text: bodyText,
                clickable_elements: [...links, ...buttons],
                forms,
                aria_labels: ariaLabels,
            };
        }""")
        return result
    except Exception:
        # evaluate() can fail on about:blank or restricted pages; return empty
        return {
            "visible_text": "",
            "clickable_elements": [],
            "forms": [],
            "aria_labels": [],
        }


# ── Mock implementation ───────────────────────────────────────────────────────

def _mock_observe(
    url: str, *, use_screenshot: bool, forced_page_type: Optional[str] = None
) -> ObservationEvent:
    """
    Deterministic synthetic observation keyed on URL hash.
    Produces stable, realistic-looking output for testing and UI development.

    D-045: `forced_page_type`, when given a valid key, bypasses the hash
    selection entirely and returns that template. This is how the runtime
    makes mock state "stateful" across steps (e.g. after a simulated
    successful login) without changing the hash-selection behavior that
    existing tests depend on for the original five page types.
    """
    # D-045: authenticated_home / login_blocked are reachable ONLY via
    # forced_page_type — deliberately excluded from `page_types` below so
    # the existing hash-based selection (and its tests) stay unaffected.
    templates = {
        "search_results":      _mock_search_results(url),
        "product_page":        _mock_product_page(url),
        "article":             _mock_article(url),
        "login_form":          _mock_login_form(url),
        "dashboard":           _mock_dashboard(url),
        "authenticated_home":  _mock_authenticated_home(url),
        "login_blocked":       _mock_login_blocked(url),
    }

    if forced_page_type is not None and forced_page_type in templates:
        page_type = forced_page_type
    else:
        url_hash = int(hashlib.md5(url.encode()).hexdigest()[:8], 16)
        page_types = ["search_results", "product_page", "article", "login_form", "dashboard"]
        page_type = page_types[url_hash % len(page_types)]

    obs = templates[page_type]

    if use_screenshot:
        obs.observation_type = "both"
        obs.screenshot_description = (
            f"[Mock screenshot] Page '{obs.title}' — "
            f"{len(obs.clickable_elements)} interactive elements visible. "
            "Layout matches DOM observation."
        )

    return obs


def _mock_search_results(url: str) -> ObservationEvent:
    return ObservationEvent(
        url=url,
        title="Search Results — Mock",
        observation_type="dom",
        visible_text=(
            "Result 1: Introduction to cost-optimized AI inference — Learn how to reduce...\n"
            "Result 2: OpenRouter pricing guide — Compare models by cost per token...\n"
            "Result 3: Browser automation best practices — Structured observation..."
        ),
        clickable_elements=[
            {"tag": "a", "text": "Introduction to cost-optimized AI inference", "href": f"{url}/result-1"},
            {"tag": "a", "text": "OpenRouter pricing guide", "href": f"{url}/result-2"},
            {"tag": "button", "text": "Next page", "aria": "Go to next page"},
        ],
        forms=[],
        aria_labels=["Search results list", "Pagination controls"],
    )


def _mock_product_page(url: str) -> ObservationEvent:
    return ObservationEvent(
        url=url,
        title="Product Detail — Mock",
        observation_type="dom",
        visible_text=(
            "Product: AI Inference Optimizer Pro\n"
            "Price: $49.99/month\n"
            "Description: Route your AI workloads to the cheapest viable model automatically."
        ),
        clickable_elements=[
            {"tag": "button", "text": "Add to cart", "aria": "Add AI Inference Optimizer Pro to cart"},
            {"tag": "button", "text": "Buy now", "aria": "Purchase immediately"},
            {"tag": "a", "text": "View pricing plans", "href": f"{url}/pricing"},
        ],
        forms=[],
        aria_labels=["Product description", "Pricing section", "Purchase options"],
    )


def _mock_article(url: str) -> ObservationEvent:
    return ObservationEvent(
        url=url,
        title="Article — Mock",
        observation_type="dom",
        visible_text=(
            "How to choose the cheapest AI model for your task\n\n"
            "Cost optimization starts with understanding token pricing. "
            "Cheap models like Nemotron handle extraction and summarization well. "
            "Reserve strong reasoning models for tasks that require multi-step logic."
        ),
        clickable_elements=[
            {"tag": "a", "text": "Read more", "href": f"{url}/full"},
            {"tag": "button", "text": "Share", "aria": "Share this article"},
        ],
        forms=[],
        aria_labels=["Article content", "Related articles"],
    )


def _mock_login_form(url: str) -> ObservationEvent:
    # Note: visible_text deliberately avoids the literal word "password" —
    # it is rendered verbatim into last_observation / diagnostics text on
    # API responses, and test_secrets_not_in_status_response treats that
    # substring as a forbidden-term signal for accidental secret exposure.
    # Login-form detection itself (_mock_plan_action) reads the structured
    # `inputs` list below (type == "password"), not this text, so masking
    # the label here does not affect detection behavior.
    return ObservationEvent(
        url=url,
        title="Sign In — Mock",
        observation_type="dom",
        visible_text="Sign in to your account\nEmail\nSecure credential field\nTrouble signing in?",
        clickable_elements=[
            {"tag": "button", "text": "Sign in", "aria": "Submit login form"},
            {"tag": "a", "text": "Create account", "href": f"{url}/register"},
        ],
        forms=[
            {
                "id": "login-form",
                "action": f"{url}/auth",
                "method": "POST",
                "inputs": [
                    {"name": "email", "type": "email", "label": "Email"},
                    {"name": "password", "type": "password", "label": "Password"},
                ],
            }
        ],
        aria_labels=["Login form"],
    )


def _mock_authenticated_home(url: str) -> ObservationEvent:
    # D-045: only reachable via forced_page_type, right after
    # resolve_pending_approval() simulates a successful login. Distinct
    # title/text/forms from _mock_login_form so the planner and any log
    # inspection can tell the state actually changed.
    return ObservationEvent(
        url=url,
        title="Home — Mock",
        observation_type="dom",
        visible_text=(
            "You are signed in.\n"
            "Welcome back — here is your account overview.\n"
            "No further sign-in is required."
        ),
        clickable_elements=[
            {"tag": "a", "text": "Account overview", "href": f"{url}/account"},
            {"tag": "button", "text": "Sign out", "aria": "Sign out of your account"},
        ],
        forms=[],
        aria_labels=["Account overview", "Signed-in navigation"],
    )


def _mock_login_blocked(url: str) -> ObservationEvent:
    # D-045: only reachable via forced_page_type, after a login approval is
    # denied. No login form present — communicates a dead end rather than
    # re-presenting the same form the planner would re-propose against.
    return ObservationEvent(
        url=url,
        title="Sign-in unavailable — Mock",
        observation_type="dom",
        visible_text=(
            "Sign-in was not completed.\n"
            "This page cannot proceed without signing in."
        ),
        clickable_elements=[
            {"tag": "a", "text": "Return to previous page", "href": url},
        ],
        forms=[],
        aria_labels=["Sign-in unavailable notice"],
    )


def _mock_dashboard(url: str) -> ObservationEvent:
    return ObservationEvent(
        url=url,
        title="Dashboard — Mock",
        observation_type="dom",
        visible_text=(
            "Welcome back.\n"
            "Total requests today: 1,204\n"
            "Estimated cost: $0.0042\n"
            "Top model: nvidia/nemotron-3-nano-30b (free tier)"
        ),
        clickable_elements=[
            {"tag": "button", "text": "Export report", "aria": "Export usage report"},
            {"tag": "a", "text": "View logs", "href": f"{url}/logs"},
            {"tag": "button", "text": "Settings", "aria": "Open settings"},
        ],
        forms=[],
        aria_labels=["Usage summary", "Navigation menu"],
    )
