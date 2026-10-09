"""CAPTCHA detection and optional CapSolver-backed solving for the search layer.

This module is scoped to the Scrapling search provider and the browser-agent
challenge handoff. Browser-agent sessions try Theriad-owned Scrapling stealth
handling first, then preserve a visible human handoff when the challenge is
not cleared.

Flow
----
1. After a stealth fetch, detect whether the returned HTML is a CAPTCHA page.
2. If yes and CAPTCHA_SOLVER_API_KEY is set, call the CapSolver API to get a
   token for the detected challenge type.
3. Use Playwright to inject the token and await the post-challenge page, then
   return the final HTML for Scrapling to parse.

Env vars
--------
CAPTCHA_SOLVER_API_KEY   CapSolver API key (capsolver.com).  Required; solver
                         is disabled when absent.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import Optional

from theriad.core.logging.logger import get_logger

logger = get_logger("search.captcha_solver")


# ── CAPTCHA type detection ─────────────────────────────────────────────────────

class CaptchaType(Enum):
    CLOUDFLARE_TURNSTILE = auto()
    HCAPTCHA = auto()
    RECAPTCHA_V2 = auto()


@dataclass
class CaptchaChallenge:
    type: CaptchaType
    url: str
    site_key: Optional[str] = None


# Cloudflare: "Just a moment..." / legacy JS challenge / Turnstile widget
_CF_BODY_RE = re.compile(
    r"(just a moment|checking your browser|cf-browser-verification|"
    r"cloudflare ray id|cf_clearance|enable javascript and cookies)",
    re.IGNORECASE,
)
_CF_TURNSTILE_RE = re.compile(
    r'<[^>]+class="[^"]*cf-turnstile[^"]*"[^>]*data-sitekey="([^"]+)"',
    re.IGNORECASE,
)
_HCAPTCHA_RE = re.compile(
    r'data-sitekey="([^"]+)"[^>]*class="[^"]*h-captcha|'
    r'class="[^"]*h-captcha[^"]*"[^>]*data-sitekey="([^"]+)"',
    re.IGNORECASE,
)
_RECAPTCHA_RE = re.compile(
    r'class="[^"]*g-recaptcha[^"]*"[^>]*data-sitekey="([^"]+)"',
    re.IGNORECASE,
)


def detect_captcha(html: str, url: str) -> Optional[CaptchaChallenge]:
    """Return the detected CAPTCHA challenge, or None if the page is clean."""
    if not html:
        return None

    cf_match = _CF_TURNSTILE_RE.search(html)
    if cf_match or _CF_BODY_RE.search(html):
        return CaptchaChallenge(
            type=CaptchaType.CLOUDFLARE_TURNSTILE,
            url=url,
            site_key=cf_match.group(1) if cf_match else None,
        )

    hc_match = _HCAPTCHA_RE.search(html)
    if hc_match:
        site_key = hc_match.group(1) or hc_match.group(2)
        return CaptchaChallenge(type=CaptchaType.HCAPTCHA, url=url, site_key=site_key)

    rc_match = _RECAPTCHA_RE.search(html)
    if rc_match:
        return CaptchaChallenge(type=CaptchaType.RECAPTCHA_V2, url=url, site_key=rc_match.group(1))

    return None


# ── CapSolver API client ───────────────────────────────────────────────────────

class CaptchaSolver:
    """Thin wrapper around the CapSolver API (capsolver.com).

    Install the optional extra: pip install "theriad[captcha-solver]"
    """

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    @classmethod
    def from_environment(cls) -> Optional["CaptchaSolver"]:
        key = os.environ.get("CAPTCHA_SOLVER_API_KEY", "").strip()
        return cls(key) if key else None

    def solve(self, challenge: CaptchaChallenge) -> Optional[str]:
        """Call CapSolver and return the solution token, or None on any failure."""
        task = self._build_task(challenge)
        if task is None:
            logger.warning(
                "Challenge type %s is not supported by the CapSolver integration. "
                "Supported types: Cloudflare Turnstile, hCaptcha, reCAPTCHA v2.",
                challenge.type.name,
            )
            return None
        try:
            import capsolver
        except ImportError:
            logger.warning(
                "The capsolver package is required for automatic CAPTCHA solving but is "
                "not installed. Run: pip install \"theriad[captcha-solver]\""
            )
            return None

        capsolver.api_key = self.api_key
        try:
            solution = capsolver.solve(task)
            token = solution.get("token") or solution.get("gRecaptchaResponse")
            if token:
                logger.debug("CAPTCHA solved: type=%s url=%s", challenge.type, challenge.url)
            else:
                logger.warning(
                    "CapSolver API responded but returned no solution token "
                    "(response: %s). The challenge may be unsolvable with the current "
                    "site key, or the task type may not match the actual widget on the page.",
                    solution,
                )
            return token
        except Exception as exc:
            logger.warning(
                "CapSolver API request failed: %s. Verify that CAPTCHA_SOLVER_API_KEY "
                "is correct and that your capsolver.com account has sufficient balance.",
                exc,
            )
            return None

    def _build_task(self, challenge: CaptchaChallenge) -> Optional[dict]:
        site_key = challenge.site_key or ""
        if challenge.type == CaptchaType.CLOUDFLARE_TURNSTILE:
            return {
                "type": "AntiTurnstileTaskProxyLess",
                "websiteURL": challenge.url,
                "websiteKey": site_key,
            }
        if challenge.type == CaptchaType.HCAPTCHA:
            return {
                "type": "HCaptchaTaskProxyLess",
                "websiteURL": challenge.url,
                "websiteKey": site_key,
            }
        if challenge.type == CaptchaType.RECAPTCHA_V2:
            return {
                "type": "ReCaptchaV2TaskProxyLess",
                "websiteURL": challenge.url,
                "websiteKey": site_key,
            }
        return None


# ── Playwright token injection ─────────────────────────────────────────────────

# JS snippets that inject a solved token into the DOM, then trigger whatever
# submission callback the challenge widget expects.
_INJECT_JS: dict[CaptchaType, str] = {
    CaptchaType.CLOUDFLARE_TURNSTILE: """
        (token) => {
            const input = document.querySelector('[name="cf-turnstile-response"]');
            if (input) input.value = token;
            // Trigger the Turnstile callback if registered.
            if (window.__cfTurnstileCallback) window.__cfTurnstileCallback(token);
        }
    """,
    CaptchaType.HCAPTCHA: """
        (token) => {
            const r = document.querySelector('[name="h-captcha-response"]');
            if (r) r.value = token;
            if (window.hcaptcha) {
                const id = document.querySelector('.h-captcha')?.dataset?.hcaptchaWidgetId;
                if (id !== undefined) window.hcaptcha.execute(id, {async: true});
            }
        }
    """,
    CaptchaType.RECAPTCHA_V2: """
        (token) => {
            const r = document.getElementById('g-recaptcha-response');
            if (r) { r.innerHTML = token; r.value = token; }
            const cb = document.querySelector('[data-callback]');
            if (cb?.dataset?.callback) {
                try { window[cb.dataset.callback](token); } catch(e) {}
            }
        }
    """,
}

_SUBMIT_JS = """
    () => {
        const btn = document.querySelector(
            'input[type="submit"], button[type="submit"], button.challenge-form__btn'
        );
        if (btn) btn.click();
    }
"""


def solve_with_playwright(
    url: str,
    challenge: CaptchaChallenge,
    solver: CaptchaSolver,
    *,
    user_data_dir: str | None = None,
) -> Optional[str]:
    """Solve the CAPTCHA, inject the token via Playwright, return post-challenge HTML.

    Returns None when solving or injection fails so the caller can fall back.
    This runs synchronously — wrap in asyncio.to_thread when calling from async.
    """
    token = solver.solve(challenge)
    if not token:
        return None

    inject_js = _INJECT_JS.get(challenge.type)
    if inject_js is None:
        return None

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.warning(
            "Playwright is required to inject the CAPTCHA solution token but is not "
            "installed. Run: pip install playwright && playwright install chromium"
        )
        return None

    try:
        with sync_playwright() as pw:
            browser = None
            persistent_context = bool(user_data_dir)
            if persistent_context:
                profile_dir = Path(user_data_dir).expanduser()
                profile_dir.mkdir(parents=True, exist_ok=True)
                context = pw.chromium.launch_persistent_context(
                    str(profile_dir),
                    headless=True,
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    java_script_enabled=True,
                )
                browser = getattr(context, "browser", None)
                pages = getattr(context, "pages", [])
                page = pages[0] if pages else context.new_page()
            else:
                browser = pw.chromium.launch(headless=True)
                context = browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    java_script_enabled=True,
                )
                page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=20_000)
            page.evaluate(inject_js, token)
            page.evaluate(_SUBMIT_JS)
            page.wait_for_load_state("networkidle", timeout=15_000)
            html = page.content()
            context.close()
            if browser is not None:
                browser.close()
            return html
    except Exception as exc:
        logger.warning(
            "Failed to inject the CAPTCHA solution token via Playwright: %s. "
            "The page structure may have changed after navigation, the token may have "
            "expired, or the challenge form could not be located or submitted.",
            exc,
        )
        return None
