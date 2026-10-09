"""
Unit tests for theriad.search.captcha_solver.

All external calls (CapSolver API, Playwright) are mocked.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch, call

import pytest

from theriad.search.captcha_solver import (
    CaptchaChallenge,
    CaptchaSolver,
    CaptchaType,
    detect_captcha,
    solve_with_playwright,
)


# ── detect_captcha ─────────────────────────────────────────────────────────────

URL = "https://html.duckduckgo.com/html/?q=test"


class TestDetectCaptcha:
    def test_clean_page_returns_none(self):
        html = "<html><body><div class='result'>normal page</div></body></html>"
        assert detect_captcha(html, URL) is None

    def test_empty_html_returns_none(self):
        assert detect_captcha("", URL) is None

    def test_cloudflare_just_a_moment(self):
        html = "<html><head><title>Just a moment...</title></head><body>checking your browser</body></html>"
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.CLOUDFLARE_TURNSTILE
        assert ch.url == URL
        assert ch.site_key is None

    def test_cloudflare_with_turnstile_sitekey(self):
        html = """
        <div class="cf-turnstile" data-sitekey="0x4AAAAAAAxyz"></div>
        """
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.CLOUDFLARE_TURNSTILE
        assert ch.site_key == "0x4AAAAAAAxyz"

    def test_cloudflare_ray_id(self):
        html = "<html><body>Cloudflare Ray ID: 9abc</body></html>"
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.CLOUDFLARE_TURNSTILE

    def test_hcaptcha_detected(self):
        html = '<div class="h-captcha" data-sitekey="abc123"></div>'
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.HCAPTCHA
        assert ch.site_key == "abc123"

    def test_recaptcha_detected(self):
        html = '<div class="g-recaptcha" data-sitekey="6Lcr_abc"></div>'
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.RECAPTCHA_V2
        assert ch.site_key == "6Lcr_abc"

    def test_cloudflare_takes_priority_over_recaptcha(self):
        html = """
        Just a moment...
        <div class="g-recaptcha" data-sitekey="6Lcr_abc"></div>
        """
        ch = detect_captcha(html, URL)
        assert ch is not None
        assert ch.type == CaptchaType.CLOUDFLARE_TURNSTILE


# ── CaptchaSolver.from_environment ────────────────────────────────────────────

class TestCaptchaSolverFromEnv:
    def test_none_when_no_key(self, monkeypatch):
        monkeypatch.delenv("CAPTCHA_SOLVER_API_KEY", raising=False)
        assert CaptchaSolver.from_environment() is None

    def test_returns_solver_with_key(self, monkeypatch):
        monkeypatch.setenv("CAPTCHA_SOLVER_API_KEY", "test-key-123")
        s = CaptchaSolver.from_environment()
        assert isinstance(s, CaptchaSolver)
        assert s.api_key == "test-key-123"


# ── CaptchaSolver._build_task ─────────────────────────────────────────────────

class TestBuildTask:
    def setup_method(self):
        self.solver = CaptchaSolver("dummy-key")

    def test_turnstile_task(self):
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site123")
        task = self.solver._build_task(ch)
        assert task["type"] == "AntiTurnstileTaskProxyLess"
        assert task["websiteURL"] == URL
        assert task["websiteKey"] == "site123"

    def test_hcaptcha_task(self):
        ch = CaptchaChallenge(CaptchaType.HCAPTCHA, URL, "hkey")
        task = self.solver._build_task(ch)
        assert task["type"] == "HCaptchaTaskProxyLess"
        assert task["websiteKey"] == "hkey"

    def test_recaptcha_task(self):
        ch = CaptchaChallenge(CaptchaType.RECAPTCHA_V2, URL, "rkey")
        task = self.solver._build_task(ch)
        assert task["type"] == "ReCaptchaV2TaskProxyLess"
        assert task["websiteKey"] == "rkey"

    def test_empty_sitekey_uses_empty_string(self):
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, site_key=None)
        task = self.solver._build_task(ch)
        assert task["websiteKey"] == ""


# ── CaptchaSolver.solve ───────────────────────────────────────────────────────

class TestCaptchaSolverSolve:
    def _make_capsolver_stub(self, return_value: dict):
        stub = types.ModuleType("capsolver")
        stub.api_key = None
        stub.solve = MagicMock(return_value=return_value)
        return stub

    def test_returns_token_field(self, monkeypatch):
        stub = self._make_capsolver_stub({"token": "solved-token-xyz"})
        monkeypatch.setitem(sys.modules, "capsolver", stub)
        solver = CaptchaSolver("key")
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solver.solve(ch) == "solved-token-xyz"

    def test_falls_back_to_gRecaptchaResponse(self, monkeypatch):
        stub = self._make_capsolver_stub({"gRecaptchaResponse": "rc-token"})
        monkeypatch.setitem(sys.modules, "capsolver", stub)
        solver = CaptchaSolver("key")
        ch = CaptchaChallenge(CaptchaType.RECAPTCHA_V2, URL, "site")
        assert solver.solve(ch) == "rc-token"

    def test_returns_none_on_api_error(self, monkeypatch):
        stub = types.ModuleType("capsolver")
        stub.api_key = None
        stub.solve = MagicMock(side_effect=RuntimeError("API error"))
        monkeypatch.setitem(sys.modules, "capsolver", stub)
        solver = CaptchaSolver("key")
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solver.solve(ch) is None

    def test_returns_none_when_capsolver_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "capsolver", None)
        solver = CaptchaSolver("key")
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solver.solve(ch) is None

    def test_sets_api_key_on_module(self, monkeypatch):
        stub = self._make_capsolver_stub({"token": "t"})
        monkeypatch.setitem(sys.modules, "capsolver", stub)
        solver = CaptchaSolver("my-key")
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        solver.solve(ch)
        assert stub.api_key == "my-key"


# ── solve_with_playwright ─────────────────────────────────────────────────────

class TestSolveWithPlaywright:
    def _make_pw_stack(self, page_html: str = "<html><body>done</body></html>"):
        """Build a minimal playwright sync_api stub."""
        page = MagicMock()
        page.content.return_value = page_html
        context = MagicMock()
        context.new_page.return_value = page
        browser = MagicMock()
        browser.new_context.return_value = context
        chromium = MagicMock()
        chromium.launch.return_value = browser
        pw = MagicMock()
        pw.chromium = chromium
        pw.__enter__ = MagicMock(return_value=pw)
        pw.__exit__ = MagicMock(return_value=False)

        sync_playwright = MagicMock(return_value=pw)

        playwright_mod = types.ModuleType("playwright.sync_api")
        playwright_mod.sync_playwright = sync_playwright

        return playwright_mod, page, browser

    def test_returns_none_when_solver_gives_no_token(self):
        solver = MagicMock(spec=CaptchaSolver)
        solver.solve.return_value = None
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solve_with_playwright(URL, ch, solver) is None

    def test_returns_html_after_injection(self, monkeypatch):
        pw_mod, page, browser = self._make_pw_stack("<html>clean</html>")
        monkeypatch.setitem(sys.modules, "playwright.sync_api", pw_mod)

        solver = MagicMock(spec=CaptchaSolver)
        solver.solve.return_value = "solved-token"
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")

        result = solve_with_playwright(URL, ch, solver)
        assert result == "<html>clean</html>"
        page.goto.assert_called_once_with(URL, wait_until="domcontentloaded", timeout=20_000)
        page.evaluate.assert_called()

    def test_returns_none_when_playwright_not_installed(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "playwright.sync_api", None)
        solver = MagicMock(spec=CaptchaSolver)
        solver.solve.return_value = "token"
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solve_with_playwright(URL, ch, solver) is None

    def test_returns_none_on_playwright_error(self, monkeypatch):
        pw_mod = types.ModuleType("playwright.sync_api")
        pw_mod.sync_playwright = MagicMock(side_effect=RuntimeError("crash"))
        monkeypatch.setitem(sys.modules, "playwright.sync_api", pw_mod)

        solver = MagicMock(spec=CaptchaSolver)
        solver.solve.return_value = "token"
        ch = CaptchaChallenge(CaptchaType.CLOUDFLARE_TURNSTILE, URL, "site")
        assert solve_with_playwright(URL, ch, solver) is None


# ── ScraplingSearchProvider integration ───────────────────────────────────────

class TestScraplingWithCaptchaSolving:
    """Verify that ScraplingSearchProvider calls the solver when a CAPTCHA is detected."""

    def test_solver_invoked_on_captcha_page(self, monkeypatch):
        from theriad.search.scrapling_provider import ScraplingSearchProvider, _parse_html

        monkeypatch.setenv("CAPTCHA_SOLVER_API_KEY", "test-key")

        captcha_html = "<html><body>Just a moment... checking your browser</body></html>"
        clean_html = (
            '<html><body>'
            '<div class="result">'
            '<a class="result__a" href="https://example.com">Title</a>'
            '<a class="result__snippet">Snippet</a>'
            '</div></body></html>'
        )

        provider = ScraplingSearchProvider(stealth=True)
        assert provider._solver is not None

        fake_page = MagicMock()
        fake_page.html = captcha_html

        with (
            patch.object(provider, "_fetch_stealth_browser", return_value=fake_page),
            patch(
                "theriad.search.scrapling_provider.solve_with_playwright",
                return_value=clean_html,
            ) as mock_solve,
            patch(
                "theriad.search.scrapling_provider._parse_html",
                wraps=lambda h, u: MagicMock(css=MagicMock(return_value=[])),
            ),
        ):
            provider._search_sync("test query")
            mock_solve.assert_called_once()

    def test_solver_not_invoked_when_no_captcha(self, monkeypatch):
        from theriad.search.scrapling_provider import ScraplingSearchProvider

        monkeypatch.setenv("CAPTCHA_SOLVER_API_KEY", "test-key")

        clean_html = "<html><body><div class='result'>normal</div></body></html>"
        fake_page = MagicMock()
        fake_page.html = clean_html

        provider = ScraplingSearchProvider(stealth=True)

        with (
            patch.object(provider, "_fetch_stealth_browser", return_value=fake_page),
            patch("theriad.search.scrapling_provider.solve_with_playwright") as mock_solve,
        ):
            provider._fetch_stealth("https://html.duckduckgo.com/html/?q=test")
            mock_solve.assert_not_called()
