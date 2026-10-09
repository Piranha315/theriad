"""Stress and contract tests for the canonical OpenSandbox runtime."""
from __future__ import annotations

import asyncio
from types import SimpleNamespace

import httpx
import pytest

import theriad.browser.opensandbox_runtime as runtime_module
from theriad.browser.opensandbox_runtime import (
    OpenSandboxBrowserRuntime,
    _cdp_connection_url,
    _rewrite_cdp_websocket_url,
)


class FakePage:
    url = "about:blank"

    def __init__(self) -> None:
        self.closed = False

    async def evaluate(self, *_args):
        return {"visible_text": "OpenSandbox test page", "elements": [], "title": "Test"}

    async def screenshot(self, **_kwargs):
        return b"png"

    async def close(self):
        self.closed = True


class FakeContext:
    def __init__(self) -> None:
        self.pages = []
        self.closed = False

    async def new_page(self):
        page = FakePage()
        self.pages.append(page)
        return page

    async def close(self):
        self.closed = True


class FakeBrowser:
    def __init__(self) -> None:
        self.contexts = []
        self.closed = False

    async def new_context(self, **_kwargs):
        context = FakeContext()
        self.contexts.append(context)
        return context

    async def close(self):
        self.closed = True

    async def disconnect(self):
        self.closed = True


class FakePlaywright:
    class Chromium:
        async def connect_over_cdp(self, *_args, **_kwargs):
            return FakeBrowser()

    chromium = Chromium()

    async def stop(self):
        return None


class FakePlaywrightManager:
    def start(self):
        return FakePlaywright()


class FakeSandbox:
    created = 0
    destroyed = 0

    def __init__(self) -> None:
        type(self).created += 1
        self.id = f"sandbox-{self.created}"

    async def get_endpoint(self, _port):
        return SimpleNamespace(endpoint="http://sandbox.test:9222")

    async def destroy(self):
        type(self).destroyed += 1


@pytest.fixture()
def fake_runtime():
    FakeSandbox.created = 0
    FakeSandbox.destroyed = 0
    return OpenSandboxBrowserRuntime(
        sandbox_factory=lambda **_kwargs: FakeSandbox(),
        playwright_factory=lambda: FakePlaywrightManager(),
    )


def test_rewrite_cdp_websocket_url_preserves_execd_proxy_route():
    endpoint = "http://127.0.0.1:40618/proxy/9222"
    websocket = "ws://127.0.0.1:40618/devtools/browser/browser-id"

    assert _rewrite_cdp_websocket_url(endpoint, websocket) == (
        "ws://127.0.0.1:40618/proxy/9222/devtools/browser/browser-id"
    )


def test_rewrite_cdp_websocket_url_leaves_unproxied_endpoint_unchanged():
    websocket = "ws://127.0.0.1:9222/devtools/browser/browser-id"

    assert _rewrite_cdp_websocket_url("http://127.0.0.1:9222", websocket) == websocket


@pytest.mark.asyncio
async def test_cdp_connection_url_retries_transient_proxy_failures(monkeypatch):
    request = httpx.Request("GET", "http://127.0.0.1:40000/proxy/9222/json/version")
    responses = iter(
        [
            httpx.Response(502, request=request),
            httpx.Response(
                200,
                json={
                    "webSocketDebuggerUrl": (
                        "ws://127.0.0.1:40000/devtools/browser/browser-id"
                    )
                },
                request=request,
            ),
        ]
    )

    class Client:
        calls = 0

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            type(self).calls += 1
            return next(responses)

    monkeypatch.setenv("THERIAD_OPENSANDBOX_CDP_READY_TIMEOUT_SECONDS", "1")
    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", lambda **_kwargs: Client())

    result = await _cdp_connection_url(
        "http://127.0.0.1:40000/proxy/9222", {}
    )

    assert result == (
        "ws://127.0.0.1:40000/proxy/9222/devtools/browser/browser-id"
    )
    assert Client.calls == 2


@pytest.mark.asyncio
async def test_repeated_start_stop_releases_sessions_and_keeps_terminal_status(fake_runtime):
    for index in range(128):
        session_id = f"stress-{index}"
        started = await fake_runtime.start_session(session_id=session_id)
        assert started["ok"] is True
        assert started["sandbox_id"] == f"sandbox-{index + 1}"

        stopped = await fake_runtime.stop_session(session_id)
        assert stopped["status"] == "stopped"
        assert stopped["sandbox_id"] == started["sandbox_id"]

    status = await fake_runtime.status()
    assert status["active_sessions"] == 0
    assert fake_runtime.sessions == {}
    assert len(fake_runtime._terminal_snapshots) == 128
    assert (await fake_runtime.status("stress-127"))["status"] == "stopped"
    assert FakeSandbox.created == FakeSandbox.destroyed == 128


@pytest.mark.asyncio
async def test_concurrent_same_session_id_creates_one_sandbox(fake_runtime):
    async def delayed_factory(**_kwargs):
        await asyncio.sleep(0)
        return FakeSandbox()

    fake_runtime._sandbox_factory = delayed_factory
    results = await asyncio.gather(
        fake_runtime.start_session(session_id="same"),
        fake_runtime.start_session(session_id="same"),
    )

    assert [result["ok"] for result in results] == [True, True]
    assert {result["sandbox_id"] for result in results} == {"sandbox-1"}
    assert FakeSandbox.created == 1

    await fake_runtime.stop_session("same")
    assert FakeSandbox.destroyed == 1


@pytest.mark.asyncio
async def test_different_session_ids_can_provision_concurrently(fake_runtime):
    active = 0
    peak = 0

    async def concurrent_factory(**_kwargs):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return FakeSandbox()

    fake_runtime._sandbox_factory = concurrent_factory
    await asyncio.gather(
        fake_runtime.start_session(session_id="first"),
        fake_runtime.start_session(session_id="second"),
    )

    assert peak == 2
    await fake_runtime.stop_all()
    assert FakeSandbox.destroyed == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("health_payload", "expected_available"),
    [({"status": "healthy"}, True), ({"status": "degraded"}, False), ({}, False)],
)
async def test_availability_requires_healthy_server_response(
    monkeypatch, health_payload, expected_available
):
    class Response:
        status_code = 200

        def json(self):
            return health_payload

    class Client:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return False

        async def get(self, *_args, **_kwargs):
            return Response()

    runtime = OpenSandboxBrowserRuntime()
    monkeypatch.setattr(
        runtime,
        "_dependency_status",
        lambda: {
            "available": True,
            "error": None,
            "opensandbox_installed": True,
            "playwright_installed": True,
        },
    )
    monkeypatch.setattr(runtime_module.httpx, "AsyncClient", lambda **_kwargs: Client())

    result = await runtime.availability()

    assert result["available"] is expected_available
    assert result["health_status"] == health_payload.get("status")
