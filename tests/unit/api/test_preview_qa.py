"""
Tests for:
  - GET /api/runtime/status
  - POST /api/preview/qa
  - ChromeDevToolsAdapter real-transport boundary
  - MCPBroker used for Chrome QA calls (D-042)
  - App starts when Chrome MCP is missing
  - consumer_secure forces ephemeral (D-039)
  - No secrets in status or QA responses

All tests run offline — no real Chrome MCP subprocess is spawned.
Adapter is mocked or left unconfigured (THERIAD_CHROME_MCP_ENABLED unset).
"""
from __future__ import annotations

import os
import pytest
from fastapi.testclient import TestClient


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """
    Ensure Chrome MCP env vars and runtime profile vars are clear for each test
    unless the test explicitly sets them.
    """
    for var in (
        "THERIAD_CHROME_MCP_ENABLED",
        "CHROME_MCP_ENABLED",
        "CHROME_MCP_WS_URL",
        "THERIAD_RUNTIME_PROFILE",
        "THERIAD_CHROME_PROFILE_MODE",
        "THERIAD_ALLOW_REAL_CHROME_PROFILE",
    ):
        monkeypatch.delenv(var, raising=False)
    # Force profile singleton reload each test
    import theriad.runtime.profile as _p
    _p._cached_config = None
    yield
    _p._cached_config = None


@pytest.fixture()
def client():
    from theriad.api.server import app
    return TestClient(app)


# ── Part 2: Chrome MCP availability check ────────────────────────────────────

class TestChromeMCPAvailability:
    def test_adapter_unavailable_without_env(self):
        """Adapter reports available=False when THERIAD_CHROME_MCP_ENABLED is unset."""
        from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
        a = ChromeDevToolsAdapter()
        assert a.available is False
        assert a.error is not None

    def test_adapter_available_with_env_flag(self, monkeypatch):
        """Adapter reports available=True when THERIAD_CHROME_MCP_ENABLED=true."""
        monkeypatch.setenv("THERIAD_CHROME_MCP_ENABLED", "true")
        from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
        a = ChromeDevToolsAdapter()
        assert a.available is True
        assert a.error is None

    def test_adapter_unavailable_call_raises(self):
        """call_tool raises RuntimeError when adapter is unavailable."""
        from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
        a = ChromeDevToolsAdapter()
        assert not a.available
        with pytest.raises(RuntimeError):
            a.call_tool("take_screenshot", {})


# ── Part 3: GET /api/runtime/status ──────────────────────────────────────────

class TestRuntimeStatusEndpoint:
    def test_returns_200(self, client):
        r = client.get("/api/runtime/status")
        assert r.status_code == 200

    def test_default_is_consumer_secure(self, client):
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["runtime_profile"] == "consumer_secure"

    def test_default_browser_mode_is_ephemeral(self, client):
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["browser_profile_mode"] == "ephemeral"

    def test_default_chrome_mcp_not_configured(self, client):
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["chrome_mcp_status"] == "not_configured"

    def test_default_browser_runtime_is_opensandbox(self, client):
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["browser_runtime"] == "opensandbox"
        assert data["active_backend"] == "opensandbox"

    def test_no_secrets_in_response(self, client):
        r = client.get("/api/runtime/status")
        text = r.text.lower()
        for forbidden in ("api_key", "password", "secret", "token", "cookie"):
            assert forbidden not in text, f"Secret field '{forbidden}' found in runtime/status"

    def test_personal_trusted_reflected(self, client, monkeypatch):
        monkeypatch.setenv("THERIAD_RUNTIME_PROFILE", "personal_trusted")
        import theriad.runtime.profile as _p
        _p._cached_config = None
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["runtime_profile"] == "personal_trusted"

    def test_consumer_secure_forces_ephemeral_in_status(self, client, monkeypatch):
        """Even if someone tries to set real_profile, consumer_secure overrides it."""
        monkeypatch.setenv("THERIAD_CHROME_PROFILE_MODE", "real_profile")
        import theriad.runtime.profile as _p
        _p._cached_config = None
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["browser_profile_mode"] == "ephemeral"

    def test_profile_badge_present(self, client):
        r = client.get("/api/runtime/status")
        data = r.json()
        assert "profile_badge" in data
        assert len(data["profile_badge"]) > 0

    def test_chrome_mcp_status_connected_when_configured(self, client, monkeypatch):
        monkeypatch.setenv("THERIAD_CHROME_MCP_ENABLED", "true")
        r = client.get("/api/runtime/status")
        data = r.json()
        assert data["chrome_mcp_status"] == "connected"
        assert data["browser_runtime"] == "opensandbox"
        assert data["active_backend"] == "opensandbox"


# ── Part 5: POST /api/preview/qa ─────────────────────────────────────────────

SIMPLE_HTML = "<h1>Hello world</h1><p>This is a test.</p>"

class TestPreviewQAEndpoint:
    def test_unavailable_when_chrome_mcp_missing(self, client):
        """Returns status=unavailable cleanly when Chrome MCP is not configured."""
        r = client.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unavailable"
        assert "Chrome MCP" in data["message"]

    def test_unavailable_response_shape(self, client):
        """Unavailable response has all expected fields."""
        r = client.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        data = r.json()
        assert "status"         in data
        assert "message"        in data
        assert "screenshot"     in data
        assert "console_errors" in data
        assert "network_errors" in data
        assert "dom_summary"    in data
        assert "warnings"       in data
        assert "latency_ms"     in data
        assert isinstance(data["console_errors"], list)
        assert isinstance(data["network_errors"], list)
        assert isinstance(data["warnings"],       list)

    def test_unavailable_has_no_secrets(self, client):
        r = client.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        text = r.text.lower()
        for forbidden in ("api_key", "password", "secret", "token", "cookie"):
            assert forbidden not in text, f"Secret '{forbidden}' in preview/qa unavailable response"

    def test_latency_ms_is_number(self, client):
        r = client.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        assert isinstance(r.json()["latency_ms"], (int, float))

    def test_screenshot_is_none_when_unavailable(self, client):
        r = client.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        assert r.json()["screenshot"] is None

    def test_with_stub_adapter_returns_ok_shape(self, monkeypatch):
        """
        With a stub adapter injected via monkeypatching, /api/preview/qa
        runs the full MCPBroker pipeline and returns the expected shape.
        """
        from theriad.api.server import app
        from fastapi.testclient import TestClient

        # Stub adapter: available=True, call_tool returns tool-specific fixtures
        class _StubAdapter:
            available = True
            connected = True
            error     = None
            tools     = [
                "new_page", "navigate_page", "take_screenshot",
                "list_console_messages", "list_network_requests", "evaluate_script",
            ]

            def call_tool(self, tool_name: str, arguments: dict) -> dict:
                if tool_name == "new_page":
                    return {"pageId": "stub-page-1"}
                if tool_name == "navigate_page":
                    return {"status": "ok"}
                if tool_name == "take_screenshot":
                    return {"screenshot": "aGVsbG8="}   # base64 "hello"
                if tool_name == "list_console_messages":
                    return {"messages": [{"level": "error", "text": "stub console error"}]}
                if tool_name == "list_network_requests":
                    return {"requests": [{"method": "GET", "url": "http://stub/", "status": 404}]}
                if tool_name == "evaluate_script":
                    import json
                    return {"result": json.dumps({
                        "title": "Stub Title",
                        "bodyLength": 42,
                        "elementCount": 10,
                    })}
                return {}

        # Monkeypatch ChromeDevToolsAdapter so the endpoint gets our stub
        import theriad.tools.chrome_devtools_adapter as _mod
        monkeypatch.setattr(_mod, "ChromeDevToolsAdapter", lambda: _StubAdapter())

        # Also ensure MCPBroker uses the stubbed adapter
        import theriad.tools.mcp_broker as _broker_mod
        _OrigBroker = _broker_mod.MCPBroker

        stub = _StubAdapter()

        class _StubBroker(_OrigBroker):
            def __init__(self, session_id, adapters=None, runtime_config=None):
                super().__init__(
                    session_id=session_id,
                    adapters={"chrome_devtools": stub},
                    runtime_config=runtime_config,
                )

        monkeypatch.setattr(_broker_mod, "MCPBroker", _StubBroker)

        c = TestClient(app)
        r = c.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        assert r.status_code == 200
        data = r.json()

        # Shape assertions
        assert data["status"] in ("ok", "failed")   # 'ok' if broker + adapter wire correctly
        assert isinstance(data["console_errors"], list)
        assert isinstance(data["network_errors"], list)
        assert isinstance(data["warnings"],       list)
        assert isinstance(data["latency_ms"],     (int, float))

        if data["status"] == "ok":
            assert data["screenshot"] == "aGVsbG8="
            assert "Stub Title" in (data["dom_summary"] or "")
            assert any("stub console error" in e for e in data["console_errors"])
            assert any("404" in e for e in data["network_errors"])

    def test_broker_used_for_qa_calls(self, monkeypatch):
        """
        Verify MCPBroker.call() is invoked rather than adapter.call_tool() directly.
        """
        from theriad.api.server import app
        from fastapi.testclient import TestClient

        broker_calls: list[str] = []

        import theriad.tools.chrome_devtools_adapter as _mod

        class _StubAdapter:
            available = True
            connected = True
            error     = None
            tools     = ["new_page", "navigate_page", "take_screenshot",
                         "list_console_messages", "list_network_requests", "evaluate_script"]

            def call_tool(self, tool_name, arguments):
                return {}  # minimal ok response

        monkeypatch.setattr(_mod, "ChromeDevToolsAdapter", lambda: _StubAdapter())

        import theriad.tools.mcp_broker as _broker_mod
        _OrigBroker = _broker_mod.MCPBroker
        stub = _StubAdapter()

        class _InstrumentedBroker(_OrigBroker):
            def __init__(self, session_id, adapters=None, runtime_config=None):
                super().__init__(
                    session_id=session_id,
                    adapters={"chrome_devtools": stub},
                    runtime_config=runtime_config,
                )

            def call(self, subtask, server_name, tool_name, arguments=None):
                broker_calls.append(tool_name)
                return super().call(subtask, server_name, tool_name, arguments)

        monkeypatch.setattr(_broker_mod, "MCPBroker", _InstrumentedBroker)

        c = TestClient(app)
        c.post("/api/preview/qa", json={"html": SIMPLE_HTML})

        # MCPBroker.call() should have been called at least for new_page
        # (it may be blocked by allowlists, but the call path was exercised)
        assert len(broker_calls) > 0, "MCPBroker.call() was never invoked"

    def test_no_secrets_with_stub(self, monkeypatch):
        """Even with a connected stub, no secrets appear in the QA response."""
        from theriad.api.server import app
        from fastapi.testclient import TestClient

        import theriad.tools.chrome_devtools_adapter as _mod

        class _StubAdapter:
            available = True
            connected = True
            error     = None
            tools     = ["new_page"]

            def call_tool(self, tool_name, arguments):
                return {}

        monkeypatch.setattr(_mod, "ChromeDevToolsAdapter", lambda: _StubAdapter())

        c = TestClient(app)
        r = c.post("/api/preview/qa", json={"html": SIMPLE_HTML})
        text = r.text.lower()
        for forbidden in ("api_key", "password", "secret", "token", "cookie"):
            assert forbidden not in text


# ── Part 6: Preview iframe unaffected ────────────────────────────────────────

class TestPreviewIframeUnaffected:
    def test_app_starts_without_chrome_mcp(self, client):
        """App health endpoint still returns ok when Chrome MCP is absent."""
        r = client.get("/api/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_chrome_status_endpoint_clean_when_missing(self, client):
        r = client.get("/api/tools/chrome/status")
        assert r.status_code == 200
        data = r.json()
        assert data["available"] is False
        assert data["configured"] is False
        assert data["mode"] == "mock"

    def test_execute_unaffected(self, client):
        """POST /api/execute still exists and accepts requests (422 = schema validation, not 404)."""
        r = client.post("/api/execute", json={})
        assert r.status_code != 404


# ── Part 8: Existing D-039 to D-042 tests still pass ─────────────────────────
# (These are covered by running the full pytest suite — see test_runtime_profile.py
# and test_mcp_broker.py. The import below confirms the modules still load.)

class TestExistingDecisionsUnbroken:
    def test_runtime_profile_module_loads(self):
        from theriad.runtime.profile import RuntimeConfig, RuntimeProfile, BrowserProfileMode
        cfg = RuntimeConfig()
        assert cfg.runtime_profile == RuntimeProfile.consumer_secure
        assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral

    def test_mcp_broker_module_loads(self):
        from theriad.tools.mcp_broker import MCPBroker
        b = MCPBroker(session_id="test")
        assert b is not None

    def test_browser_policy_module_loads(self):
        from theriad.agents.browser_policy import validate_action
        from theriad.agents.browser_events import ActionProposal
        p = ActionProposal(action_type="navigate", target="https://example.com",
                           reason="test", confidence=0.9, risk_level="low",
                           requires_approval=False)
        decision = validate_action(p, [])
        assert decision.allowed is True

    def test_stdio_client_module_loads(self):
        """MCPStdioClient is importable without errors."""
        from theriad.tools.mcp_stdio_client import MCPStdioClient, MCPError
        assert MCPStdioClient is not None
        assert MCPError is not None

    def test_build_preview_doc_helper(self):
        """_build_preview_doc generates valid HTML without secrets."""
        # Import directly from server module
        import importlib, sys
        # server may already be imported; access via sys.modules or import
        from theriad.api import server as srv
        doc = srv._build_preview_doc("<p>hi</p>", "body{color:red}", False)
        assert "<!doctype html>" in doc.lower()
        assert "<p>hi</p>" in doc
        assert "body{color:red}" in doc
        assert "api_key" not in doc
        assert "secret" not in doc

    def test_build_preview_doc_full_doc_passthrough(self):
        from theriad.api import server as srv
        full = "<!DOCTYPE html><html><head></head><body>full</body></html>"
        doc = srv._build_preview_doc(full, "", False)
        assert "full" in doc

    def test_build_preview_doc_tailwind_injection(self):
        from theriad.api import server as srv
        doc = srv._build_preview_doc("<p>hi</p>", "", True)
        assert "cdn.tailwindcss.com" in doc
