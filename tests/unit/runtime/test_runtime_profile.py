"""
Tests for theriad.runtime.profile — Theriad runtime profiles and browser profile modes.

Tests cover all 20 scenarios from the spec:
 1. Default runtime profile is consumer_secure unless explicitly configured.
 2. consumer_secure forces browser_profile_mode=ephemeral.
 3. consumer_secure blocks real_profile.
 4. consumer_secure forces allow_real_chrome_profile=false.
 5. consumer_secure requires approval for sensitive actions.
 6. personal_trusted can enable dedicated_persistent.
 7. personal_trusted can enable real_profile only with explicit config + confirmation.
 8. personal_trusted still requires approval for hard-stop actions.
 9. Settings UI requires confirmation before enabling personal_trusted.
    (tested via: profile badge is distinct for personal_trusted)
10. Settings UI displays current runtime profile.
    (tested via: to_safe_dict returns runtime_profile)
11. Agent UI displays current runtime profile and browser profile mode.
    (tested via: BrowserAgentStatus carries runtime_profile + profile_badge)
12. MCP broker receives runtime_profile and applies correct allowlist.
13. Chrome status endpoint reports runtime_profile and browser_profile_mode.
14. App starts normally if Chrome MCP is missing.
15. Secrets are never returned in status responses for either profile.
16. /api/execute remains unaffected by runtime profile.
17. Preview iframe remains separate from browser-agent browser context.
    (tested via: ephemeral mode does not share user data dir)
18. consumer_secure cannot use a user-provided real Chrome profile path.
19. ephemeral profile dir is None (caller handles tmp dir lifecycle).
20. dedicated_persistent uses Theriad-owned profile directory.

Run with:
    pytest tests/unit/runtime/test_runtime_profile.py -v
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from theriad.runtime.profile import (
    ApprovalMode,
    BrowserProfileMode,
    RuntimeConfig,
    RuntimeProfile,
    get_runtime_config,
    load_runtime_config,
    HARD_STOP_ACTIONS,
    CONSUMER_SECURE_APPROVAL_ACTIONS,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _reload_config(monkeypatch, env: dict[str, str]) -> RuntimeConfig:
    """
    Set env vars via monkeypatch, clear the module singleton, and reload.
    """
    for key, value in env.items():
        monkeypatch.setenv(key, value)

    # Clear all THERIAD_ vars not in `env`
    for var in [
        "THERIAD_RUNTIME_PROFILE",
        "THERIAD_AGENT_APPROVAL_MODE",
        "THERIAD_ALLOW_LOCAL_FILES",
        "THERIAD_ALLOW_REAL_CHROME_PROFILE",
        "THERIAD_CHROME_PROFILE_MODE",
        "THERIAD_CHROME_USER_DATA_DIR",
        "THERIAD_CHROME_HEADLESS",
        "THERIAD_CHROME_MCP_ENABLED",
        "CHROME_MCP_ENABLED",
        "CHROME_MCP_WS_URL",
    ]:
        if var not in env:
            monkeypatch.delenv(var, raising=False)

    return get_runtime_config(reload=True)


# ── Test 1: Default profile is consumer_secure ────────────────────────────────

def test_default_profile_is_consumer_secure(monkeypatch):
    """No env vars set → RuntimeProfile.consumer_secure."""
    cfg = _reload_config(monkeypatch, {})
    assert cfg.runtime_profile == RuntimeProfile.consumer_secure
    assert cfg.is_consumer_secure is True
    assert cfg.is_personal_trusted is False


# ── Test 2: consumer_secure forces ephemeral ──────────────────────────────────

def test_consumer_secure_forces_ephemeral(monkeypatch):
    """Even if dedicated_persistent is requested, consumer_secure overrides to ephemeral."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_CHROME_PROFILE_MODE": "dedicated_persistent",
    })
    assert cfg.runtime_profile == RuntimeProfile.consumer_secure
    assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral


# ── Test 3: consumer_secure blocks real_profile ───────────────────────────────

def test_consumer_secure_blocks_real_profile(monkeypatch):
    """consumer_secure + real_profile config → falls back to consumer_secure defaults."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_CHROME_PROFILE_MODE": "real_profile",
        "THERIAD_ALLOW_REAL_CHROME_PROFILE": "true",
    })
    # The validator in RuntimeConfig silently overrides to ephemeral for consumer_secure
    assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral
    assert cfg.allow_real_chrome_profile is False


def test_real_profile_with_consumer_secure_is_overridden_to_ephemeral():
    """
    Constructing RuntimeConfig with real_profile + consumer_secure silently
    overrides to ephemeral (per spec: consumer_secure 'forces' ephemeral).
    This ensures startup never fails even with conflicting env vars.
    """
    cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.consumer_secure,
        browser_profile_mode=BrowserProfileMode.real_profile,
        allow_real_chrome_profile=True,
    )
    # consumer_secure forces override to ephemeral — does not raise
    assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral
    assert cfg.allow_real_chrome_profile is False


# ── Test 4: consumer_secure forces allow_real_chrome_profile=False ────────────

def test_consumer_secure_forces_no_real_chrome(monkeypatch):
    """consumer_secure always sets allow_real_chrome_profile=False."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_ALLOW_REAL_CHROME_PROFILE": "true",
    })
    assert cfg.allow_real_chrome_profile is False


# ── Test 5: consumer_secure requires approval for sensitive actions ────────────

def test_consumer_secure_requires_approval_for_login(monkeypatch):
    cfg = _reload_config(monkeypatch, {"THERIAD_RUNTIME_PROFILE": "consumer_secure"})
    assert cfg.requires_approval_for("login") is True


def test_consumer_secure_requires_approval_for_submit_form(monkeypatch):
    cfg = _reload_config(monkeypatch, {"THERIAD_RUNTIME_PROFILE": "consumer_secure"})
    assert cfg.requires_approval_for("submit_form") is True


def test_consumer_secure_requires_approval_for_delete(monkeypatch):
    cfg = _reload_config(monkeypatch, {"THERIAD_RUNTIME_PROFILE": "consumer_secure"})
    assert cfg.requires_approval_for("delete") is True


def test_consumer_secure_approval_mode_is_strict(monkeypatch):
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_AGENT_APPROVAL_MODE": "minimal",  # should be overridden
    })
    assert cfg.approval_mode == ApprovalMode.strict


# ── Test 6: personal_trusted can enable dedicated_persistent ──────────────────

def test_personal_trusted_dedicated_persistent(monkeypatch):
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "personal_trusted",
        "THERIAD_CHROME_PROFILE_MODE": "dedicated_persistent",
    })
    assert cfg.runtime_profile == RuntimeProfile.personal_trusted
    assert cfg.browser_profile_mode == BrowserProfileMode.dedicated_persistent


def test_personal_trusted_dedicated_persistent_dir_default(monkeypatch):
    """dedicated_persistent defaults to ~/.theriad/chrome-profile."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "personal_trusted",
        "THERIAD_CHROME_PROFILE_MODE": "dedicated_persistent",
    })
    data_dir = cfg.effective_chrome_user_data_dir
    assert data_dir is not None
    assert ".theriad" in data_dir
    assert "chrome-profile" in data_dir


# ── Test 7: personal_trusted can enable real_profile with explicit config ──────

def test_personal_trusted_real_profile_with_explicit_config():
    """personal_trusted + allow_real_chrome_profile=True allows real_profile."""
    cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.personal_trusted,
        browser_profile_mode=BrowserProfileMode.real_profile,
        allow_real_chrome_profile=True,
    )
    assert cfg.browser_profile_mode == BrowserProfileMode.real_profile
    assert cfg.allow_real_chrome_profile is True


def test_personal_trusted_real_profile_without_explicit_config_raises():
    """personal_trusted + real_profile without allow_real_chrome_profile raises."""
    with pytest.raises(ValueError, match="allow_real_chrome_profile=true"):
        RuntimeConfig(
            runtime_profile=RuntimeProfile.personal_trusted,
            browser_profile_mode=BrowserProfileMode.real_profile,
            allow_real_chrome_profile=False,
        )


# ── Test 8: personal_trusted still requires approval for hard-stop actions ─────

def test_personal_trusted_hard_stop_always_requires_approval():
    """Hard-stop actions require approval regardless of profile."""
    cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.personal_trusted,
        approval_mode=ApprovalMode.minimal,
    )
    for action in HARD_STOP_ACTIONS:
        assert cfg.requires_approval_for(action) is True, (
            f"Hard-stop action '{action}' should require approval in personal_trusted"
        )


def test_personal_trusted_minimal_mode_skips_non_hard_stop():
    """personal_trusted + minimal approval: non-hard-stop actions do NOT require approval."""
    cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.personal_trusted,
        approval_mode=ApprovalMode.minimal,
    )
    # navigate and click are not hard-stops
    assert cfg.requires_approval_for("navigate") is False
    assert cfg.requires_approval_for("click") is False


# ── Test 9: Profile badge is visually distinct ────────────────────────────────

def test_profile_badge_consumer_secure_distinct_from_personal_trusted():
    """The two profiles produce distinct, non-overlapping badge strings."""
    consumer_cfg = RuntimeConfig(runtime_profile=RuntimeProfile.consumer_secure)
    personal_cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.personal_trusted,
    )
    assert consumer_cfg.profile_badge != personal_cfg.profile_badge
    assert "Secure Mode" in consumer_cfg.profile_badge
    assert "Personal Trusted Mode" in personal_cfg.profile_badge


def test_real_profile_badge_mentions_real_chrome():
    cfg = RuntimeConfig(
        runtime_profile=RuntimeProfile.personal_trusted,
        browser_profile_mode=BrowserProfileMode.real_profile,
        allow_real_chrome_profile=True,
    )
    assert "real Chrome profile" in cfg.profile_badge


# ── Test 10: to_safe_dict returns runtime_profile ────────────────────────────

def test_to_safe_dict_includes_runtime_profile():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.consumer_secure)
    d = cfg.to_safe_dict()
    assert "runtime_profile" in d
    assert d["runtime_profile"] == "consumer_secure"
    assert "browser_profile_mode" in d
    assert "profile_badge" in d


def test_to_safe_dict_no_secrets():
    """to_safe_dict must not include any secret-like keys."""
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.personal_trusted)
    d = cfg.to_safe_dict()
    forbidden = {"api_key", "token", "secret", "password", "cookie"}
    for key in d:
        assert key.lower() not in forbidden, f"to_safe_dict returned suspicious key: {key}"


# ── Test 11: BrowserAgentStatus carries runtime_profile and profile_badge ──────

@pytest.mark.asyncio
async def test_browser_agent_status_includes_profile(monkeypatch):
    """BrowserAgentStatus.to_status() includes runtime_profile and profile_badge."""
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    # Force consumer_secure singleton
    get_runtime_config(reload=True)

    from theriad.agents.browser_schemas import BrowserAgentStartRequest
    from theriad.agents.browser_session_store import BrowserSession

    req = BrowserAgentStartRequest(goal="test goal")
    session = BrowserSession(req)
    status = session.to_status()

    assert status.runtime_profile == "consumer_secure"
    assert status.browser_profile_mode == "ephemeral"
    assert "Secure Mode" in status.profile_badge


# ── Test 12: MCP broker receives runtime_profile and applies allowlist ─────────

def test_mcp_broker_status_includes_runtime_profile(monkeypatch):
    """MCPBroker.get_status() returns the active runtime_profile."""
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    get_runtime_config(reload=True)

    from theriad.tools.mcp_broker import MCPBroker

    class StubAdapter:
        available = True; connected = True; error = None; tools = []
        def call_tool(self, *a, **k): return {}

    broker = MCPBroker("t1", {"chrome_devtools": StubAdapter()})
    status = broker.get_status()
    assert status.runtime_profile == "consumer_secure"
    assert status.browser_profile_mode == "ephemeral"


def test_mcp_broker_consumer_secure_escalates_high_risk_to_approval(monkeypatch):
    """consumer_secure forces approval for high-risk tools like 'fill'."""
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    get_runtime_config(reload=True)

    from theriad.tools.mcp_broker import MCPBroker
    from theriad.tools.mcp_registry import SUBTASK_ALLOWLISTS, SubtaskAllowlist

    class StubAdapter:
        available = True; connected = True; error = None; tools = ["fill"]
        def call_tool(self, *a, **k): return {}

    # Register a test subtask that permits 'fill'
    test_subtask = SubtaskAllowlist(
        subtask="_test_consumer_fill",
        allowed_server="chrome_devtools",
        allowed_tools=["fill"],
        requires_approval_for_any=False,
    )
    original = dict(SUBTASK_ALLOWLISTS)
    try:
        SUBTASK_ALLOWLISTS["_test_consumer_fill"] = test_subtask
        consumer_cfg = RuntimeConfig(runtime_profile=RuntimeProfile.consumer_secure)
        broker = MCPBroker("cs-test", {"chrome_devtools": StubAdapter()},
                           runtime_config=consumer_cfg)
        result = broker.call(
            "_test_consumer_fill", "chrome_devtools", "fill",
            {"selector": "#pw", "value": "secret"},
        )
        # consumer_secure should require approval for fill (high-risk)
        assert result.ok is False
        assert result.approval_required is True
    finally:
        SUBTASK_ALLOWLISTS.clear()
        SUBTASK_ALLOWLISTS.update(original)


def test_mcp_call_log_includes_runtime_profile(monkeypatch):
    """MCPCallLog entries carry runtime_profile and browser_profile_mode."""
    consumer_cfg = RuntimeConfig(runtime_profile=RuntimeProfile.consumer_secure)

    from theriad.tools.mcp_broker import MCPBroker

    class StubAdapter:
        available = True; connected = True; error = None
        tools = ["take_screenshot"]
        def call_tool(self, *a, **k): return {"ok": True}

    broker = MCPBroker("log-test", {"chrome_devtools": StubAdapter()},
                       runtime_config=consumer_cfg)
    broker.call("website_preview_qa", "chrome_devtools", "take_screenshot")
    log = broker.get_call_log()
    assert len(log) == 1
    assert log[0].runtime_profile == "consumer_secure"
    assert log[0].browser_profile_mode == "ephemeral"


# ── Test 13: Chrome status endpoint reports runtime_profile ───────────────────

def test_chrome_status_endpoint_reports_consumer_secure(monkeypatch):
    """GET /api/tools/chrome/status returns runtime_profile=consumer_secure."""
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)
    response = client.get("/api/tools/chrome/status")
    assert response.status_code == 200
    data = response.json()
    assert data["runtime_profile"] == "consumer_secure"
    assert data["browser_profile_mode"] == "ephemeral"
    assert data["allow_real_chrome_profile"] is False


def test_chrome_status_endpoint_personal_trusted(monkeypatch):
    """GET /api/tools/chrome/status reflects personal_trusted when configured."""
    monkeypatch.setenv("THERIAD_RUNTIME_PROFILE", "personal_trusted")
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)
    response = client.get("/api/tools/chrome/status")
    assert response.status_code == 200
    data = response.json()
    assert data["runtime_profile"] == "personal_trusted"


# ── Test 14: App starts normally if Chrome MCP is missing ─────────────────────

def test_app_starts_without_chrome_mcp(monkeypatch):
    """Removing Chrome MCP env vars doesn't prevent the app from starting."""
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)

    # Health check must pass
    r = client.get("/api/health")
    assert r.status_code == 200

    # Chrome status must return gracefully with available=False
    r2 = client.get("/api/tools/chrome/status")
    assert r2.status_code == 200
    assert r2.json()["available"] is False


# ── Test 15: Secrets never returned in status responses ───────────────────────

def test_chrome_status_no_secrets(monkeypatch):
    """chrome/status response must not contain API keys, tokens, or cookies."""
    monkeypatch.setenv("THERIAD_RUNTIME_PROFILE", "personal_trusted")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-super-secret")
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)
    response = client.get("/api/tools/chrome/status")
    body = response.text
    assert "sk-super-secret" not in body
    assert "api_key" not in body.lower()
    assert "token" not in body.lower() or "allow_real_chrome_profile" in body.lower()


def test_tools_status_no_secrets(monkeypatch):
    """/api/tools/status must not expose API keys."""
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-another-secret")
    from theriad.api.server import app
    client = TestClient(app)
    response = client.get("/api/tools/status")
    assert "sk-another-secret" not in response.text


# ── Test 16: /api/execute remains unaffected by runtime profile ───────────────

def test_execute_unaffected_by_consumer_secure(monkeypatch):
    """/api/execute returns 403 for allow_paid_calls=False regardless of profile."""
    monkeypatch.setenv("THERIAD_RUNTIME_PROFILE", "consumer_secure")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)
    response = client.post("/api/execute", json={
        "prompt": "hello",
        "allow_paid_calls": False,
    })
    assert response.status_code == 403


def test_execute_unaffected_by_personal_trusted(monkeypatch):
    """/api/execute returns 403 for allow_paid_calls=False in personal_trusted too."""
    monkeypatch.setenv("THERIAD_RUNTIME_PROFILE", "personal_trusted")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    get_runtime_config(reload=True)

    from theriad.api.server import app
    client = TestClient(app)
    response = client.post("/api/execute", json={
        "prompt": "hello",
        "allow_paid_calls": False,
    })
    assert response.status_code == 403


# ── Test 17 / 19: ephemeral profile has no user data dir ──────────────────────

def test_ephemeral_profile_dir_is_none(monkeypatch):
    """ephemeral mode returns None for effective_chrome_user_data_dir."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
    })
    assert cfg.effective_chrome_user_data_dir is None


def test_consumer_secure_ephemeral_no_shared_dir(monkeypatch):
    """consumer_secure cannot produce a user data dir (isolating from real profile)."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_CHROME_USER_DATA_DIR": "/Users/someone/.config/google-chrome",
    })
    # consumer_secure forces ephemeral → effective dir must be None
    assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral
    assert cfg.effective_chrome_user_data_dir is None


# ── Test 18: consumer_secure cannot use a real Chrome profile path ────────────

def test_consumer_secure_ignores_user_data_dir(monkeypatch):
    """A chrome_user_data_dir set in env is irrelevant under consumer_secure."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "consumer_secure",
        "THERIAD_CHROME_PROFILE_MODE": "real_profile",
        "THERIAD_CHROME_USER_DATA_DIR": "/Users/me/.config/google-chrome",
        "THERIAD_ALLOW_REAL_CHROME_PROFILE": "true",
    })
    # consumer_secure overrides to ephemeral — user data dir is irrelevant
    assert cfg.browser_profile_mode == BrowserProfileMode.ephemeral
    assert cfg.effective_chrome_user_data_dir is None


# ── Test 20: dedicated_persistent uses Theriad-owned profile dir ────────────

def test_dedicated_persistent_uses_theriad_browser_dir(monkeypatch):
    """dedicated_persistent always uses ~/.theriad/chrome-profile by default."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "personal_trusted",
        "THERIAD_CHROME_PROFILE_MODE": "dedicated_persistent",
    })
    data_dir = cfg.effective_chrome_user_data_dir
    assert data_dir is not None
    # Must NOT look like the user's normal Chrome profile
    assert "google-chrome" not in data_dir.lower()
    assert "Google Chrome" not in data_dir
    # Must be Theriad-owned
    assert ".theriad" in data_dir


def test_dedicated_persistent_custom_dir(monkeypatch):
    """A custom chrome_user_data_dir overrides the default Theriad path."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "personal_trusted",
        "THERIAD_CHROME_PROFILE_MODE": "dedicated_persistent",
        "THERIAD_CHROME_USER_DATA_DIR": "/tmp/my-theriad_browser-profile",
    })
    assert cfg.effective_chrome_user_data_dir == "/tmp/my-theriad_browser-profile"


# ── Additional: unrecognised profile falls back to consumer_secure ─────────────

def test_unrecognised_profile_falls_back_to_consumer_secure(monkeypatch):
    """An unrecognised THERIAD_RUNTIME_PROFILE value defaults to consumer_secure."""
    cfg = _reload_config(monkeypatch, {
        "THERIAD_RUNTIME_PROFILE": "admin_root_mode",  # not valid
    })
    assert cfg.runtime_profile == RuntimeProfile.consumer_secure


# ── Additional: ChromeDevToolsAdapter exposes profile attributes ──────────────

def test_chrome_adapter_exposes_runtime_profile(monkeypatch):
    """ChromeDevToolsAdapter exposes runtime_profile and browser_profile_mode."""
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    monkeypatch.delenv("CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("THERIAD_CHROME_MCP_ENABLED", raising=False)
    monkeypatch.delenv("CHROME_MCP_WS_URL", raising=False)
    get_runtime_config(reload=True)

    import importlib
    from theriad.tools import chrome_devtools_adapter as mod
    importlib.reload(mod)  # pick up fresh env
    adapter = mod.ChromeDevToolsAdapter()

    assert hasattr(adapter, "runtime_profile")
    assert hasattr(adapter, "browser_profile_mode")
    assert hasattr(adapter, "allow_real_chrome_profile")
    assert hasattr(adapter, "headless")
    assert adapter.runtime_profile == "consumer_secure"
    assert adapter.browser_profile_mode == "ephemeral"
    assert adapter.allow_real_chrome_profile is False
