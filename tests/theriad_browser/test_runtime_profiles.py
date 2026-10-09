from pathlib import Path

import pytest

from theriad_browser.browser_agent import BrowserAgentStartRequest, STORE
from theriad_browser.chrome import chrome_status
from theriad_browser.mcp_broker import MCPBroker
from theriad_browser.runtime_config import BrowserProfileMode, RuntimeConfig, RuntimeProfile


def test_default_runtime_profile_is_consumer_secure(monkeypatch):
    monkeypatch.delenv("THERIAD_RUNTIME_PROFILE", raising=False)
    assert RuntimeConfig.from_env({}).runtime_profile == RuntimeProfile.CONSUMER_SECURE


def test_consumer_secure_forces_ephemeral_and_blocks_real_profile():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE, browser_profile_mode=BrowserProfileMode.REAL_PROFILE, allow_real_chrome_profile=True).validated()
    assert cfg.browser_profile_mode == BrowserProfileMode.EPHEMERAL
    assert cfg.allow_real_chrome_profile is False
    assert cfg.approval_mode == "strict"


def test_consumer_secure_requires_sensitive_approval():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE).validated()
    session = STORE.start(BrowserAgentStartRequest(goal="submit form"), cfg)
    for _ in range(8):
        session._last_tick = 0
        session.tick()
    assert session.status == "waiting_for_approval"


def test_personal_trusted_can_enable_dedicated_persistent():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.PERSONAL_TRUSTED, browser_profile_mode=BrowserProfileMode.DEDICATED_PERSISTENT).validated()
    assert cfg.browser_profile_mode == BrowserProfileMode.DEDICATED_PERSISTENT
    assert cfg.effective_user_data_dir.endswith(".theriad/chrome-profile")


def test_personal_trusted_real_profile_requires_explicit_confirmation(tmp_path):
    with pytest.raises(ValueError):
        RuntimeConfig(runtime_profile=RuntimeProfile.PERSONAL_TRUSTED, browser_profile_mode=BrowserProfileMode.REAL_PROFILE, allow_real_chrome_profile=True, chrome_user_data_dir=str(tmp_path)).validated()
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.PERSONAL_TRUSTED, browser_profile_mode=BrowserProfileMode.REAL_PROFILE, allow_real_chrome_profile=True, real_profile_confirmation=True, chrome_user_data_dir=str(tmp_path)).validated()
    assert cfg.browser_profile_mode == BrowserProfileMode.REAL_PROFILE


def test_personal_trusted_still_requires_hard_stop_approval():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.PERSONAL_TRUSTED).validated()
    assert "purchase" in cfg.approval_required_actions


def test_mcp_broker_receives_profile_and_applies_allowlist():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE).validated()
    broker = MCPBroker(cfg)
    entry = broker.record_call("s1", "chrome", "filesystem.read_project", {"api_key": "secret"}, "use_local_filesystem")
    assert entry.runtime_profile == "consumer_secure"
    assert entry.status == "blocked"
    assert entry.redacted_args["api_key"] == "[REDACTED]"


def test_chrome_status_endpoint_shape_and_no_secrets():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE, chrome_mcp_enabled=True).validated()
    status = chrome_status(cfg)
    assert status["runtime_profile"] == "consumer_secure"
    assert status["browser_profile_mode"] == "ephemeral"
    assert "chrome_user_data_dir" not in status
    assert status["mode"] in {"mock", "chrome_mcp"}


def test_app_starts_normally_if_chrome_mcp_missing():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE, chrome_mcp_enabled=False).validated()
    status = chrome_status(cfg)
    assert status["available"] is False
    assert status["mode"] == "mock"


def test_execute_route_unaffected_contract_documented():
    server_text = Path("theriad_browser/server.py").read_text()
    assert "/api/execute" in server_text
    assert "execute route unaffected" in server_text


def test_preview_iframe_separate_contract_documented():
    server_text = Path("theriad_browser/server.py").read_text()
    assert "sandbox" in server_text
    assert "isolated from browser-agent session data" in server_text


def test_consumer_secure_rejects_real_chrome_profile_path():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE, chrome_user_data_dir="/home/me/.config/google-chrome/Default").validated()
    assert cfg.chrome_user_data_dir is None


def test_ephemeral_profile_is_cleared_after_session():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.CONSUMER_SECURE).validated()
    path = Path(cfg.effective_user_data_dir)
    assert path.exists()
    cfg.cleanup()
    assert not path.exists()


def test_dedicated_persistent_uses_theriad_browser_owned_profile_dir():
    cfg = RuntimeConfig(runtime_profile=RuntimeProfile.PERSONAL_TRUSTED, browser_profile_mode=BrowserProfileMode.DEDICATED_PERSISTENT).validated()
    assert str(Path.home() / ".theriad" / "chrome-profile") == cfg.effective_user_data_dir
