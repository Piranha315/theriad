"""
Tests for theriad.settings.secrets

Covers:
  - OpenRouter, OpenAI, and Anthropic env/local resolution
  - environment values take precedence over local values
  - masking never exposes full key
  - provider-isolated save / delete roundtrips
  - active assistant provider preference
  - file permissions (chmod 600/700)
  - get_openrouter_key_status() returns correct source
  - empty key raises ValueError on save
"""

from __future__ import annotations

import json
import stat

import pytest

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture()
def tmp_theriad_dir(tmp_path, monkeypatch):
    """Redirect ~/.theriad to a temp directory for the duration of a test."""
    theriad_dir = tmp_path / ".theriad"
    monkeypatch.setattr("theriad.settings.secrets._THERIAD_DIR", theriad_dir)
    monkeypatch.setattr("theriad.settings.secrets._SECRETS_FILE", theriad_dir / "secrets.json")
    return theriad_dir


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    """Ensure provider API keys do not bleed between tests."""
    for env_name in ("OPENROUTER_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(env_name, raising=False)


# ── Resolution order ──────────────────────────────────────────────────────────


class TestResolutionOrder:
    def test_returns_none_when_nothing_configured(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_api_key

        assert get_openrouter_api_key() is None

    def test_env_var_returned_when_set(self, tmp_theriad_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        from theriad.settings.secrets import get_openrouter_api_key

        assert get_openrouter_api_key() == "sk-or-env-key"

    def test_local_file_returned_when_env_absent(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_api_key, save_openrouter_api_key

        save_openrouter_api_key("sk-or-file-key")
        assert get_openrouter_api_key() == "sk-or-file-key"

    def test_env_takes_precedence_over_file(self, tmp_theriad_dir, monkeypatch):
        from theriad.settings.secrets import get_openrouter_api_key, save_openrouter_api_key

        save_openrouter_api_key("sk-or-file-key")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-env-key")
        # env var wins
        assert get_openrouter_api_key() == "sk-or-env-key"

    def test_strips_whitespace_from_env(self, tmp_theriad_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "  sk-or-v1-trimmed  ")
        from theriad.settings.secrets import get_openrouter_api_key

        assert get_openrouter_api_key() == "sk-or-v1-trimmed"

    def test_hex_encoded_env_key_is_decoded(self, tmp_theriad_dir, monkeypatch):
        # `security find-generic-password -w` prints a key saved with a trailing newline as hex.
        from theriad.settings.secrets import get_openrouter_api_key, get_openrouter_key_status
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-abc123\n".encode().hex())
        assert get_openrouter_api_key() == "sk-or-v1-abc123"
        assert get_openrouter_key_status()["masked_key"].endswith("c123")

    def test_hex_looking_value_that_is_not_a_key_is_left_alone(self, tmp_theriad_dir, monkeypatch):
        from theriad.settings.secrets import get_openrouter_api_key
        monkeypatch.setenv("OPENROUTER_API_KEY", "736b2dzz-not-hex")
        assert get_openrouter_api_key() == "736b2dzz-not-hex"

    def test_strips_whitespace_from_file(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_api_key, save_openrouter_api_key

        save_openrouter_api_key("  sk-or-file-ws  ")
        assert get_openrouter_api_key() == "sk-or-file-ws"


# ── Save / delete ─────────────────────────────────────────────────────────────


class TestSaveDelete:
    def test_provider_values_are_encrypted_at_rest(self, tmp_theriad_dir):
        from theriad.settings.secrets import _SECRETS_FILE, save_openrouter_api_key

        full_key = "sk-or-v1-never-plaintext-on-disk"
        save_openrouter_api_key(full_key)

        stored = _SECRETS_FILE.read_text()
        assert full_key not in stored
        assert "enc:v1:" in stored

    def test_save_and_retrieve(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_api_key, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-test1234")
        assert get_openrouter_api_key() == "sk-or-v1-test1234"

    def test_save_overwrites_previous(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_api_key, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-first")
        save_openrouter_api_key("sk-or-v1-second")
        assert get_openrouter_api_key() == "sk-or-v1-second"

    def test_delete_removes_key(self, tmp_theriad_dir):
        from theriad.settings.secrets import (
            delete_openrouter_api_key,
            get_openrouter_api_key,
            save_openrouter_api_key,
        )

        save_openrouter_api_key("sk-or-v1-todelete")
        delete_openrouter_api_key()
        assert get_openrouter_api_key() is None

    def test_delete_is_idempotent_when_not_set(self, tmp_theriad_dir):
        from theriad.settings.secrets import delete_openrouter_api_key

        # Should not raise
        delete_openrouter_api_key()

    def test_save_empty_key_raises(self, tmp_theriad_dir):
        from theriad.settings.secrets import save_openrouter_api_key

        with pytest.raises(ValueError):
            save_openrouter_api_key("")

    def test_save_whitespace_only_raises(self, tmp_theriad_dir):
        from theriad.settings.secrets import save_openrouter_api_key

        with pytest.raises(ValueError):
            save_openrouter_api_key("   ")

    def test_other_keys_preserved_on_delete(self, tmp_theriad_dir):
        """Deleting the OpenRouter key must not clobber other keys in the file."""
        from theriad.settings.secrets import (
            _SECRETS_FILE,
            _write_secrets,
            delete_openrouter_api_key,
        )

        _write_secrets({"openrouter_api_key": "sk-or-v1-abc", "other_key": "preserve_me"})
        delete_openrouter_api_key()
        data = json.loads(_SECRETS_FILE.read_text())
        assert "other_key" in data
        assert data["other_key"] == "preserve_me"
        assert "openrouter_api_key" not in data


# ── File permissions ──────────────────────────────────────────────────────────


class TestFilePermissions:
    def test_directory_chmod_700(self, tmp_theriad_dir):
        from theriad.settings.secrets import save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-test")
        mode = stat.S_IMODE(tmp_theriad_dir.stat().st_mode)
        assert mode == 0o700, f"Expected 0o700, got {oct(mode)}"

    def test_file_chmod_600(self, tmp_theriad_dir):
        from theriad.settings.secrets import _SECRETS_FILE, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-test")
        mode = stat.S_IMODE(_SECRETS_FILE.stat().st_mode)
        assert mode == 0o600, f"Expected 0o600, got {oct(mode)}"


# ── Masking ───────────────────────────────────────────────────────────────────


class TestMasking:
    def test_masked_key_never_exposes_full_key(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key

        full_key = "sk-or-v1-supersecretapikey1234"
        save_openrouter_api_key(full_key)
        status = get_openrouter_key_status()
        assert status["masked_key"] is not None
        assert status["masked_key"] != full_key

    def test_masked_key_shows_last_4_chars(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-abcdef")
        status = get_openrouter_key_status()
        assert status["masked_key"].endswith("cdef")

    def test_masked_key_contains_bullet_chars(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-longkey")
        status = get_openrouter_key_status()
        # Must contain masking characters
        assert "•" in status["masked_key"]

    def test_env_var_masked_key_hides_value(self, tmp_theriad_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-envvalue")
        from theriad.settings.secrets import get_openrouter_key_status

        status = get_openrouter_key_status()
        assert status["masked_key"] is not None
        assert status["masked_key"] != "sk-or-v1-envvalue"


# ── Key status source ─────────────────────────────────────────────────────────


class TestKeyStatus:
    def test_status_missing_when_no_key(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_key_status

        status = get_openrouter_key_status()
        assert status["configured"] is False
        assert status["source"] == "missing"
        assert status["masked_key"] is None

    def test_status_env_when_env_var_set(self, tmp_theriad_dir, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-envtest")
        from theriad.settings.secrets import get_openrouter_key_status

        status = get_openrouter_key_status()
        assert status["configured"] is True
        assert status["source"] == "env"

    def test_status_local_settings_when_file_set(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-filetest")
        status = get_openrouter_key_status()
        assert status["configured"] is True
        assert status["source"] == "local_settings"

    def test_status_env_trumps_file(self, tmp_theriad_dir, monkeypatch):
        from theriad.settings.secrets import get_openrouter_key_status, save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-filetest")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-envtest")
        status = get_openrouter_key_status()
        assert status["source"] == "env"


# -- Multi-provider settings -------------------------------------------------


class TestMultiProviderSettings:
    @pytest.mark.parametrize(
        ("provider", "env_name", "value"),
        [
            ("openrouter", "OPENROUTER_API_KEY", "sk-or-provider-1234"),
            ("openai", "OPENAI_API_KEY", "sk-openai-provider-5678"),
            ("anthropic", "ANTHROPIC_API_KEY", "sk-ant-provider-9012"),
        ],
    )
    def test_resolves_each_provider_from_environment(
        self,
        tmp_theriad_dir,
        monkeypatch,
        provider,
        env_name,
        value,
    ):
        from theriad.settings.secrets import get_provider_api_key, get_provider_key_status

        monkeypatch.setenv(env_name, value)

        assert get_provider_api_key(provider) == value
        status = get_provider_key_status(provider)
        assert status["configured"] is True
        assert status["source"] == "env"
        assert status["masked_key"] != value

    def test_provider_writes_preserve_other_keys_and_preference(self, tmp_theriad_dir):
        from theriad.settings.secrets import (
            delete_provider_api_key,
            get_active_assistant_provider,
            get_provider_api_key,
            save_provider_api_key,
            set_active_provider,
        )

        save_provider_api_key("openai", "sk-openai-local-1234")
        save_provider_api_key("anthropic", "sk-anthropic-local-5678")
        set_active_provider("anthropic")
        delete_provider_api_key("openai")

        assert get_provider_api_key("openai") is None
        assert get_provider_api_key("anthropic") == "sk-anthropic-local-5678"
        assert get_active_assistant_provider() == "anthropic"

    def test_environment_still_wins_after_local_delete(self, tmp_theriad_dir, monkeypatch):
        from theriad.settings.secrets import (
            delete_provider_api_key,
            get_provider_api_key,
            save_provider_api_key,
        )

        save_provider_api_key("openai", "sk-openai-local-1234")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-openai-env-5678")
        delete_provider_api_key("openai")

        assert get_provider_api_key("openai") == "sk-openai-env-5678"

    def test_aggregate_status_never_contains_plaintext_keys(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_provider_settings_status, save_provider_api_key

        keys = {
            "openrouter": "sk-or-secret-1234",
            "openai": "sk-openai-secret-5678",
            "anthropic": "sk-anthropic-secret-9012",
        }
        for provider, key in keys.items():
            save_provider_api_key(provider, key)

        status = get_provider_settings_status()
        serialized = json.dumps(status)

        assert status["active_provider"] == "theriad"
        assert set(status["providers"]) == {
            "openrouter",
            "openai",
            "anthropic",
        }
        assert all(status["providers"][provider]["configured"] for provider in keys)
        assert all(key not in serialized for key in keys.values())

    @pytest.mark.parametrize("value", ["", "   ", "bad key", "bad\nkey", "bad\x00key"])
    def test_rejects_invalid_key_values(self, tmp_theriad_dir, value):
        from theriad.settings.secrets import save_provider_api_key

        with pytest.raises(ValueError):
            save_provider_api_key("openai", value)

    def test_rejects_unknown_provider(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_provider_key_status

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_provider_key_status("unknown")

    def test_short_key_is_fully_masked(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_provider_key_status, save_provider_api_key

        save_provider_api_key("openai", "abcd")
        assert get_provider_key_status("openai")["masked_key"] == "••••"


class TestActiveAssistantProvider:
    def test_defaults_to_theriad(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_active_assistant_provider

        assert get_active_assistant_provider() == "theriad"

    @pytest.mark.parametrize("provider", ["theriad", "openai", "anthropic"])
    def test_roundtrip(self, tmp_theriad_dir, provider):
        from theriad.settings.secrets import get_active_assistant_provider, set_active_provider

        assert set_active_provider(provider) == provider
        assert get_active_assistant_provider() == provider

    def test_openrouter_alias_normalizes_to_theriad(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_active_assistant_provider, set_active_provider

        assert set_active_provider("openrouter") == "theriad"
        assert get_active_assistant_provider() == "theriad"

    def test_rejects_unknown_value_without_changing_preference(self, tmp_theriad_dir):
        from theriad.settings.secrets import get_active_assistant_provider, set_active_provider

        set_active_provider("openai")
        with pytest.raises(ValueError, match="Unsupported assistant provider"):
            set_active_provider("other")
        assert get_active_assistant_provider() == "openai"

    @pytest.mark.parametrize("provider", ["groq", "deepinfra"])
    def test_removed_provider_is_not_selectable(self, tmp_theriad_dir, provider):
        from theriad.settings.secrets import get_provider_key_status, set_active_provider

        with pytest.raises(ValueError, match="Unsupported"):
            get_provider_key_status(provider)
        with pytest.raises(ValueError, match="Unsupported"):
            set_active_provider(provider)


# ── API endpoints ─────────────────────────────────────────────────────────────


class TestSettingsEndpoints:
    """Integration tests for the three /api/settings/* endpoints."""

    @pytest.fixture()
    def client(self, tmp_theriad_dir, monkeypatch):
        """FastAPI test client with secrets redirected to tmp dir."""
        # Patch secrets module paths before importing app
        import theriad.settings.secrets as sec_mod

        monkeypatch.setattr(sec_mod, "_THERIAD_DIR", tmp_theriad_dir)
        monkeypatch.setattr(sec_mod, "_SECRETS_FILE", tmp_theriad_dir / "secrets.json")

        from fastapi.testclient import TestClient

        from theriad.api.server import app

        return TestClient(app)

    def test_provider_status_missing(self, client, monkeypatch):
        # server.py calls load_dotenv() at import time, which may have loaded
        # OPENROUTER_API_KEY from .env.  Clear it explicitly here so the test
        # sees the "no key" state regardless of the local environment.
        monkeypatch.setenv("OPENROUTER_API_KEY", "")
        r = client.get("/api/settings/provider-status")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is False
        assert data["source"] == "missing"
        assert data["masked_key"] is None

    def test_save_key_returns_masked_status(self, client):
        r = client.post(
            "/api/settings/openrouter-key",
            json={"api_key": "sk-or-v1-testkey1234"},
        )
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is True
        assert data["masked_key"] is not None
        assert data["masked_key"] != "sk-or-v1-testkey1234"
        assert data["masked_key"].endswith("1234")

    def test_save_key_never_echoes_full_key(self, client):
        full = "sk-or-v1-verysecretvalue9999"
        r = client.post("/api/settings/openrouter-key", json={"api_key": full})
        assert full not in r.text

    def test_save_empty_key_returns_422(self, client):
        r = client.post("/api/settings/openrouter-key", json={"api_key": ""})
        assert r.status_code == 422

    def test_delete_key_returns_missing_status(self, client):
        # Save first
        client.post("/api/settings/openrouter-key", json={"api_key": "sk-or-v1-abc1234"})
        # Then delete
        r = client.delete("/api/settings/openrouter-key")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is False

    def test_provider_status_after_save(self, client):
        client.post("/api/settings/openrouter-key", json={"api_key": "sk-or-v1-abc5678"})
        r = client.get("/api/settings/provider-status")
        assert r.status_code == 200
        data = r.json()
        assert data["configured"] is True
        assert data["source"] == "local_settings"

    def test_env_var_source_when_env_set(self, client, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-v1-envonly")
        r = client.get("/api/settings/provider-status")
        assert r.status_code == 200
        data = r.json()
        assert data["source"] == "env"
        assert data["configured"] is True

    def test_execute_endpoint_uses_saved_key(self, client, monkeypatch):
        """
        /api/execute with allow_paid_calls=True should use the saved key.
        We mock the provider so no real HTTP call is made.
        """
        import theriad.api.server as srv
        from theriad.settings.secrets import save_openrouter_api_key

        save_openrouter_api_key("sk-or-v1-savedkey1234")

        captured = {}

        class FakeProvider:
            def __init__(self, api_key, **kwargs):
                captured["api_key"] = api_key
                self.name = "openrouter"

            async def complete(self, req):
                from theriad.core.providers.interfaces import CompletionResponse

                return CompletionResponse(
                    model="test/model",
                    content="hello",
                    prompt_tokens=5,
                    completion_tokens=3,
                    latency_ms=100.0,
                    provider="openrouter",
                    raw={},
                    finish_reason="stop",
                )

        monkeypatch.setattr(srv, "OpenRouterProvider", FakeProvider, raising=False)
        # Monkeypatch inside the execute handler's import scope
        import theriad.core.providers.openrouter as or_mod

        monkeypatch.setattr(or_mod, "OpenRouterProvider", FakeProvider)

        client.post(
            "/api/execute",
            json={
                "prompt": "hello",
                "allow_paid_calls": True,
            },
        )
        # Key should have been resolved from local settings
        assert captured.get("api_key") == "sk-or-v1-savedkey1234"
