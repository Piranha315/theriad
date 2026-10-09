"""
Theriad runtime profiles and browser profile modes.

Runtime profiles control the trust level and capability set for the
current Theriad instance:

    personal_trusted  — local/dev machine owned by the operator.
                        Higher-authority browser access is allowed only by
                        explicit opt-in.  Hard-stop actions still require
                        approval.

    consumer_secure   — default.  Ordinary users or public-facing builds.
                        Isolated/ephemeral browser, strict approval gates,
                        narrow MCP tool allowlists.

Browser profile modes determine how Chrome/Chromium sessions are launched:

    ephemeral            — temporary profile cleared after session (required
                           for consumer_secure).
    dedicated_persistent — Theriad-owned persistent profile at
                           ~/.theriad/chrome-profile (personal_trusted only).
    real_profile         — user's actual Chrome profile (personal_trusted only,
                           requires explicit opt-in).

Configuration (environment variables)
--------------------------------------
    THERIAD_RUNTIME_PROFILE        personal_trusted | consumer_secure
    THERIAD_AGENT_APPROVAL_MODE    minimal | strict
    THERIAD_ALLOW_LOCAL_FILES      true | false
    THERIAD_ALLOW_REAL_CHROME_PROFILE  true | false
    THERIAD_CHROME_PROFILE_MODE    ephemeral | dedicated_persistent | real_profile
    THERIAD_CHROME_USER_DATA_DIR   filesystem path (optional)
    THERIAD_CHROME_HEADLESS        true | false
    THERIAD_CHROME_MCP_ENABLED     true | false
    THERIAD_CHROME_MCP_COMMAND     e.g. npx
    THERIAD_CHROME_MCP_ARGS        comma-separated, e.g. -y,chrome-devtools-mcp@latest

Validation rules (enforced by validate_runtime_config)
-------------------------------------------------------
    consumer_secure  forces browser_profile_mode=ephemeral,
                     allow_real_chrome_profile=False, approval_mode=strict,
                     and rejects any real_profile setting.

    real_profile     requires personal_trusted + allow_real_chrome_profile=True.

    dedicated_persistent  creates/uses ~/.theriad/chrome-profile and must
                          not point to the user's default Chrome profile.

Security invariants
-------------------
    - Never switch silently into personal_trusted.
    - Default (no config) is always consumer_secure.
    - Packaged/consumer builds always default to consumer_secure.
    - Secrets (API keys, cookies, tokens) are never stored in this config.
"""
from __future__ import annotations

import logging
import os
from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── Enums ─────────────────────────────────────────────────────────────────────


class RuntimeProfile(str, Enum):
    """Trust level for the current Theriad instance."""

    personal_trusted = "personal_trusted"
    """
    Local machine owned by the operator.  Higher-authority browser access
    may be enabled by explicit opt-in.  Hard-stop actions still require
    approval.
    """

    consumer_secure = "consumer_secure"
    """
    Default.  Ordinary users or public-facing builds.  Isolated browser,
    strict approval gates, narrow MCP tool allowlists.
    """


class BrowserProfileMode(str, Enum):
    """How the browser session is launched."""

    ephemeral = "ephemeral"
    """Temporary profile cleared after session.  Required for consumer_secure."""

    dedicated_persistent = "dedicated_persistent"
    """Theriad-owned persistent profile (~/.theriad/chrome-profile)."""

    real_profile = "real_profile"
    """User's actual Chrome profile.  personal_trusted only, explicit opt-in."""


class ApprovalMode(str, Enum):
    strict = "strict"
    """All sensitive actions require explicit approval (default)."""

    minimal = "minimal"
    """Fewer approval prompts — only hard-stop actions require approval."""


# ── Hard-stop actions (always require approval in both profiles) ───────────────

HARD_STOP_ACTIONS: frozenset[str] = frozenset({
    "purchase",
    "checkout",
    "send_message",
    "send_email",
    "delete_local_files",
    "delete_cloud_data",
    "change_account_settings",
    "submit_legal_form",
    "submit_financial_form",
    "submit_medical_form",
    "submit_job_form",
    "submit_school_form",
    "trading_action",
    "investment_action",
    "expose_api_key",
    "expose_secret",
    "install_software",
    "grant_permission",
    "upload_sensitive_file",
})

# ── Consumer-secure extra approval actions ─────────────────────────────────────
# Actions that consumer_secure requires approval for (in addition to hard-stops).

CONSUMER_SECURE_APPROVAL_ACTIONS: frozenset[str] = frozenset({
    "login",
    "submit_form",
    "send_message",
    "purchase",
    "checkout",
    "delete",
    "edit_account_data",
    "upload_file",
    "download_file",
    "enter_sensitive_info",
    "grant_permission",
    "access_financial_portal",
    "access_medical_portal",
    "access_legal_portal",
    "use_local_filesystem",
    "connect_external_account",
}).union(HARD_STOP_ACTIONS)


# ── Runtime configuration model ───────────────────────────────────────────────


class RuntimeConfig(BaseModel):
    """
    Full runtime configuration for one Theriad process.

    Constructed by load_runtime_config() from environment variables.
    Validated by validate_runtime_config() which enforces cross-field rules.

    Never store secrets (API keys, tokens, passwords) here.
    """

    runtime_profile: RuntimeProfile = Field(
        default=RuntimeProfile.consumer_secure,
        description="Trust level for this instance. Default: consumer_secure.",
    )

    browser_profile_mode: BrowserProfileMode = Field(
        default=BrowserProfileMode.ephemeral,
        description=(
            "Deprecated compatibility field. OpenSandbox always owns an isolated "
            "ephemeral Chromium session."
        ),
    )

    approval_mode: ApprovalMode = Field(
        default=ApprovalMode.strict,
        description="Approval gate strictness. Default: strict.",
    )

    allow_local_files: bool = Field(
        default=False,
        description="Whether the agent may access local files without prompting.",
    )

    allow_real_chrome_profile: bool = Field(
        default=False,
        description=(
            "Whether to allow real_profile mode. "
            "Requires personal_trusted + explicit opt-in."
        ),
    )

    chrome_user_data_dir: Optional[str] = Field(
        default=None,
        description="Explicit path for Chrome user data dir (overrides auto-detection).",
    )

    chrome_headless: bool = Field(
        default=False,
        description="Whether to run Chrome headless.",
    )

    chrome_mcp_enabled: bool = Field(
        default=False,
        description="Whether Chrome MCP is enabled.",
    )

    chrome_mcp_command: str = Field(
        default="npx",
        description="Command to launch Chrome MCP server.",
    )

    chrome_mcp_args: list[str] = Field(
        default_factory=lambda: ["-y", "chrome-devtools-mcp@latest"],
        description="Args for the Chrome MCP server command.",
    )

    chrome_mcp_attach_mode: str = Field(
        default="isolated",
        description=(
            "Chrome DevTools MCP connection mode: isolated (temporary profile), "
            "auto_connect (user-approved active Chrome), or user_data_dir."
        ),
    )

    # ── Browser action / observation backends (Part 7) ────────────────────────
    # Controls which backend executes browser actions and observations.
    # Allowed values: "mock" | "playwright" | "chrome_mcp"
    # consumer_secure default: action=mock (must be explicitly enabled),
    #                          observation=playwright if available else mock.
    # personal_trusted: both may be set to playwright or chrome_mcp.
    browser_action_backend: str = Field(
        default="opensandbox",
        description="Canonical browser action backend: OpenSandbox.",
    )

    browser_observation_backend: str = Field(
        default="opensandbox",
        description="Canonical browser observation backend: OpenSandbox.",
    )

    # ── Cross-field validation ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _enforce_profile_rules(self) -> "RuntimeConfig":
        """Apply cross-field rules based on runtime_profile."""
        if self.browser_action_backend != "opensandbox" or self.browser_observation_backend != "opensandbox":
            raise ValueError(
                "Legacy browser backends are deprecated; use browser backend 'opensandbox'."
            )
        if self.runtime_profile == RuntimeProfile.consumer_secure:
            # consumer_secure always forces these — no exceptions.
            object.__setattr__(self, "browser_profile_mode", BrowserProfileMode.ephemeral)
            object.__setattr__(self, "allow_real_chrome_profile", False)
            object.__setattr__(self, "approval_mode", ApprovalMode.strict)

        # These profile fields are retained as migration metadata for the
        # legacy Chrome MCP surface. They never alter the active OpenSandbox
        # browser, which is isolated regardless of this configuration.
        if self.browser_profile_mode == BrowserProfileMode.real_profile:
            if self.runtime_profile != RuntimeProfile.personal_trusted:
                raise ValueError(
                    "browser_profile_mode=real_profile requires "
                    "runtime_profile=personal_trusted."
                )
            if not self.allow_real_chrome_profile:
                raise ValueError(
                    "browser_profile_mode=real_profile requires "
                    "allow_real_chrome_profile=true."
                )

        return self

    # ── Convenience helpers ───────────────────────────────────────────────────

    @property
    def is_personal_trusted(self) -> bool:
        return self.runtime_profile == RuntimeProfile.personal_trusted

    @property
    def is_consumer_secure(self) -> bool:
        return self.runtime_profile == RuntimeProfile.consumer_secure

    @property
    def effective_chrome_user_data_dir(self) -> Optional[str]:
        """
        Return the Chrome user data dir to use, or None for ephemeral.

        For ephemeral: None (caller creates a tmp dir and clears it).
        For dedicated_persistent: ~/.theriad/chrome-profile.
        For real_profile: self.chrome_user_data_dir (must be set explicitly).
        """
        if self.browser_profile_mode == BrowserProfileMode.ephemeral:
            return None
        if self.browser_profile_mode == BrowserProfileMode.dedicated_persistent:
            custom = self.chrome_user_data_dir
            return custom or str(Path.home() / ".theriad" / "chrome-profile")
        # real_profile — must be explicitly configured
        return self.chrome_user_data_dir

    @property
    def profile_badge(self) -> str:
        """Short UI badge string for the current profile."""
        if self.runtime_profile == RuntimeProfile.personal_trusted:
            if self.browser_profile_mode == BrowserProfileMode.real_profile:
                return (
                    "Personal Trusted Mode — legacy real Chrome profile setting "
                    "retained; OpenSandbox remains the active isolated browser."
                )
            if self.browser_profile_mode == BrowserProfileMode.dedicated_persistent:
                return (
                    "Personal Trusted Mode — legacy dedicated Chrome profile setting "
                    "retained; OpenSandbox remains the active isolated browser."
                )
            return "Personal Trusted Mode — OpenSandbox isolated browser active."
        return "Secure Mode — isolated browser and approval gates enabled."

    def requires_approval_for(self, action_type: str) -> bool:
        """
        Return True if the given action_type requires explicit user approval
        under the current profile.

        Hard-stop actions always return True regardless of profile.
        """
        action = action_type.lower()
        if action in HARD_STOP_ACTIONS:
            return True
        if self.runtime_profile == RuntimeProfile.consumer_secure:
            return action in CONSUMER_SECURE_APPROVAL_ACTIONS
        # personal_trusted + minimal approval — only hard-stops (handled above)
        if self.approval_mode == ApprovalMode.minimal:
            return False
        # personal_trusted + strict — same hard-stops
        return False

    def to_safe_dict(self) -> dict:
        """
        Return a dict safe to include in API responses.

        Never includes secrets, tokens, file contents, or env vars.
        """
        return {
            "runtime_profile": self.runtime_profile.value,
            "browser_profile_mode": self.browser_profile_mode.value,
            "approval_mode": self.approval_mode.value,
            "allow_local_files": self.allow_local_files,
            "allow_real_chrome_profile": self.allow_real_chrome_profile,
            "chrome_headless": self.chrome_headless,
            "chrome_mcp_enabled": self.chrome_mcp_enabled,
            "chrome_mcp_attach_mode": self.chrome_mcp_attach_mode,
            "profile_badge": self.profile_badge,
            "browser_action_backend": self.browser_action_backend,
            "browser_observation_backend": self.browser_observation_backend,
            "browser_runtime": "opensandbox",
        }


# ── Env var loading ───────────────────────────────────────────────────────────

_ENV_RUNTIME_PROFILE            = "THERIAD_RUNTIME_PROFILE"
_ENV_APPROVAL_MODE              = "THERIAD_AGENT_APPROVAL_MODE"
_ENV_ALLOW_LOCAL_FILES          = "THERIAD_ALLOW_LOCAL_FILES"
_ENV_ALLOW_REAL_CHROME_PROFILE  = "THERIAD_ALLOW_REAL_CHROME_PROFILE"
_ENV_CHROME_PROFILE_MODE        = "THERIAD_CHROME_PROFILE_MODE"
_ENV_CHROME_USER_DATA_DIR       = "THERIAD_CHROME_USER_DATA_DIR"
_ENV_CHROME_HEADLESS            = "THERIAD_CHROME_HEADLESS"
_ENV_CHROME_MCP_ENABLED         = "THERIAD_CHROME_MCP_ENABLED"
_ENV_CHROME_MCP_COMMAND         = "THERIAD_CHROME_MCP_COMMAND"
_ENV_CHROME_MCP_ARGS            = "THERIAD_CHROME_MCP_ARGS"
_ENV_CHROME_MCP_ATTACH_MODE     = "THERIAD_CHROME_MCP_ATTACH_MODE"
_ENV_BROWSER_ACTION_BACKEND     = "THERIAD_BROWSER_ACTION_BACKEND"
_ENV_BROWSER_OBSERVATION_BACKEND = "THERIAD_BROWSER_OBSERVATION_BACKEND"

# Legacy env var still honoured for Chrome MCP
_ENV_CHROME_MCP_ENABLED_LEGACY  = "CHROME_MCP_ENABLED"

_VALID_BROWSER_BACKENDS = frozenset({"opensandbox"})
_VALID_CHROME_MCP_ATTACH_MODES = frozenset({"isolated", "auto_connect", "user_data_dir"})


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in ("1", "true", "yes")


def load_runtime_config() -> RuntimeConfig:
    """
    Build RuntimeConfig from environment variables.

    Default (no env vars set) → consumer_secure with ephemeral browser.
    Never silently promotes to personal_trusted — must be explicit.

    Validation rules are applied inside RuntimeConfig's model_validator so
    conflicting settings (e.g. consumer_secure + real_profile) raise
    ValueError before the config object is returned.
    """
    raw_profile = os.environ.get(_ENV_RUNTIME_PROFILE, "").strip().lower()
    if raw_profile == RuntimeProfile.personal_trusted.value:
        profile = RuntimeProfile.personal_trusted
        logger.warning(
            "[runtime_profile] personal_trusted mode active — "
            "high-authority local browser access enabled."
        )
    else:
        # Default and any unrecognised value → consumer_secure
        if raw_profile and raw_profile != RuntimeProfile.consumer_secure.value:
            logger.warning(
                "[runtime_profile] unrecognised THERIAD_RUNTIME_PROFILE=%r — "
                "defaulting to consumer_secure.",
                raw_profile,
            )
        profile = RuntimeProfile.consumer_secure

    raw_approval = os.environ.get(_ENV_APPROVAL_MODE, "").strip().lower()
    approval = (
        ApprovalMode.minimal
        if raw_approval == ApprovalMode.minimal.value
        else ApprovalMode.strict
    )

    allow_local = _parse_bool(os.environ.get(_ENV_ALLOW_LOCAL_FILES, "false"))
    # Legacy profile/MCP settings remain parseable for status and migration
    # clients. OpenSandbox never consumes the local profile path or attaches
    # to the user's Chrome process.
    allow_real_chrome = _parse_bool(
        os.environ.get(_ENV_ALLOW_REAL_CHROME_PROFILE, "false")
    )
    raw_browser_mode = os.environ.get(_ENV_CHROME_PROFILE_MODE, "").strip().lower()
    if raw_browser_mode == BrowserProfileMode.real_profile.value:
        browser_mode = BrowserProfileMode.real_profile
    elif raw_browser_mode == BrowserProfileMode.dedicated_persistent.value:
        browser_mode = BrowserProfileMode.dedicated_persistent
    else:
        browser_mode = BrowserProfileMode.ephemeral
    chrome_user_data_dir = os.environ.get(_ENV_CHROME_USER_DATA_DIR, "").strip() or None
    chrome_headless = _parse_bool(os.environ.get(_ENV_CHROME_HEADLESS, "false"))

    chrome_mcp_raw = os.environ.get(_ENV_CHROME_MCP_ENABLED)
    if chrome_mcp_raw is None:
        chrome_mcp_raw = os.environ.get(_ENV_CHROME_MCP_ENABLED_LEGACY, "")
    chrome_mcp_raw = chrome_mcp_raw.strip().lower()
    chrome_mcp_enabled = chrome_mcp_raw in ("1", "true", "yes")
    chrome_mcp_command = os.environ.get(_ENV_CHROME_MCP_COMMAND, "npx").strip() or "npx"
    raw_args = os.environ.get(_ENV_CHROME_MCP_ARGS, "").strip()
    chrome_mcp_args = (
        [arg.strip() for arg in raw_args.split(",") if arg.strip()]
        if raw_args
        else ["-y", "chrome-devtools-mcp@latest"]
    )
    raw_attach_mode = os.environ.get(
        _ENV_CHROME_MCP_ATTACH_MODE, "isolated"
    ).strip().lower()
    chrome_mcp_attach_mode = (
        raw_attach_mode
        if raw_attach_mode in _VALID_CHROME_MCP_ATTACH_MODES
        else "isolated"
    )

    # Browser action/observation backends (Part 7)
    # Env var takes precedence; fall back to the user-saved preference in
    # the scoped Theriad settings store so the Settings UI works without env vars.
    raw_action_backend = os.environ.get(_ENV_BROWSER_ACTION_BACKEND, "").strip().lower()
    if not raw_action_backend:
        from theriad.settings.secrets import get_browser_backend as _get_saved_backend
        raw_action_backend = _get_saved_backend()
    if raw_action_backend and raw_action_backend not in _VALID_BROWSER_BACKENDS:
        logger.warning(
            "[runtime_profile] deprecated browser backend %r ignored; using OpenSandbox.",
            raw_action_backend,
        )
    browser_action_backend = "opensandbox"
    raw_obs_backend = os.environ.get(_ENV_BROWSER_OBSERVATION_BACKEND, "").strip().lower()
    if raw_obs_backend and raw_obs_backend not in _VALID_BROWSER_BACKENDS:
        logger.warning(
            "[runtime_profile] deprecated browser observation backend %r ignored; using OpenSandbox.",
            raw_obs_backend,
        )
    browser_observation_backend = "opensandbox"

    try:
        config = RuntimeConfig(
            runtime_profile=profile,
            browser_profile_mode=browser_mode,
            approval_mode=approval,
            allow_local_files=allow_local,
            allow_real_chrome_profile=allow_real_chrome,
            chrome_user_data_dir=chrome_user_data_dir,
            chrome_headless=chrome_headless,
            chrome_mcp_enabled=chrome_mcp_enabled,
            chrome_mcp_command=chrome_mcp_command,
            chrome_mcp_args=chrome_mcp_args,
            chrome_mcp_attach_mode=chrome_mcp_attach_mode,
            browser_action_backend=browser_action_backend,
            browser_observation_backend=browser_observation_backend,
        )
    except ValueError as exc:
        logger.error(
            "[runtime_profile] invalid configuration: %s — "
            "falling back to consumer_secure defaults.",
            exc,
        )
        config = RuntimeConfig()  # safe defaults

    logger.info(
        "[runtime_profile] loaded profile=%s browser_mode=%s approval=%s",
        config.runtime_profile.value,
        config.browser_profile_mode.value,
        config.approval_mode.value,
    )
    return config


# ── Module-level singleton (lazy, re-loadable in tests) ───────────────────────

_cached_config: Optional[RuntimeConfig] = None


def get_runtime_config(*, reload: bool = False) -> RuntimeConfig:
    """
    Return the process-wide RuntimeConfig singleton.

    Pass reload=True to force re-reading env vars (useful in tests).
    """
    global _cached_config
    if _cached_config is None or reload:
        _cached_config = load_runtime_config()
    return _cached_config
