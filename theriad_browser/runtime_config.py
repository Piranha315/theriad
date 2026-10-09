"""Runtime profile and browser profile validation for Theriad."""
from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from dotenv import load_dotenv as _load_dotenv
_load_dotenv(Path(__file__).resolve().parent.parent / ".env")


class RuntimeProfile(str, Enum):
    PERSONAL_TRUSTED = "personal_trusted"
    CONSUMER_SECURE = "consumer_secure"


class ApprovalMode(str, Enum):
    MINIMAL = "minimal"
    STRICT = "strict"


class BrowserProfileMode(str, Enum):
    EPHEMERAL = "ephemeral"
    DEDICATED_PERSISTENT = "dedicated_persistent"
    REAL_PROFILE = "real_profile"


HARD_STOP_ACTIONS = {
    "purchase", "checkout", "send_message", "delete_local_files",
    "delete_cloud_data", "edit_account_data", "submit_legal_form",
    "submit_financial_form", "submit_medical_form", "submit_school_form",
    "submit_job_form", "trading_action", "expose_secret", "install_software",
    "grant_permission", "upload_sensitive_file",
}

CONSUMER_APPROVAL_ACTIONS = {
    "login", "submit_form", "send_message", "purchase", "checkout", "delete",
    "edit_account_data", "upload_file", "download_file", "enter_sensitive_info",
    "grant_permission", "access_financial_portal", "access_medical_portal",
    "access_legal_portal", "use_local_filesystem", "connect_external_accounts",
} | HARD_STOP_ACTIONS

REAL_PROFILE_MARKERS = (
    ".config/google-chrome", ".config/chromium", "Library/Application Support/Google/Chrome",
    "AppData/Local/Google/Chrome/User Data", "Google/Chrome/User Data",
)


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


@dataclass
class RuntimeConfig:
    runtime_profile: RuntimeProfile = RuntimeProfile.CONSUMER_SECURE
    approval_mode: ApprovalMode = ApprovalMode.STRICT
    browser_profile_mode: BrowserProfileMode = BrowserProfileMode.EPHEMERAL
    allow_local_files: bool = False
    allow_real_chrome_profile: bool = False
    real_profile_confirmation: bool = False
    chrome_user_data_dir: str | None = None
    chrome_headless: bool = False
    chrome_mcp_enabled: bool = False
    chrome_mcp_command: str = "npx"
    chrome_mcp_args: list[str] = field(default_factory=lambda: ["-y", "chrome-devtools-mcp@latest"])
    effective_user_data_dir: str | None = None
    warnings: list[str] = field(default_factory=list)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "RuntimeConfig":
        env = env or os.environ
        profile = RuntimeProfile(env.get("THERIAD_RUNTIME_PROFILE", RuntimeProfile.CONSUMER_SECURE))
        args = env.get("THERIAD_CHROME_MCP_ARGS", "-y,chrome-devtools-mcp@latest")
        return cls(
            runtime_profile=profile,
            approval_mode=ApprovalMode(env.get("THERIAD_AGENT_APPROVAL_MODE", ApprovalMode.STRICT)),
            browser_profile_mode=BrowserProfileMode(env.get("THERIAD_CHROME_PROFILE_MODE", BrowserProfileMode.EPHEMERAL)),
            allow_local_files=_bool(env.get("THERIAD_ALLOW_LOCAL_FILES")),
            allow_real_chrome_profile=_bool(env.get("THERIAD_ALLOW_REAL_CHROME_PROFILE")),
            real_profile_confirmation=_bool(env.get("THERIAD_REAL_CHROME_PROFILE_CONFIRMATION")),
            chrome_user_data_dir=env.get("THERIAD_CHROME_USER_DATA_DIR") or None,
            chrome_headless=_bool(env.get("THERIAD_CHROME_HEADLESS")),
            chrome_mcp_enabled=_bool(env.get("THERIAD_CHROME_MCP_ENABLED")),
            chrome_mcp_command=env.get("THERIAD_CHROME_MCP_COMMAND", "npx"),
            chrome_mcp_args=[part for part in args.split(",") if part],
        ).validated()

    def validated(self) -> "RuntimeConfig":
        cfg = RuntimeConfig(**{k: v for k, v in asdict(self).items() if k != "warnings"})
        cfg.runtime_profile = RuntimeProfile(cfg.runtime_profile)
        cfg.approval_mode = ApprovalMode(cfg.approval_mode)
        cfg.browser_profile_mode = BrowserProfileMode(cfg.browser_profile_mode)
        cfg.warnings = list(self.warnings)
        if cfg.runtime_profile == RuntimeProfile.CONSUMER_SECURE:
            if cfg.browser_profile_mode != BrowserProfileMode.EPHEMERAL:
                cfg.warnings.append("consumer_secure forces browser_profile_mode=ephemeral")
            cfg.browser_profile_mode = BrowserProfileMode.EPHEMERAL
            cfg.allow_real_chrome_profile = False
            cfg.approval_mode = ApprovalMode.STRICT
            cfg.allow_local_files = False
            if cfg.chrome_user_data_dir and looks_like_real_profile(cfg.chrome_user_data_dir):
                cfg.warnings.append("consumer_secure rejected a real Chrome profile path")
                cfg.chrome_user_data_dir = None
        if cfg.browser_profile_mode == BrowserProfileMode.REAL_PROFILE:
            if cfg.runtime_profile != RuntimeProfile.PERSONAL_TRUSTED:
                raise ValueError("real_profile requires personal_trusted")
            if not (cfg.allow_real_chrome_profile and cfg.real_profile_confirmation and cfg.chrome_user_data_dir):
                raise ValueError("real_profile requires explicit config, allow flag, confirmation, and userDataDir")
        if cfg.browser_profile_mode == BrowserProfileMode.DEDICATED_PERSISTENT:
            cfg.effective_user_data_dir = str(Path.home() / ".theriad" / "chrome-profile")
        elif cfg.browser_profile_mode == BrowserProfileMode.EPHEMERAL:
            cfg.effective_user_data_dir = tempfile.mkdtemp(prefix="theriad-chrome-")
        else:
            cfg.effective_user_data_dir = cfg.chrome_user_data_dir
        return cfg

    @property
    def mcp_allowlist(self) -> list[str]:
        if self.runtime_profile == RuntimeProfile.CONSUMER_SECURE:
            return ["browser.navigate", "browser.snapshot", "browser.click", "browser.type"]
        return ["browser.*", "filesystem.read_project", "shell.safe", "mcp.account_tools"]

    @property
    def approval_required_actions(self) -> set[str]:
        return HARD_STOP_ACTIONS if self.runtime_profile == RuntimeProfile.PERSONAL_TRUSTED else CONSUMER_APPROVAL_ACTIONS

    def cleanup(self) -> None:
        if self.browser_profile_mode == BrowserProfileMode.EPHEMERAL and self.effective_user_data_dir:
            shutil.rmtree(self.effective_user_data_dir, ignore_errors=True)

    def safe_status(self) -> dict[str, Any]:
        return {
            "runtime_profile": self.runtime_profile.value,
            "approval_mode": self.approval_mode.value,
            "browser_profile_mode": self.browser_profile_mode.value,
            "allow_real_chrome_profile": self.allow_real_chrome_profile,
            "allow_local_files": self.allow_local_files,
            "headless": self.chrome_headless,
            "chrome_mcp_enabled": self.chrome_mcp_enabled,
            "mcp_tools_enabled": self.mcp_allowlist,
            "warnings": self.warnings,
        }


def looks_like_real_profile(path: str) -> bool:
    normalized = str(Path(path).expanduser())
    return any(marker in normalized for marker in REAL_PROFILE_MARKERS)


def current_config() -> RuntimeConfig:
    return RuntimeConfig.from_env()
