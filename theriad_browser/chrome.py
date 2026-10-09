"""Chrome MCP status helpers."""
from __future__ import annotations

import shutil
from typing import Any

from .runtime_config import RuntimeConfig, current_config


def chrome_status(config: RuntimeConfig | None = None) -> dict[str, Any]:
    cfg = config or current_config()
    command_available = shutil.which(cfg.chrome_mcp_command) is not None
    configured = cfg.chrome_mcp_enabled
    available = configured and command_available
    message = "Chrome MCP is available." if available else "Chrome MCP is not configured or unavailable; mock browser-agent mode remains available."
    return {
        "configured": configured,
        "available": available,
        "runtime_profile": cfg.runtime_profile.value,
        "browser_profile_mode": cfg.browser_profile_mode.value,
        "allow_real_chrome_profile": cfg.allow_real_chrome_profile,
        "headless": cfg.chrome_headless,
        "mode": "chrome_mcp" if available else "mock",
        "message": message,
    }
