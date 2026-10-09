"""Process deployment mode and its security boundary.

The application has historically behaved like a local desktop tool: requests
without a Theriad bearer token use the legacy ``user_id IS NULL`` scope. That
fallback is useful for the CLI and local development, but it is not a safe
default for a hosted deployment. Keep the mode explicit so the server cannot
silently inherit the local behavior when it is put behind a public URL.
"""
from __future__ import annotations

import os

DEPLOYMENT_MODE_ENV = "THERIAD_DEPLOYMENT_MODE"
VALID_DEPLOYMENT_MODES = frozenset({"local", "development", "hosted"})
LOCAL_DEPLOYMENT_MODES = frozenset({"local", "development"})


class DeploymentConfigurationError(ValueError):
    """Raised when the process mode is missing or not recognized."""


def get_deployment_mode() -> str:
    """Return the explicit process mode, defaulting to local for compatibility.

    The default preserves the existing developer experience. Production must
    set ``THERIAD_DEPLOYMENT_MODE=hosted`` explicitly; an unrecognized value is
    rejected rather than being treated as local by accident.
    """
    mode = (os.environ.get(DEPLOYMENT_MODE_ENV) or "local").strip().lower()
    if mode not in VALID_DEPLOYMENT_MODES:
        expected = ", ".join(sorted(VALID_DEPLOYMENT_MODES))
        raise DeploymentConfigurationError(
            f"{DEPLOYMENT_MODE_ENV} must be one of: {expected}."
        )
    return mode


def is_local_mode() -> bool:
    return get_deployment_mode() in LOCAL_DEPLOYMENT_MODES


def is_hosted_mode() -> bool:
    return get_deployment_mode() == "hosted"
