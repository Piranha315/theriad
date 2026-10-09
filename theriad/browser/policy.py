"""Deterministic policy shared by browser bridge adapters."""
from __future__ import annotations

from dataclasses import dataclass


SAFE_LOW_RISK = frozenset({
    "navigate", "observe", "read_page", "scroll", "click", "click_link",
    "click_button", "open_page", "screenshot", "get_status", "go_back", "wait",
    "press_key",
})
APPROVAL_REQUIRED = frozenset({
    "type", "fill", "submit", "submit_form", "login", "send_message", "send_email",
    "purchase", "payment", "delete", "edit_data", "upload", "download", "install",
    "access_sensitive_account", "reveal_secret", "copy_secret",
})
BLOCKED = frozenset({
    "bypass_captcha", "bypass_paywall", "bypass_access_control",
    "credential_theft", "unauthorized_scrape", "exfiltrate_secrets",
    "destructive_system_action", "destructive_account_action",
})


@dataclass(frozen=True)
class BrowserPolicyDecision:
    status: str
    reason: str

    @property
    def allowed(self) -> bool:
        return self.status == "success"


class BrowserPolicy:
    """Classify an action without consulting page content or a model."""

    def classify(self, action: str, *, approved: bool = False) -> BrowserPolicyDecision:
        normalized = action.strip().lower().replace(" ", "_")
        if normalized in BLOCKED:
            return BrowserPolicyDecision("blocked", f"Action '{normalized}' is blocked by browser policy.")
        if normalized in APPROVAL_REQUIRED and not approved:
            return BrowserPolicyDecision(
                "approval_required",
                f"Action '{normalized}' requires explicit user approval.",
            )
        if normalized in SAFE_LOW_RISK or (normalized in APPROVAL_REQUIRED and approved):
            return BrowserPolicyDecision("success", f"Action '{normalized}' is allowed.")
        return BrowserPolicyDecision(
            "approval_required",
            f"Unknown browser action '{normalized}' requires explicit approval.",
        )
