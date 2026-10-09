"""
Browser-agent policy / action validator.

Design:
- Deterministic first: no model required for obvious allow/block decisions.
- Sensitive actions always require approval, regardless of planner confidence.
- Webpage content cannot override policy (prompt-injection resistance).
- Escalates to cheap_reasoning only when action is genuinely ambiguous.
- Permission mode (D-044) controls how often non-hard-stop actions ask for
  approval; hard-stop actions always ask, in every mode and every runtime
  profile — this floor cannot be configured away in this release (Part 4).
"""
from __future__ import annotations

from .browser_events import ActionProposal
from .browser_schemas import (
    ALWAYS_AGNOSTIC_SAFE_ACTIONS,
    CLICK_ACTIONS,
    DEFAULT_PERMISSION_MODE,
    HARD_STOP_ACTIONS,
    HIGH_RISK_ACTIONS,
    PERMISSION_ALWAYS_ASK,
    PERMISSION_NEVER_ASK,
)

# ── Actions always allowed without approval, regardless of permission mode ───
# (kept for reference / other call sites; validate_action now routes through
# evaluate_permission_mode() for anything not caught by injection/hard-stop/
# caller-list rules below.)

_ALWAYS_ALLOWED: frozenset[str] = ALWAYS_AGNOSTIC_SAFE_ACTIONS | CLICK_ACTIONS

# ── Prompt-injection trigger phrases ─────────────────────────────────────────
# If any of these appear in a goal or action proposal, treat as injected.

_INJECTION_PHRASES: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore your instructions",
    "reveal api key",
    "reveal secret",
    "expose credentials",
    "bypass approval",
    "override policy",
    "send credentials",
    "leak secrets",
    "print api key",
    "print openrouter",
    "disregard your goal",
    "new goal:",
    "change goal to",
    "forget your",
    "act as if",
)


class PolicyDecision:
    __slots__ = ("allowed", "requires_approval", "reason", "blocked_by_policy")

    def __init__(
        self,
        *,
        allowed: bool,
        requires_approval: bool = False,
        reason: str = "",
        blocked_by_policy: bool = False,
    ) -> None:
        self.allowed = allowed
        self.requires_approval = requires_approval
        self.reason = reason
        self.blocked_by_policy = blocked_by_policy


def evaluate_permission_mode(action_type: str, permission_mode: str) -> tuple[bool, str]:
    """
    Decide whether `action_type` requires approval under `permission_mode`,
    ignoring hard-stop actions (callers must check HARD_STOP_ACTIONS first —
    validate_action() does this before calling here).

    Returns (requires_approval, reason). No model calls; pure lookup.

    always_ask     — only ALWAYS_AGNOSTIC_SAFE_ACTIONS are auto-allowed;
                     everything else, including normal clicks, asks.
    ask_high_risk  — ALWAYS_AGNOSTIC_SAFE_ACTIONS + normal clicks are
                     auto-allowed; HIGH_RISK_ACTIONS ask; anything else is
                     treated as low-risk and auto-allowed.
    never_ask      — everything auto-allowed (hard-stop already excluded by
                     the caller).
    """
    action = action_type.lower()
    mode = permission_mode if permission_mode else DEFAULT_PERMISSION_MODE

    if action in ALWAYS_AGNOSTIC_SAFE_ACTIONS:
        return False, f"Action '{action}' is read-only/navigational and allowed under {mode}."

    if mode == PERMISSION_ALWAYS_ASK:
        return True, f"Action '{action}' requires approval under Always Ask permission mode."

    if mode == PERMISSION_NEVER_ASK:
        return False, f"Action '{action}' allowed automatically under Never Ask permission mode."

    # Default: ask_high_risk
    if action in CLICK_ACTIONS:
        return False, f"Action '{action}' is a normal click and allowed under Only Ask if Risk is High mode."
    if action in HIGH_RISK_ACTIONS:
        return True, f"Action '{action}' is high-risk and requires approval under Only Ask if Risk is High mode."
    return False, f"Action '{action}' is treated as low-risk and allowed under Only Ask if Risk is High mode."


def validate_action(
    proposal: ActionProposal,
    approval_required_actions: list[str],
    page_content: str = "",
    *,
    permission_mode: str = DEFAULT_PERMISSION_MODE,
) -> PolicyDecision:
    """
    Deterministic policy check.  Returns a PolicyDecision with no model calls.

    Rules (in priority order):
    1. Prompt-injection in page content → block.
    2. Hard-stop actions → always require approval, in every permission mode
       and every runtime profile. Not configurable in this release (Part 4;
       "never_ask must not bypass hard-stop actions").
    3. If action_type is in the caller-supplied approval_required_actions
       (an optional, explicit, extra safety list) → require approval.
    4. Permission-mode policy (see evaluate_permission_mode) → decides for
       everything else based on the user's selected permission_mode.
    5. If proposal.requires_approval flag is set (planner-level signal for
       action types our static classification doesn't recognize) → require
       approval, defense-in-depth.
    6. Otherwise → allow.
    """
    action = proposal.action_type.lower()

    # Rule 1: Prompt-injection guard (page content is untrusted)
    if _contains_injection(page_content):
        return PolicyDecision(
            allowed=False,
            blocked_by_policy=True,
            reason="Prompt-injection attempt detected in page content.",
        )

    # Rule 2: Hard-stop actions — unconditional, regardless of permission_mode
    if action in HARD_STOP_ACTIONS:
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            reason=f"Action '{action}' is a hard-stop action and always requires approval, "
                   f"regardless of permission mode.",
        )

    # Rule 3: Caller-supplied extra approval list (defense-in-depth / opt-in)
    if action in approval_required_actions:
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            reason=f"Action '{action}' requires user approval per session policy.",
        )

    # Rule 4: Permission-mode policy
    requires, reason = evaluate_permission_mode(action, permission_mode)
    if requires:
        return PolicyDecision(allowed=True, requires_approval=True, reason=reason)

    # Rule 5: Planner flagged it as requiring approval
    if proposal.requires_approval:
        return PolicyDecision(
            allowed=True,
            requires_approval=True,
            reason="Planner flagged action as requiring approval.",
        )

    # Rule 6: Allow
    return PolicyDecision(allowed=True, reason=reason or "Action passed policy check.")


def _contains_injection(text: str) -> bool:
    if not text:
        return False
    lower = text.lower()
    return any(phrase in lower for phrase in _INJECTION_PHRASES)


def detect_injection_in_goal(goal: str) -> bool:
    """Check user-supplied goal for prompt-injection attempts."""
    return _contains_injection(goal)
