"""
Event types for the browser-agent execution loop.

Events are plain dataclasses (not Pydantic) because they live on the
hot path inside the runtime loop, not on the API surface.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ObservationEvent:
    """Result of observing the current page state."""
    url: str
    title: str
    observation_type: str          # "dom" | "screenshot" | "both" | "none"
    visible_text: str = ""
    clickable_elements: list[dict[str, str]] = field(default_factory=list)
    forms: list[dict[str, Any]] = field(default_factory=list)
    aria_labels: list[str] = field(default_factory=list)
    screenshot_description: Optional[str] = None
    raw_dom_snippet: Optional[str] = None
    captcha_detected: bool = False
    captcha_reason: Optional[str] = None


@dataclass
class ActionProposal:
    """Proposed browser action from the planner."""
    action_type: str               # navigate | click | scroll | extract | search | etc.
    target: Optional[str]
    reason: str
    confidence: float              # 0.0–1.0
    risk_level: str                # low | medium | high | critical
    requires_approval: bool
    fallback_action: Optional[str] = None


@dataclass
class ActionResult:
    """Outcome of executing a browser action."""
    action_type: str
    target: Optional[str]
    url: Optional[str]
    success: bool
    result_summary: str
    blocked_by_policy: bool = False
    approval_requested: bool = False
    error: Optional[str] = None
