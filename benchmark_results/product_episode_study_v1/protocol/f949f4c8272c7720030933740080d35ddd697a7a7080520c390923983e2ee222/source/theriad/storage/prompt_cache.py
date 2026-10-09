"""Stable, locally cached system-prompt assembly."""

from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=256)
def cached_system_prompt(
    base_prompt: str,
    project_instructions: str = "",
    enabled_skills: str = "",
) -> str:
    """Build a stable prefix once so provider prompt caches see identical text."""
    parts = [base_prompt.strip()]
    if project_instructions.strip():
        parts.append("Project instructions:\n" + project_instructions.strip())
    if enabled_skills.strip():
        parts.append("Enabled skills:\n" + enabled_skills.strip())
    return "\n\n".join(parts)


def prompt_cache_extra(provider_name: str) -> dict:
    """Request automatic/explicit caching where the provider supports it."""
    if provider_name in {"openrouter", "anthropic"}:
        return {"cache_control": {"type": "ephemeral"}}
    return {}
