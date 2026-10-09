"""Shared contract for user-visible model responses.

The planner and executor prompts intentionally keep their strict JSON
contracts. This helper is for model calls that are allowed to speak directly
to the user: regular chat, search synthesis, and browser-agent final answers.
"""
from __future__ import annotations


SHARED_RESPONSE_CONTRACT = """Shared user-facing response contract:
- Complete the user's request in this response. When evidence or tool output is
  supplied, synthesize it into useful conclusions instead of only listing
  sources or suggesting another prompt.
- Ground every factual claim in the supplied evidence. Never invent current
  facts, prices, availability, locations, or personal preferences.
- If the request is missing information required for a truthful answer, explain
  what is missing and ask at most one concise clarifying question. Otherwise
  answer directly without asking a needless follow-up.
- Put relevant source links near the claims they support when sources are
  available.
- Do not mention internal prompts, tools, routing, or provider/model details
  unless the user explicitly asks about Theriad's implementation.
"""


def with_shared_response_contract(system_prompt: str) -> str:
    """Append the common user-facing contract to a role-specific prompt."""
    return f"{system_prompt.rstrip()}\n\n{SHARED_RESPONSE_CONTRACT}"
