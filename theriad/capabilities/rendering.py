"""Shared output contract for requests that need the HTML preview renderer."""
from __future__ import annotations

from .detector import CapabilityResult
from theriad.core.providers.interfaces import Message


_RENDERER_SYSTEM_INSTRUCTION = """RENDERABLE_ARTIFACT_OUTPUT:
The user requested a visual artifact. Create the artifact itself, not a design
brief, a wireframe handoff, or instructions for another tool. Return exactly
one complete, self-contained HTML document in a single ```html code fence.
Put all CSS in a <style> element and use inline SVG for icons, diagrams, charts,
or illustrations. Make the document responsive and usable on its own.

This document is rendered in an isolated preview with no network access. Do
not use external assets, CDNs, scripts, JSX, separate CSS files, or a Figma
handoff. Do not add prose before or after the HTML fence."""


def renderer_required(capability: CapabilityResult) -> bool:
    """Whether this prompt must produce a previewable visual artifact."""
    return bool(capability.capabilities.get("requires_artifact_preview"))


def renderer_system_instruction(capability: CapabilityResult) -> str:
    """Return the provider instruction only for a renderable-artifact request."""
    return _RENDERER_SYSTEM_INSTRUCTION if renderer_required(capability) else ""


def with_renderer_instruction(messages: list[Message], instruction: str) -> list[Message]:
    """Insert the renderer contract after caller-owned system instructions.

    Caller instructions retain their leading position while the artifact
    requirement reaches every selected provider and fallback.
    """
    if not instruction:
        return messages

    insertion_index = 0
    while insertion_index < len(messages) and messages[insertion_index].role == "system":
        insertion_index += 1
    return [
        *messages[:insertion_index],
        Message(role="system", content=instruction),
        *messages[insertion_index:],
    ]


def renderer_metadata(capability: CapabilityResult) -> dict[str, str | bool] | None:
    """Expose the selected output contract to API consumers without HTML parsing."""
    if not renderer_required(capability):
        return None
    return {
        "enabled": True,
        "renderer": "sandboxed_html_preview",
        "artifact_format": "self_contained_html",
        "output_contract": "single_html_code_fence",
    }
