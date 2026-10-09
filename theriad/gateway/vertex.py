"""Explicit, location-bound managed Google deployment construction."""

from __future__ import annotations

from .contracts import Deployment, managed_google_endpoint
from .releases import digest


def vertex_deployment(
    *,
    project,
    location,
    model,
    revision,
    context_window,
    max_output_tokens,
    input_price,
    output_price,
    priced_at,
    tiers,
    tools=True,
    schemas=False,
    vision=False,
    reasoning=False,
    reasoning_config=None,
    supported_parameters=None,
):
    if not project or not location or location == "global" or "/" in project or "/" in location:
        raise ValueError("a concrete project and non-global location are required")
    if (
        not model.startswith("google/")
        or not revision
        or revision in {"latest", "default", "unknown"}
    ):
        raise ValueError("an exact verified Google model revision is required")
    endpoint = managed_google_endpoint(project, location)
    identity = {
        "project": project,
        "location": location,
        "model": model,
        "revision": revision,
        "endpoint": endpoint,
        "reasoning_config": reasoning_config or {},
    }
    return Deployment(
        id="vertex-" + digest(identity)[:24],
        model=model,
        revision=revision,
        provider="vertex",
        endpoint=endpoint,
        project=project,
        region=location,
        tiers=tiers,
        context_window=context_window,
        max_output_tokens=max_output_tokens,
        input_price=input_price,
        output_price=output_price,
        priced_at=priced_at,
        tools=tools,
        schemas=schemas,
        vision=vision,
        reasoning=reasoning,
        reasoning_config=reasoning_config or {},
        supported_parameters=supported_parameters,
        origin="US",
        serving_variant="google-managed",
        hosting_provider="google",
    )
