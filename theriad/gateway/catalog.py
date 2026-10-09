"""Select an initial concrete OpenRouter host; never infer a physical region."""

from .contracts import GatewayError


def initial_endpoint(model, catalog):
    endpoints = catalog.get("endpoints", [])
    candidates = [
        e
        for e in endpoints
        if e.get("tag")
        and e.get("quantization") not in {None, "unknown"}
        and isinstance(e.get("context_length"), int)
        and isinstance(e.get("max_completion_tokens"), int)
        and {"tools", "tool_choice"}.issubset(e.get("supported_parameters", []))
        and e["context_length"] >= 64000
        and e["max_completion_tokens"] >= 4096
    ]
    priced = []
    for endpoint in candidates:
        try:
            pricing = endpoint["pricing"]
            score = float(pricing["prompt"]) + float(pricing["completion"])
            if score >= 0:
                priced.append((score, endpoint["tag"], endpoint))
        except (KeyError, TypeError, ValueError):
            continue
    if not priced:
        raise GatewayError(
            "no_eligible_model", "No priced, verified agent-capable host for " + model, 422
        )
    return min(priced, key=lambda entry: entry[:2])[2]
