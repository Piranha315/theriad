"""
Shared display formatters for benchmark output.
"""
from __future__ import annotations


def format_usd(cost: float | None) -> str:
    """
    Format a USD cost value for terminal display.

    States:
        None           → "?"           (pricing unavailable / model unknown)
        0.0 exactly    → "$0"          (known true zero, e.g. Ollama)
        0 < x < 1e-6   → "<$0.000001" (sub-microdollar, avoids misleading zeros)
        1e-6 ≤ x < 0.01 → adaptive 8-digit precision, trailing zeros stripped
        x ≥ 0.01       → "$x.xxxx"    (4 decimal places)
    """
    if cost is None:
        return "?"
    if cost == 0.0:
        return "$0"
    if cost < 0.000001:
        return "<$0.000001"
    if cost < 0.01:
        return f"${cost:.8f}".rstrip("0").rstrip(".")
    return f"${cost:.4f}"
