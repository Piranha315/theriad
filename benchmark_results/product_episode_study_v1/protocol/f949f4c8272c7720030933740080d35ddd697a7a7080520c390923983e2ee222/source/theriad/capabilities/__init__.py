"""
Capability detection for the Theriad routing pipeline.

Determines *what kind of execution* a prompt requires before the model/tier
router runs.  The capability layer is strictly rule-based and deterministic —
no model calls, no embeddings.

Public API
----------
    from theriad.capabilities.detector import CapabilityDetector, CapabilityResult

    result = CapabilityDetector().detect("Find buntings near Bethesda in stock")
    result.recommended_execution_mode  # "browser_agent"
    result.capabilities["requires_browser_navigation"]  # True
"""
from theriad.capabilities.detector import CapabilityDetector, CapabilityResult

__all__ = ["CapabilityDetector", "CapabilityResult"]
