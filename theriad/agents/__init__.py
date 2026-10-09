"""
theriad.agents — optional browser-agent execution mode.

This package is intentionally isolated from the core /api/execute path.
Importing this package will NOT fail if browser-use is not installed;
individual modules defer that check to runtime.
"""
