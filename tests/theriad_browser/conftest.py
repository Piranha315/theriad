"""
Conftest for theriad_browser integration tests.

These tests were copied from Theriad and exercise theriad_browser-specific
endpoints and modules not yet fully wired into the Theriad server:

  - OpenAI-compat routes (/v1/models, /v1/chat/completions) not yet mounted
  - Browser session route schema differs from test fixtures
  - Theriad Browser extension path references need updating to apps/theriad-extension/src/
  - theriad.execution / theriad.context_layer are Theriad thin-shim modules

All collected tests are auto-marked xfail until integration is complete (Task #2).
"""
import pytest

collect_ignore = [
    "test_captcha_and_search.py",   # needs theriad.execution
    "test_chat_agents.py",           # needs theriad.context_layer
    "test_context_layer.py",         # needs theriad.context_layer
    "test_execute_flow.py",          # needs theriad.execution
    "test_ui_contract.py",           # needs static/index.html (built frontend)
]


def pytest_collection_modifyitems(config, items):
    """Mark all theriad_browser tests as xfail pending full route integration."""
    xfail_mark = pytest.mark.xfail(
        reason="theriad_browser route integration incomplete — pending Task #2 browser integration",
        strict=False,
    )
    for item in items:
        if "tests/theriad_browser" in str(item.fspath):
            item.add_marker(xfail_mark)
