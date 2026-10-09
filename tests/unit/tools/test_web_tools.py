"""Canonical web tools shared by chat agents and autonomous tasks."""
from __future__ import annotations

from unittest.mock import patch

import pytest

from theriad.agents.tool_broker import ToolBroker
from theriad.search.service import SearchResult


@pytest.mark.asyncio
async def test_tool_broker_web_search_uses_canonical_service():
    broker = ToolBroker(project_root="/tmp")

    async def search(_service, query):
        assert query == "python async"
        return [
            SearchResult(
                title="Asyncio docs",
                url="https://docs.python.org/3/library/asyncio.html",
                snippet="Concurrent code with async/await.",
                source="duckduckgo",
            )
        ]

    with patch("theriad.search.service.SearchService.search", new=search):
        result = await broker.call("web.search", {"query": "python async"})

    assert result.ok is True
    assert result.output["provider"] == "duckduckgo"
    assert result.output["results"][0]["snippet"].startswith("Concurrent")


@pytest.mark.asyncio
async def test_tool_broker_web_search_returns_structured_failure():
    broker = ToolBroker(project_root="/tmp")

    async def fail(_service, _query):
        raise RuntimeError("providers unavailable")

    with patch("theriad.search.service.SearchService.search", new=fail):
        result = await broker.call("web.search", {"query": "test"})

    assert result.ok is False
    assert "web.search unavailable" in (result.error or "")


@pytest.mark.asyncio
async def test_tool_broker_web_search_requires_query():
    result = await ToolBroker(project_root="/tmp").call("web.search", {})
    assert result.ok is False
    assert "query is required" in (result.error or "")
