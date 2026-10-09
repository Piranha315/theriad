"""
MCP tool broker layer for Theriad.

Provides a controlled gateway between Theriad subtasks and MCP servers
(e.g. Chrome DevTools MCP). Raw MCP tools are never exposed directly to
model prompts — every call goes through the broker with allowlist enforcement,
budget tracking, approval gates, and diagnostics.

Public surface
--------------
    from theriad.tools.mcp_broker import MCPBroker
    from theriad.tools.mcp_schemas import MCPCallRequest, MCPBrokerStatus
    from theriad.tools.mcp_registry import SUBTASK_ALLOWLISTS, get_allowlist
    from theriad.tools.chrome_devtools_adapter import ChromeDevToolsAdapter
    from theriad.tools.playwright_mcp_adapter import PlaywrightMCPAdapter
    from theriad.tools.scrapling_mcp_adapter import ScraplingMCPAdapter
    from theriad.tools.openrouter_mcp_adapter import OpenRouterMCPAdapter
    from theriad.tools.llm_mcp_adapter import LLMMCPAdapter, GroqMCPAdapter, DeepInfraMCPAdapter
"""
