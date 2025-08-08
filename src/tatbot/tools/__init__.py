"""Unified tools module for tatbot operations and utilities.

This module provides a clean, decorator-based approach to defining tools that can be
executed via MCP across multiple nodes. It replaces the previous split between
mcp/handlers and ops modules with a unified architecture.

Key features:
- Decorator-based tool registration (@tool)
- Node-specific tool availability 
- Async generator pattern for progress reporting
- Type-safe input/output validation with Pydantic
- Auto-discovery and registration of tools
"""

from tatbot.tools.registry import get_tools_for_node, register_all_tools, tool

__all__ = ["tool", "get_tools_for_node", "register_all_tools"]

# Auto-register all tools when module is imported
register_all_tools()