"""Pydantic models for system tools."""

from typing import List, Optional

from tatbot.tools.base import ToolInput, ToolOutput


class PingNodesInput(ToolInput):
    """Input for ping_nodes tool."""
    nodes: Optional[List[str]] = None


class PingNodesOutput(ToolOutput):
    """Output for ping_nodes tool."""
    details: List[str]
    all_success: bool


class ListScenesInput(ToolInput):
    """Input for list_scenes tool."""
    pass


class ListScenesOutput(ToolOutput):
    """Output for list_scenes tool."""
    scenes: List[str]
    count: int


class ListNodesInput(ToolInput):
    """Input for list_nodes tool."""
    pass


class ListNodesOutput(ToolOutput):
    """Output for list_nodes tool."""
    nodes: List[str]
    count: int


class ListOpsInput(ToolInput):
    """Input for list_ops tool."""
    node_name: Optional[str] = None


class ListOpsOutput(ToolOutput):
    """Output for list_ops tool."""
    ops: List[str]
    count: int
    node_name: Optional[str] = None