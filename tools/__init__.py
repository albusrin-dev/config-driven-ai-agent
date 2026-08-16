"""Tools layer: registry and built-in tools.

The contracts (Tool ABC, ToolContext, ToolResult) live in ``core.base`` so
that core never imports this package; they are re-exported here for
convenience.
"""

from core.base import Tool, ToolContext, ToolResult

from .registry import ToolRegistry

__all__ = ["Tool", "ToolContext", "ToolResult", "ToolRegistry"]
