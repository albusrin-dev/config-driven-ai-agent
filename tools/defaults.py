"""The default toolset: every built-in tool, registered once.

Both clients (CLI and web server) get their registry from here, so the two
can never drift into offering different capabilities. What an individual
agent may use is still decided by its allowlist — this is the shelf, not
the permission (Rule 3).
"""

from __future__ import annotations

from .builtins.documents import ReadDocxTool, ReadPdfTool
from .builtins.files import DeleteFileTool, ReadFileTool, WriteFileTool
from .builtins.web import WebFetchTool, WebSearchTool
from .registry import ToolRegistry


def build_registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(DeleteFileTool())
    registry.register(ReadPdfTool())
    registry.register(ReadDocxTool())
    registry.register(WebSearchTool())
    registry.register(WebFetchTool())
    return registry
